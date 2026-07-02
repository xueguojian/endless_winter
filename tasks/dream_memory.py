from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from loguru import logger

from core.adb_client import AdbClient
from core.dream_memory.config import (
    DreamMemoryConfig,
    PK_ITEM_FILTER_LABELS,
    load_dream_memory_config,
    normalize_pk_item_filter,
    pk_item_filter_matches,
    sample_tap_between_delay,
)
from core.dream_memory.maps import DreamMemoryMap, load_map
from core.dream_memory.misclick import PseudoRandomMisclickScheduler
from core.dream_memory.ocr_engine import ocr_engine_available, resolve_ocr_engine, warmup_ocr
from core.dream_memory.vision import chip_is_active, read_target_chips, resolve_item_coord

StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class _BatchTap:
    slot_index: int
    text: str
    x: int
    y: int


class _PKTapQueue:
    """PK 扫描/点击解耦：扫描追加识别结果，点击线程持续消费。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: collections.deque[_BatchTap] = collections.deque()

    def extend(self, items: list[_BatchTap]) -> int:
        if not items:
            return 0
        added = 0
        with self._lock:
            existing = {(item.text, item.x, item.y) for item in self._items}
            for item in items:
                key = (item.text, item.x, item.y)
                if key in existing:
                    continue
                self._items.append(item)
                existing.add(key)
                added += 1
        return added

    def pop(self) -> _BatchTap | None:
        with self._lock:
            if not self._items:
                return None
            return self._items.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class DreamMemorySession:
    """用户手动进入关卡后，循环识别底栏并点击物品，直到 stop。"""

    name = "寻梦记忆"

    def __init__(
        self,
        adb: AdbClient,
        game_map: DreamMemoryMap,
        *,
        config: DreamMemoryConfig | None = None,
        on_status: StatusCallback | None = None,
    ):
        self.adb = adb
        self.game_map = game_map
        self.config = config or load_dream_memory_config()
        self.on_status = on_status
        self._stop_event = threading.Event()
        if self.config.pk_mode:
            self.name = "寻梦记忆PK"
        self._misclick: PseudoRandomMisclickScheduler | None = None
        if self.config.enable_misclick:
            self._misclick = PseudoRandomMisclickScheduler(
                interval_min=self.config.misclick_interval_min,
                interval_max=self.config.misclick_interval_max,
                center_x=self.config.misclick_center_x,
                center_y=self.config.misclick_center_y,
                radius_x=self.config.misclick_radius_x,
                radius_y=self.config.misclick_radius_y,
            )

    def stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

    def _interrupted(self) -> bool:
        return self._stop_event.is_set()

    def _emit(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
        if self.on_status:
            self.on_status(message)

    def _map_keys(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (*self.game_map.items.keys(), *self.game_map.aliases.keys())
            )
        )

    def _slot_patch(self, screen, slot_index: int):
        if slot_index >= len(self.config.target_slots):
            return np.array([])
        x1, y1, x2, y2 = self.config.target_slots[slot_index]
        h, w = screen.shape[:2]
        return screen[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]

    @staticmethod
    def _patch_mean(patch) -> float:
        if patch.size == 0:
            return 0.0
        gray = patch
        if patch.ndim == 3:
            gray = patch.mean(axis=2)
        return float(gray.mean())

    def _slot_fingerprints(self, screen, batch: list[_BatchTap]) -> dict[int, float]:
        return {item.slot_index: self._patch_mean(self._slot_patch(screen, item.slot_index)) for item in batch}

    def _lookup_coord(self, label: str) -> tuple[int, int] | None:
        if self.config.pk_mode:
            return self.game_map.lookup_strict(label)
        return resolve_item_coord(self.game_map, label)

    def _scan_batch(self, screen) -> list[_BatchTap]:
        chips = read_target_chips(
            screen,
            self.config.target_slots,
            map_keys=self._map_keys(),
            map_aliases=self.game_map.aliases,
            tesseract_cmd=self.config.tesseract_cmd,
            ocr_engine=self.config.ocr_engine,
            min_brightness=self.config.chip_active_min_brightness,
            refs_dir=self.config.chip_refs_dir,
            template_min_score=self.config.chip_template_min_score,
            template_min_margin=self.config.chip_template_min_margin,
            fuzzy_min_ratio=self.config.chip_fuzzy_min_ratio,
            pk_mode=self.config.pk_mode,
        )
        active_count = sum(1 for chip in chips if chip.active and chip.text)
        if self.config.pk_mode and active_count:
            logger.debug(f"PK 当前亮槽 {active_count}/{len(chips)}")

        batch: list[_BatchTap] = []
        for chip in sorted(chips, key=lambda c: c.slot_index):
            if not chip.active or not chip.text:
                if self.config.pk_mode and chip.active and not chip.text:
                    logger.debug(f"PK 槽位 {chip.slot_index + 1} 有内容但未识别，跳过")
                continue
            coord = self._lookup_coord(chip.text)
            if coord is None:
                if self.config.pk_mode:
                    logger.debug(
                        f"槽位 {chip.slot_index + 1} OCR「{chip.text}」— 未匹配地图，跳过"
                    )
                else:
                    self._emit(
                        f"槽位 {chip.slot_index + 1} OCR「{chip.text}」"
                        f" — 未匹配地图，请标定"
                    )
                continue
            if self.config.pk_mode and self._pk_item_filter_active():
                ordinal = self.game_map.item_ordinal_index(chip.text)
                if ordinal is None or not pk_item_filter_matches(
                    ordinal, self.config.pk_item_filter
                ):
                    logger.debug(
                        f"PK 物品「{chip.text}」为第 {ordinal} 个标定点，"
                        f"分工={PK_ITEM_FILTER_LABELS.get(normalize_pk_item_filter(self.config.pk_item_filter), '全部')}，跳过"
                    )
                    continue
            x, y = coord
            batch.append(_BatchTap(chip.slot_index, chip.text, x, y))
        return batch

    def _bar_changed(
        self,
        screen,
        batch: list[_BatchTap],
        before_means: dict[int, float],
    ) -> bool:
        cleared = 0
        changed = 0
        delta_floor = self.config.bar_change_mean_delta
        for item in batch:
            patch = self._slot_patch(screen, item.slot_index)
            if patch.size == 0:
                continue
            if not chip_is_active(
                patch,
                min_brightness=self.config.chip_active_min_brightness,
            ):
                cleared += 1
                continue
            before = before_means.get(item.slot_index, 0.0)
            if abs(self._patch_mean(patch) - before) >= delta_floor:
                changed += 1
        return cleared > 0 or changed > 0

    def _wait_bar_refresh(self, batch: list[_BatchTap], before_means: dict[int, float]) -> None:
        """点完一批后等待底栏变灰或换目标，避免同一批 OCR 连点两轮。"""
        if not batch:
            return
        floor = max(0.12, self.config.bar_refresh_min_wait)
        time.sleep(floor)
        deadline = time.time() + self.config.bar_refresh_timeout
        poll = max(0.05, self.config.bar_refresh_poll)
        while time.time() < deadline:
            if self._interrupted():
                return
            try:
                screen = self.adb.screenshot()
            except Exception:
                time.sleep(poll)
                continue
            if self._bar_changed(screen, batch, before_means):
                logger.debug("底栏已刷新")
                return
            time.sleep(poll)
        logger.debug("等待底栏刷新超时，继续下一轮")

    def _fire_misclick_if_due(self, normal_click_count: int) -> None:
        if self._misclick is None or normal_click_count <= 0 or self._interrupted():
            return
        if not self._misclick.register_normal_clicks(normal_click_count):
            return
        x, y = self._misclick.sample_point()
        self._emit(f"误点 ({x},{y})")
        self.adb.tap(x, y)
        time.sleep(0.15)

    def _click_batch(self, batch: list[_BatchTap], *, before_means: dict[int, float]) -> int:
        """普通模式：同一批识别结果连续点击。"""
        labels = "、".join(item.text for item in batch)
        self._emit(f"本批 {len(batch)} 个: {labels}")

        clicked = 0
        for index, item in enumerate(batch):
            if self._interrupted():
                break
            self._emit(f"点击「{item.text}」@ ({item.x},{item.y})")
            self.adb.tap(item.x, item.y)
            clicked += 1
            if index < len(batch) - 1:
                time.sleep(sample_tap_between_delay(self.config))

        if clicked:
            self._wait_bar_refresh(batch, before_means)
            self._fire_misclick_if_due(clicked)
        return clicked

    def _pk_item_key(self, item: _BatchTap) -> str:
        return self.game_map.resolve_label(item.text) or item.text

    def _pk_filter_new_items(
        self,
        batch: list[_BatchTap],
        seen: set[str],
    ) -> list[_BatchTap]:
        """PK 每个物品整局只出现一次，已记录过的不再入队。"""
        fresh: list[_BatchTap] = []
        for item in batch:
            key = self._pk_item_key(item)
            if key in seen:
                logger.debug(f"PK 重复扫描「{key}」，跳过入队")
                continue
            seen.add(key)
            fresh.append(item)
        return fresh

    def _pk_item_filter_active(self) -> bool:
        return normalize_pk_item_filter(self.config.pk_item_filter) != "all"

    def _pk_scan_loop(self, queue: _PKTapQueue, seen: set[str]) -> None:
        """PK 扫描线程：定时 OCR，仅首次见到的物品入队。"""
        while not self._interrupted():
            try:
                screen = self.adb.screenshot()
            except Exception as exc:
                self._emit(f"截图失败: {exc}")
                time.sleep(0.5)
                continue

            batch = self._scan_batch(screen)
            fresh = self._pk_filter_new_items(batch, seen)
            if fresh:
                added = queue.extend(fresh)
                if added:
                    logger.info(
                        f"PK 扫描入队 +{added}（新 {len(fresh)}/{len(batch)}，"
                        f"已见 {len(seen)}，队列 {len(queue)}）"
                    )

            interval = max(0.0, self.config.scan_interval)
            if interval > 0:
                deadline = time.time() + interval
                while time.time() < deadline:
                    if self._interrupted():
                        return
                    time.sleep(min(0.05, deadline - time.time()))

    def _pk_click_loop(self, queue: _PKTapQueue) -> None:
        """PK 点击线程：持续从队列取目标点击，与扫描无关。"""
        while not self._interrupted():
            item = queue.pop()
            if item is None:
                time.sleep(0.01)
                continue
            self._emit(f"点击「{item.text}」@ ({item.x},{item.y})")
            self.adb.tap(item.x, item.y)
            delay = sample_tap_between_delay(self.config)
            if delay > 0:
                time.sleep(delay)

    def _run_pk_dual_loop(self) -> None:
        queue = _PKTapQueue()
        seen: set[str] = set()
        click_thread = threading.Thread(
            target=self._pk_click_loop,
            args=(queue,),
            name="dream-pk-click",
            daemon=True,
        )
        click_thread.start()
        try:
            self._pk_scan_loop(queue, seen)
        finally:
            click_thread.join(timeout=3.0)

    def run_until_stopped(self) -> None:
        if not ocr_engine_available(self.config.ocr_engine):
            engine = resolve_ocr_engine(self.config.ocr_engine)
            if engine == "rapidocr":
                raise FileNotFoundError(
                    "未安装 RapidOCR，请运行:\n"
                    "  .venv\\Scripts\\pip.exe install rapidocr-onnxruntime onnxruntime"
                )
            raise FileNotFoundError(
                f"未找到 Tesseract: {self.config.tesseract_cmd}\n"
                "安装: https://github.com/UB-Mannheim/tesseract/wiki"
            )
        if not self.game_map.items:
            raise ValueError(
                f"地图「{self.game_map.name}」尚无标定物品，"
                f"请运行 tools/calibrate_dream_memory_map.py"
            )

        engine = resolve_ocr_engine(self.config.ocr_engine)
        warmup_ocr(self.config.ocr_engine)
        mode_hint = "PK·队列" if self.config.pk_mode else "普通"
        misclick_hint = "·含误点" if self.config.enable_misclick else ""
        pk_hint = (
            f"·扫描/点击解耦（扫描 {self.config.scan_interval}s，连点 {self.config.tap_between_delay}s）"
            if self.config.pk_mode
            else ""
        )
        filter_hint = ""
        if self.config.pk_mode:
            filter_mode = normalize_pk_item_filter(self.config.pk_item_filter)
            if filter_mode != "all":
                filter_hint = f"·分工={PK_ITEM_FILTER_LABELS.get(filter_mode, filter_mode)}"
        self._emit(
            f"开始({mode_hint}{misclick_hint}{pk_hint}{filter_hint}) — 地图「{self.game_map.name}」"
            f"（{len(self.game_map.items)} 个标定物品，"
            f"识别区 {len(self.config.target_slots)} 槽，OCR={engine}）"
        )

        if self.config.pk_mode:
            self._run_pk_dual_loop()
            self._emit("已结束")
            return

        while not self._interrupted():
            try:
                screen = self.adb.screenshot()
            except Exception as exc:
                self._emit(f"截图失败: {exc}")
                time.sleep(0.5)
                continue

            batch = self._scan_batch(screen)
            if not batch:
                time.sleep(self.config.scan_interval)
                continue

            before_means = self._slot_fingerprints(screen, batch)
            self._click_batch(batch, before_means=before_means)
            if self._interrupted():
                break

        self._emit("已结束")
