from __future__ import annotations

import collections
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.dream_memory.config import (
    DEFAULT_PK_TARGET_BAR,
    DreamMemoryConfig,
    TURBO_PK_SCAN_FAST,
    TURBO_PK_SCAN_SLOW,
    format_tap_interval_hint,
    load_dream_memory_config,
    sample_tap_between_delay,
)
from core.dream_memory.maps import DreamMemoryMap, load_map
from core.dream_memory.misclick import PseudoRandomMisclickScheduler
from core.dream_memory.ocr_engine import (
    ocr_chip_text,
    ocr_engine_available,
    resolve_ocr_engine,
    warmup_ocr,
)
from core.dream_memory.vision import (
    read_target_chips,
    resolve_item_coord,
    split_bar_grid_slots,
)
from core.window_capture import WindowCapture, touch_roi_to_client_xywh

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
        turbo_pk: bool = False,
        window_hwnd: int | None = None,
        turbo_bar_roi: tuple[int, int, int, int] | None = None,
        turbo_scan_fast: float = TURBO_PK_SCAN_FAST,
        turbo_scan_slow: float = TURBO_PK_SCAN_SLOW,
    ):
        self.adb = adb
        self.game_map = game_map
        self.config = config or load_dream_memory_config()
        self.on_status = on_status
        self._stop_event = threading.Event()
        self.turbo_pk = bool(turbo_pk) and bool(self.config.pk_mode)
        self._turbo_capture: WindowCapture | None = None
        self._turbo_bar_roi: tuple[int, int, int, int] | None = None
        self._turbo_scan_fast = max(0.0, float(turbo_scan_fast))
        self._turbo_scan_slow = max(0.0, float(turbo_scan_slow))
        if self.turbo_pk:
            self.name = "极速寻梦PK"
            if window_hwnd is None:
                raise ValueError("极速寻梦PK 需要选择雷电窗口")
            self._turbo_capture = WindowCapture()
            self._turbo_capture.set_window(int(window_hwnd))
            if turbo_bar_roi is not None:
                x, y, w, h = (int(v) for v in turbo_bar_roi)
                if w <= 0 or h <= 0:
                    raise ValueError("极速寻梦PK 底栏 ROI 无效")
                self._turbo_bar_roi = (x, y, w, h)
            else:
                crect = self._turbo_capture.client_rect()
                self._turbo_bar_roi = touch_roi_to_client_xywh(
                    DEFAULT_PK_TARGET_BAR,
                    client_w=crect.width,
                    client_h=crect.height,
                )
        elif self.config.pk_mode:
            self.name = "寻梦记忆PK"
        self._unmatched_logged: set[str] = set()
        # 普通模式：刚点过的槽位短时抑制，防止划线未检出时连点两轮
        self._recent_taps: dict[int, tuple[str, float]] = {}
        self._recent_tap_ttl = 2.0
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

    def _warn_unmatched_map(self, slot_index: int, raw: str) -> None:
        """OCR 有字，但地图无此物品/无相似项 → 提示可能未标定。"""
        key = (raw or "").strip()
        if not key:
            return
        msg = (
            f"槽位 {slot_index + 1} OCR「{key}」未匹配地图（无相似项），"
            f"可能尚未标定"
        )
        if key not in self._unmatched_logged:
            self._unmatched_logged.add(key)
            logger.warning(f"[{self.name}] {msg}")
            self._emit(msg)
        else:
            logger.debug(f"[{self.name}] {msg}（已提示过）")

    def _map_keys(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (*self.game_map.items.keys(), *self.game_map.aliases.keys())
            )
        )

    def _lookup_coord(self, label: str) -> tuple[int, int] | None:
        if self.config.pk_mode:
            return self.game_map.lookup_strict(label)
        return resolve_item_coord(self.game_map, label)

    def _sleep_interruptible(self, seconds: float) -> bool:
        """可中断等待；返回 False 表示被 stop。"""
        deadline = time.time() + max(0.0, float(seconds))
        while time.time() < deadline:
            if self._interrupted():
                return False
            time.sleep(min(0.1, max(0.0, deadline - time.time())))
        return not self._interrupted()

    def _roi_center(self, roi: tuple[int, int, int, int]) -> tuple[int, int]:
        x1, y1, x2, y2 = roi
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _crop_roi(self, screen, roi: tuple[int, int, int, int]):
        x1, y1, x2, y2 = roi
        h, w = screen.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return screen[y1:y2, x1:x2]

    def _ocr_roi_text(
        self,
        screen,
        roi: tuple[int, int, int, int],
    ) -> str:
        patch = self._crop_roi(screen, roi)
        if patch is None or getattr(patch, "size", 0) == 0:
            return ""
        text, _engine = ocr_chip_text(
            patch,
            engine=self.config.ocr_engine,
            tesseract_cmd=self.config.tesseract_cmd,
        )
        return re.sub(r"\s+", "", text or "")

    def _ocr_aux_rois(
        self,
        screen,
        rois: dict[str, tuple[int, int, int, int]],
    ) -> dict[str, str]:
        """辅助按钮区一次截图、多 ROI 一起 OCR（不走底栏三槽逻辑）。"""
        keys = list(rois.keys())
        patches = []
        valid_keys: list[str] = []
        for key in keys:
            patch = self._crop_roi(screen, rois[key])
            if patch is None or getattr(patch, "size", 0) == 0:
                continue
            patches.append(patch)
            valid_keys.append(key)

        results = {key: "" for key in keys}
        if not patches:
            return results

        texts: list[str] = []
        if resolve_ocr_engine(self.config.ocr_engine) == "rapidocr":
            try:
                from core.dream_memory.ocr_rapid import ocr_slots_batch

                texts = ocr_slots_batch(patches)
            except Exception as exc:
                logger.warning(f"辅助区批量 OCR 失败，回退逐个: {exc}")
                texts = []

        if len(texts) != len(patches):
            texts = []
            for patch in patches:
                text, _ = ocr_chip_text(
                    patch,
                    engine=self.config.ocr_engine,
                    tesseract_cmd=self.config.tesseract_cmd,
                )
                texts.append(text)

        for key, text in zip(valid_keys, texts):
            results[key] = re.sub(r"\s+", "", text or "")
        logger.debug(f"辅助区 OCR: {results}")
        return results

    def _dismiss_continue_overlay(self) -> bool:
        """「您的奖励增加了」但无「继续」时，按回退关掉遮挡面板。"""
        self._emit("检测到奖励结算但无「继续」，按回退关闭遮挡…")
        self.adb.back()
        return self._sleep_interruptible(0.6)

    def _try_auto_advance(self) -> str:
        """过关推进监视循环：各辅助 ROI 每轮一起 OCR，按结果分支。

        返回:
          - ``started``：已进入关卡，应恢复底栏 OCR
          - ``stop_chapter``：出现「开启章节/领取」，应结束任务
          - ``stopped``：用户手动停止
        """
        self._emit(
            f"连续 {self.config.empty_rounds_before_advance} 次未识别到底栏目标，"
            f"暂停底栏 OCR，批量检查过关界面…"
        )
        wait = max(0.0, float(self.config.advance_wait_sec))
        aux_rois = {
            "continue": tuple(int(v) for v in self.config.continue_btn_roi),
            "chapter": tuple(int(v) for v in self.config.chapter_btn_roi),
            "start_tip": tuple(int(v) for v in self.config.start_tip_roi),
            "claim": tuple(int(v) for v in self.config.claim_btn_roi),
            "reward_title": tuple(int(v) for v in self.config.reward_title_roi),
        }

        while not self._interrupted():
            try:
                screen = self.adb.screenshot()
            except Exception as exc:
                self._emit(f"截图失败: {exc}")
                if not self._sleep_interruptible(0.5):
                    return "stopped"
                continue

            texts = self._ocr_aux_rois(screen, aux_rois)
            continue_text = texts.get("continue", "")
            chapter_text = texts.get("chapter", "")
            tip_text = texts.get("start_tip", "")
            claim_text = texts.get("claim", "")
            reward_title = texts.get("reward_title", "")
            has_reward_title = (
                "您的奖励增加了" in reward_title or "奖励增加了" in reward_title
            )
            has_continue = "继续" in continue_text

            # 1) 结束类：领取 / 开启章节
            if "领取" in claim_text:
                self._emit(f"检测到「领取」（{claim_text}），自动结束")
                return "stop_chapter"
            if "开启章节" in chapter_text:
                self._emit(f"检测到「开启章节」（{chapter_text}），自动结束")
                return "stop_chapter"

            # 2) 已在开局提示界面 → 点提示区，立刻恢复 OCR
            if "点击任意位置开始" in tip_text or "任意位置开始" in tip_text:
                tx, ty = self._roi_center(aux_rois["start_tip"])
                self._emit(f"检测到「点击任意位置开始」，点击 @ ({tx},{ty})")
                self.adb.tap(tx, ty)
                self._recent_taps.clear()
                self._emit("已进入关卡，立即恢复底栏识别")
                return "started"

            # 3) 大厅「开始游戏」→ 点击后等动画，下一轮再一起 OCR
            if "开始游戏" in chapter_text:
                bx, by = self._roi_center(aux_rois["chapter"])
                self._emit(f"检测到「开始游戏」，点击 @ ({bx},{by})")
                self.adb.tap(bx, by)
                if not self._sleep_interruptible(wait):
                    return "stopped"
                continue

            # 4) 小关结算：先看标题「您的奖励增加了」
            if has_reward_title:
                if has_continue:
                    bx, by = self._roi_center(aux_rois["continue"])
                    self._emit(f"检测到「继续」，点击 @ ({bx},{by})")
                    self.adb.tap(bx, by)
                    if not self._sleep_interruptible(wait):
                        return "stopped"
                    continue
                # 有奖励标题但没有继续 → 被遮挡
                if not self._dismiss_continue_overlay():
                    return "stopped"
                continue

            # 5) 无奖励标题时仍看到继续（兜底）
            if has_continue:
                bx, by = self._roi_center(aux_rois["continue"])
                self._emit(f"检测到「继续」，点击 @ ({bx},{by})")
                self.adb.tap(bx, by)
                if not self._sleep_interruptible(wait):
                    return "stopped"
                continue

            if not self._sleep_interruptible(1.0):
                return "stopped"

        return "stopped"

    def _grab_pk_frame(self) -> tuple[object, tuple[tuple[int, int, int, int], ...]]:
        """返回 (图像, 槽位ROI)。极速模式抓窗口底栏并均分六格。"""
        if self.turbo_pk and self._turbo_capture is not None and self._turbo_bar_roi is not None:
            x, y, w, h = self._turbo_bar_roi
            bar = self._turbo_capture.grab_region(x, y, w, h)
            if bar is None or getattr(bar, "size", 0) == 0:
                raise RuntimeError("窗口抓屏为空（窗口是否最小化？）")
            height, width = bar.shape[:2]
            slots = split_bar_grid_slots(width, height, rows=2, cols=3)
            return bar, slots
        return self.adb.screenshot(), self.config.target_slots

    def _scan_batch(
        self,
        screen,
        slots: tuple[tuple[int, int, int, int], ...] | None = None,
    ) -> list[_BatchTap]:
        chips = read_target_chips(
            screen,
            slots if slots is not None else self.config.target_slots,
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
        now = time.time()
        for chip in sorted(chips, key=lambda c: c.slot_index):
            if not chip.active:
                self._recent_taps.pop(chip.slot_index, None)
                continue
            raw = (chip.ocr_raw or chip.text or "").strip()
            if not chip.text:
                if raw:
                    self._warn_unmatched_map(chip.slot_index, raw)
                elif self.config.pk_mode:
                    logger.debug(f"槽位 {chip.slot_index + 1} 有内容但未识别，跳过")
                continue
            # 普通模式：刚点过且文字未变 → 视为划线未检出，跳过
            if not self.config.pk_mode:
                recent = self._recent_taps.get(chip.slot_index)
                if recent is not None:
                    recent_text, recent_ts = recent
                    if now - recent_ts > self._recent_tap_ttl:
                        self._recent_taps.pop(chip.slot_index, None)
                    elif recent_text == chip.text:
                        logger.debug(
                            f"槽位 {chip.slot_index + 1}「{chip.text}」刚点过，"
                            f"跳过（防重复）"
                        )
                        continue
                    else:
                        # 槽位已换成新目标
                        self._recent_taps.pop(chip.slot_index, None)
            coord = self._lookup_coord(chip.text)
            if coord is None:
                self._warn_unmatched_map(chip.slot_index, raw or chip.text)
                continue
            x, y = coord
            batch.append(_BatchTap(chip.slot_index, chip.text, x, y))
        return batch

    def _fire_misclick_if_due(self, normal_click_count: int) -> None:
        if self._misclick is None or normal_click_count <= 0 or self._interrupted():
            return
        if not self._misclick.register_normal_clicks(normal_click_count):
            return
        x, y = self._misclick.sample_point()
        self._emit(f"误点 ({x},{y})")
        self.adb.tap(x, y)
        time.sleep(0.15)

    def _click_batch(self, batch: list[_BatchTap]) -> int:
        """普通模式：截图识别后连点；已划线槽由 chip_is_active 跳过，不做点后确认。

        tap_delay：本批点完后等待再进入下一轮截图，给游戏划线/换目标时间。
        """
        labels = "、".join(item.text for item in batch)
        self._emit(f"本批 {len(batch)} 个: {labels}")

        clicked = 0
        for index, item in enumerate(batch):
            if self._interrupted():
                break
            self._emit(f"点击「{item.text}」@ ({item.x},{item.y})")
            self.adb.tap(item.x, item.y)
            if not self.config.pk_mode:
                self._recent_taps[item.slot_index] = (item.text, time.time())
            clicked += 1
            if index < len(batch) - 1:
                time.sleep(sample_tap_between_delay(self.config))

        if clicked:
            settle = max(0.0, float(self.config.tap_delay))
            if settle > 0:
                time.sleep(settle)
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

    def _pk_scan_loop(self, queue: _PKTapQueue, seen: set[str]) -> None:
        """PK 扫描线程：定时 OCR，仅首次见到的物品入队。"""
        while not self._interrupted():
            try:
                screen, slots = self._grab_pk_frame()
            except Exception as exc:
                self._emit(f"截图失败: {exc}")
                time.sleep(0.5)
                continue

            batch = self._scan_batch(screen, slots=slots)
            fresh = self._pk_filter_new_items(batch, seen)
            if fresh:
                added = queue.extend(fresh)
                if added:
                    logger.info(
                        f"PK 扫描入队 +{added}（新 {len(fresh)}/{len(batch)}，"
                        f"已见 {len(seen)}，队列 {len(queue)}）"
                    )

            if self.turbo_pk:
                # 未见字：0.1s 快扫；识别出文字后：0.5s
                interval = self._turbo_scan_fast if not seen else self._turbo_scan_slow
            elif not seen:
                # 普通 PK：首帧未识别到任何物品时不等待，连续重扫
                continue
            else:
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
        if self.turbo_pk:
            mode_hint = "极速PK·窗口抓屏"
            roi = self._turbo_bar_roi
            backend = self._turbo_capture.backend if self._turbo_capture else "?"
            pk_hint = (
                f"·ROI {roi}·{backend}"
                f"·快扫 {self._turbo_scan_fast:g}s / 慢扫 {self._turbo_scan_slow:g}s"
                f"·{format_tap_interval_hint(self.config)}"
            )
        elif self.config.pk_mode:
            mode_hint = "PK·队列"
            pk_hint = (
                f"·扫描/点击解耦（扫描 {self.config.scan_interval:g}s，"
                f"{format_tap_interval_hint(self.config)}）"
            )
        else:
            mode_hint = "普通"
            pk_hint = f"·{format_tap_interval_hint(self.config)}·扫描 {self.config.scan_interval:g}s"
            if self.config.auto_advance:
                pk_hint += (
                    f"·空识别{self.config.empty_rounds_before_advance}次后自动过关"
                )
        misclick_hint = "·含误点" if self.config.enable_misclick else ""
        self._emit(
            f"开始({mode_hint}{misclick_hint}{pk_hint}) — 地图「{self.game_map.name}」"
            f"（{len(self.game_map.items)} 个标定物品，"
            f"识别区 {len(self.config.target_slots)} 槽，OCR={engine}）"
        )

        try:
            if self.config.pk_mode:
                self._run_pk_dual_loop()
                self._emit("已结束")
                return

            empty_rounds = 0
            while not self._interrupted():
                try:
                    screen = self.adb.screenshot()
                except Exception as exc:
                    self._emit(f"截图失败: {exc}")
                    time.sleep(0.5)
                    continue

                batch = self._scan_batch(screen)
                if batch:
                    empty_rounds = 0
                    self._click_batch(batch)
                    if self._interrupted():
                        break
                    continue

                empty_rounds += 1
                if (
                    self.config.auto_advance
                    and empty_rounds >= int(self.config.empty_rounds_before_advance)
                ):
                    outcome = self._try_auto_advance()
                    if outcome == "stop_chapter":
                        break
                    if outcome == "stopped":
                        break
                    # started → 恢复底栏 OCR
                    empty_rounds = 0
                    continue

                time.sleep(self.config.scan_interval)

            self._emit("已结束")
        finally:
            if self._turbo_capture is not None:
                try:
                    self._turbo_capture.close()
                except Exception:
                    pass
                self._turbo_capture = None
