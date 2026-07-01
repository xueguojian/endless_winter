from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable
from loguru import logger

from core.adb_client import AdbClient
from core.dream_memory.config import DreamMemoryConfig, load_dream_memory_config
from core.dream_memory.maps import DreamMemoryMap, load_map
from core.dream_memory.ocr_engine import ocr_engine_available, resolve_ocr_engine, warmup_ocr
from core.dream_memory.vision import read_target_chips, resolve_item_coord

StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class _BatchTap:
    slot_index: int
    text: str
    x: int
    y: int


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
        self._last_clicked_sig: tuple[str, ...] | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()
        self._last_clicked_sig = None

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
        )
        batch: list[_BatchTap] = []
        for chip in sorted(chips, key=lambda c: c.slot_index):
            if not chip.active or not chip.text:
                continue
            coord = resolve_item_coord(self.game_map, chip.text)
            if coord is None:
                self._emit(
                    f"槽位 {chip.slot_index + 1} OCR「{chip.text}」"
                    f" — 地图中无坐标，请标定"
                )
                continue
            x, y = coord
            batch.append(_BatchTap(chip.slot_index, chip.text, x, y))
        return batch

    @staticmethod
    def _batch_signature(batch: list[_BatchTap]) -> tuple[str, ...]:
        return tuple(item.text for item in batch)

    def _click_batch(self, batch: list[_BatchTap]) -> None:
        """同一批识别结果连续点击，中间不再 OCR、不再反复截图。"""
        gap = max(0.25, self.config.tap_between_delay)
        labels = "、".join(item.text for item in batch)
        self._emit(f"本批 {len(batch)} 个: {labels}")

        for index, item in enumerate(batch):
            if self._interrupted():
                return
            self._emit(f"点击「{item.text}」@ ({item.x},{item.y})")
            self.adb.tap(item.x, item.y)
            if index < len(batch) - 1:
                time.sleep(gap)

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
        self._emit(
            f"开始 — 地图「{self.game_map.name}」"
            f"（{len(self.game_map.items)} 个标定物品，"
            f"底栏 {len(self.config.target_slots)} 槽，OCR={engine}）"
        )

        while not self._interrupted():
            try:
                screen = self.adb.screenshot()
            except Exception as exc:
                self._emit(f"截图失败: {exc}")
                time.sleep(0.5)
                continue

            batch = self._scan_batch(screen)
            if not batch:
                self._last_clicked_sig = None
                time.sleep(self.config.scan_interval)
                continue

            sig = self._batch_signature(batch)
            if self._last_clicked_sig is not None and sig == self._last_clicked_sig:
                time.sleep(0.12)
                continue

            self._click_batch(batch)
            self._last_clicked_sig = sig
            if self._interrupted():
                break

        self._emit("已结束")
