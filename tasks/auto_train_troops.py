"""自动练兵：扫描城镇状态面板，对可练兵部队依次训练。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from loguru import logger

from core.adb_client import AdbClient
from core.navigation import WildernessNavigator
from core.vision import MatchResult, Vision

StatusCallback = Callable[[str], None]

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"

TRAIN_READY_COLLECT = "training/train_ready_collect.png"
TRAIN_READY_BTN = "training/train_ready_btn.png"

TRAIN_SCAN_ROI = (370, 530, 434, 732)
TRAIN_SCAN_ROWS = 3
TRAIN_READY_THRESHOLD = 0.65
TRAIN_PREAMBLE_TAP_DELAY = 0.8

DEFAULT_COORDS: dict[str, list[int]] = {
    "status_open": [16, 546],
    "train_collect_tap": [356, 566],
    "barracks_enter": [480, 868],
    "train_confirm": [524, 1126],
    "dialog_cancel": [250, 780],
}

TRAIN_READY_KINDS: tuple[tuple[str, str, str], ...] = (
    ("collect", "待收取", TRAIN_READY_COLLECT),
    ("train", "可练兵", TRAIN_READY_BTN),
)

DEFAULT_STEP_DELAY = 1.5
DEFAULT_INTERVAL = 3 * 3600  # 3 小时
TRAIN_CYCLE_MAX_ATTEMPTS = 10


@dataclass(frozen=True)
class TrainReadyMatch:
    result: MatchResult
    kind: str
    label: str


def merge_task_config(cfg: dict) -> dict:
    coords = {**cfg.get("coords", {}), **DEFAULT_COORDS}
    return {
        "step_delay": cfg.get("step_delay", DEFAULT_STEP_DELAY),
        "coords": coords,
        "train_ready_threshold": float(
            cfg.get("train_ready_threshold", TRAIN_READY_THRESHOLD)
        ),
        "max_cycle_attempts": int(cfg.get("max_cycle_attempts", TRAIN_CYCLE_MAX_ATTEMPTS)),
    }


class AutoTrainTroopsTask:
    """城镇主界面 → 状态面板 → 发现可练兵 → 兵营训练 → 循环直到无待练兵。"""

    def __init__(
        self,
        adb: AdbClient,
        coords: dict[str, list[int]] | None = None,
        interval: float = DEFAULT_INTERVAL,
        step_delay: float = DEFAULT_STEP_DELAY,
        train_ready_threshold: float = TRAIN_READY_THRESHOLD,
        max_cycle_attempts: int = TRAIN_CYCLE_MAX_ATTEMPTS,
        on_status: StatusCallback | None = None,
    ):
        merged = merge_task_config(
            {
                "coords": coords or {},
                "step_delay": step_delay,
                "train_ready_threshold": train_ready_threshold,
                "max_cycle_attempts": max_cycle_attempts,
            }
        )
        self.adb = adb
        self.coords = merged["coords"]
        self.interval = interval
        self.step_delay = merged["step_delay"]
        self.train_ready_threshold = merged["train_ready_threshold"]
        self.max_cycle_attempts = merged["max_cycle_attempts"]
        self.on_status = on_status
        self._last_run = 0.0
        self._stop_event = threading.Event()
        self.vision = Vision(TEMPLATE_DIR, threshold=self.train_ready_threshold)
        self._wilderness = WildernessNavigator.from_task(self)

    @property
    def name(self) -> str:
        return "自动练兵"

    def stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

    def _emit(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
        if self.on_status:
            self.on_status(message)

    def _interrupted(self) -> bool:
        return self._stop_event.is_set()

    def should_run(self) -> bool:
        return time.time() - self._last_run >= self.interval

    def _tap_xy(self, x: int, y: int, delay: float | None = None) -> None:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        self.adb.tap(x, y)
        time.sleep(delay if delay is not None else self.step_delay)

    def _tap(self, key: str, delay: float | None = None) -> None:
        x, y = self.coords[key]
        logger.debug(f"[{self.name}] 点击 {key} ({x}, {y})")
        self._tap_xy(x, y, delay)

    def _back(self, times: int = 1) -> None:
        for _ in range(times):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            self.adb.back()
            time.sleep(0.5)

    def _ensure_main_town(self) -> None:
        self._emit("确保从野外进入城镇…")
        self._wilderness.ensure_wilderness()
        self._wilderness.switch_to_town()
        self._emit("已在城镇主界面")
        time.sleep(0.5)

    def _return_to_wilderness(self) -> None:
        self._emit("返回野外主界面…")
        self._wilderness.try_return_to_wilderness()

    def _has_red_notification_dot(
        self, screen, top_left: tuple[int, int], size: tuple[int, int]
    ) -> bool:
        """可练兵 / 待收取图标右上角有红点；训练中图标无红点。"""
        x, y = top_left
        w, h = size
        rx1 = x + int(w * 0.55)
        ry1 = y
        rx2 = min(x + w, screen.shape[1])
        ry2 = y + int(h * 0.45)
        if rx2 <= rx1 or ry2 <= ry1:
            return False
        patch = screen[ry1:ry2, rx1:rx2]
        red_mask = (patch[:, :, 2] > 180) & (patch[:, :, 1] < 100) & (patch[:, :, 0] < 100)
        return int(red_mask.sum()) > 15

    def _find_train_ready(self, screen) -> TrainReadyMatch | None:
        x1, y1, x2, y2 = TRAIN_SCAN_ROI
        roi = screen[y1:y2, x1:x2]
        roi_h = roi.shape[0]
        row_h = max(roi_h // TRAIN_SCAN_ROWS, 1)
        train_vision = Vision(TEMPLATE_DIR, threshold=self.train_ready_threshold)
        best: TrainReadyMatch | None = None
        best_conf = 0.0

        for row in range(TRAIN_SCAN_ROWS):
            ry1 = row * row_h
            ry2 = roi_h if row == TRAIN_SCAN_ROWS - 1 else (row + 1) * row_h
            sub = roi[ry1:ry2, :]

            for kind, label, template_name in TRAIN_READY_KINDS:
                result = train_vision.match_template_multiscale(sub, template_name)
                best_conf = max(best_conf, result.confidence)
                if not result.found:
                    continue
                if not self._has_red_notification_dot(sub, result.top_left, result.size):
                    logger.debug(
                        f"第 {row + 1} 行 {label} {result.confidence:.2f}，无红点，跳过"
                    )
                    continue

                cx, cy = result.center
                match = TrainReadyMatch(
                    result=MatchResult(
                        found=True,
                        confidence=result.confidence,
                        center=(x1 + cx, y1 + ry1 + cy),
                        top_left=(x1 + result.top_left[0], y1 + ry1 + result.top_left[1]),
                        size=result.size,
                    ),
                    kind=kind,
                    label=label,
                )
                if best is None or result.confidence > best.result.confidence:
                    best = match

        if best is None:
            logger.debug(f"练兵扫描最高匹配度 {best_conf:.2f}")
        return best

    def _handle_train_ready(self, ready: TrainReadyMatch) -> None:
        gx, gy = ready.result.center
        self._emit(
            f"发现{ready.label}图标 ({gx},{gy})，匹配度 {ready.result.confidence:.2f}"
        )
        self._tap_xy(gx, gy, delay=1.0)

        if ready.kind == "collect":
            tx, ty = self.coords["train_collect_tap"]
            self._emit(f"收取练兵结果，点击 ({tx},{ty})")
            self._tap_xy(tx, ty, delay=TRAIN_PREAMBLE_TAP_DELAY)
            self._emit(f"再次点击 ({tx},{ty})")
            self._tap_xy(tx, ty, delay=TRAIN_PREAMBLE_TAP_DELAY)

        self._emit("进入兵营")
        self._tap("barracks_enter", delay=2.0)

        self._emit("确认练兵")
        self._tap("train_confirm", delay=1.5)

    def run_train_cycle(self) -> int:
        trained_count = 0
        attempt_count = 0
        hit_attempt_limit = False

        while not self._interrupted():
            if attempt_count >= self.max_cycle_attempts:
                self._emit(
                    f"本轮识别已达上限 {self.max_cycle_attempts} 次，"
                    f"停止当前任务，等待下次执行"
                )
                hit_attempt_limit = True
                break

            attempt_count += 1
            self._emit(
                f"第 {attempt_count}/{self.max_cycle_attempts} 次识别扫描"
            )

            self._ensure_main_town()
            self._emit("打开城镇状态面板")
            self._tap("status_open", delay=1.5)

            screen = self.adb.screenshot()
            ready = self._find_train_ready(screen)
            if ready is None:
                debug_dir = TEMPLATE_DIR.parent / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                x1, y1, x2, y2 = TRAIN_SCAN_ROI
                cv2.imwrite(str(debug_dir / "train_scan_fail_roi.png"), screen[y1:y2, x1:x2])
                self._emit("没有可练兵项目，扫描结束")
                self._back(1)
                break

            self._handle_train_ready(ready)

            trained_count += 1
            self._emit(f"已完成第 {trained_count} 项练兵")
            self._return_to_wilderness()
            self._ensure_main_town()

        self._return_to_wilderness()
        return trained_count

    def run_once(self, *, force: bool = False) -> bool:
        if not force and not self.should_run():
            return False

        self._last_run = time.time()
        try:
            count = self.run_train_cycle()
            hours = max(1, int(self.interval // 3600))
            if count:
                self._emit(f"本轮共练兵 {count} 项，{hours} 小时后再次扫描")
            else:
                self._emit(f"本轮无需练兵，{hours} 小时后再次扫描")
            return True
        except InterruptedError:
            self._emit("任务已停止")
            raise
        except Exception as exc:
            logger.exception(f"[{self.name}] 执行失败")
            self._emit(f"执行失败：{exc}")
            self._wilderness.try_return_to_wilderness()
            return False
