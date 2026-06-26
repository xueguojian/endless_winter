"""一键领取统帅物资。"""

from __future__ import annotations

import threading
import time
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.navigation import WildernessNavigator

StatusCallback = Callable[[str], None]

DEFAULT_COORDS: dict[str, list[int]] = {
    "commander_open": [482, 76],
    "claim_first": [616, 288],
    "claim_second": [584, 818],
    "dialog_cancel": [250, 780],
}

DEFAULT_STEP_DELAY = 1.5
DEFAULT_DOUBLE_TAP_DELAY = 1.0


def merge_task_config(cfg: dict) -> dict:
    coords = {**cfg.get("coords", {}), **DEFAULT_COORDS}
    return {
        "step_delay": cfg.get("step_delay", DEFAULT_STEP_DELAY),
        "double_tap_delay": float(cfg.get("double_tap_delay", DEFAULT_DOUBLE_TAP_DELAY)),
        "coords": coords,
    }


class CollectCommanderSuppliesTask:
    """野外 → 统帅界面 → 两次领取 → 回野外。"""

    def __init__(
        self,
        adb: AdbClient,
        coords: dict[str, list[int]] | None = None,
        step_delay: float = DEFAULT_STEP_DELAY,
        double_tap_delay: float = DEFAULT_DOUBLE_TAP_DELAY,
        on_status: StatusCallback | None = None,
    ):
        merged = merge_task_config(
            {
                "coords": coords or {},
                "step_delay": step_delay,
                "double_tap_delay": double_tap_delay,
            }
        )
        self.adb = adb
        self.coords = merged["coords"]
        self.step_delay = merged["step_delay"]
        self.double_tap_delay = merged["double_tap_delay"]
        self.on_status = on_status
        self._last_run = 0.0
        self._stop_event = threading.Event()
        self._wilderness = WildernessNavigator.from_task(self)

    @property
    def name(self) -> str:
        return "一键领取统帅物资"

    def _emit(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
        if self.on_status:
            self.on_status(message)

    def stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

    def _interrupted(self) -> bool:
        return self._stop_event.is_set()

    def _check_stop(self) -> None:
        if self._stop_event.is_set():
            raise InterruptedError("任务已停止")

    def _ensure_wilderness(self) -> None:
        self._emit("确保在野外主界面…")
        self._wilderness.ensure_wilderness()

    def _return_to_wilderness(self) -> None:
        self._wilderness.try_return_to_wilderness()

    def _tap(self, key: str, delay: float | None = None) -> None:
        self._check_stop()
        if key not in self.coords:
            raise KeyError(f"缺少坐标配置: {key}")
        x, y = self.coords[key]
        logger.debug(f"[{self.name}] 点击 {key} ({x}, {y})")
        self.adb.tap(x, y)
        time.sleep(delay if delay is not None else self.step_delay)

    def _tap_twice(self, key: str) -> None:
        x, y = self.coords[key]
        self._emit(f"点击 {key} ({x}, {y}) ×2")
        self._tap(key, delay=self.double_tap_delay)
        self._tap(key, delay=self.step_delay)

    def execute(self) -> None:
        """野外 → 统帅界面 → 领取物资。"""
        self._ensure_wilderness()

        self._emit("打开统帅界面")
        self._tap("commander_open", delay=2.0)

        self._tap_twice("claim_first")

        self._emit("等待界面刷新…")
        time.sleep(self.step_delay)

        self._tap_twice("claim_second")
        self._emit("领取完成")

    def run_once(self, *, force: bool = False) -> bool:
        _ = force
        self._stop_event.clear()
        try:
            self.execute()
            self._return_to_wilderness()
            self._emit("已回到野外")
            return True
        except InterruptedError:
            self._emit("任务已停止")
            raise
        except Exception as exc:
            logger.exception(f"[{self.name}] 执行失败")
            self._emit(f"执行失败：{exc}")
            self._return_to_wilderness()
            return False
