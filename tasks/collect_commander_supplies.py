"""一键领取统帅物资。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.navigation import return_to_main_screen

StatusCallback = Callable[[str], None]

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"
BTN_TOWN_LABEL = "btn_town_label.png"
BTN_WILDERNESS_LABEL = "btn_wilderness_label.png"
MAIN_SCREEN_ENSURE_BACKS = 18

DEFAULT_COORDS: dict[str, list[int]] = {
    "commander_open": [482, 76],
    "claim_first": [616, 288],
    "claim_second": [584, 818],
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
    """主界面 → 统帅界面 → 两次领取操作 → 退回主界面。"""

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
        self._stop_requested = False

    @property
    def name(self) -> str:
        return "一键领取统帅物资"

    def _emit(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
        if self.on_status:
            self.on_status(message)

    def stop(self) -> None:
        self._stop_requested = True

    def _check_stop(self) -> None:
        if self._stop_requested:
            raise InterruptedError("任务已停止")

    def _has_scene_templates(self) -> bool:
        return (TEMPLATE_DIR / BTN_TOWN_LABEL).is_file() or (
            TEMPLATE_DIR / BTN_WILDERNESS_LABEL
        ).is_file()

    def _ensure_main_screen(self) -> None:
        self._emit("返回主界面…")
        if self._has_scene_templates():
            return_to_main_screen(self.adb, on_status=self.on_status)
            self._emit("已确认在主界面")
            time.sleep(0.8)
            return

        for _ in range(MAIN_SCREEN_ENSURE_BACKS):
            self._check_stop()
            self.adb.back()
            time.sleep(0.55)
        time.sleep(1.0)
        self._emit("已返回主界面，准备开始领取")

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
        """打开统帅界面 → 领取物资 → 结束。"""
        self._stop_requested = False
        self._ensure_main_screen()

        self._emit("打开统帅界面")
        self._tap("commander_open", delay=2.0)

        self._tap_twice("claim_first")

        self._emit("等待界面刷新…")
        time.sleep(self.step_delay)

        self._tap_twice("claim_second")
        self._emit("领取完成")
