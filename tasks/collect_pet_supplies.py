"""一键领取宠物物资。"""

from __future__ import annotations

import threading
import time
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.navigation import WildernessNavigator

StatusCallback = Callable[[str], None]

DEFAULT_COORDS: dict[str, list[int]] = {
    "pet_open": [664, 950],
    "pet_slot_1": [282, 298],
    "pet_slot_2": [430, 306],
    "pet_slot_3": [570, 300],
    "pet_slot_4": [272, 444],
    "pet_confirm": [522, 1092],
    "dialog_cancel": [250, 780],
}

PET_SLOT_KEYS: tuple[str, ...] = (
    "pet_slot_1",
    "pet_slot_2",
    "pet_slot_3",
    "pet_slot_4",
)

DEFAULT_STEP_DELAY = 1.5


def merge_task_config(cfg: dict) -> dict:
    coords = {**DEFAULT_COORDS, **cfg.get("coords", {})}
    return {
        "step_delay": cfg.get("step_delay", DEFAULT_STEP_DELAY),
        "coords": coords,
    }


class CollectPetSuppliesTask:
    """野外 → 宠物界面 → 逐格领取 → 回野外。"""

    def __init__(
        self,
        adb: AdbClient,
        coords: dict[str, list[int]] | None = None,
        step_delay: float = DEFAULT_STEP_DELAY,
        on_status: StatusCallback | None = None,
    ):
        merged = merge_task_config({"coords": coords or {}, "step_delay": step_delay})
        self.adb = adb
        self.coords = merged["coords"]
        self.step_delay = merged["step_delay"]
        self.on_status = on_status
        self._stop_event = threading.Event()
        self._wilderness = WildernessNavigator.from_task(self)

    @property
    def name(self) -> str:
        return "领取宠物物资"

    def _emit(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
        if self.on_status:
            self.on_status(message)

    def stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

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

    def execute(self) -> None:
        """野外 → 打开宠物 → 逐格领取。"""
        self._ensure_wilderness()

        self._emit("打开宠物界面")
        self._tap("pet_open", delay=2.0)

        for index, slot_key in enumerate(PET_SLOT_KEYS, start=1):
            x, y = self.coords[slot_key]
            self._emit(f"领取第 {index} 格 @ ({x},{y})")
            self._tap(slot_key)
            self._tap("pet_confirm")

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
