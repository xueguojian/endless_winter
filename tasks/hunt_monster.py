"""自动打野怪任务。"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.navigation import WildernessNavigator
from core.vision import Vision

StatusCallback = Callable[[str], None]

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"

DEFAULT_COORDS: dict[str, list[int]] = {
    "dialog_cancel": [250, 780],
}


class HuntMonsterTask:
    """野外 → 搜索野兽 → 攻击出征 → 回野外。"""

    def __init__(
        self,
        adb: AdbClient,
        coords: dict[str, list[int]],
        interval: float = 300.0,
        monster_level: int = 30,
        max_monster_level: int = 30,
        skip_hour: int = 21,
        step_delay: float = 1.5,
        on_status: StatusCallback | None = None,
    ):
        self.adb = adb
        self.coords = {**DEFAULT_COORDS, **coords}
        self.interval = interval
        self.monster_level = monster_level
        self.max_monster_level = max_monster_level
        self.skip_hour = skip_hour
        self.step_delay = step_delay
        self.on_status = on_status
        self._last_run = 0.0
        self._stop_event = threading.Event()
        self.vision = Vision(TEMPLATE_DIR, threshold=0.70)
        self._wilderness = WildernessNavigator.from_task(self)

    @property
    def name(self) -> str:
        return "自动打野怪"

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
        if datetime.now().hour == self.skip_hour:
            return False
        return time.time() - self._last_run >= self.interval

    def _tap(self, key: str, delay: float | None = None) -> None:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        x, y = self.coords[key]
        logger.debug(f"[{self.name}] 点击 {key} ({x}, {y})")
        self.adb.tap(x, y)
        time.sleep(delay if delay is not None else self.step_delay)

    def _return_to_wilderness(self) -> None:
        self._wilderness.try_return_to_wilderness()

    def _close_popups(self) -> None:
        if "close_popup" in self.coords:
            self._tap("close_popup", delay=0.8)

    def _set_monster_level(self) -> None:
        diff = self.max_monster_level - self.monster_level
        if diff <= 0:
            return

        key = "level_minus" if diff > 0 else "level_plus"
        for _ in range(abs(diff)):
            self._tap(key, delay=0.2)

    def run_hunt_cycle(self) -> None:
        """野外 → 搜索 → 攻击 → 出征。"""
        self._emit(f"开始打怪，目标等级 {self.monster_level}")
        self._wilderness.ensure_wilderness()
        self._close_popups()

        self._tap("search_open")
        self._set_monster_level()
        self._tap("search_confirm", delay=2.5)
        self._tap("attack", delay=2.0)
        self._tap("march", delay=1.5)

        self._emit("已派出部队")

    def run_once(self, *, force: bool = False) -> bool:
        if not force and not self.should_run():
            return False

        self._last_run = time.time()
        try:
            self.run_hunt_cycle()
            self._return_to_wilderness()
            self._emit(f"本轮完成，{int(self.interval // 60)} 分钟后再次打野")
            return True
        except InterruptedError:
            self._emit("任务已停止")
            raise
        except Exception as exc:
            logger.exception(f"[{self.name}] 执行失败")
            self._emit(f"执行失败：{exc}")
            self._return_to_wilderness()
            return False
