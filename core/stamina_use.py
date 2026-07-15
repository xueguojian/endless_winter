"""领主体力道具：批量点击 + 罐头用量上限。"""

from __future__ import annotations

import time
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient

# 体力不足弹窗一次连点次数（减少反复弹窗）
STAMINA_BATCH_CLICKS = 20
DEFAULT_STAMINA_CAN_LIMIT = 800
STAMINA_USE_XY = (576, 522)
STAMINA_TAP_INTERVAL = 0.12
STAMINA_AFTER_BATCH_DELAY = 0.5


class StaminaCanLimitReached(InterruptedError):
    """已用完配置的体力罐头数量，应终止任务循环。"""


class StaminaCanBudget:
    """勾选「使用体力」后生效：累计点击次数达到上限则停止。"""

    def __init__(self, *, enabled: bool, limit: int = DEFAULT_STAMINA_CAN_LIMIT):
        self.enabled = bool(enabled)
        self.limit = max(1, int(limit))
        self.used = 0

    @property
    def remaining(self) -> int:
        if not self.enabled:
            return STAMINA_BATCH_CLICKS
        return max(0, self.limit - self.used)

    def record_click(self) -> None:
        if not self.enabled:
            return
        self.used += 1
        if self.used >= self.limit:
            raise StaminaCanLimitReached(
                f"体力罐头已达上限 {self.used}/{self.limit}，停止循环"
            )


def use_stamina_cans_batch(
    adb: AdbClient,
    budget: StaminaCanBudget,
    *,
    tap_xy: tuple[int, int] = STAMINA_USE_XY,
    emit: Callable[[str], None] | None = None,
    interrupted: Callable[[], bool] | None = None,
    close_with_back: bool = True,
) -> int:
    """在体力弹窗内连点使用道具。返回本次实际点击次数。"""
    if interrupted and interrupted():
        raise InterruptedError("任务已停止")

    clicks = min(STAMINA_BATCH_CLICKS, budget.remaining)
    if clicks <= 0:
        raise StaminaCanLimitReached(
            f"体力罐头已达上限 {budget.used}/{budget.limit}，停止循环"
        )

    cx, cy = tap_xy
    msg = (
        f"使用领主体力 ×{clicks} @ ({cx},{cy})"
        f"（本次前累计 {budget.used}/{budget.limit if budget.enabled else '∞'}）"
    )
    logger.info(msg)
    if emit:
        emit(msg)

    for index in range(clicks):
        if interrupted and interrupted():
            raise InterruptedError("任务已停止")
        adb.tap(cx, cy)
        budget.record_click()
        time.sleep(STAMINA_TAP_INTERVAL)
        if index == clicks - 1:
            break

    time.sleep(STAMINA_AFTER_BATCH_DELAY)
    if close_with_back:
        if emit:
            emit("按返回键关闭弹窗")
        adb.back()
        time.sleep(0.8)
    return clicks
