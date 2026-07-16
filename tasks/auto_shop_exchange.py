"""自动换资源：游荡商人用资源购尽商品，并耗尽免费刷新。"""

from __future__ import annotations

import threading
import time
from typing import Callable, Literal

import cv2
import numpy as np
from loguru import logger

from core.adb_client import AdbClient
from core.dream_memory.ocr_engine import ocr_chip_text
from core.navigation import WildernessNavigator

StatusCallback = Callable[[str], None]

PriceKind = Literal["diamond", "resource", "empty"]

DEFAULT_COORDS: dict[str, list[int]] = {
    "shop_open": [418, 1220],
    "dialog_cancel": [250, 780],
}

# 「免费刷新」按钮文字区
FREE_REFRESH_ROI = (504, 232, 676, 282)

# 六个商品价格区（含货币图标 + 数量）
PRICE_ROIS: tuple[tuple[int, int, int, int], ...] = (
    (48, 634, 232, 680),
    (264, 638, 454, 682),
    (488, 628, 672, 678),
    (46, 920, 234, 966),
    (266, 930, 454, 970),
    (486, 922, 672, 976),
)

DEFAULT_STEP_DELAY = 1.5
BUY_TAP_DELAY = 1.2
REFRESH_TAP_DELAY = 2.0
MAX_BUY_PASSES = 12
MAX_REFRESH_ROUNDS = 20


def merge_task_config(cfg: dict) -> dict:
    coords = {**DEFAULT_COORDS, **cfg.get("coords", {})}
    return {
        "step_delay": cfg.get("step_delay", DEFAULT_STEP_DELAY),
        "coords": coords,
    }


def _normalize_ocr(text: str | None) -> str:
    if not text:
        return ""
    return (
        str(text)
        .replace(" ", "")
        .replace("\n", "")
        .replace("　", "")
    )


def is_free_refresh_available(screen: np.ndarray) -> bool:
    """ROI 内 OCR 是否包含「免费刷新」。"""
    x1, y1, x2, y2 = FREE_REFRESH_ROI
    h, w = screen.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False
    crop = screen[y1:y2, x1:x2]
    big = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    text, engine = ocr_chip_text(big)
    normalized = _normalize_ocr(text)
    ok = "免费" in normalized and "刷新" in normalized
    logger.debug(f"免费刷新 OCR({engine}) raw={text!r} → {ok}")
    return ok


def _extract_price_icon(crop: np.ndarray) -> np.ndarray | None:
    """从价格按钮中取出货币图标区域。

    小数额时整块（图标+数字）会居中，左侧多为空白；先找非底色内容列，
    再取内容块最左侧一段作为图标，避免漏识别铁矿等窄价签。
    """
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # 按钮底多为浅色低饱和；图标/数字更暗或更艳
    interesting = (sat > 40) | (val < 170)
    col_ratio = interesting.mean(axis=0)
    # 忽略零星噪点，要求该列有足够内容像素
    xs = np.flatnonzero(col_ratio >= 0.18)
    if xs.size == 0:
        return None
    left = int(xs[0])
    right = min(crop.shape[1], left + 52)
    if right - left < 8:
        return None
    return crop[:, left:right]


def _score_currency_masks(icon: np.ndarray) -> tuple[float, float]:
    """返回 (cyan_ratio, resource_ratio)，仅统计非底色像素。"""
    hsv = cv2.cvtColor(icon, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    content = (sat > 35) | (val < 175)
    content_n = int(content.sum())
    if content_n < 12:
        return 0.0, 0.0

    cyan_m = cv2.inRange(hsv, (85, 70, 100), (120, 255, 255)) > 0
    wood_m = cv2.inRange(hsv, (5, 45, 50), (30, 255, 255)) > 0
    coal_m = (sat < 95) & (val > 25) & (val < 155)
    iron_m = (
        (sat >= 12)
        & (sat < 120)
        & (val > 35)
        & (val < 180)
        & (
            ((hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 28))
            | ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 140))
        )
    )
    meat_m = (cv2.inRange(hsv, (0, 55, 55), (14, 255, 255)) > 0) | (
        cv2.inRange(hsv, (160, 55, 55), (180, 255, 255)) > 0
    )
    # 排除青色宝石像素，避免钻石暗部被煤/铁矿掩码吃掉
    resource_m = (wood_m | coal_m | iron_m | meat_m) & ~cyan_m

    cyan = float((cyan_m & content).sum()) / content_n
    resource = float((resource_m & content).sum()) / content_n
    return cyan, resource


def classify_price_kind(crop: np.ndarray) -> PriceKind:
    """根据价格区货币图标颜色判断：钻石 / 资源 / 空。"""
    icon = _extract_price_icon(crop)
    if icon is None:
        return "empty"

    cyan, resource = _score_currency_masks(icon)
    # 铁矿等资源色可能与冷灰接近，资源分优先于弱青色
    if resource >= 0.12 and resource >= cyan * 0.75:
        return "resource"
    if cyan >= 0.10:
        return "diamond"
    if resource >= 0.08:
        return "resource"
    return "empty"


def list_resource_price_centers(
    screen: np.ndarray,
) -> list[tuple[int, int, int]]:
    """返回 [(slot_index_1based, tap_x, tap_y), ...] 仅资源价。"""
    found: list[tuple[int, int, int]] = []
    for index, (x1, y1, x2, y2) in enumerate(PRICE_ROIS, start=1):
        h, w = screen.shape[:2]
        xa, ya = max(0, x1), max(0, y1)
        xb, yb = min(w, x2), min(h, y2)
        crop = screen[ya:yb, xa:xb]
        kind = classify_price_kind(crop)
        logger.debug(f"商店价格格{index} → {kind}")
        if kind == "resource":
            found.append((index, (x1 + x2) // 2, (y1 + y2) // 2))
    return found


class AutoShopExchangeTask:
    """野外 → 游荡商人 → 资源购尽 + 耗尽免费刷新。"""

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
        return "自动换资源"

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

    def _tap_xy(self, x: int, y: int, delay: float | None = None) -> None:
        self._check_stop()
        logger.debug(f"[{self.name}] 点击 ({x}, {y})")
        self.adb.tap(x, y)
        time.sleep(delay if delay is not None else self.step_delay)

    def _tap(self, key: str, delay: float | None = None) -> None:
        if key not in self.coords:
            raise KeyError(f"缺少坐标配置: {key}")
        x, y = self.coords[key]
        self._tap_xy(x, y, delay=delay)

    def _buy_all_resource_items(self) -> int:
        """反复购买当前页所有资源价商品，直到只剩钻石/空。返回购买次数。"""
        bought = 0
        for pass_index in range(1, MAX_BUY_PASSES + 1):
            self._check_stop()
            screen = self.adb.screenshot()
            targets = list_resource_price_centers(screen)
            if not targets:
                self._emit(f"第 {pass_index} 轮扫描：无资源价商品")
                break
            self._emit(
                f"第 {pass_index} 轮扫描：发现 {len(targets)} 个资源价，开始购买"
            )
            for slot, x, y in targets:
                self._check_stop()
                self._emit(f"购买资源商品 格{slot} @ ({x},{y})")
                self._tap_xy(x, y, delay=BUY_TAP_DELAY)
                bought += 1
        else:
            self._emit(f"单页购买已达上限 {MAX_BUY_PASSES} 轮，停止本页")
        return bought

    def _tap_free_refresh(self) -> None:
        x1, y1, x2, y2 = FREE_REFRESH_ROI
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        self._emit(f"点击免费刷新 @ ({cx},{cy})")
        self._tap_xy(cx, cy, delay=REFRESH_TAP_DELAY)

    def execute(self) -> None:
        self._ensure_wilderness()

        self._emit("打开商店")
        self._tap("shop_open", delay=2.5)

        screen = self.adb.screenshot()
        if not is_free_refresh_available(screen):
            self._emit("当前不是「免费刷新」，结束任务")
            return

        self._emit("检测到免费刷新，开始换资源")
        total_bought = 0
        for round_index in range(1, MAX_REFRESH_ROUNDS + 1):
            self._check_stop()
            self._emit(f"—— 刷新轮次 {round_index} ——")
            total_bought += self._buy_all_resource_items()

            screen = self.adb.screenshot()
            if is_free_refresh_available(screen):
                self._tap_free_refresh()
                continue

            self._emit("免费刷新已用尽，任务结束")
            break
        else:
            self._emit(f"免费刷新轮次已达上限 {MAX_REFRESH_ROUNDS}，停止")

        self._emit(f"自动换资源完成，共购买 {total_bought} 次")

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
