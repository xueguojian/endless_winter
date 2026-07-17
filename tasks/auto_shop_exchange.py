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

# 六个商品价格区（含货币图标 + 数量）；略加宽以覆盖居中窄价签（煤/铁）
PRICE_ROIS: tuple[tuple[int, int, int, int], ...] = (
    (40, 628, 240, 686),
    (256, 632, 462, 688),
    (476, 622, 684, 686),
    (40, 916, 240, 972),
    (256, 924, 462, 976),
    (476, 916, 684, 982),
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


def _content_column_segments(
    col_ratio: np.ndarray, *, min_ratio: float = 0.18, max_gap: int = 2
) -> list[tuple[int, int]]:
    """把有内容的列合并成连续段 [(start, end), ...]。"""
    xs = np.flatnonzero(col_ratio >= min_ratio)
    if xs.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = prev = int(xs[0])
    for x in xs[1:]:
        xi = int(x)
        if xi <= prev + max_gap:
            prev = xi
            continue
        segments.append((start, prev))
        start = prev = xi
    segments.append((start, prev))
    return segments


def _extract_price_icon(crop: np.ndarray) -> np.ndarray | None:
    """从价格按钮中取出货币图标区域。

    小数额时整块会居中；大额宽价签左侧是图标、右侧是长数字。
    用暗色列定位起点后只取较短窗口，避免把深蓝数字吃进图标区。
    """
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_ratio = (gray < 135).mean(axis=0)
    segments = _content_column_segments(dark_ratio, min_ratio=0.10)
    if not segments:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        vivid = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 80)
        segments = _content_column_segments(vivid.mean(axis=0), min_ratio=0.12)
    if not segments:
        return None

    left: int | None = None
    for start, end in segments:
        if end - start + 1 >= 3:
            left = start
            break
    if left is None:
        left = segments[0][0]

    left = max(0, left - 4)
    # 货币图标本身约 24~36px；宽价签切太长会把深蓝数字算进青色
    right = min(crop.shape[1], left + 36)
    if right - left < 10:
        left = max(0, right - 28)
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

    # 钻石：亮青宝石。V 要够高，避免宽价签里深蓝数字被当成钻石
    cyan_m = cv2.inRange(hsv, (85, 70, 150), (120, 255, 255)) > 0
    wood_m = cv2.inRange(hsv, (5, 45, 50), (30, 255, 255)) > 0
    # 煤矿：暗灰块
    coal_m = (sat < 110) & (val > 20) & (val < 170)
    # 铁矿：偏褐灰，不用冷蓝 Hue（易与价格数字撞车）
    iron_m = (
        (sat >= 12)
        & (sat < 120)
        & (val > 35)
        & (val < 180)
        & (hsv[:, :, 0] >= 0)
        & (hsv[:, :, 0] <= 35)
    )
    meat_m = (cv2.inRange(hsv, (0, 55, 55), (14, 255, 255)) > 0) | (
        cv2.inRange(hsv, (160, 55, 55), (180, 255, 255)) > 0
    )
    resource_m = (wood_m | coal_m | iron_m | meat_m) & ~cyan_m

    cyan = float((cyan_m & content).sum()) / content_n
    resource = float((resource_m & content).sum()) / content_n
    return cyan, resource


def _decide_kind(cyan: float, resource: float) -> PriceKind:
    """同一窗口内比较青色(钻石)与资源色。"""
    if resource >= 0.14 and resource >= cyan * 0.5:
        return "resource"
    if resource >= 0.10 and resource >= cyan * 0.75:
        return "resource"
    if cyan >= 0.12 and cyan >= resource * 1.15:
        return "diamond"
    if resource >= 0.08:
        return "resource"
    if cyan >= 0.08:
        return "diamond"
    return "empty"


def classify_price_kind(crop: np.ndarray) -> PriceKind:
    """根据价格区货币图标颜色判断：钻石 / 资源 / 空。"""
    if crop.size == 0:
        return "empty"

    windows: list[np.ndarray] = []
    icon = _extract_price_icon(crop)
    if icon is not None:
        windows.append(icon)
    windows.append(crop)

    w = crop.shape[1]
    # 宽价签：优先看左侧图标区；窄价签：多扫几个居中位置
    for x0 in (0, 8, 16, 24, 36, max(0, w // 2 - 40), max(0, w // 3)):
        win = crop[:, x0 : min(w, x0 + 40)]
        if win.shape[1] >= 12:
            windows.append(win)

    saw_resource = False
    saw_diamond = False
    best_resource = 0.0
    best_cyan = 0.0
    for win in windows:
        cyan, resource = _score_currency_masks(win)
        best_resource = max(best_resource, resource)
        best_cyan = max(best_cyan, cyan)
        kind = _decide_kind(cyan, resource)
        if kind == "resource":
            saw_resource = True
        elif kind == "diamond":
            saw_diamond = True

    # 任一窗口明确是资源则买；避免「别的窗口青色更高」把宽价签打成钻石
    if saw_resource:
        return "resource"
    if saw_diamond:
        return "diamond"
    return _decide_kind(best_cyan, best_resource)


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

        # 无论是否免费刷新，先买光当前页资源价（避免漏掉煤/铁窄价签商品）
        self._emit("扫描并购买当前页资源价商品…")
        total_bought = self._buy_all_resource_items()

        screen = self.adb.screenshot()
        if not is_free_refresh_available(screen):
            self._emit("当前不是「免费刷新」，已处理本页资源价后结束")
            self._emit(f"自动换资源完成，共购买 {total_bought} 次")
            return

        self._emit("检测到免费刷新，继续刷新换资源")
        for round_index in range(1, MAX_REFRESH_ROUNDS + 1):
            self._check_stop()
            self._emit(f"—— 刷新轮次 {round_index} ——")
            self._tap_free_refresh()
            total_bought += self._buy_all_resource_items()

            screen = self.adb.screenshot()
            if is_free_refresh_available(screen):
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
