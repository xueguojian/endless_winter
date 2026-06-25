"""出征界面：编队选择、出征、体力不足处理。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.vision import MatchResult, Vision

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"

MARCH_BTN = "march_btn.png"
STAMINA_USE_BTN = "stamina_use_btn.png"
STAMINA_GET_MORE_TITLE = "stamina_get_more_title.png"

STAMINA_TITLE_ROI = (150, 120, 570, 280)
STAMINA_USE_ROW_ROI = (500, 520, 710, 660)
STAMINA_USE_CENTER = (630, 570)
STAMINA_USE_Y_MIN = 520
STAMINA_USE_Y_MAX = 660
STAMINA_MATCH_THRESHOLD = 0.58
STAMINA_TITLE_THRESHOLD = 0.62

MARCH_MATCH_THRESHOLD = 0.72
MARCH_MIN_Y = 1050
MARCH_CENTER = (560, 1200)

FORMATION_SLOTS: dict[int, tuple[int, int]] = {
    1: (44, 129),
    2: (121, 129),
    3: (198, 129),
    4: (275, 129),
    5: (352, 129),
    6: (429, 129),
    7: (506, 129),
    8: (583, 129),
}


class StaminaInsufficientError(RuntimeError):
    """体力不足且未启用自动使用体力。"""


class DeployMarchHelper:
    def __init__(
        self,
        adb: AdbClient,
        *,
        formation_slot: int,
        use_stamina: bool,
        coords: dict[str, list[int]] | None = None,
        on_status: Callable[[str], None] | None = None,
        interrupted: Callable[[], bool] | None = None,
        step_delay: float = 1.5,
    ):
        self.adb = adb
        self.formation_slot = formation_slot
        self.use_stamina = use_stamina
        self.coords = coords or {}
        self.on_status = on_status
        self._interrupted = interrupted or (lambda: False)
        self.step_delay = step_delay

    def _emit(self, message: str) -> None:
        logger.info(message)
        if self.on_status:
            self.on_status(message)

    def _tap_xy(self, x: int, y: int, delay: float | None = None) -> None:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        self.adb.tap(x, y)
        time.sleep(delay if delay is not None else self.step_delay)

    def is_stamina_popup(self, screen) -> bool:
        """是否出现「获取更多」体力不足弹窗（与冰原巨兽逻辑一致，仅认标题）。"""
        title_path = TEMPLATE_DIR / STAMINA_GET_MORE_TITLE
        if title_path.exists():
            x1, y1, x2, y2 = STAMINA_TITLE_ROI
            title_vision = Vision(TEMPLATE_DIR, threshold=STAMINA_TITLE_THRESHOLD)
            title = title_vision.match_template(screen[y1:y2, x1:x2], STAMINA_GET_MORE_TITLE)
            if title.found and title.confidence >= STAMINA_TITLE_THRESHOLD:
                logger.info(
                    f"体力弹窗：标题「获取更多」匹配 {title.confidence:.2f} "
                    f"(阈值 {STAMINA_TITLE_THRESHOLD})"
                )
                return True
            return False

        btn = self._find_stamina_use_button(screen)
        if btn.found and btn.confidence >= STAMINA_MATCH_THRESHOLD:
            logger.info(
                f"体力弹窗：使用按钮匹配 {btn.confidence:.2f} "
                f"(阈值 {STAMINA_MATCH_THRESHOLD}，无标题模板时的兜底)"
            )
            return True
        return False

    def _find_stamina_use_button(self, screen) -> MatchResult:
        if not (TEMPLATE_DIR / STAMINA_USE_BTN).exists():
            return MatchResult(found=False)

        x1, y1, x2, y2 = STAMINA_USE_ROW_ROI
        crop = screen[y1:y2, x1:x2]
        stamina_vision = Vision(TEMPLATE_DIR, threshold=STAMINA_MATCH_THRESHOLD)
        result = stamina_vision.match_template_multiscale(crop, STAMINA_USE_BTN)
        if not result.found:
            result = stamina_vision.match_template(crop, STAMINA_USE_BTN)
        if result.found:
            cx, cy = result.center
            global_center = (x1 + cx, y1 + cy)
            if not (STAMINA_USE_Y_MIN <= global_center[1] <= STAMINA_USE_Y_MAX):
                return MatchResult(found=False, confidence=result.confidence)
            return MatchResult(
                found=True,
                confidence=result.confidence,
                center=global_center,
                top_left=(x1 + result.top_left[0], y1 + result.top_left[1]),
                size=result.size,
            )
        return MatchResult(found=False, confidence=result.confidence)

    def _get_stamina_use_tap_point(self, screen) -> tuple[int, int]:
        btn = self._find_stamina_use_button(screen)
        if btn.found:
            return btn.center
        if "stamina_use" in self.coords:
            return tuple(self.coords["stamina_use"])
        return STAMINA_USE_CENTER

    def use_stamina_items(self) -> None:
        for click_idx in range(2):
            time.sleep(0.5)
            screen = self.adb.screenshot()
            if not self.is_stamina_popup(screen):
                self._emit(f"体力弹窗已关闭（第 {click_idx + 1} 次前）")
                break
            cx, cy = self._get_stamina_use_tap_point(screen)
            self._emit(f"使用领主体力 ({click_idx + 1}/2) @ ({cx},{cy})")
            if self._interrupted():
                raise InterruptedError("任务已停止")
            self.adb.tap(cx, cy)
            time.sleep(0.15)
            self.adb.tap(cx, cy)
            time.sleep(1.4)

    def is_deploy_screen(self, screen) -> bool:
        if not (TEMPLATE_DIR / MARCH_BTN).exists():
            return False
        march_vision = Vision(TEMPLATE_DIR, threshold=MARCH_MATCH_THRESHOLD)
        result = march_vision.match_template_multiscale(screen, MARCH_BTN)
        return result.found and result.center[1] >= MARCH_MIN_Y

    def wait_for_deploy_screen(self, timeout: float = 15.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._interrupted():
                raise InterruptedError("任务已停止")
            screen = self.adb.screenshot()
            if self.is_deploy_screen(screen):
                return
            time.sleep(0.5)
        raise RuntimeError("未检测到出征界面")

    def select_formation(self) -> None:
        slot = self.formation_slot
        if not (1 <= slot <= 8):
            raise RuntimeError(f"编队槽位 {slot} 无效，请填写 1~8")
        self._emit(f"选择编队槽位 {slot}…")
        self.wait_for_deploy_screen()
        sx, sy = FORMATION_SLOTS[slot]
        self._emit(f"点击编队槽位 {slot} @ ({sx},{sy})")
        self._tap_xy(sx, sy, delay=1.0)

    def _resolve_march_button(self) -> tuple[int, int, float, bool]:
        march_vision = Vision(TEMPLATE_DIR, threshold=MARCH_MATCH_THRESHOLD)
        cx, cy = (
            tuple(self.coords["march"])
            if "march" in self.coords
            else MARCH_CENTER
        )
        matched = False
        match_conf = 0.0

        for attempt in range(6):
            screen = self.adb.screenshot()
            if (TEMPLATE_DIR / MARCH_BTN).exists():
                result = march_vision.match_template_multiscale(screen, MARCH_BTN)
                if result.found and result.center[1] >= MARCH_MIN_Y:
                    cx, cy = result.center
                    matched = True
                    match_conf = result.confidence
                    break
            time.sleep(0.5)
            logger.debug(f"等待出征按钮 {attempt + 1}/6")

        return cx, cy, match_conf, matched

    def _click_march_button(self) -> None:
        cx, cy, match_conf, matched = self._resolve_march_button()
        if matched:
            self._emit(f"点击出征 @ ({cx},{cy})（匹配 {match_conf:.2f}）")
        else:
            self._emit(f"点击出征 @ ({cx},{cy})（固定坐标）")
        self._tap_xy(cx, cy, delay=0.6)

    def _wait_march_stamina_popup(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._interrupted():
                raise InterruptedError("任务已停止")
            if self.is_stamina_popup(self.adb.screenshot()):
                return True
            time.sleep(0.35)
        return self.is_stamina_popup(self.adb.screenshot())

    def tap_march(self) -> None:
        """出征；体力不足时按配置处理。"""
        self._click_march_button()
        time.sleep(0.8)
        if not self._wait_march_stamina_popup():
            self._emit("队伍已出征")
            return

        self._emit("检测到体力不足弹窗")
        if not self.use_stamina:
            raise StaminaInsufficientError("体力不足，任务结束")

        self._emit("自动使用领主体力道具…")
        self.use_stamina_items()
        time.sleep(0.8)
        self.wait_for_deploy_screen(timeout=12.0)
        self._click_march_button()
        time.sleep(0.8)
        if self._wait_march_stamina_popup():
            raise StaminaInsufficientError("使用体力后仍体力不足，任务结束")
        self._emit("队伍已出征")

    def handle_stamina_popup_if_any(self) -> bool:
        """检测体力弹窗。返回 True=已处理并应重新扫描；False=无弹窗。

        未启用体力时抛出 StaminaInsufficientError。
        """
        time.sleep(0.8)
        if not self.is_stamina_popup(self.adb.screenshot()):
            return False
        self._emit("检测到体力不足弹窗")
        if not self.use_stamina:
            raise StaminaInsufficientError("体力不足，任务结束")
        self.use_stamina_items()
        return True
