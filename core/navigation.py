"""回到游戏主界面（城镇或野外均可）。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.vision import Vision

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"
SCENE_TOGGLE_ROI = (500, 1150, 720, 1280)
BTN_TOWN_LABEL = "btn_town_label.png"
BTN_WILDERNESS_LABEL = "btn_wilderness_label.png"
SEARCH_TEMPLATES = ("beast_tab.png", "search_confirm_btn.png")
MAX_ATTEMPTS = 20
FALLBACK_BACKS = 18


def _match_in_roi(vision: Vision, screen, template: str, roi: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi
    return vision.match_template(screen[y1:y2, x1:x2], template).found


def _is_main_screen(vision: Vision, screen) -> bool:
    if _match_in_roi(vision, screen, BTN_TOWN_LABEL, SCENE_TOGGLE_ROI):
        return True
    return _match_in_roi(vision, screen, BTN_WILDERNESS_LABEL, SCENE_TOGGLE_ROI)


def _overlay_open(vision: Vision, screen) -> bool:
    return any(vision.match_template(screen, tpl).found for tpl in SEARCH_TEMPLATES)


def return_to_main_screen(
    adb: AdbClient,
    on_status: Callable[[str], None] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> None:
    """关闭子界面/弹窗，直到右下角出现「城镇」或「野外」切换按钮。"""
    vision = Vision(TEMPLATE_DIR, threshold=0.70)
    has_scene_templates = (TEMPLATE_DIR / BTN_TOWN_LABEL).is_file() or (
        TEMPLATE_DIR / BTN_WILDERNESS_LABEL
    ).is_file()

    if not has_scene_templates:
        logger.debug("未找到场景模板，使用固定次数返回")
        for _ in range(FALLBACK_BACKS):
            adb.back()
            time.sleep(0.5)
        if on_status:
            on_status("已尝试返回主界面")
        return

    for _ in range(max_attempts):
        screen = adb.screenshot()
        if _is_main_screen(vision, screen):
            if on_status:
                on_status("已回到主界面")
            return
        if _overlay_open(vision, screen):
            adb.back()
            time.sleep(0.6)
            continue
        adb.back()
        time.sleep(0.6)

    if on_status:
        on_status("已尝试返回主界面（未确认场景）")
    logger.warning("return_to_main_screen: 未能确认主界面")
