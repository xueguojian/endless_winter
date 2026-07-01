"""灯塔任务图标识别。"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.coords import PORTRAIT_HEIGHT, PORTRAIT_WIDTH
from loguru import logger

from core.vision import MatchResult, Vision

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"

# 情报页中间地图区域，排除顶部刷新条与底部等级栏
LIGHTHOUSE_SCAN_ROI = (20, 130, 700, 1020)
# 情报页背景：平常 / 活动期间（含红色晶簇）
LIGHTHOUSE_MAP_BG_NORMAL = "lighthouse/lighthouse_map_bg.png"
LIGHTHOUSE_MAP_BG_EVENT = "lighthouse/lighthouse_map_bg_1.png"
_active_map_bg_name: str = LIGHTHOUSE_MAP_BG_NORMAL
# 情报页顶部刷新条，用于区分「情报页」与「野外大地图」
LIGHTHOUSE_HEADER_ROI = (0, 155, 720, 210)
LIGHTHOUSE_HEADER_MEAN_DIFF_MAX = 18.0
# 与背景图差分：仅保留相对背景新出现且颜色鲜艳的图钉
BG_DIFF_THRESHOLD = 28
BG_DIFF_VIVID_MIN = 0.10
BG_DIFF_PIN_SUPPORT_MIN = 0.15
BG_MAX_PINS_PER_DIFF_BLOB = 2
BG_PIN_AREA_MIN = 60
BG_PIN_AREA_MAX = 1800
BG_DIFF_BLOB_MIN_AREA = 800
BG_TEARDROP_TOP_BOTTOM_MIN = 1.05
BG_TEARDROP_ASPECT_MIN = 0.85
BG_TEARDROP_ASPECT_MAX = 2.8
SUPER_BOSS_Y_MAX = 320
UI_EXCLUDE_Y_MAX = 230
UI_EXCLUDE_X_MAX = 120
PLANE_EXCLUDE_Y_MIN = 820
PLANE_EXCLUDE_X_MIN = 480
BG_PIN_DIFF_SEARCH_RADIUS = 50
LIGHTHOUSE_PIN_PATCH_HALF = 36
LIGHTHOUSE_SLOT_MERGE_DISTANCE = 18
LIGHTHOUSE_MAX_PIN_CANDIDATES = 18
PIN_BLOB_OPEN_KERNEL = 5
PIN_LARGE_BLOB_MIN_AREA = 900
PIN_BLOB_MAX_WHITE_RATIO = 0.28
PIN_BLOB_TENT_MAX_WHITE_RATIO = 0.85
TENT_PIN_BLOB_MIN_AREA = 80
TENT_PIN_BLOB_MAX_AREA = 1200
MISSION_PIN_MIN_AREA = 100
MISSION_PIN_MAX_AREA = 700
MISSION_PIN_MIN_CIRCULARITY = 0.20
MISSION_PIN_MAX_CIRCULARITY = 0.95
MISSION_PIN_MIN_EXTENT = 0.20
MISSION_PIN_MAX_EXTENT = 0.85
MISSION_PIN_MIN_ASPECT = 0.9
MISSION_PIN_MAX_ASPECT = 2.2
MISSION_PIN_MERGE_DISTANCE = 18
MAP_PIN_MIN_SCREEN_Y = 390
MAP_PIN_MAX_SCREEN_Y = 980
MAP_PIN_MIN_SCREEN_X = 50
MAP_PIN_MAX_SCREEN_X = 670
BOSS_ZONE_Y_MAX = 400
BOSS_ZONE_X = (160, 560)
BOSS_ZONE_MIN_AREA = 480
PLANE_ZONE_Y_MIN = 640
PLANE_ZONE_X_MAX = 220
BASE_EXCLUDE_CENTER = (360, 530)
BASE_EXCLUDE_RADIUS = 95

# 任务详情页（点击「立即查看」后）：底部行动按钮与标题副文案
MISSION_DETAIL_ACTION_ROI = (238, 592, 482, 666)
MISSION_DETAIL_SUBTITLE_ROI = (272, 380, 452, 412)
MISSION_BTN_MARCH = "lighthouse/mission_btn_march.png"
MISSION_BTN_ADVENTURE = "lighthouse/mission_btn_adventure.png"
MISSION_BTN_RESCUE = "lighthouse/mission_btn_rescue.png"
BOUNTY_MASTER_LABEL = "lighthouse/bounty_master_label.png"
BOUNTY_KEYWORD_DASHI_XUANSHANG = "lighthouse/bounty_keyword_dashi_xuanshang.png"
# 仅匹配「大师/宗师悬赏」专属词，不用单独的「悬赏：」（易与等级怪副标题混淆）
BOUNTY_SUBTITLE_TEMPLATES: tuple[tuple[str, float], ...] = (
    (BOUNTY_KEYWORD_DASHI_XUANSHANG, 0.48),
    (BOUNTY_MASTER_LABEL, 0.45),
)
MISSION_DETAIL_BOUNTY_THRESHOLD = 0.45
BOUNTY_SUBTITLE_SEARCH_WIDTH_RATIO = 0.40
BOUNTY_SUBTITLE_MAX_MATCH_X_RATIO = 0.35
BOUNTY_SUBTITLE_SCALES = (0.82, 0.90, 0.96, 1.0, 1.06, 1.14, 1.22)
MONSTER_KEYWORD_LEVEL = "lighthouse/monster_keyword_level.png"
BEAST_KEYWORD_HAO = "lighthouse/beast_keyword_hao.png"
MONSTER_LEVEL_SUBTITLE_MIN = 0.55
MONSTER_LEVEL_MAX_MATCH_X_RATIO = 0.12
BEAST_HAO_SUBTITLE_MIN = 0.70
BEAST_HAO_SEARCH_WIDTH_RATIO = 0.58
BEAST_HAO_MIN_MATCH_X_RATIO = 0.55
BEAST_HAO_SOFT_MIN = 0.65
LEVEL_OVERRIDE_HAO_MIN = 0.55
BEAST_PIN_TEMPLATE = "lighthouse/small_monster_beast.png"
BEAST_PIN_MATCH_MIN = 0.48
SKIP_MISSION_KINDS = frozenset({"small_monster_beast"})
MISSION_DETAIL_BTN_THRESHOLD = 0.65
MISSION_DETAIL_BTN_ACCEPT_SCORE = 0.60
MISSION_DETAIL_BTN_MIN_MARGIN = 0.04
MISSION_DETAIL_ABSOLUTE_MIN = 0.55
MISSION_DETAIL_COLOR_BACKUP_MIN_TEMPLATE = 0.32
MISSION_DETAIL_COLOR_STRONG_RATIO = 0.22
MISSION_DETAIL_COLOR_STRONG_MIN_TEMPLATE = 0.30
MISSION_DETAIL_BTN_SCALES = (0.88, 0.94, 1.0, 1.06, 1.12)
MISSION_DETAIL_COLOR_MIN_RATIO = 0.14
MISSION_DETAIL_COLOR_MIN_MARGIN = 0.04

_ACTION_BTN_TEMPLATE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}

# HSV 行动按钮主色（橙=出征，蓝=探险，绿=营救）
_DETAIL_BTN_HSV: tuple[tuple[str, str, str, tuple[int, int, int], tuple[int, int, int]], ...] = (
    ("small_monster", "出征", "orange", (5, 90, 130), (30, 255, 255)),
    ("hero_journey", "探险", "blue", (95, 70, 110), (125, 255, 255)),
    ("tent", "营救", "green", (32, 55, 70), (88, 255, 255)),
)


@dataclass(frozen=True)
class MissionDetailClassification:
    """详情页行动按钮 + 副标题联合分类结果。"""

    kind: str
    label: str
    confidence: float = 0.0
    action_center: tuple[int, int] | None = None
    beast_explicit: bool = False


@dataclass(frozen=True)
class LighthouseMission:
    kind: str
    label: str
    template: str
    center: tuple[int, int]
    confidence: float
    top_left: tuple[int, int] = (0, 0)
    size: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class LighthouseScanResult:
    mission: LighthouseMission | None
    missions: tuple[LighthouseMission, ...] = ()
    best_confidence: float = 0.0
    best_label: str = ""
    candidate_locations: int = 0

_MAP_BG_ROI: np.ndarray | None = None
_MAP_BG_SCREEN: np.ndarray | None = None
_event_period: bool = False
_scan_interrupt_cb: Callable[[], bool] | None = None


def _check_scan_interrupted() -> None:
    if _scan_interrupt_cb and _scan_interrupt_cb():
        raise InterruptedError("任务已停止")


def _normalize_screen_for_scan(screen: np.ndarray) -> np.ndarray:
    h, w = screen.shape[:2]
    if (w, h) == (PORTRAIT_WIDTH, PORTRAIT_HEIGHT):
        return screen
    return cv2.resize(
        screen,
        (PORTRAIT_WIDTH, PORTRAIT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )

def _is_ignored_background_pin(
    center: tuple[int, int],
    *,
    contour_area: float = 0.0,
) -> bool:
    """排除中央固定基地、左上角 UI 头像等非任务元素。"""
    x, y = center
    bx, by = BASE_EXCLUDE_CENTER
    if abs(x - bx) <= BASE_EXCLUDE_RADIUS and abs(y - by) <= BASE_EXCLUDE_RADIUS:
        return True
    if x <= UI_EXCLUDE_X_MAX and y <= UI_EXCLUDE_Y_MAX:
        return True
    return False


def _is_super_boss_point(
    center: tuple[int, int],
    roi: np.ndarray,
    hsv: np.ndarray,
    roi_offset: tuple[int, int],
) -> bool:
    """顶部超级大怪：位置靠上 + 大橙色块 + 内部白色兽头。"""
    x, y = center
    if y > SUPER_BOSS_Y_MAX:
        return False
    rx, ry = x - roi_offset[0], y - roi_offset[1]
    radius = 48
    y1 = max(0, ry - radius)
    y2 = min(roi.shape[0], ry + radius)
    x1 = max(0, rx - radius)
    x2 = min(roi.shape[1], rx + radius)
    patch_hsv = hsv[y1:y2, x1:x2]
    if patch_hsv.size == 0:
        return False
    orange = cv2.inRange(
        patch_hsv, np.array((8, 100, 140)), np.array((28, 255, 255))
    )
    if float(orange.mean()) / 255.0 < 0.22:
        return False
    patch_gray = cv2.cvtColor(roi[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    white_ratio = float((patch_gray > 200).mean())
    return white_ratio >= 0.05 or y <= 290


def _is_plane_point(
    center: tuple[int, int],
    *,
    blob_area: float = 0.0,
    blob_width: int = 0,
    blob_height: int = 0,
) -> bool:
    """排除右下角移动小飞机（宽扁、面积大）。"""
    x, y = center
    if y >= 880 and x >= 500:
        return True
    if y >= PLANE_EXCLUDE_Y_MIN and x >= PLANE_EXCLUDE_X_MIN:
        if blob_area >= 3000 or blob_width > blob_height * 1.15:
            return True
    return False


def _is_ui_diff_blob(
    center_y: int, *, blob_height: int, blob_width: int
) -> bool:
    if center_y <= UI_EXCLUDE_Y_MAX:
        return True
    return blob_height <= 20 and blob_width >= 80


def _patch_vivid_ratio(hsv: np.ndarray, center: tuple[int, int], radius: int = 22) -> float:
    y, x = center[1], center[0]
    y1 = max(0, y - radius)
    y2 = min(hsv.shape[0], y + radius)
    x1 = max(0, x - radius)
    x2 = min(hsv.shape[1], x + radius)
    patch = hsv[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    sat, val = patch[:, :, 1], patch[:, :, 2]
    return float(((sat > 100) & (val > 130)).mean())


def _pin_diff_support(
    center: tuple[int, int],
    diff_mask: np.ndarray,
    roi_offset: tuple[int, int],
    *,
    radius: int = BG_PIN_DIFF_SEARCH_RADIUS,
) -> tuple[float, tuple[int, int]]:
    """在图钉附近搜索差分最强的位置，返回 (差分占比,  refined_center)。"""
    ox, oy = roi_offset
    base_rx, base_ry = center[0] - ox, center[1] - oy
    best_ratio = 0.0
    best_center = center
    patch_r = 20

    for dy in range(-radius, radius + 1, 4):
        for dx in range(-radius, radius + 1, 4):
            rx, ry = base_rx + dx, base_ry + dy
            if (
                rx < patch_r
                or ry < patch_r
                or rx >= diff_mask.shape[1] - patch_r
                or ry >= diff_mask.shape[0] - patch_r
            ):
                continue
            patch = diff_mask[ry - patch_r : ry + patch_r, rx - patch_r : rx + patch_r]
            ratio = float(patch.mean()) / 255.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_center = (ox + rx, oy + ry)
    return best_ratio, best_center


def _merge_nearby_pin_candidates(
    candidates: list[tuple[tuple[int, int], float]],
    *,
    merge_distance: int = MISSION_PIN_MERGE_DISTANCE,
) -> list[tuple[tuple[int, int], float]]:
    """合并邻近图钉候选，保留置信度最高者。"""
    ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
    merged: list[tuple[tuple[int, int], float]] = []
    for center, confidence in ordered:
        if any(
            abs(center[0] - existing[0]) < merge_distance
            and abs(center[1] - existing[1]) < merge_distance
            for existing, _ in merged
        ):
            continue
        merged.append((center, confidence))
    return merged


def _contour_white_ratio(gray: np.ndarray, contour: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0.0
    sub_gray = gray[y : y + h, x : x + w]
    return float((sub_gray > 190).mean())

def _pin_contour_passes_filters(
    contour: np.ndarray,
    gray: np.ndarray,
    *,
    area_min: int,
    area_max: int,
    white_max: float,
    aspect_min: float = MISSION_PIN_MIN_ASPECT,
    aspect_max: float = MISSION_PIN_MAX_ASPECT,
    circularity_min: float = MISSION_PIN_MIN_CIRCULARITY,
) -> bool:
    area = cv2.contourArea(contour)
    if area < area_min or area > area_max:
        return False
    if _contour_white_ratio(gray, contour) > white_max:
        return False
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    if (
        circularity < circularity_min
        or circularity > MISSION_PIN_MAX_CIRCULARITY
    ):
        return False
    x, y, w, h = cv2.boundingRect(contour)
    extent = area / max(w * h, 1)
    aspect = h / max(w, 1)
    if extent < MISSION_PIN_MIN_EXTENT or extent > MISSION_PIN_MAX_EXTENT:
        return False
    if aspect < aspect_min or aspect > aspect_max:
        return False
    return True


def _contour_vivid_peak_center(
    contour: np.ndarray,
    hsv: np.ndarray,
    roi_offset: tuple[int, int],
    *,
    local_offset: tuple[int, int] = (0, 0),
) -> tuple[int, int] | None:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None
    lx, ly = local_offset
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    sat = hsv[y : y + h, x : x + w, 1].astype(np.float32)
    val = hsv[y : y + h, x : x + w, 2]
    score_map = sat * (val > 130)
    score_map[mask == 0] = 0.0
    if float(score_map.max()) <= 0:
        return None
    _, _, _, max_loc = cv2.minMaxLoc(score_map)
    ox, oy = roi_offset
    return (ox + lx + x + max_loc[0], oy + ly + y + max_loc[1])


def _contour_screen_center(
    contour: np.ndarray,
    roi_offset: tuple[int, int],
    *,
    local_offset: tuple[int, int] = (0, 0),
    hsv: np.ndarray | None = None,
    use_vivid_peak: bool = False,
) -> tuple[int, int] | None:
    if use_vivid_peak and hsv is not None:
        peak = _contour_vivid_peak_center(
            contour, hsv, roi_offset, local_offset=local_offset
        )
        if peak is not None:
            return peak
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    ox, oy = roi_offset
    lx, ly = local_offset
    cx = int(moments["m10"] / moments["m00"]) + lx
    cy = int(moments["m01"] / moments["m00"]) + ly
    return (ox + cx, oy + cy)


def _screen_center_in_map(center: tuple[int, int]) -> bool:
    x, y = center
    return (
        MAP_PIN_MIN_SCREEN_Y <= y <= MAP_PIN_MAX_SCREEN_Y
        and MAP_PIN_MIN_SCREEN_X <= x <= MAP_PIN_MAX_SCREEN_X
    )

def _append_pin_candidates_from_contours(
    contours: list[np.ndarray],
    gray: np.ndarray,
    roi_offset: tuple[int, int],
    candidates: list[tuple[tuple[int, int], float]],
    *,
    area_min: int = MISSION_PIN_MIN_AREA,
    area_max: int = MISSION_PIN_MAX_AREA,
    white_max: float = PIN_BLOB_MAX_WHITE_RATIO,
    aspect_min: float = MISSION_PIN_MIN_ASPECT,
    aspect_max: float = MISSION_PIN_MAX_ASPECT,
    circularity_min: float = MISSION_PIN_MIN_CIRCULARITY,
    local_offset: tuple[int, int] = (0, 0),
    screen_y_min: int = MAP_PIN_MIN_SCREEN_Y,
    hsv: np.ndarray | None = None,
    vivid_peak_min_area: float = 0.0,
) -> None:
    for contour in contours:
        _check_scan_interrupted()
        if not _pin_contour_passes_filters(
            contour,
            gray,
            area_min=area_min,
            area_max=area_max,
            white_max=white_max,
            aspect_min=aspect_min,
            aspect_max=aspect_max,
            circularity_min=circularity_min,
        ):
            continue
        area = cv2.contourArea(contour)
        center = _contour_screen_center(
            contour,
            roi_offset,
            local_offset=local_offset,
            hsv=hsv,
            use_vivid_peak=(
                hsv is not None
                and vivid_peak_min_area > 0
                and area >= vivid_peak_min_area
            ),
        )
        if center is None or center[1] < screen_y_min:
            continue
        if not _screen_center_in_map(center):
            continue
        if _is_ignored_background_pin(center, contour_area=area):
            continue
        candidates.append((center, cv2.contourArea(contour)))


def _append_fragment_centers_from_large_blobs(
    pin_mask: np.ndarray,
    gray: np.ndarray,
    roi_offset: tuple[int, int],
    candidates: list[tuple[tuple[int, int], float]],
    *,
    hsv: np.ndarray | None = None,
    min_area: float = PIN_LARGE_BLOB_MIN_AREA,
    erode_iters: int = 2,
) -> None:
    """大地形色块内用腐蚀切分，找回被粘连的任务图钉。"""
    kernel = np.ones((7, 7), np.uint8)
    for contour in cv2.findContours(
        pin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        local = pin_mask[y : y + h, x : x + w]
        if local.size == 0:
            continue
        eroded = cv2.erode(local, kernel, iterations=erode_iters)
        eroded = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        sub_gray = gray[y : y + h, x : x + w]
        sub_hsv = hsv[y : y + h, x : x + w] if hsv is not None else None
        _append_pin_candidates_from_contours(
            cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            sub_gray,
            roi_offset,
            candidates,
            area_min=60,
            area_max=600,
            local_offset=(x, y),
            hsv=sub_hsv,
            vivid_peak_min_area=120.0,
        )


def _find_mission_pin_centers(
    roi: np.ndarray,
    roi_offset: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """情报地图上彩色任务图钉中心（橙/紫/蓝紧凑水滴形）。"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pin_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    for lower, upper in (
        ((10, 130, 150), (28, 255, 255)),
        ((125, 60, 100), (155, 255, 255)),
        ((98, 70, 110), (118, 255, 255)),
    ):
        pin_mask = cv2.bitwise_or(
            pin_mask, cv2.inRange(hsv, np.array(lower), np.array(upper))
        )

    pin_closed = cv2.morphologyEx(pin_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    pin_opened = cv2.morphologyEx(
        pin_closed,
        cv2.MORPH_OPEN,
        np.ones((PIN_BLOB_OPEN_KERNEL, PIN_BLOB_OPEN_KERNEL), np.uint8),
    )

    candidates: list[tuple[tuple[int, int], float]] = []
    _append_pin_candidates_from_contours(
        cv2.findContours(pin_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        candidates,
        area_min=60,
        hsv=hsv,
        vivid_peak_min_area=180.0,
    )
    _append_fragment_centers_from_large_blobs(
        pin_closed, gray, roi_offset, candidates, hsv=hsv, min_area=950.0
    )

    orange_hero_mask = cv2.inRange(
        hsv,
        np.array((8, 90, 130)),
        np.array((28, 255, 255)),
    )
    orange_hero_mask = cv2.morphologyEx(
        orange_hero_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    priority_candidates: list[tuple[tuple[int, int], float]] = []
    _append_pin_candidates_from_contours(
        cv2.findContours(orange_hero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        priority_candidates,
        area_min=50,
        area_max=450,
    )

    hero_mask = cv2.inRange(
        hsv,
        np.array((155, 85, 150)),
        np.array((175, 255, 255)),
    )
    hero_mask = cv2.morphologyEx(hero_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    hero_mask = cv2.morphologyEx(hero_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _append_pin_candidates_from_contours(
        cv2.findContours(hero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        candidates,
        area_min=60,
        area_max=500,
    )

    blue_claw_mask = cv2.inRange(
        hsv,
        np.array((98, 80, 120)),
        np.array((118, 255, 255)),
    )
    blue_claw_mask = cv2.morphologyEx(
        blue_claw_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    _append_pin_candidates_from_contours(
        cv2.findContours(blue_claw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        priority_candidates,
        area_min=50,
        area_max=400,
    )

    orange_claw_mask = cv2.inRange(
        hsv,
        np.array((10, 100, 140)),
        np.array((26, 255, 255)),
    )
    orange_claw_mask = cv2.morphologyEx(
        orange_claw_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    _append_pin_candidates_from_contours(
        cv2.findContours(orange_claw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        priority_candidates,
        area_min=50,
        area_max=450,
    )
    candidates.extend(priority_candidates)

    tent_mask = cv2.inRange(
        hsv,
        np.array((150, 18, 170)),
        np.array((175, 55, 255)),
    )
    tent_mask = cv2.morphologyEx(tent_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    tent_mask = cv2.morphologyEx(tent_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _append_pin_candidates_from_contours(
        cv2.findContours(tent_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        candidates,
        area_min=TENT_PIN_BLOB_MIN_AREA,
        area_max=TENT_PIN_BLOB_MAX_AREA,
        white_max=PIN_BLOB_TENT_MAX_WHITE_RATIO,
        aspect_max=3.2,
        circularity_min=0.15,
    )

    blue_tent_mask = cv2.inRange(
        hsv,
        np.array((98, 70, 120)),
        np.array((118, 255, 255)),
    )
    blue_tent_mask = cv2.morphologyEx(
        blue_tent_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )
    blue_tent_mask = cv2.morphologyEx(
        blue_tent_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    _append_pin_candidates_from_contours(
        cv2.findContours(blue_tent_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        candidates,
        area_min=60,
        area_max=900,
        white_max=PIN_BLOB_TENT_MAX_WHITE_RATIO,
        aspect_max=3.5,
        circularity_min=0.12,
    )
    _append_fragment_centers_from_large_blobs(
        blue_tent_mask,
        gray,
        roi_offset,
        candidates,
        hsv=hsv,
        min_area=750.0,
        erode_iters=3,
    )

    candidates.sort(key=lambda item: item[1], reverse=True)
    merged: list[tuple[int, int]] = []
    priority_centers = [c for c, _ in priority_candidates]
    for center in priority_centers:
        if any(
            abs(center[0] - existing[0]) < MISSION_PIN_MERGE_DISTANCE
            and abs(center[1] - existing[1]) < MISSION_PIN_MERGE_DISTANCE
            for existing in merged
        ):
            continue
        merged.append(center)
    for center, _area in candidates:
        if any(
            abs(center[0] - existing[0]) < MISSION_PIN_MERGE_DISTANCE
            and abs(center[1] - existing[1]) < MISSION_PIN_MERGE_DISTANCE
            for existing in merged
        ):
            continue
        merged.append(center)

    if len(merged) > LIGHTHOUSE_MAX_PIN_CANDIDATES:
        priority_set = set(priority_centers)
        priority_kept = [c for c in merged if c in priority_set]
        others = [c for c in merged if c not in priority_set]
        merged = priority_kept + others[: max(0, LIGHTHOUSE_MAX_PIN_CANDIDATES - len(priority_kept))]
    return tuple(merged)
def _mission_slot_distance(a: LighthouseMission, b: LighthouseMission) -> float:
    return max(abs(a.center[0] - b.center[0]), abs(a.center[1] - b.center[1]))


def _dedupe_missions_by_slot(
    missions: list[LighthouseMission],
    *,
    merge_distance: float = LIGHTHOUSE_SLOT_MERGE_DISTANCE,
) -> list[LighthouseMission]:
    ordered = sorted(missions, key=lambda item: item.confidence, reverse=True)
    kept: list[LighthouseMission] = []
    for mission in ordered:
        if any(
            _mission_slot_distance(mission, existing) < merge_distance
            for existing in kept
        ):
            continue
        kept.append(mission)
    return kept
def _detail_action_roi_patch(
    screen: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    screen = _normalize_screen_for_scan(screen)
    x1, y1, x2, y2 = roi
    return screen[y1:y2, x1:x2], (x1, y1)


def _prepare_action_button_template(template_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """裁掉模板四周浅灰背景，只保留按钮本体（含白字）用于匹配。"""
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    color_mask = ((sat > 45) & (val > 85)).astype(np.uint8) * 255
    white_mask = ((sat < 45) & (val > 165)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(color_mask, white_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    coords = cv2.findNonZero(mask)
    if coords is None:
        h, w = template_bgr.shape[:2]
        return template_bgr, np.full((h, w), 255, dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    pad = 2
    y1 = max(0, y - pad)
    x1 = max(0, x - pad)
    y2 = min(template_bgr.shape[0], y + h + pad)
    x2 = min(template_bgr.shape[1], x + w + pad)
    return template_bgr[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def _load_action_button_template(template_name: str, template_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    cache_key = f"{template_dir}/{template_name}"
    if cache_key in _ACTION_BTN_TEMPLATE_CACHE:
        return _ACTION_BTN_TEMPLATE_CACHE[cache_key]

    template_path = template_dir / template_name
    if not template_path.is_file():
        return None
    template = cv2.imread(str(template_path))
    if template is None:
        return None

    prepared = _prepare_action_button_template(template)
    _ACTION_BTN_TEMPLATE_CACHE[cache_key] = prepared
    return prepared


def _score_masked_at(
    patch_f: np.ndarray,
    t_z: np.ndarray,
    t_norm: float,
    m: np.ndarray,
    n: float,
    x: int,
    y: int,
    th: int,
    tw: int,
) -> float:
    sub = patch_f[y : y + th, x : x + tw]
    s_mean = (sub * m[..., None]).sum(axis=(0, 1)) / n
    s_z = (sub - s_mean) * m[..., None]
    s_norm = float(np.sqrt((s_z ** 2).sum())) + 1e-6
    return float((s_z * t_z).sum() / (s_norm * t_norm))


def _match_masked_ncc(
    patch: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    *,
    search_radius: int = 24,
) -> tuple[float, int, int, int, int]:
    """仅在 mask 覆盖的按钮像素上计算归一化互相关（忽略模板背景）。"""
    m = (mask > 0).astype(np.float32)
    n = float(m.sum())
    th, tw = template.shape[:2]
    if n < 20:
        return 0.0, 0, 0, tw, th

    tpl_f = template.astype(np.float32)
    t_mean = (tpl_f * m[..., None]).sum(axis=(0, 1)) / n
    t_z = (tpl_f - t_mean) * m[..., None]
    t_norm = float(np.sqrt((t_z ** 2).sum())) + 1e-6

    ph, pw = patch.shape[:2]
    if th > ph or tw > pw:
        return 0.0, 0, 0, tw, th

    patch_f = patch.astype(np.float32)
    cx = (pw - tw) // 2
    cy = (ph - th) // 2
    best_score = -1.0
    best_x = best_y = 0

    for dy in range(-search_radius, search_radius + 1, 2):
        for dx in range(-search_radius, search_radius + 1, 2):
            x, y = cx + dx, cy + dy
            if x < 0 or y < 0 or x + tw > pw or y + th > ph:
                continue
            score = _score_masked_at(patch_f, t_z, t_norm, m, n, x, y, th, tw)
            if score > best_score:
                best_score = score
                best_x, best_y = x, y

    if best_score >= 0:
        return best_score, best_x, best_y, tw, th
    return 0.0, cx, cy, tw, th


def _match_masked_ncc_multiscale(
    patch: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    scales: tuple[float, ...],
) -> tuple[float, tuple[int, int, int, int]]:
    best_score = 0.0
    best_box = (0, 0, template.shape[1], template.shape[0])
    for scale in scales:
        if scale == 1.0:
            tpl, m = template, mask
        else:
            tpl = cv2.resize(
                template,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
            m = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        score, x, y, tw, th = _match_masked_ncc(patch, tpl, m)
        if score > best_score:
            best_score = score
            best_box = (x, y, tw, th)
    return best_score, best_box

def _match_action_button_in_screen_roi(
    screen: np.ndarray,
    template_name: str,
    roi: tuple[int, int, int, int],
    *,
    template_dir: Path = TEMPLATE_DIR,
    threshold: float = MISSION_DETAIL_BTN_THRESHOLD,
    scales: tuple[float, ...] = MISSION_DETAIL_BTN_SCALES,
) -> MatchResult:
    """行动按钮：去背景 + 掩膜匹配（避免模板浅灰底拉低得分）。"""
    patch, (ox, oy) = _detail_action_roi_patch(screen, roi)
    if patch.size == 0:
        return MatchResult(found=False)

    prepared = _load_action_button_template(template_name, template_dir)
    if prepared is None:
        logger.warning(f"灯塔详情模板缺失: {template_name}")
        return MatchResult(found=False)

    template, mask = prepared
    best_conf, (x, y, tw, th) = _match_masked_ncc_multiscale(patch, template, mask, scales)
    logger.debug(f"行动按钮 {template_name}: masked={best_conf:.2f}")

    center = (ox + x + tw // 2, oy + y + th // 2)
    return MatchResult(
        found=best_conf >= threshold,
        confidence=best_conf,
        center=center,
        top_left=(ox + x, oy + y),
        size=(tw, th),
    )


def _match_template_in_screen_roi(
    screen: np.ndarray,
    template_name: str,
    roi: tuple[int, int, int, int],
    *,
    template_dir: Path = TEMPLATE_DIR,
    threshold: float = MISSION_DETAIL_BTN_THRESHOLD,
    scales: tuple[float, ...] = MISSION_DETAIL_BTN_SCALES,
) -> MatchResult:
    """在屏幕固定 ROI 内做多尺度模板匹配（灰度 + 彩色取较高分）。"""
    patch, (ox, oy) = _detail_action_roi_patch(screen, roi)
    if patch.size == 0:
        return MatchResult(found=False)

    template_path = template_dir / template_name
    if not template_path.is_file():
        logger.warning(f"灯塔详情模板缺失: {template_name}")
        return MatchResult(found=False)

    template = cv2.imread(str(template_path))
    if template is None:
        return MatchResult(found=False)

    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    best_conf = 0.0
    best_loc: tuple[int, int, int, int] | None = None

    for scale in scales:
        scaled_gray = cv2.resize(
            tpl_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        scaled_bgr = cv2.resize(
            template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        th, tw = scaled_gray.shape[:2]
        if th > patch_gray.shape[0] or tw > patch_gray.shape[1]:
            continue

        gray_result = cv2.matchTemplate(patch_gray, scaled_gray, cv2.TM_CCOEFF_NORMED)
        color_result = cv2.matchTemplate(patch, scaled_bgr, cv2.TM_CCOEFF_NORMED)
        gray_score = float(gray_result.max())
        color_score = float(color_result.max())
        if color_score >= gray_score:
            max_val = color_score
            _, _, _, max_loc = cv2.minMaxLoc(color_result)
        else:
            max_val = gray_score
            _, _, _, max_loc = cv2.minMaxLoc(gray_result)
        if max_val <= best_conf:
            continue
        x, y = max_loc
        best_conf = max_val
        best_loc = (x, y, tw, th)

    if best_loc is None:
        return MatchResult(found=False, confidence=0.0)

    x, y, tw, th = best_loc
    center = (ox + x + tw // 2, oy + y + th // 2)
    found = best_conf >= threshold
    return MatchResult(
        found=found,
        confidence=best_conf,
        center=center,
        top_left=(ox + x, oy + y),
        size=(tw, th),
    )


def _classify_detail_action_by_color(
    screen: np.ndarray,
    action_roi: tuple[int, int, int, int],
) -> tuple[str, str, float, tuple[int, int]] | None:
    """按行动按钮主色兜底分类（模板得分偏低时）。"""
    patch, (ox, oy) = _detail_action_roi_patch(screen, action_roi)
    if patch.size == 0:
        return None

    h, w = patch.shape[:2]
    margin_x = max(2, int(w * 0.04))
    margin_y = max(2, int(h * 0.08))
    center_patch = patch[margin_y : h - margin_y, margin_x : w - margin_x]
    if center_patch.size == 0:
        center_patch = patch

    hsv = cv2.cvtColor(center_patch, cv2.COLOR_BGR2HSV)
    scored: list[tuple[str, str, float]] = []

    for kind, label, _color_key, lower, upper in _DETAIL_BTN_HSV:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        ratio = float(mask.mean()) / 255.0
        scored.append((kind, label, ratio))
        logger.debug(f"详情按钮颜色 {label} ratio={ratio:.3f}")

    scored.sort(key=lambda item: item[2], reverse=True)
    if len(scored) < 2:
        return None

    best_kind, best_label, best_ratio = scored[0]
    second_ratio = scored[1][2]
    if best_ratio < MISSION_DETAIL_COLOR_MIN_RATIO:
        return None
    if best_ratio < second_ratio + MISSION_DETAIL_COLOR_MIN_MARGIN:
        return None

    # 色块质心作为点击位置
    hsv_full = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    for kind, _label, _key, lower, upper in _DETAIL_BTN_HSV:
        if kind != best_kind:
            continue
        mask = cv2.inRange(hsv_full, np.array(lower), np.array(upper))
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            tap = (ox + cx, oy + cy)
        else:
            tap = ((action_roi[0] + action_roi[2]) // 2, (action_roi[1] + action_roi[3]) // 2)
        return best_kind, best_label, best_ratio, tap

    return None


def _template_conf_for_kind(
    scored: list[tuple[str, str, str, MatchResult]], kind: str
) -> float:
    for k, _label, _tpl, res in scored:
        if k == kind:
            return res.confidence
    return 0.0


def _refine_action_with_button_color(
    resolved_kind: str,
    resolved_label: str,
    resolved_conf: float,
    action_center: tuple[int, int] | None,
    scored: list[tuple[str, str, str, MatchResult]],
    screen: np.ndarray,
    action_roi: tuple[int, int, int, int],
) -> tuple[str, str, float, tuple[int, int] | None]:
    """模板与按钮主色不一致时，以颜色为准（蓝=探险，绿=营救，橙=出征）。"""
    if resolved_kind not in ("tent", "hero_journey", "small_monster"):
        return resolved_kind, resolved_label, resolved_conf, action_center

    color_hit = _classify_detail_action_by_color(screen, action_roi)
    if color_hit is None:
        return resolved_kind, resolved_label, resolved_conf, action_center

    c_kind, c_label, c_ratio, c_center = color_hit
    if c_kind == resolved_kind:
        return resolved_kind, resolved_label, resolved_conf, action_center or c_center

    tpl_for_color = _template_conf_for_kind(scored, c_kind)
    color_kinds = {resolved_kind, c_kind}
    is_blue_green_swap = color_kinds == {"tent", "hero_journey"}

    strong_color = (
        c_ratio >= MISSION_DETAIL_COLOR_STRONG_RATIO
        and tpl_for_color >= MISSION_DETAIL_COLOR_STRONG_MIN_TEMPLATE
    )
    soft_color = (
        is_blue_green_swap
        and c_ratio >= MISSION_DETAIL_COLOR_MIN_RATIO
        and tpl_for_color >= MISSION_DETAIL_COLOR_BACKUP_MIN_TEMPLATE
    )

    if strong_color or soft_color:
        logger.info(
            f"行动按钮颜色修正：模板 {resolved_label}({resolved_conf:.2f}) "
            f"→ {c_label}（color={c_ratio:.2f} tpl={tpl_for_color:.2f}）"
        )
        return (
            c_kind,
            c_label,
            max(resolved_conf, c_ratio, tpl_for_color),
            c_center,
        )

    return resolved_kind, resolved_label, resolved_conf, action_center


def _detail_action_center_fallback(
    action_roi: tuple[int, int, int, int],
) -> tuple[int, int]:
    return ((action_roi[0] + action_roi[2]) // 2, (action_roi[1] + action_roi[3]) // 2)


def save_mission_detail_debug(
    screen: np.ndarray,
    *,
    action_roi: tuple[int, int, int, int] = MISSION_DETAIL_ACTION_ROI,
    subtitle_roi: tuple[int, int, int, int] = MISSION_DETAIL_SUBTITLE_ROI,
) -> None:
    """识别失败时保存详情页 ROI 便于调模板。"""
    screen = _normalize_screen_for_scan(screen)
    debug_dir = TEMPLATE_DIR.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ax1, ay1, ax2, ay2 = action_roi
    sx1, sy1, sx2, sy2 = subtitle_roi
    cv2.imwrite(str(debug_dir / "lighthouse_detail_action_roi.png"), screen[ay1:ay2, ax1:ax2])
    cv2.imwrite(str(debug_dir / "lighthouse_detail_subtitle_roi.png"), screen[sy1:sy2, sx1:sx2])


def _match_subtitle_template(
    screen: np.ndarray,
    template_name: str,
    subtitle_roi: tuple[int, int, int, int],
    *,
    template_dir: Path = TEMPLATE_DIR,
    threshold: float = MISSION_DETAIL_BOUNTY_THRESHOLD,
    scales: tuple[float, ...] = BOUNTY_SUBTITLE_SCALES,
    search_side: str = "left",
    search_width_ratio: float = BOUNTY_SUBTITLE_SEARCH_WIDTH_RATIO,
) -> MatchResult:
    """副标题文字模板匹配。

    search_side:
      left  — 仅在 ROI 左侧搜索（「等级」、悬赏等靠左文案）
      right — 仅在 ROI 右侧搜索（大怪「号」靠右）
      full  — 全宽搜索
    """
    patch, (ox, oy) = _detail_action_roi_patch(screen, subtitle_roi)
    if patch.size == 0:
        return MatchResult(found=False)

    pw = patch.shape[1]
    search_w = max(1, int(pw * search_width_ratio))
    if search_side == "right":
        search_start = max(0, pw - search_w)
        search_patch = patch[:, search_start:]
        x_offset = search_start
    elif search_side == "full":
        search_patch = patch
        x_offset = 0
    else:
        search_patch = patch[:, :search_w]
        x_offset = 0

    template_path = template_dir / template_name
    if not template_path.is_file():
        logger.warning(f"灯塔详情模板缺失: {template_name}")
        return MatchResult(found=False)

    template = cv2.imread(str(template_path))
    if template is None:
        return MatchResult(found=False)

    patch_gray = cv2.cvtColor(search_patch, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    best_conf = 0.0
    best_loc: tuple[int, int, int, int] | None = None

    for scale in scales:
        scaled_gray = cv2.resize(
            tpl_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        scaled_bgr = cv2.resize(
            template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        th, tw = scaled_gray.shape[:2]
        if th > patch_gray.shape[0] or tw > patch_gray.shape[1]:
            continue

        gray_result = cv2.matchTemplate(patch_gray, scaled_gray, cv2.TM_CCOEFF_NORMED)
        color_result = cv2.matchTemplate(search_patch, scaled_bgr, cv2.TM_CCOEFF_NORMED)
        gray_score = float(gray_result.max())
        color_score = float(color_result.max())
        if color_score >= gray_score:
            max_val = color_score
            _, _, _, max_loc = cv2.minMaxLoc(color_result)
        else:
            max_val = gray_score
            _, _, _, max_loc = cv2.minMaxLoc(gray_result)
        if max_val <= best_conf:
            continue
        x, y = max_loc
        best_conf = max_val
        best_loc = (x, y, tw, th)

    if best_loc is None:
        return MatchResult(found=False, confidence=0.0)

    x, y, tw, th = best_loc
    center = (ox + x_offset + x + tw // 2, oy + y + th // 2)
    found = best_conf >= threshold
    return MatchResult(
        found=found,
        confidence=best_conf,
        center=center,
        top_left=(ox + x_offset + x, oy + y),
        size=(tw, th),
    )


def _subtitle_is_bounty(
    screen: np.ndarray,
    *,
    subtitle_roi: tuple[int, int, int, int] = MISSION_DETAIL_SUBTITLE_ROI,
    template_dir: Path = TEMPLATE_DIR,
) -> tuple[bool, str, float]:
    """副标题含「大师悬赏」或「宗师悬赏」（须匹配专属关键词，非单独「悬赏：」）。

    返回 (是否悬赏, 命中模板名, 置信度)。
    """
    x1, _y1, x2, _y2 = subtitle_roi
    roi_width = x2 - x1
    max_match_x = int(roi_width * BOUNTY_SUBTITLE_MAX_MATCH_X_RATIO)

    best_name = ""
    best_conf = 0.0
    for template_name, min_conf in BOUNTY_SUBTITLE_TEMPLATES:
        result = _match_subtitle_template(
            screen,
            template_name,
            subtitle_roi,
            template_dir=template_dir,
            threshold=0.0,
            scales=BOUNTY_SUBTITLE_SCALES,
        )
        rel_x = result.top_left[0] - x1 if result.top_left != (0, 0) else max_match_x + 1
        accepted = (
            result.confidence >= min_conf
            and rel_x <= max_match_x
        )
        logger.debug(
            f"悬赏副标题 {template_name} conf={result.confidence:.2f} "
            f"rel_x={rel_x} min={min_conf} accepted={accepted}"
        )
        if accepted and result.confidence > best_conf:
            best_conf = result.confidence
            best_name = template_name
    if best_name:
        return True, best_name, best_conf
    return False, "", best_conf


def _subtitle_is_beast_number(
    screen: np.ndarray,
    *,
    subtitle_roi: tuple[int, int, int, int] = MISSION_DETAIL_SUBTITLE_ROI,
    template_dir: Path = TEMPLATE_DIR,
) -> tuple[bool, float, bool]:
    """副标题右侧含「号」（如 …1号）→ 特殊大怪。

    返回 (accepted, confidence, position_ok)。
    """
    if not (template_dir / BEAST_KEYWORD_HAO).is_file():
        return False, 0.0, False

    x1, _y1, x2, _y2 = subtitle_roi
    roi_width = x2 - x1
    min_match_x = int(roi_width * BEAST_HAO_MIN_MATCH_X_RATIO)
    result = _match_subtitle_template(
        screen,
        BEAST_KEYWORD_HAO,
        subtitle_roi,
        template_dir=template_dir,
        threshold=0.0,
        scales=BOUNTY_SUBTITLE_SCALES,
        search_side="right",
        search_width_ratio=BEAST_HAO_SEARCH_WIDTH_RATIO,
    )
    rel_x = result.top_left[0] - x1 if result.top_left != (0, 0) else 0
    position_ok = rel_x >= min_match_x
    accepted = result.confidence >= BEAST_HAO_SUBTITLE_MIN and position_ok
    logger.debug(
        f"大怪副标题 {BEAST_KEYWORD_HAO} conf={result.confidence:.2f} "
        f"rel_x={rel_x} min_x={min_match_x} accepted={accepted}"
    )
    return accepted, result.confidence, position_ok


def _subtitle_is_level_monster(
    screen: np.ndarray,
    *,
    subtitle_roi: tuple[int, int, int, int] = MISSION_DETAIL_SUBTITLE_ROI,
    template_dir: Path = TEMPLATE_DIR,
) -> tuple[bool, float]:
    """副标题以「等级」开头 → 普通灯塔小怪（非大怪/悬赏）。"""
    x1, _y1, x2, _y2 = subtitle_roi
    max_match_x = int((x2 - x1) * MONSTER_LEVEL_MAX_MATCH_X_RATIO)
    result = _match_subtitle_template(
        screen,
        MONSTER_KEYWORD_LEVEL,
        subtitle_roi,
        template_dir=template_dir,
        threshold=0.0,
        scales=BOUNTY_SUBTITLE_SCALES,
    )
    rel_x = result.top_left[0] - x1 if result.top_left != (0, 0) else max_match_x + 1
    accepted = (
        result.confidence >= MONSTER_LEVEL_SUBTITLE_MIN and rel_x <= max_match_x
    )
    logger.debug(
        f"等级副标题 {MONSTER_KEYWORD_LEVEL} conf={result.confidence:.2f} "
        f"rel_x={rel_x} max_x={max_match_x} accepted={accepted}"
    )
    return accepted, result.confidence


def _classify_march_subtitle(
    screen: np.ndarray,
    *,
    subtitle_roi: tuple[int, int, int, int] = MISSION_DETAIL_SUBTITLE_ROI,
    template_dir: Path = TEMPLATE_DIR,
    base_confidence: float = 0.0,
) -> MissionDetailClassification | None:
    """出征按钮已确认后，仅根据副标题区分悬赏 / 小怪 / 特殊大怪。"""
    is_bounty, _bounty_tpl, bounty_conf = _subtitle_is_bounty(
        screen, subtitle_roi=subtitle_roi, template_dir=template_dir
    )
    if is_bounty:
        return MissionDetailClassification(
            kind="bounty_skip",
            label="大师/宗师悬赏",
            confidence=bounty_conf,
        )

    is_level, level_conf = _subtitle_is_level_monster(
        screen, subtitle_roi=subtitle_roi, template_dir=template_dir
    )
    if is_level:
        return None

    is_hao, hao_conf, hao_pos_ok = _subtitle_is_beast_number(
        screen, subtitle_roi=subtitle_roi, template_dir=template_dir
    )
    # 「等级」靠左且匹配足够强时，忽略右侧「号」误匹配（如 大角鹿）
    if is_hao and level_conf >= LEVEL_OVERRIDE_HAO_MIN:
        x1, _y1, x2, _y2 = subtitle_roi
        level_left = _match_subtitle_template(
            screen,
            MONSTER_KEYWORD_LEVEL,
            subtitle_roi,
            template_dir=template_dir,
            threshold=0.0,
            scales=BOUNTY_SUBTITLE_SCALES,
            search_side="left",
        )
        rel_x = (
            level_left.top_left[0] - x1 if level_left.top_left != (0, 0) else 999
        )
        max_left_x = int((x2 - x1) * MONSTER_LEVEL_MAX_MATCH_X_RATIO)
        if rel_x <= max_left_x:
            logger.info(
                f"左侧已有「等级」匹配（conf={level_conf:.2f} rel_x={rel_x}），"
                f"忽略右侧「号」误匹配 conf={hao_conf:.2f}"
            )
            return None

    if is_hao or (hao_pos_ok and hao_conf >= BEAST_HAO_SOFT_MIN):
        logger.info(
            f"副标题右侧含「号」，识别为特殊大怪（conf={hao_conf:.2f}）"
        )
        return MissionDetailClassification(
            kind="beast_skip",
            label="特殊大怪",
            confidence=max(base_confidence, hao_conf),
            beast_explicit=is_hao,
        )

    logger.info(
        f"出征副标题非悬赏、无「等级」、无「号」（bounty={bounty_conf:.2f} "
        f"level={level_conf:.2f} hao={hao_conf:.2f}），视为特殊大怪"
    )
    return MissionDetailClassification(
        kind="beast_skip",
        label="特殊大怪",
        confidence=max(base_confidence, level_conf, bounty_conf, hao_conf),
        beast_explicit=False,
    )


def classify_mission_detail_screen(
    screen: np.ndarray,
    *,
    template_dir: Path = TEMPLATE_DIR,
    action_roi: tuple[int, int, int, int] = MISSION_DETAIL_ACTION_ROI,
    subtitle_roi: tuple[int, int, int, int] = MISSION_DETAIL_SUBTITLE_ROI,
) -> MissionDetailClassification:
    """详情页底部行动按钮识别任务类型；出征需再判悬赏/等级怪。"""
    candidates: tuple[tuple[str, str, str], ...] = (
        ("small_monster", "出征", MISSION_BTN_MARCH),
        ("hero_journey", "探险", MISSION_BTN_ADVENTURE),
        ("tent", "营救", MISSION_BTN_RESCUE),
    )

    scored: list[tuple[str, str, str, MatchResult]] = []
    for kind, label, template_name in candidates:
        result = _match_action_button_in_screen_roi(
            screen,
            template_name,
            action_roi,
            template_dir=template_dir,
        )
        logger.debug(
            f"详情行动按钮 {label} ({template_name}) conf={result.confidence:.2f}"
        )
        scored.append((kind, label, template_name, result))

    scored.sort(key=lambda item: item[3].confidence, reverse=True)
    best_kind, best_label, _best_tpl, best_result = scored[0]
    second_conf = scored[1][3].confidence if len(scored) > 1 else 0.0
    margin = best_result.confidence - second_conf

    score_summary = "，".join(
        f"{label}={res.confidence:.2f}" for _, label, _, res in scored
    )
    logger.info(
        f"详情行动按钮得分: {score_summary}（最高 {best_label} "
        f"{best_result.confidence:.2f} margin={margin:.2f}）"
    )

    resolved_kind: str | None = None
    resolved_label = best_label
    resolved_conf = best_result.confidence
    action_center = best_result.center if best_result.center != (0, 0) else None

    # 仅当掩膜匹配真正达标（>=0.65）且领先第二名时才采纳模板结果
    if best_result.found and margin >= MISSION_DETAIL_BTN_MIN_MARGIN:
        resolved_kind = best_kind
    else:
        color_hit = _classify_detail_action_by_color(screen, action_roi)
        if color_hit is not None:
            c_kind, c_label, c_ratio, c_center = color_hit
            kind_tpl_conf = next(
                (res.confidence for k, _, _, res in scored if k == c_kind),
                0.0,
            )
            color_ok = c_ratio >= MISSION_DETAIL_COLOR_MIN_RATIO and (
                (
                    c_kind == best_kind
                    and kind_tpl_conf >= MISSION_DETAIL_COLOR_BACKUP_MIN_TEMPLATE
                )
                or (
                    c_ratio >= MISSION_DETAIL_COLOR_STRONG_RATIO
                    and kind_tpl_conf >= MISSION_DETAIL_COLOR_STRONG_MIN_TEMPLATE
                )
            )
            if color_ok:
                logger.info(
                    f"详情按钮模板未达标 (best={best_result.confidence:.2f})，"
                    f"颜色确认 {c_label} ratio={c_ratio:.2f} tpl={kind_tpl_conf:.2f}"
                )
                resolved_kind = c_kind
                resolved_label = c_label
                resolved_conf = max(best_result.confidence, c_ratio, kind_tpl_conf)
                action_center = c_center

    if resolved_kind is None or resolved_conf < MISSION_DETAIL_ABSOLUTE_MIN:
        is_bounty, bounty_tpl, bounty_conf = _subtitle_is_bounty(
            screen, subtitle_roi=subtitle_roi, template_dir=template_dir
        )
        if is_bounty:
            logger.info(
                f"行动按钮未达标（最高 {best_result.confidence:.2f}）但副标题识别为悬赏 "
                f"conf={bounty_conf:.2f} tpl={bounty_tpl}"
            )
            return MissionDetailClassification(
                kind="bounty_skip",
                label="大师/宗师悬赏",
                confidence=bounty_conf,
            )
        march_sub = _classify_march_subtitle(
            screen,
            subtitle_roi=subtitle_roi,
            template_dir=template_dir,
            base_confidence=best_result.confidence,
        )
        if march_sub is not None and march_sub.kind == "beast_skip":
            return march_sub
        logger.info(
            f"详情页识别不足（最高 {best_result.confidence:.2f} < "
            f"{MISSION_DETAIL_ABSOLUTE_MIN}），视为未识别"
        )
        return MissionDetailClassification(
            kind="unknown",
            label="未识别",
            confidence=best_result.confidence,
        )

    if action_center is None:
        action_center = _detail_action_center_fallback(action_roi)

    resolved_kind, resolved_label, resolved_conf, action_center = (
        _refine_action_with_button_color(
            resolved_kind,
            resolved_label,
            resolved_conf,
            action_center,
            scored,
            screen,
            action_roi,
        )
    )

    if resolved_kind == "small_monster":
        march_sub = _classify_march_subtitle(
            screen,
            subtitle_roi=subtitle_roi,
            template_dir=template_dir,
            base_confidence=resolved_conf,
        )
        if march_sub is not None:
            if march_sub.kind == "beast_skip":
                save_mission_detail_debug(screen)
            return march_sub
        return MissionDetailClassification(
            kind="small_monster",
            label="灯塔小怪",
            confidence=resolved_conf,
            action_center=action_center,
        )

    return MissionDetailClassification(
        kind=resolved_kind,
        label=resolved_label,
        confidence=resolved_conf,
        action_center=action_center,
    )
def configure_lighthouse_scan(*, event_period: bool = False) -> str:
    """按是否活动期间切换差分背景图，切换时清空缓存。"""
    global _active_map_bg_name, _MAP_BG_SCREEN, _MAP_BG_ROI, _event_period
    desired = LIGHTHOUSE_MAP_BG_EVENT if event_period else LIGHTHOUSE_MAP_BG_NORMAL
    _event_period = event_period
    if desired != _active_map_bg_name:
        _active_map_bg_name = desired
        _MAP_BG_SCREEN = None
        _MAP_BG_ROI = None
    logger.info(f"灯塔扫描背景图: {_active_map_bg_name}")
    return _active_map_bg_name


def _is_event_period() -> bool:
    return _event_period


def _extract_pin_patch(
    roi: np.ndarray,
    screen_center: tuple[int, int],
    roi_offset: tuple[int, int],
    *,
    half: int = LIGHTHOUSE_PIN_PATCH_HALF,
) -> np.ndarray:
    cx = screen_center[0] - roi_offset[0]
    cy = screen_center[1] - roi_offset[1]
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(roi.shape[1], cx + half)
    y2 = min(roi.shape[0], cy + half)
    return roi[y1:y2, x1:x2]


def is_beast_map_pin(
    screen: np.ndarray,
    center: tuple[int, int],
    *,
    scan_roi: tuple[int, int, int, int] = LIGHTHOUSE_SCAN_ROI,
) -> MatchResult:
    """地图图钉位置是否匹配特殊大怪（熊头/兽形橙钉）模板。"""
    screen = _normalize_screen_for_scan(screen)
    x1, y1, x2, y2 = scan_roi
    roi = screen[y1:y2, x1:x2]
    patch = _extract_pin_patch(roi, center, (x1, y1))
    if patch.size == 0:
        return MatchResult(found=False)
    template_path = TEMPLATE_DIR / BEAST_PIN_TEMPLATE
    if not template_path.is_file():
        return MatchResult(found=False)
    vision = Vision(TEMPLATE_DIR, threshold=BEAST_PIN_MATCH_MIN)
    result = vision.match_template_multiscale(patch, BEAST_PIN_TEMPLATE)
    if result.found:
        logger.debug(
            f"地图图钉 ({center[0]},{center[1]}) 匹配特殊大怪 "
            f"conf={result.confidence:.2f}"
        )
    return result


def tag_scanned_missions(
    screen: np.ndarray,
    missions: tuple[LighthouseMission, ...],
    *,
    scan_roi: tuple[int, int, int, int] = LIGHTHOUSE_SCAN_ROI,
) -> tuple[LighthouseMission, ...]:
    """扫描后补类型：标记特殊大怪图钉，供任务层跳过。"""
    if not missions:
        return missions

    tagged: list[LighthouseMission] = []
    for mission in missions:
        beast = is_beast_map_pin(screen, mission.center, scan_roi=scan_roi)
        if beast.found:
            tagged.append(
                LighthouseMission(
                    kind="small_monster_beast",
                    label="特殊大怪",
                    template=BEAST_PIN_TEMPLATE,
                    center=mission.center,
                    confidence=beast.confidence,
                    top_left=mission.top_left,
                    size=mission.size,
                )
            )
        else:
            tagged.append(mission)
    return tuple(tagged)


def scan_mission_icons(
    screen: np.ndarray,
    *,
    scan_roi: tuple[int, int, int, int] = LIGHTHOUSE_SCAN_ROI,
    interrupted: Callable[[], bool] | None = None,
) -> LighthouseScanResult:
    """检测地图上的任务图钉位置，不区分任务类型。

    相对固定背景图做差分，再结合鲜艳色 + 倒水滴轮廓过滤；
    自动排除背景中的飞机、超级大怪及中央基地等干扰。
    """
    global _scan_interrupt_cb
    prev_interrupt = _scan_interrupt_cb
    _scan_interrupt_cb = interrupted
    try:
        return _scan_icons_impl(screen, scan_roi=scan_roi)
    except InterruptedError:
        return LighthouseScanResult(mission=None, best_label="已停止")
    finally:
        _scan_interrupt_cb = prev_interrupt


def _load_map_background_screen() -> np.ndarray | None:
    """加载完整灯塔情报页背景（720×1280）。"""
    global _MAP_BG_SCREEN
    if _MAP_BG_SCREEN is not None:
        return _MAP_BG_SCREEN

    path = TEMPLATE_DIR / _active_map_bg_name
    image = cv2.imread(str(path))
    if image is None:
        logger.warning(f"灯塔地图背景缺失: {path}")
        return None

    _MAP_BG_SCREEN = _normalize_screen_for_scan(image)
    return _MAP_BG_SCREEN


def _load_map_background_roi() -> np.ndarray | None:
    """加载灯塔情报页地图背景（720×1280 竖屏 ROI 切片）。"""
    global _MAP_BG_ROI
    if _MAP_BG_ROI is not None:
        return _MAP_BG_ROI

    screen = _load_map_background_screen()
    if screen is None:
        return None

    x1, y1, x2, y2 = LIGHTHOUSE_SCAN_ROI
    _MAP_BG_ROI = screen[y1:y2, x1:x2].copy()
    return _MAP_BG_ROI


def is_lighthouse_intel_screen(screen: np.ndarray) -> bool:
    """当前截图是否为灯塔情报页（非野外大地图/出征界面）。"""
    normalized = _normalize_screen_for_scan(screen)
    bg = _load_map_background_screen()
    if bg is None:
        return False

    x1, y1, x2, y2 = LIGHTHOUSE_HEADER_ROI
    patch = normalized[y1:y2, x1:x2]
    reference = bg[y1:y2, x1:x2]
    if patch.shape != reference.shape:
        return False
    mean_diff = float(cv2.absdiff(patch, reference).mean())
    return mean_diff <= LIGHTHOUSE_HEADER_MEAN_DIFF_MAX


def _align_background_roi(
    bg_roi: np.ndarray, roi_shape: tuple[int, ...]
) -> np.ndarray:
    if bg_roi.shape[:2] == roi_shape[:2]:
        return bg_roi
    return cv2.resize(
        bg_roi,
        (roi_shape[1], roi_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def _build_vivid_pin_mask(hsv: np.ndarray) -> np.ndarray:
    """任务图钉常见高饱和色（橙 / 紫 / 蓝）。"""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in (
        ((10, 120, 140), (30, 255, 255)),
        ((125, 55, 100), (155, 255, 255)),
        ((98, 65, 110), (118, 255, 255)),
        ((155, 80, 140), (175, 255, 255)),
    ):
        mask = cv2.bitwise_or(
            mask, cv2.inRange(hsv, np.array(lower), np.array(upper))
        )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _contour_vivid_ratio(contour: np.ndarray, hsv: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0.0
    patch_hsv = hsv[y : y + h, x : x + w]
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    sat = patch_hsv[:, :, 1]
    val = patch_hsv[:, :, 2]
    vivid = ((sat > 100) & (val > 130)).astype(np.float32)
    vivid[mask == 0] = 0.0
    if mask.sum() <= 0:
        return 0.0
    return float(vivid.sum() / (mask.sum() / 255.0))


def _is_teardrop_contour(contour: np.ndarray, *, relaxed: bool = False) -> bool:
    """倒水滴：上宽下窄、竖向略长。"""
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    area_max = BG_PIN_AREA_MAX if not relaxed else 3200
    if area < BG_PIN_AREA_MIN or area > area_max:
        return False
    aspect = h / max(w, 1)
    aspect_min = BG_TEARDROP_ASPECT_MIN if not relaxed else 0.75
    if aspect < aspect_min or aspect > BG_TEARDROP_ASPECT_MAX:
        return False

    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)

    top_band = mask[: max(1, h // 4), :]
    bottom_band = mask[3 * h // 4 :, :]
    top_width = int(np.count_nonzero(top_band.any(axis=0)))
    bottom_width = int(np.count_nonzero(bottom_band.any(axis=0)))
    if bottom_width <= 0:
        return False
    ratio_min = BG_TEARDROP_TOP_BOTTOM_MIN if not relaxed else 1.0
    return (top_width / bottom_width) >= ratio_min


def _snap_to_vivid_peak(
    hsv: np.ndarray,
    center: tuple[int, int],
    roi_offset: tuple[int, int],
    *,
    radius: int = 36,
) -> tuple[tuple[int, int], float]:
    """将点击点吸附到附近的鲜艳色图钉头部。"""
    ox, oy = roi_offset
    base_rx, base_ry = center[0] - ox, center[1] - oy
    best_vivid = 0.0
    best_center = center
    for dy in range(-radius, radius + 1, 4):
        for dx in range(-radius, radius + 1, 4):
            rx, ry = base_rx + dx, base_ry + dy
            if rx < 0 or ry < 0 or rx >= hsv.shape[1] or ry >= hsv.shape[0]:
                continue
            vivid = _patch_vivid_ratio(hsv, (rx, ry), radius=14)
            if vivid > best_vivid:
                best_vivid = vivid
                best_center = (ox + rx, oy + ry)
    return best_center, best_vivid


def _accept_pin_candidate(
    center: tuple[int, int],
    confidence: float,
    roi: np.ndarray,
    hsv: np.ndarray,
    roi_offset: tuple[int, int],
    diff_mask: np.ndarray,
    *,
    blob_area: float = 0.0,
    blob_width: int = 0,
    blob_height: int = 0,
) -> tuple[tuple[int, int], float] | None:
    if not _screen_center_in_map(center):
        return None
    if _is_ignored_background_pin(center):
        return None
    if _is_super_boss_point(center, roi, hsv, roi_offset):
        return None
    if _is_plane_point(
        center,
        blob_area=blob_area,
        blob_width=blob_width,
        blob_height=blob_height,
    ):
        return None

    diff_ratio, refined = _pin_diff_support(center, diff_mask, roi_offset)
    if diff_ratio < BG_DIFF_PIN_SUPPORT_MIN:
        return None

    snapped, vivid = _snap_to_vivid_peak(hsv, refined, roi_offset)
    if vivid >= BG_DIFF_VIVID_MIN:
        refined = snapped
    elif confidence < BG_DIFF_VIVID_MIN and diff_ratio < 0.40:
        return None
    if _is_plane_point(refined):
        return None
    return refined, float(max(confidence, vivid, diff_ratio))


def _extract_pins_from_diff_blob(
    blob_contour: np.ndarray,
    search_mask: np.ndarray,
    hsv: np.ndarray,
    roi: np.ndarray,
    roi_offset: tuple[int, int],
    diff_mask: np.ndarray,
) -> list[tuple[tuple[int, int], float]]:
    bx, by, bw, bh = cv2.boundingRect(blob_contour)
    pad = 6
    lx = max(0, bx - pad)
    ly = max(0, by - pad)
    rx = min(search_mask.shape[1], bx + bw + pad)
    ry = min(search_mask.shape[0], by + pad + bh)
    local_search = search_mask[ly:ry, lx:rx]
    local_hsv = hsv[ly:ry, lx:rx]
    if local_search.size == 0:
        return []

    blob_area = cv2.contourArea(blob_contour)
    found: list[tuple[tuple[int, int], float]] = []
    seen: set[tuple[int, int]] = set()

    for erode_iters in (0, 2, 4, 6):
        mask = local_search
        if erode_iters:
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=erode_iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        sub_contours = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        for sub in sub_contours:
            sub_area = cv2.contourArea(sub)
            if sub_area < 50:
                continue
            center: tuple[int, int] | None = None
            confidence = _contour_vivid_ratio(sub, local_hsv)
            if _is_teardrop_contour(sub):
                center = _contour_screen_center(
                    sub,
                    (roi_offset[0] + lx, roi_offset[1] + ly),
                    hsv=local_hsv,
                    use_vivid_peak=True,
                )
            elif sub_area >= 120:
                sx, sy, sw, sh = cv2.boundingRect(sub)
                sub_hsv = local_hsv[sy : sy + sh, sx : sx + sw]
                sat = sub_hsv[:, :, 1].astype(np.float32)
                val = sub_hsv[:, :, 2]
                score = sat * (val > 130)
                if float(score.max()) > 0:
                    _, _, _, peak = cv2.minMaxLoc(score)
                    center = (
                        roi_offset[0] + lx + sx + peak[0],
                        roi_offset[1] + ly + sy + peak[1],
                    )
            if center is None:
                continue
            key = (center[0] // 8, center[1] // 8)
            if key in seen:
                continue
            accepted = _accept_pin_candidate(
                center,
                confidence,
                roi,
                hsv,
                roi_offset,
                diff_mask,
                blob_area=blob_area,
                blob_width=bw,
                blob_height=bh,
            )
            if accepted is None:
                continue
            seen.add(key)
            found.append(accepted)
    return _merge_nearby_pin_candidates(found)[:BG_MAX_PINS_PER_DIFF_BLOB]


def _find_mission_pins_bg_diff(
    roi: np.ndarray,
    bg_roi: np.ndarray,
    roi_offset: tuple[int, int],
) -> list[tuple[tuple[int, int], float]]:
    """背景差分定位新增任务图钉（图钉 + 地面光晕会连成大片，需分块提取）。"""
    bg_roi = _align_background_roi(bg_roi, roi.shape)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    diff = cv2.absdiff(roi, bg_roi)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_mask = cv2.threshold(
        diff_gray, BG_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    diff_mask = cv2.morphologyEx(
        diff_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    search_mask = cv2.bitwise_and(_build_vivid_pin_mask(hsv), diff_mask)

    candidates: list[tuple[tuple[int, int], float]] = []
    rejected = {"ui": 0, "plane": 0, "boss": 0, "support": 0, "merged": 0}

    for blob in cv2.findContours(
        diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]:
        _check_scan_interrupted()
        blob_area = cv2.contourArea(blob)
        if blob_area < BG_DIFF_BLOB_MIN_AREA:
            continue

        bx, by, bw, bh = cv2.boundingRect(blob)
        moments = cv2.moments(blob)
        if moments["m00"] <= 0:
            continue
        blob_cx = int(moments["m10"] / moments["m00"]) + roi_offset[0]
        blob_cy = int(moments["m01"] / moments["m00"]) + roi_offset[1]

        if _is_ui_diff_blob(blob_cy, blob_height=bh, blob_width=bw):
            rejected["ui"] += 1
            continue
        if _is_plane_point(
            (blob_cx, blob_cy),
            blob_area=blob_area,
            blob_width=bw,
            blob_height=bh,
        ):
            rejected["plane"] += 1
            continue
        if _is_super_boss_point((blob_cx, blob_cy), roi, hsv, roi_offset):
            rejected["boss"] += 1
            continue

        for item in _extract_pins_from_diff_blob(
            blob, search_mask, hsv, roi, roi_offset, diff_mask
        ):
            candidates.append(item)

    color_pins = _find_mission_pin_centers(roi, roi_offset)
    for pin in color_pins:
        _check_scan_interrupted()
        accepted = _accept_pin_candidate(
            pin,
            _patch_vivid_ratio(
                hsv,
                (pin[0] - roi_offset[0], pin[1] - roi_offset[1]),
            ),
            roi,
            hsv,
            roi_offset,
            diff_mask,
        )
        if accepted is None:
            rejected["support"] += 1
            continue
        if any(
            abs(accepted[0][0] - c[0][0]) < MISSION_PIN_MERGE_DISTANCE
            and abs(accepted[0][1] - c[0][1]) < MISSION_PIN_MERGE_DISTANCE
            for c in candidates
        ):
            rejected["merged"] += 1
            continue
        candidates.append(accepted)

    candidates = _merge_nearby_pin_candidates(candidates)

    if candidates or any(rejected.values()):
        logger.debug(
            "灯塔差分扫描：保留 {} 个，剔除 ui={} plane={} boss={} "
            "support={} merged={}",
            len(candidates),
            rejected["ui"],
            rejected["plane"],
            rejected["boss"],
            rejected["support"],
            rejected["merged"],
        )
    return candidates


def _scan_icons_impl(
    screen: np.ndarray,
    *,
    scan_roi: tuple[int, int, int, int],
) -> LighthouseScanResult:
    """背景差分定位任务图钉，不区分类型。"""
    screen = _normalize_screen_for_scan(screen)
    x1, y1, x2, y2 = scan_roi
    roi = screen[y1:y2, x1:x2]
    if roi.size == 0:
        return LighthouseScanResult(mission=None)

    bg_roi = _load_map_background_roi()
    if bg_roi is None:
        logger.error(
            f"缺少灯塔背景图 {_active_map_bg_name}，无法扫描任务图标"
        )
        return LighthouseScanResult(mission=None)

    pin_candidates = _find_mission_pins_bg_diff(roi, bg_roi, (x1, y1))
    total_candidates = len(pin_candidates)
    if not pin_candidates:
        return LighthouseScanResult(mission=None, candidate_locations=0)

    half = LIGHTHOUSE_PIN_PATCH_HALF
    missions: list[LighthouseMission] = []
    for center, confidence in pin_candidates:
        roi_cx = center[0] - x1
        roi_cy = center[1] - y1
        x1p = max(0, roi_cx - half)
        y1p = max(0, roi_cy - half)
        x2p = min(roi.shape[1], roi_cx + half)
        y2p = min(roi.shape[0], roi_cy + half)
        missions.append(
            LighthouseMission(
                kind="",
                label="图标",
                template="",
                center=center,
                confidence=confidence,
                top_left=(x1 + x1p, y1 + y1p),
                size=(x2p - x1p, y2p - y1p),
            )
        )

    missions = _dedupe_missions_by_slot(missions, merge_distance=28)
    missions.sort(key=lambda item: item.center[1])

    best = missions[0] if missions else None
    return LighthouseScanResult(
        mission=best,
        missions=tuple(missions),
        best_confidence=best.confidence if best else 0.0,
        best_label="图标",
        candidate_locations=total_candidates,
    )
