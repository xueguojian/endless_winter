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

from core.vision import MatchResult

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"

# 情报页中间地图区域，排除顶部刷新条与底部等级栏
LIGHTHOUSE_SCAN_ROI = (20, 130, 700, 1020)
LIGHTHOUSE_MAP_BG_NAME = "lighthouse/lighthouse_map_bg.png"
# 情报页顶部刷新条，用于区分「情报页」与「野外大地图」
LIGHTHOUSE_HEADER_ROI = (0, 155, 720, 210)
LIGHTHOUSE_HEADER_MEAN_DIFF_MAX = 18.0
# 与背景图差分：仅保留相对背景新出现且颜色鲜艳的图钉
BG_DIFF_THRESHOLD = 28
BG_DIFF_VIVID_MIN = 0.10
BG_DIFF_PIN_SUPPORT_MIN = 0.12
BG_PIN_AREA_MIN = 60
BG_PIN_AREA_MAX = 1800
BG_DIFF_BLOB_MIN_AREA = 800
BG_TEARDROP_TOP_BOTTOM_MIN = 1.05
BG_TEARDROP_ASPECT_MIN = 0.85
BG_TEARDROP_ASPECT_MAX = 2.8
BOSS_CONTOUR_AREA_MIN = 4500
PLANE_CONTOUR_AREA_MIN = 8000
SUPER_BOSS_Y_MAX = 320
UI_EXCLUDE_Y_MAX = 230
UI_EXCLUDE_X_MAX = 120
PLANE_EXCLUDE_Y_MIN = 820
PLANE_EXCLUDE_X_MIN = 480
BG_PIN_DIFF_SEARCH_RADIUS = 50
LIGHTHOUSE_MATCH_THRESHOLD = 0.55
LIGHTHOUSE_CANDIDATE_THRESHOLD = 0.28
LIGHTHOUSE_HERO_CANDIDATE_THRESHOLD = 0.30
LIGHTHOUSE_TYPE_SCORE_MARGIN = 0.02
LIGHTHOUSE_MIN_LOCATION_SCORE = 0.18
LIGHTHOUSE_RELAXED_THRESHOLD_FLOOR = 0.20
LIGHTHOUSE_FAST_SCALES = (0.85, 1.0, 1.15)
LIGHTHOUSE_MIN_MATCH_DISTANCE = 40
LIGHTHOUSE_SLOT_MERGE_DISTANCE = 18
LIGHTHOUSE_NEIGHBOR_DISTANCE = 52
LIGHTHOUSE_CENTER_PATCH_HALF = 32
LIGHTHOUSE_MAX_RAW_MATCHES_PER_KIND = 16
LIGHTHOUSE_MAX_CLASSIFY_CANDIDATES = 14
WHITE_SYMBOL_THRESHOLD = 175
PATCH_EXTRACT_PAD = 8
WHITE_SCORE_WEIGHT = 0.2
EDGE_SCORE_WEIGHT = 0.8
HERO_JOURNEY_MIN_EDGE_SCORE = 0.62
HERO_JOURNEY_MIN_CONFIDENCE = 0.72
TENT_MIN_CONFIDENCE = 0.55
TENT_MIN_EDGE_SCORE = 0.55
MONSTER_MIN_EDGE_SCORE = 0.70
MONSTER_MIN_CONFIDENCE = 0.65
MONSTER_TEXTURE_WHITE_MIN = 0.50
MONSTER_TEXTURE_EDGE_MAX = 0.35
BEAST_CANDIDATE_THRESHOLD = 0.50
PIN_SCAN_MIN_CONFIDENCE = 0.28
PIN_CLASSIFY_MARGIN = 0.06
HERO_MONSTER_CLASSIFY_MARGIN = 0.12
HERO_EDGE_OVER_MONSTER_MIN = 0.12
HERO_COLOR_RATIO_MIN = 0.22
HERO_JOURNEY_PIN_MIN_CONFIDENCE = 0.28
MONSTER_PIN_MIN_CONFIDENCE = 0.25
TENT_PIN_MIN_CONFIDENCE = 0.28
LIGHTHOUSE_PIN_PATCH_HALF = 36
PIN_REFINE_TRIGGER_SCORE = 0.48
PIN_REFINE_MIN_SCORE = 0.18
PIN_REFINE_SEARCH_RADIUS = 22
PIN_REFINE_FINE_RADIUS = 10
PIN_REFINE_SEARCH_STEP = 8
PIN_REFINE_FINE_STEP = 2
PIN_REFINE_MAX_DRIFT = 56
PIN_REFINE_CHASE_PASSES = 0
LIGHTHOUSE_MAX_PIN_CANDIDATES = 18
LIGHTHOUSE_DIRECT_MATCH_MIN = 0.55
LIGHTHOUSE_PIN_PROXIMITY_RADIUS = 50
LIGHTHOUSE_PIN_PROXIMITY_BOOST = 0.18
HERO_ORANGE_RATIO_MIN = 0.12
HERO_MIN_EDGE_FOR_WEAK_COLOR = 0.30
HERO_PURPLE_BLUE_EDGE_MIN = 0.32
PIN_TEMPLATE_REFINE_DISABLED_AREA = 100_000
PIN_VIVID_COLOR_MIN = 0.15
PIN_SATURATION_PEAK_MIN = 150
PIN_SATURATION_PEAK_RELAXED = 110
PIN_VIVID_RELAXED_MIN = 0.05
PIN_STRONG_SCORE_WITHOUT_VIVID = 0.55
PIN_BLOB_OPEN_KERNEL = 5
PIN_LARGE_BLOB_MIN_AREA = 900
SNOW_TEXTURE_WHITE_MIN = 0.72
SNOW_TEXTURE_EDGE_MIN = 0.42
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
# 背景干扰：顶部强力大怪 / 左下移动飞机 / 中央固定基地
BOSS_ZONE_Y_MAX = 400
BOSS_ZONE_X = (160, 560)
BOSS_ZONE_MIN_AREA = 480
PLANE_ZONE_Y_MIN = 640
PLANE_ZONE_X_MAX = 220
BASE_EXCLUDE_CENTER = (360, 530)
BASE_EXCLUDE_RADIUS = 95


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


MISSION_DEFINITIONS: tuple[tuple[str, str, str, str], ...] = (
    ("hero_journey", "英雄之旅", "lighthouse/hero_journey.png", "lighthouse/hero_journey_edges.png"),
    ("tent", "帐篷", "lighthouse/tent.png", "lighthouse/tent_edges.png"),
    ("small_monster", "小怪", "lighthouse/small_monster.png", "lighthouse/small_monster_edges.png"),
    ("small_monster_beast", "小怪", "lighthouse/small_monster_beast.png", "lighthouse/small_monster_beast_edges.png"),
)

MONSTER_KINDS = frozenset({"small_monster", "small_monster_beast"})

_ALT_TEMPLATE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("hero_journey", "lighthouse/hero_journey_alt.png", "lighthouse/hero_journey_alt_edges.png"),
    ("hero_journey", "lighthouse/hero_journey_orange.png", "lighthouse/hero_journey_orange_edges.png"),
    ("tent", "lighthouse/tent_blue.png", "lighthouse/tent_blue_edges.png"),
    ("tent", "lighthouse/tent_blue_alt.png", "lighthouse/tent_blue_alt_edges.png"),
    ("small_monster", "lighthouse/small_monster_blue.png", "lighthouse/small_monster_blue_edges.png"),
    ("small_monster", "lighthouse/small_monster_purple.png", "lighthouse/small_monster_purple_edges.png"),
    ("small_monster", "lighthouse/small_monster_purple_alt.png", "lighthouse/small_monster_purple_alt_edges.png"),
    ("small_monster", "lighthouse/small_monster_orange.png", "lighthouse/small_monster_orange_edges.png"),
    ("small_monster", "lighthouse/small_monster_orange_alt.png", "lighthouse/small_monster_orange_alt_edges.png"),
)

_TYPE_TEMPLATES: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str, str]] | None = None
_ALT_TYPE_TEMPLATES: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] | None = None
_MAP_BG_ROI: np.ndarray | None = None
_MAP_BG_SCREEN: np.ndarray | None = None
_scan_interrupt_cb: Callable[[], bool] | None = None


def _check_scan_interrupted() -> None:
    if _scan_interrupt_cb and _scan_interrupt_cb():
        raise InterruptedError("任务已停止")


def _white_symbol_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, WHITE_SYMBOL_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _load_type_templates() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str, str]]:
    global _TYPE_TEMPLATES
    if _TYPE_TEMPLATES is not None:
        return _TYPE_TEMPLATES

    loaded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str, str]] = {}
    for kind, label, symbol_name, edge_name in MISSION_DEFINITIONS:
        symbol_path = TEMPLATE_DIR / symbol_name
        edge_path = TEMPLATE_DIR / edge_name
        symbol = cv2.imread(str(symbol_path))
        edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        if symbol is None or edge is None:
            logger.warning(f"灯塔模板缺失: {symbol_name} / {edge_name}")
            continue
        loaded[kind] = (symbol, edge, _white_symbol_mask(symbol), symbol_name, label)
    _TYPE_TEMPLATES = loaded
    return loaded


def _load_alt_type_templates() -> dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    global _ALT_TYPE_TEMPLATES
    if _ALT_TYPE_TEMPLATES is not None:
        return _ALT_TYPE_TEMPLATES

    loaded: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for kind, symbol_name, edge_name in _ALT_TEMPLATE_SPECS:
        symbol_path = TEMPLATE_DIR / symbol_name
        edge_path = TEMPLATE_DIR / edge_name
        symbol = cv2.imread(str(symbol_path))
        edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        if symbol is None or edge is None:
            continue
        loaded.setdefault(kind, []).append((symbol, edge, _white_symbol_mask(symbol)))
    _ALT_TYPE_TEMPLATES = loaded
    return loaded


def _score_kind_on_patch(
    patch: np.ndarray,
    kind: str,
    symbol: np.ndarray,
    edge: np.ndarray,
    *,
    sym_mask: np.ndarray | None = None,
) -> float:
    score = _score_patch_against_type(
        patch, symbol, edge, scales=PIN_CLASSIFY_SCALES, sym_mask=sym_mask
    )
    if score >= 0.35:
        return score
    for alt_symbol, alt_edge, alt_mask in _load_alt_type_templates().get(kind, []):
        score = max(
            score,
            _score_patch_against_type(
                patch,
                alt_symbol,
                alt_edge,
                scales=PIN_CLASSIFY_SCALES,
                sym_mask=alt_mask,
            ),
        )
    return score


def _normalize_screen_for_scan(screen: np.ndarray) -> np.ndarray:
    h, w = screen.shape[:2]
    if (w, h) == (PORTRAIT_WIDTH, PORTRAIT_HEIGHT):
        return screen
    return cv2.resize(
        screen,
        (PORTRAIT_WIDTH, PORTRAIT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )


def _to_edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.Canny(gray, 40, 120)


def _find_symbol_matches_on_mask(
    screen_mask: np.ndarray,
    template_mask: np.ndarray,
    *,
    threshold: float,
    scales: tuple[float, ...] = LIGHTHOUSE_FAST_SCALES,
) -> list[MatchResult]:
    matches: list[MatchResult] = []

    for scale in scales:
        if scale == 1.0:
            template = template_mask
        else:
            template = cv2.resize(
                template_mask,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
        th, tw = template.shape[:2]
        if th > screen_mask.shape[0] or tw > screen_mask.shape[1]:
            continue

        result = cv2.matchTemplate(screen_mask, template, cv2.TM_CCOEFF_NORMED)
        locs = np.where(result >= threshold)
        for x, y in zip(locs[1], locs[0]):
            conf = float(result[y, x])
            matches.append(
                MatchResult(
                    found=True,
                    confidence=conf,
                    center=(x + tw // 2, y + th // 2),
                    top_left=(int(x), int(y)),
                    size=(tw, th),
                )
            )

    matches.sort(key=lambda item: item.confidence, reverse=True)
    return matches[:LIGHTHOUSE_MAX_RAW_MATCHES_PER_KIND]


def _nms_tagged_locations(
    matches: list[tuple[str, MatchResult]],
) -> list[tuple[str, MatchResult]]:
    if not matches:
        return []

    ordered = sorted(matches, key=lambda item: item[1].confidence, reverse=True)
    kept: list[tuple[str, MatchResult]] = []

    for kind, candidate in ordered:
        cx, cy = candidate.center
        too_close = False
        for _, existing in kept:
            ex, ey = existing.center
            if (
                abs(cx - ex) < LIGHTHOUSE_MIN_MATCH_DISTANCE
                and abs(cy - ey) < LIGHTHOUSE_MIN_MATCH_DISTANCE
            ):
                too_close = True
                break
        if not too_close:
            kept.append((kind, candidate))

    return kept[:LIGHTHOUSE_MAX_CLASSIFY_CANDIDATES]


def _extract_patch(roi: np.ndarray, match: MatchResult) -> tuple[np.ndarray, tuple[int, int]]:
    x, y = match.top_left
    w, h = match.size
    x1 = max(0, x - PATCH_EXTRACT_PAD)
    y1 = max(0, y - PATCH_EXTRACT_PAD)
    x2 = min(roi.shape[1], x + w + PATCH_EXTRACT_PAD)
    y2 = min(roi.shape[0], y + h + PATCH_EXTRACT_PAD)
    return roi[y1:y2, x1:x2], (x1, y1)


def _score_patch_against_type(
    patch: np.ndarray,
    symbol_template: np.ndarray,
    edge_template: np.ndarray,
    *,
    scales: tuple[float, ...] = LIGHTHOUSE_FAST_SCALES,
    sym_mask: np.ndarray | None = None,
) -> float:
    """综合评分：灰度(0.60) + 边缘(0.25) + 白符号(0.15)。

    旧方案仅用边缘+白符号（权重 0.8/0.2），但游戏图钉边缘稀疏，
    得分极低（<0.20）。直接灰度模板匹配在实测中可达 0.60-0.92，
    故大幅度提升灰度权重。
    """
    patch_edges = _to_edges(patch)
    patch_mask = _white_symbol_mask(patch)
    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(symbol_template, cv2.COLOR_BGR2GRAY)
    sym_mask = sym_mask if sym_mask is not None else _white_symbol_mask(symbol_template)
    best = 0.0

    for scale in scales:
        edge_t = cv2.resize(
            edge_template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        sym_t = cv2.resize(
            sym_mask,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        gray_t = cv2.resize(
            tpl_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        eh, ew = edge_t.shape[:2]
        sh, sw = sym_t.shape[:2]
        gh, gw = gray_t.shape[:2]
        edge_score = 0.0
        white_score = 0.0
        gray_score = 0.0

        if eh <= patch_edges.shape[0] and ew <= patch_edges.shape[1]:
            edge_score = float(
                cv2.matchTemplate(patch_edges, edge_t, cv2.TM_CCOEFF_NORMED).max()
            )
        if sh <= patch_mask.shape[0] and sw <= patch_mask.shape[1]:
            white_score = float(
                cv2.matchTemplate(patch_mask, sym_t, cv2.TM_CCOEFF_NORMED).max()
            )
        if gh <= patch_gray.shape[0] and gw <= patch_gray.shape[1]:
            gray_score = float(
                cv2.matchTemplate(patch_gray, gray_t, cv2.TM_CCOEFF_NORMED).max()
            )
        combined = 0.15 * white_score + 0.25 * edge_score + 0.60 * gray_score
        best = max(best, combined)

    return best


def _edge_match_score(patch: np.ndarray, edge_template: np.ndarray) -> float:
    patch_edges = _to_edges(patch)
    if (
        edge_template.shape[0] > patch_edges.shape[0]
        or edge_template.shape[1] > patch_edges.shape[1]
    ):
        return 0.0
    return float(
        cv2.matchTemplate(patch_edges, edge_template, cv2.TM_CCOEFF_NORMED).max()
    )


def _white_match_score(
    patch: np.ndarray,
    symbol_template: np.ndarray,
    *,
    sym_mask: np.ndarray | None = None,
) -> float:
    patch_mask = _white_symbol_mask(patch)
    sym_mask = sym_mask if sym_mask is not None else _white_symbol_mask(symbol_template)
    if sym_mask.shape[0] > patch_mask.shape[0] or sym_mask.shape[1] > patch_mask.shape[1]:
        return 0.0
    return float(cv2.matchTemplate(patch_mask, sym_mask, cv2.TM_CCOEFF_NORMED).max())


def _patch_vivid_color_metrics(patch: np.ndarray) -> tuple[float, int]:
    """任务图钉中心应有高饱和彩色，雪地阴影饱和度低。"""
    h, w = patch.shape[:2]
    margin = max(4, int(min(h, w) * 0.12))
    if h <= margin * 2 or w <= margin * 2:
        center = patch
    else:
        center = patch[margin : h - margin, margin : w - margin]
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    vivid = float(((sat > 100) & (val > 130)).mean())
    peak = int(sat.max()) if sat.size else 0
    return vivid, peak


def _patch_has_vivid_pin_color(patch: np.ndarray) -> bool:
    vivid, peak = _patch_vivid_color_metrics(patch)
    if vivid >= PIN_VIVID_COLOR_MIN and peak >= PIN_SATURATION_PEAK_MIN:
        return True
    # 紫/蓝图钉饱和度峰值低于橙钉，但仍明显不是雪地
    return peak >= PIN_SATURATION_PEAK_RELAXED and vivid >= PIN_VIVID_RELAXED_MIN


def _is_snow_texture_false_positive(
    white_score: float, edge_score: float, *, kind: str
) -> bool:
    """雪地/红色晶簇纹理：高白高边且无鲜明图钉色，各类型均可能误报。"""
    if white_score < SNOW_TEXTURE_WHITE_MIN or edge_score < SNOW_TEXTURE_EDGE_MIN:
        return False
    if kind in MONSTER_KINDS:
        return True
    if kind == "tent" and white_score >= 0.78:
        return True
    return False


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


def _contour_white_ratio(gray: np.ndarray, contour: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0.0
    sub_gray = gray[y : y + h, x : x + w]
    return float((sub_gray > 190).mean())


PIN_CLASSIFY_SCALES = (0.85, 1.0, 1.15)


def _score_all_types_pin(
    patch: np.ndarray,
) -> list[tuple[str, str, str, float]]:
    templates = _load_type_templates()
    scores: list[tuple[str, str, str, float]] = []
    for kind, (symbol, edge, sym_mask, symbol_name, label) in templates.items():
        score = _score_kind_on_patch(patch, kind, symbol, edge, sym_mask=sym_mask)
        scores.append((kind, label, symbol_name, score))
    scores.sort(key=lambda item: item[3], reverse=True)
    return scores


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


def _best_template_center_in_contour(
    roi: np.ndarray,
    contour: np.ndarray,
    roi_offset: tuple[int, int],
    *,
    local_offset: tuple[int, int] = (0, 0),
    step: int = 12,
) -> tuple[tuple[int, int] | None, float]:
    """在大色块内部用模板得分找真实图钉中心。"""
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None, 0.0
    lx, ly = local_offset
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    ox, oy = roi_offset
    best_center: tuple[int, int] | None = None
    best_score = 0.0
    for py in range(0, h, step):
        for px in range(0, w, step):
            if mask[py, px] == 0:
                continue
            screen_center = (ox + lx + x + px, oy + ly + y + py)
            patch = _extract_pin_patch(roi, screen_center, roi_offset)
            scores = _score_all_types_pin(patch)
            if not scores:
                continue
            peak = _pin_refine_metric(scores)
            if peak > best_score:
                best_score = peak
                best_center = screen_center
    return best_center, best_score


def _append_pin_candidates_from_contours(
    contours: list[np.ndarray],
    gray: np.ndarray,
    roi_offset: tuple[int, int],
    candidates: list[tuple[tuple[int, int], float]],
    *,
    roi: np.ndarray | None = None,
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
    template_refine_min_area: float = 0.0,
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
        center: tuple[int, int] | None = None
        if roi is not None and area >= template_refine_min_area:
            center, template_score = _best_template_center_in_contour(
                roi,
                contour,
                roi_offset,
                local_offset=local_offset,
            )
            if template_score < PIN_REFINE_MIN_SCORE:
                center = None
        if center is None:
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
    roi: np.ndarray | None = None,
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
            roi=roi,
            area_min=60,
            area_max=600,
            local_offset=(x, y),
            hsv=sub_hsv,
            vivid_peak_min_area=120.0,
            template_refine_min_area=PIN_TEMPLATE_REFINE_DISABLED_AREA,
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
        roi=roi,
        area_min=60,
        hsv=hsv,
        vivid_peak_min_area=180.0,
        template_refine_min_area=PIN_TEMPLATE_REFINE_DISABLED_AREA,
    )
    _append_fragment_centers_from_large_blobs(
        pin_closed, gray, roi_offset, candidates, roi=roi, hsv=hsv, min_area=950.0
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
        roi=roi,
        area_min=50,
        area_max=450,
        template_refine_min_area=PIN_TEMPLATE_REFINE_DISABLED_AREA,
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
        roi=roi,
        area_min=60,
        area_max=500,
        template_refine_min_area=PIN_TEMPLATE_REFINE_DISABLED_AREA,
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
        roi=roi,
        area_min=50,
        area_max=400,
        template_refine_min_area=PIN_TEMPLATE_REFINE_DISABLED_AREA,
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
        roi=roi,
        area_min=50,
        area_max=450,
        template_refine_min_area=PIN_TEMPLATE_REFINE_DISABLED_AREA,
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
        roi=roi,
        area_min=TENT_PIN_BLOB_MIN_AREA,
        area_max=TENT_PIN_BLOB_MAX_AREA,
        white_max=PIN_BLOB_TENT_MAX_WHITE_RATIO,
        aspect_max=3.2,
        circularity_min=0.15,
        template_refine_min_area=PIN_TEMPLATE_REFINE_DISABLED_AREA,
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
        roi=roi,
        area_min=60,
        area_max=900,
        white_max=PIN_BLOB_TENT_MAX_WHITE_RATIO,
        aspect_max=3.5,
        circularity_min=0.12,
        template_refine_min_area=PIN_TEMPLATE_REFINE_DISABLED_AREA,
    )
    _append_fragment_centers_from_large_blobs(
        blue_tent_mask,
        gray,
        roi_offset,
        candidates,
        roi=roi,
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


def _match_at_screen_center(
    symbol: np.ndarray,
    screen_center: tuple[int, int],
    roi_offset: tuple[int, int],
) -> MatchResult:
    sym_mask = _white_symbol_mask(symbol)
    tw, th = sym_mask.shape[1], sym_mask.shape[0]
    cx = screen_center[0] - roi_offset[0]
    cy = screen_center[1] - roi_offset[1]
    return MatchResult(
        found=True,
        confidence=1.0,
        center=(cx, cy),
        top_left=(cx - tw // 2, cy - th // 2),
        size=(tw, th),
    )


def _is_valid_mission_patch(
    patch: np.ndarray,
    kind: str,
    confidence: float,
    *,
    from_pin: bool = False,
) -> bool:
    """过滤雪地纹理误报；图钉中心分类使用更宽松阈值。"""
    templates = _load_type_templates()
    if kind not in templates:
        return False

    symbol, edge, sym_mask, _, _ = templates[kind]
    edge_score = _edge_match_score(patch, edge)
    white_score = _white_match_score(patch, symbol, sym_mask=sym_mask)

    if _is_snow_texture_false_positive(white_score, edge_score, kind=kind):
        if not from_pin or not _patch_has_vivid_pin_color(patch):
            return False

    if from_pin and not _patch_has_vivid_pin_color(patch):
        if confidence < PIN_STRONG_SCORE_WITHOUT_VIVID:
            if kind in MONSTER_KINDS and confidence >= MONSTER_PIN_MIN_CONFIDENCE:
                pass
            else:
                return False

    if from_pin:
        if confidence < PIN_SCAN_MIN_CONFIDENCE:
            return False
        # 所有类型都需要最低边缘得分，防止背景纹理误识别
        if edge_score < 0.25:
            return False
        if kind == "tent" and confidence < TENT_PIN_MIN_CONFIDENCE:
            return False
        if kind == "hero_journey":
            if confidence < HERO_JOURNEY_PIN_MIN_CONFIDENCE:
                return False
            if confidence >= 0.45:
                pass
            else:
                orange_ratio = _patch_orange_pin_ratio(patch)
                hero_color = _patch_hero_color_ratio(patch)
                if orange_ratio >= HERO_ORANGE_RATIO_MIN:
                    pass
                elif edge_score >= HERO_PURPLE_BLUE_EDGE_MIN:
                    pass
                elif (
                    hero_color >= HERO_COLOR_RATIO_MIN
                    and edge_score >= HERO_MIN_EDGE_FOR_WEAK_COLOR
                ):
                    pass
                else:
                    return False
        if kind in MONSTER_KINDS and confidence < MONSTER_PIN_MIN_CONFIDENCE:
            return False
        if kind == "small_monster" and (
            white_score >= MONSTER_TEXTURE_WHITE_MIN
            and edge_score < MONSTER_TEXTURE_EDGE_MAX
        ):
            vivid, _peak_sat = _patch_vivid_color_metrics(patch)
            if not _patch_has_vivid_pin_color(patch):
                return False
        return True

    if kind == "hero_journey":
        if edge_score >= HERO_JOURNEY_MIN_EDGE_SCORE:
            return True
        return confidence >= HERO_JOURNEY_MIN_CONFIDENCE and edge_score >= 0.55

    if kind == "tent":
        return confidence >= TENT_MIN_CONFIDENCE and edge_score >= TENT_MIN_EDGE_SCORE

    if kind in MONSTER_KINDS:
        if (
            white_score >= MONSTER_TEXTURE_WHITE_MIN
            and edge_score < MONSTER_TEXTURE_EDGE_MAX
        ):
            return False
        if white_score >= 0.92 and edge_score < MONSTER_MIN_EDGE_SCORE:
            return False
        return confidence >= MONSTER_MIN_CONFIDENCE and edge_score >= MONSTER_MIN_EDGE_SCORE

    return confidence >= LIGHTHOUSE_MATCH_THRESHOLD


def _score_all_types(
    patch: np.ndarray,
) -> list[tuple[str, str, str, float]]:
    """对所有类型评分，使用多尺度和alt模板以提高准确性。"""
    templates = _load_type_templates()
    scores: list[tuple[str, str, str, float]] = []
    for kind, (symbol, edge, sym_mask, symbol_name, label) in templates.items():
        # 使用多尺度评分
        score = _score_patch_against_type(
            patch, symbol, edge,
            scales=LIGHTHOUSE_FAST_SCALES,
            sym_mask=sym_mask
        )
        # 尝试alt模板，取最高分
        for alt_symbol, alt_edge, alt_mask in _load_alt_type_templates().get(kind, []):
            alt_score = _score_patch_against_type(
                patch, alt_symbol, alt_edge,
                scales=LIGHTHOUSE_FAST_SCALES,
                sym_mask=alt_mask
            )
            score = max(score, alt_score)
        scores.append((kind, label, symbol_name, score))
    scores.sort(key=lambda item: item[3], reverse=True)
    return scores


def _score_entry(
    scores: list[tuple[str, str, str, float]], kind: str
) -> tuple[str, str, str, float] | None:
    return next((item for item in scores if item[0] == kind), None)


def _patch_orange_pin_ratio(patch: np.ndarray) -> float:
    """英雄之旅与橙爪小怪图钉主体为橙色高饱和。"""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    vivid = (sat > 90) & (val > 130)
    vivid_count = int(vivid.sum())
    if vivid_count < 8:
        return 0.0
    orange = ((hue >= 8) & (hue <= 32) & (sat >= 90)) & vivid
    return float(orange.sum() / vivid_count)


def _patch_hero_color_ratio(patch: np.ndarray) -> float:
    """英雄之旅图钉为橙/紫/蓝高饱和色，小怪为橙或浅紫。"""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    vivid = (sat > 70) & (val > 120)
    vivid_count = int(vivid.sum())
    if vivid_count < 8:
        return 0.0
    hero_hue = (
        ((hue >= 8) & (hue <= 32) & (sat >= 80))
        | ((hue >= 150) & (hue <= 178) & (sat >= 80))
        | ((hue >= 95) & (hue <= 118) & (sat >= 70))
    ) & vivid
    return float(hero_hue.sum() / vivid_count)


def _edge_score_for_kind(patch: np.ndarray, kind: str) -> float:
    templates = _load_type_templates()
    if kind not in templates:
        return 0.0
    _, edge, _, _, _ = templates[kind]
    return _edge_match_score(patch, edge)


def _disambiguate_hero_vs_monster(
    classified: tuple[str, str, str, float] | None,
    scores: list[tuple[str, str, str, float]],
    patch: np.ndarray,
) -> tuple[str, str, str, float] | None:
    """双斧是英雄之旅专属；紫/蓝图钉上小怪得分接近时优先英雄。"""
    if classified is None:
        return None

    kind, label, symbol_name, confidence = classified
    hero_entry = _score_entry(scores, "hero_journey")
    if hero_entry is None:
        return classified

    hero_score = hero_entry[3]
    hero_edge = _edge_score_for_kind(patch, "hero_journey")
    hero_color = _patch_hero_color_ratio(patch)

    if kind in MONSTER_KINDS:
        monster_edge = _edge_score_for_kind(patch, kind)
        if hero_edge >= monster_edge + HERO_EDGE_OVER_MONSTER_MIN:
            if hero_score >= HERO_JOURNEY_PIN_MIN_CONFIDENCE:
                return hero_entry
            if hero_score >= confidence - 0.08 and hero_score >= 0.30:
                return hero_entry
        if (
            hero_color >= HERO_COLOR_RATIO_MIN
            and hero_score >= HERO_JOURNEY_PIN_MIN_CONFIDENCE
            and hero_score >= confidence - 0.12
        ):
            return hero_entry
        if hero_color >= 0.30 and hero_edge >= 0.35 and hero_score >= 0.28:
            return hero_entry

    if kind == "hero_journey":
        orange_ratio = _patch_orange_pin_ratio(patch)
        for monster_kind in MONSTER_KINDS:
            monster_entry = _score_entry(scores, monster_kind)
            if monster_entry is None:
                continue
            monster_edge = _edge_score_for_kind(patch, monster_kind)
            monster_score = monster_entry[3]
            if (
                monster_edge > hero_edge + 0.28
                and monster_score > confidence - 0.05
                and monster_score >= MONSTER_PIN_MIN_CONFIDENCE
            ):
                return monster_entry
            if (
                orange_ratio >= HERO_ORANGE_RATIO_MIN
                and monster_kind == "small_monster"
                and monster_score >= confidence - 0.06
                and monster_score >= MONSTER_PIN_MIN_CONFIDENCE
                and hero_edge < monster_edge + 0.08
            ):
                return monster_entry

    return classified


def _effective_type_margin(kind_a: str, kind_b: str, margin: float) -> float:
    if kind_a in MONSTER_KINDS and kind_b in MONSTER_KINDS:
        return 0.0
    if (kind_a == "hero_journey" and kind_b in MONSTER_KINDS) or (
        kind_b == "hero_journey" and kind_a in MONSTER_KINDS
    ):
        return max(margin, HERO_MONSTER_CLASSIFY_MARGIN)
    return margin


def _candidate_threshold_for_kind(kind: str) -> float:
    if kind == "hero_journey":
        return LIGHTHOUSE_HERO_CANDIDATE_THRESHOLD
    if kind == "small_monster_beast":
        return BEAST_CANDIDATE_THRESHOLD
    return LIGHTHOUSE_CANDIDATE_THRESHOLD


def _pick_classified_result(
    scores: list[tuple[str, str, str, float]],
    *,
    threshold: float,
    margin: float,
    prefer_kind: str | None = None,
) -> tuple[str, str, str, float] | None:
    if not scores:
        return None

    best_kind, best_label, best_symbol, best_score = scores[0]
    second_kind = scores[1][0] if len(scores) > 1 else ""
    second_score = scores[1][3] if len(scores) > 1 else 0.0
    eff_margin = _effective_type_margin(best_kind, second_kind, margin)

    if best_score >= threshold and (best_score - second_score) >= eff_margin:
        return best_kind, best_label, best_symbol, best_score

    if prefer_kind:
        prefer = next((item for item in scores if item[0] == prefer_kind), None)
        if prefer is not None:
            p_kind, p_label, p_symbol, p_score = prefer
            relaxed = max(threshold * 0.85, LIGHTHOUSE_RELAXED_THRESHOLD_FLOOR)
            if p_score >= relaxed and p_score >= best_score - 0.04:
                if p_kind == "hero_journey" and p_score < HERO_JOURNEY_MIN_CONFIDENCE:
                    pass
                elif p_kind in MONSTER_KINDS and p_score < MONSTER_MIN_CONFIDENCE:
                    pass
                else:
                    return p_kind, p_label, p_symbol, p_score

    if best_score >= max(threshold * 0.9, LIGHTHOUSE_RELAXED_THRESHOLD_FLOOR):
        if (best_score - second_score) >= eff_margin * 0.5:
            return best_kind, best_label, best_symbol, best_score

    return None


def _classify_patch_flexible(
    patch: np.ndarray,
    *,
    threshold: float,
    margin: float,
    prefer_kind: str | None = None,
) -> tuple[tuple[str, str, str, float] | None, list[tuple[str, str, str, float]]]:
    scores = _score_all_types(patch)
    classified = _pick_classified_result(
        scores, threshold=threshold, margin=margin, prefer_kind=prefer_kind
    )
    return classified, scores


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


def _filter_noise_missions(
    missions: list[LighthouseMission],
    *,
    threshold: float,
) -> list[LighthouseMission]:
    if not missions:
        return missions

    ordered = sorted(missions, key=lambda item: item.confidence, reverse=True)
    top = ordered[0].confidence
    if top >= 0.45:
        cutoff = max(threshold, top * 0.45)
    else:
        cutoff = max(threshold * 0.9, LIGHTHOUSE_RELAXED_THRESHOLD_FLOOR)
    return [mission for mission in ordered if mission.confidence >= cutoff]


def _pin_peak_score(patch: np.ndarray) -> tuple[float, list[tuple[str, str, str, float]]]:
    """精修搜索用全量模板（含 alt）估分，避免主模板在雪地上误拉高英雄分。"""
    scores = _score_all_types_pin(patch)
    return _pin_refine_metric(scores), scores


def _pin_refine_metric(scores: list[tuple[str, str, str, float]]) -> float:
    if not scores:
        return 0.0
    return max(item[3] for item in scores)


def _refine_axis_offsets(radius: int, step: int) -> tuple[int, ...]:
    """搜索偏移；步长>1 时仍保留 0 轴，避免漏掉与色块同水平/垂直的真图钉。"""
    vals = list(range(-radius, radius + 1, step))
    if 0 not in vals:
        vals.append(0)
    return tuple(sorted(vals))


def _refine_pin_center(
    roi: np.ndarray,
    pin_center: tuple[int, int],
    roi_offset: tuple[int, int],
) -> tuple[tuple[int, int], list[tuple[str, str, str, float]]]:
    """色块中心常偏离图标，仅在得分偏低时在附近搜索真实图钉中心。"""
    patch = _extract_pin_patch(roi, pin_center, roi_offset)
    best_peak, initial_scores = _pin_peak_score(patch)
    if not initial_scores:
        return pin_center, []

    if best_peak >= PIN_REFINE_TRIGGER_SCORE:
        return pin_center, initial_scores
    if best_peak < 0.08:
        return pin_center, initial_scores

    ox, oy = roi_offset
    best_center = pin_center
    best_scores = initial_scores

    search_phases: list[tuple[int, int]] = [
        (PIN_REFINE_SEARCH_RADIUS, PIN_REFINE_SEARCH_STEP),
        (PIN_REFINE_FINE_RADIUS, PIN_REFINE_FINE_STEP),
    ]
    for _chase in range(PIN_REFINE_CHASE_PASSES):
        if best_peak >= PIN_REFINE_TRIGGER_SCORE:
            break
        search_cx = best_center[0] - ox
        search_cy = best_center[1] - oy
        for radius, step in search_phases:
            if best_peak >= PIN_REFINE_TRIGGER_SCORE:
                break
            offsets = _refine_axis_offsets(radius, step)
            for dy in offsets:
                _check_scan_interrupted()
                for dx in offsets:
                    if dx == 0 and dy == 0:
                        continue
                    px, py = search_cx + dx, search_cy + dy
                    if not (0 <= px < roi.shape[1] and 0 <= py < roi.shape[0]):
                        continue
                    screen_center = (ox + px, oy + py)
                    drift = max(
                        abs(screen_center[0] - pin_center[0]),
                        abs(screen_center[1] - pin_center[1]),
                    )
                    if drift > PIN_REFINE_MAX_DRIFT:
                        continue
                    patch = _extract_pin_patch(roi, screen_center, roi_offset)
                    peak, candidate_scores = _pin_peak_score(patch)
                    if peak <= best_peak:
                        continue
                    vivid, peak_sat = _patch_vivid_color_metrics(patch)
                    if peak < PIN_STRONG_SCORE_WITHOUT_VIVID and (
                        vivid < PIN_VIVID_COLOR_MIN
                        and peak_sat < PIN_SATURATION_PEAK_RELAXED
                    ):
                        continue
                    best_peak = peak
                    best_center = screen_center
                    best_scores = candidate_scores
        if best_peak >= 0.40:
            break

    _, best_scores = _pin_peak_score(
        _extract_pin_patch(roi, best_center, roi_offset)
    )
    return best_center, best_scores


def _resolve_mission_at_pin(
    roi: np.ndarray,
    pin_center: tuple[int, int],
    roi_offset: tuple[int, int],
) -> LighthouseMission | None:
    refined_center, scores = _refine_pin_center(roi, pin_center, roi_offset)
    if not scores:
        return None

    classified = _pick_classified_result(
        scores,
        threshold=PIN_SCAN_MIN_CONFIDENCE,
        margin=PIN_CLASSIFY_MARGIN,
    )
    patch = _extract_pin_patch(roi, refined_center, roi_offset)
    classified = _disambiguate_hero_vs_monster(classified, scores, patch)
    if classified is None:
        return None

    kind, label, symbol_name, confidence = classified
    if not _is_valid_mission_patch(patch, kind, confidence, from_pin=True):
        return None

    ox, oy = roi_offset
    cx = refined_center[0] - ox
    cy = refined_center[1] - oy
    half = LIGHTHOUSE_PIN_PATCH_HALF
    x1p = max(0, cx - half)
    y1p = max(0, cy - half)
    return LighthouseMission(
        kind=kind,
        label=label,
        template=symbol_name,
        center=refined_center,
        confidence=confidence,
        top_left=(ox + x1p, oy + y1p),
        size=(patch.shape[1], patch.shape[0]),
    )


def refine_mission_click(
    mission: LighthouseMission,
    screen: np.ndarray,
    other_missions: tuple[LighthouseMission, ...] = (),
    *,
    scan_roi: tuple[int, int, int, int] = LIGHTHOUSE_SCAN_ROI,
) -> LighthouseMission:
    """仅在相邻图标存在时微调落点，供点击前调用。"""
    if not other_missions:
        return mission

    screen = _normalize_screen_for_scan(screen)
    x1, y1, x2, y2 = scan_roi
    roi = screen[y1:y2, x1:x2]
    cx = mission.center[0] - x1
    cy = mission.center[1] - y1
    others = [
        (m.center[0] - x1, m.center[1] - y1)
        for m in other_missions
        if m is not mission
    ]
    if not any(
        abs(cx - ox) < LIGHTHOUSE_NEIGHBOR_DISTANCE
        and abs(cy - oy) < LIGHTHOUSE_NEIGHBOR_DISTANCE
        for ox, oy in others
    ):
        return mission

    templates = _load_type_templates()
    if mission.kind not in templates:
        return mission

    symbol, edge, sym_mask, _, _ = templates[mission.kind]
    best_center = (cx, cy)
    best_score = -1.0
    for dx in range(-24, 25, 8):
        for dy in range(-24, 25, 8):
            px, py = cx + dx, cy + dy
            if not (0 <= px < roi.shape[1] and 0 <= py < roi.shape[0]):
                continue
            x1p = max(0, px - LIGHTHOUSE_CENTER_PATCH_HALF)
            y1p = max(0, py - LIGHTHOUSE_CENTER_PATCH_HALF)
            x2p = min(roi.shape[1], px + LIGHTHOUSE_CENTER_PATCH_HALF)
            y2p = min(roi.shape[0], py + LIGHTHOUSE_CENTER_PATCH_HALF)
            patch = roi[y1p:y2p, x1p:x2p]
            if patch.size == 0:
                continue
            score = _score_patch_against_type(
                patch, symbol, edge, scales=(1.0,), sym_mask=sym_mask
            )
            if score > best_score:
                best_score = score
                best_center = (px, py)

    return LighthouseMission(
        kind=mission.kind,
        label=mission.label,
        template=mission.template,
        center=(x1 + best_center[0], y1 + best_center[1]),
        confidence=mission.confidence,
        top_left=mission.top_left,
        size=mission.size,
    )


def scan_lighthouse_missions(
    screen: np.ndarray,
    *,
    threshold: float = LIGHTHOUSE_MATCH_THRESHOLD,
    scan_roi: tuple[int, int, int, int] = LIGHTHOUSE_SCAN_ROI,
    interrupted: Callable[[], bool] | None = None,
) -> LighthouseScanResult:
    global _scan_interrupt_cb
    prev_interrupt = _scan_interrupt_cb
    _scan_interrupt_cb = interrupted
    try:
        return _scan_lighthouse_missions_impl(
            screen, threshold=threshold, scan_roi=scan_roi
        )
    except InterruptedError:
        return LighthouseScanResult(mission=None, best_label="已停止")
    finally:
        _scan_interrupt_cb = prev_interrupt


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

    path = TEMPLATE_DIR / LIGHTHOUSE_MAP_BG_NAME
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


def _is_plane_like_contour(contour: np.ndarray) -> bool:
    """排除小飞机：宽扁、面积大、或飞机尾迹碎片。"""
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if area >= PLANE_CONTOUR_AREA_MIN:
        return True
    aspect = h / max(w, 1)
    if w > h * 1.35 and area >= 1500:
        return True
    if aspect < 0.72 and area >= 800:
        return True
    return False


def _is_boss_like_contour(
    contour: np.ndarray, roi: np.ndarray, hsv: np.ndarray
) -> bool:
    """排除超级大怪：仅匹配顶部超大橙色块（普通任务橙色光晕不会触发）。"""
    area = cv2.contourArea(contour)
    if area < BOSS_CONTOUR_AREA_MIN:
        return False
    x, y, w, h = cv2.boundingRect(contour)
    center = (x + w // 2, y + h // 2)
    return _is_super_boss_point(center, roi, hsv, (0, 0))


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
    return found


def _contour_pin_center(
    contour: np.ndarray,
    hsv: np.ndarray,
    roi_offset: tuple[int, int],
) -> tuple[int, int] | None:
    center = _contour_screen_center(
        contour,
        roi_offset,
        hsv=hsv,
        use_vivid_peak=True,
    )
    if center is None:
        return None
    if not _screen_center_in_map(center):
        return None
    if _is_ignored_background_pin(center, contour_area=cv2.contourArea(contour)):
        return None
    return center


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
            f"缺少灯塔背景图 {LIGHTHOUSE_MAP_BG_NAME}，无法扫描任务图标"
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


def _scan_lighthouse_missions_impl(
    screen: np.ndarray,
    *,
    threshold: float,
    scan_roi: tuple[int, int, int, int],
) -> LighthouseScanResult:
    """颜色图钉定位 + 模板匹配确认分类，两者交集消除误检。

    颜色分割找到的图钉中心与图标中心有偏移，所以旧方案在色块中心
    抠 patch 做模板匹配总是失败。新方案反过来：以颜色图钉为候选，
    在其周围搜索最佳模板匹配来分类，只有模板得分足够高才保留。
    这样同时排除了：
    - 颜色误检（无模板匹配的背景色块）
    - 模板误检（无颜色图钉的石头/树木纹理）
    """
    screen = _normalize_screen_for_scan(screen)
    x1, y1, x2, y2 = scan_roi
    roi = screen[y1:y2, x1:x2]
    if roi.size == 0:
        return LighthouseScanResult(mission=None)

    templates = _load_type_templates()
    if not templates:
        logger.error("灯塔模板未加载，请检查 assets/templates/lighthouse/")
        return LighthouseScanResult(mission=None)

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 1. 颜色图钉检测（候选位置）
    color_pins = _find_mission_pin_centers(roi, (x1, y1))
    if not color_pins:
        return LighthouseScanResult(mission=None, candidate_locations=0)

    # 2. 对每个颜色图钉，在周围搜索最佳模板匹配
    missions: list[LighthouseMission] = []
    best_confidence = 0.0
    best_label = ""
    search_radius = 60  # 搜索半径，覆盖图标与色块质心的偏移

    for pin_cx, pin_cy in color_pins:
        _check_scan_interrupted()
        pin_best_score = 0.0
        pin_best_kind = ""
        pin_best_label = ""
        pin_best_sym = ""
        pin_best_cx = pin_cx
        pin_best_cy = pin_cy

        for kind, (symbol, _edge, _sym_mask, symbol_name, label) in templates.items():
            tpl_gray = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)

            for scale in LIGHTHOUSE_FAST_SCALES:
                tpl = cv2.resize(
                    tpl_gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
                )
                th, tw = tpl.shape
                if th > roi_gray.shape[0] or tw > roi_gray.shape[1]:
                    continue

                # 在图钉周围限定搜索区域
                roi_px = pin_cx - x1
                roi_py = pin_cy - y1
                sx1 = max(0, roi_px - tw // 2 - search_radius)
                sy1 = max(0, roi_py - th // 2 - search_radius)
                sx2 = min(roi_gray.shape[1] - tw + 1, roi_px - tw // 2 + search_radius)
                sy2 = min(roi_gray.shape[0] - th + 1, roi_py - th // 2 + search_radius)
                if sx2 <= sx1 or sy2 <= sy1:
                    continue

                result = cv2.matchTemplate(roi_gray, tpl, cv2.TM_CCOEFF_NORMED)
                sub = result[sy1:sy2, sx1:sx2]
                if sub.size == 0:
                    continue

                _, max_val, _, max_loc = cv2.minMaxLoc(sub)
                if max_val > pin_best_score:
                    pin_best_score = max_val
                    pin_best_kind = kind
                    pin_best_label = label
                    pin_best_sym = symbol_name
                    pin_best_cx = x1 + sx1 + max_loc[0] + tw // 2
                    pin_best_cy = y1 + sy1 + max_loc[1] + th // 2

        # 只有模板得分足够高才保留
        direct_min = max(threshold, LIGHTHOUSE_DIRECT_MATCH_MIN)
        if pin_best_score < direct_min:
            continue

        if pin_best_score > best_confidence:
            best_confidence = pin_best_score
            best_label = pin_best_label

        half = LIGHTHOUSE_PIN_PATCH_HALF
        roi_cx = pin_best_cx - x1
        roi_cy = pin_best_cy - y1
        x1p = max(0, roi_cx - half)
        y1p = max(0, roi_cy - half)

        missions.append(
            LighthouseMission(
                kind=pin_best_kind,
                label=pin_best_label,
                template=pin_best_sym,
                center=(pin_best_cx, pin_best_cy),
                confidence=pin_best_score,
                top_left=(x1 + x1p, y1 + y1p),
                size=(min(roi.shape[1], roi_cx + half) - x1p,
                      min(roi.shape[0], roi_cy + half) - y1p),
            )
        )

    missions = _dedupe_missions_by_slot(missions)

    if not missions:
        return LighthouseScanResult(
            mission=None,
            best_confidence=best_confidence,
            best_label=best_label,
            candidate_locations=len(color_pins),
        )

    missions.sort(key=lambda item: item.confidence, reverse=True)
    best = missions[0]
    return LighthouseScanResult(
        mission=best,
        missions=tuple(missions),
        best_confidence=best.confidence,
        best_label=best.label,
        candidate_locations=len(color_pins),
    )
