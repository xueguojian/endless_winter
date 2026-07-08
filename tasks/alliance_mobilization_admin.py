"""联盟总动员管理员模式：可滚动任务列表，保留高分练兵。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from loguru import logger

from core.adb_client import AdbClient
from core.dream_memory.ocr_engine import ocr_chip_text, ocr_engine_available, resolve_ocr_engine
from core.vision import Vision
from tasks.alliance_mobilization import (
    ADMIN_DETAIL_TYPE_ORDER,
    ADMIN_TARGET_TYPES,
    COUNTDOWN_TEMPLATE,
    DEFAULT_STEP_DELAY,
    TASK_TYPE_ADMIN_DETAIL_TEMPLATES,
    TASK_TYPE_ADMIN_TEMPLATES,
    TASK_TYPE_LABELS,
    TASK_TYPE_TEMPLATES,
    TASK_TYPE_TRAIN,
    TEMPLATE_DIR,
    TRAIN_ICON_ADMIN_TEMPLATE,
    TRAIN_ICON_TEMPLATE,
    _crop,
    _looks_like_countdown_text,
    _parse_score,
    _prepare_score_patch,
    _roi_center,
)

StatusCallback = Callable[[str], None]

DEFAULT_LIST_ROI = (44, 906, 714, 1264)
# 专属任务在 list_roi 上方；此处仅留小边距
DEFAULT_EXCLUDE_TOP_PX = 72
DEFAULT_COLUMN_COUNT = 3
DEFAULT_SCORE_THRESHOLD = 400
DEFAULT_SCAN_INTERVAL = 6 * 60
DEFAULT_MATCH_THRESHOLD = 0.66
DEFAULT_COUNTDOWN_THRESHOLD = 0.62
DEFAULT_TARGET_TYPES = ["train"]
# 管理员练兵模板（内圈裁剪）匹配缩放；含更小尺度以适配用户整卡裁剪图
ADMIN_TRAIN_MATCH_SCALES: tuple[float, ...] = tuple(
    round(i / 100, 2) for i in range(55, 126, 5)
)
# 旧模板仅作辅助，防止水晶/紫底误命中管理员练兵模板
ADMIN_TRAIN_LEGACY_MIN_DELTA = 0.18
# 管理员卡片结构（patch 内像素，实机标定）
ADMIN_CARD_ICON_HEIGHT = 92
# 详情弹窗练兵匹配：图标更大，阈值略低，不做 legacy 差值门槛
DEFAULT_DETAIL_MATCH_THRESHOLD = 0.55
# 最佳与次佳差距过小、或整体都在噪音区 → 判定无效
DETAIL_MIN_CONF_MARGIN = 0.08
DETAIL_NOISE_FLOOR = 0.45
DETAIL_ICON_TIMER_EXCLUDE_PX = 36
DETAIL_ICON_TIMER_RATIO = 0.28
# 卡片底色：橙/紫/蓝（管理员列表品质）
CARD_BG_ORANGE = "orange"
CARD_BG_PURPLE = "purple"
CARD_BG_BLUE = "blue"
CARD_BG_UNKNOWN = "unknown"
CARD_BG_LABELS = {
    CARD_BG_ORANGE: "橙色",
    CARD_BG_PURPLE: "紫色",
    CARD_BG_BLUE: "蓝色",
    CARD_BG_UNKNOWN: "未知",
}
# 阶段目标默认：只保留橙色练兵；可在配置/GUI 多选扩展
KEEP_ORANGE_TYPES: frozenset[str] = frozenset({TASK_TYPE_TRAIN})
TRAIN_ICON_ADMIN_INNER_TEMPLATE = "alliance_mobilization/train_icon_admin_inner.png"
ADMIN_TRAIN_DETAIL_MATCH_SCALES: tuple[float, ...] = tuple(
    round(i / 100, 2) for i in range(55, 151, 5)
)
# 任务详情弹窗（点击卡片后）
DEFAULT_DETAIL_REFRESH_BTN_ROI = (46, 778, 316, 850)
DEFAULT_DETAIL_ICON_ROI = (64, 500, 200, 632)
DEFAULT_DETAIL_SCORE_ROI = (304, 724, 456, 762)
DEFAULT_DETAIL_TITLE_ROI = (268, 418, 452, 478)
DEFAULT_DETAIL_CLOSE_BTN_ROI = (634, 406, 698, 446)
DETAIL_POPUP_WAIT_SEC = 0.85
DETAIL_CLOSE_SETTLE_SEC = 0.75

DEFAULT_COORDS: dict[str, list[int]] = {
    # 弹窗内「刷新任务」按钮中心（详情 ROI 内）
    "refresh_tap": [181, 814],
    "refresh_confirm": [512, 776],
    # 公共任务弹窗右上角 X（勿用 adb.back，会退出联盟界面）
    "detail_close": [666, 426],
    # 弹窗上方遮罩，X 无效时备用
    "detail_mask_tap": [374, 368],
}
# 用户实机标定；config 中若仍为旧默认坐标则自动纠正
CALIBRATED_DETAIL_CLOSE = (666, 426)
CALIBRATED_DETAIL_MASK_TAP = (374, 368)
DEPRECATED_DETAIL_CLOSE = frozenset({(694, 459), (684, 462)})
DEPRECATED_DETAIL_MASK_TAP = frozenset({(360, 395)})
DEFAULT_SCROLL = {
    "swipe_ms": 900,
    "swipe_x": 379,
    # 常规上滑：约半行多一点，配合较慢手势减少惯性飞滑
    "swipe_up_y1": 1240,
    "swipe_up_y2": 1100,
    "swipe_down_y1": 1160,
    "swipe_down_y2": 1240,
    # 小幅：约一行多一点（慢滑压惯性）
    "swipe_small_y1": 1240,
    "swipe_small_y2": 1105,
    "swipe_small_ms": 1200,
    # 小幅无效时的中等重试
    "swipe_medium_y1": 1240,
    "swipe_medium_y2": 1065,
    "swipe_medium_ms": 1100,
    "partial_pre_delay": 0.55,
    "partial_settle_delay": 1.35,
    "max_swipes_per_pass": 40,
    "settle_delay": 1.0,
    "screen_delay": 0.35,
    "scroll_to_top_max": 10,
}


@dataclass(frozen=True)
class AdminCard:
    """管理员列表内一张可检测任务卡。"""

    column: int
    row_key: int
    icon_roi: tuple[int, int, int, int]
    score_roi: tuple[int, int, int, int]

    @property
    def key(self) -> tuple[int, int]:
        return self.column, self.row_key


def _normalize_admin_coords(coords: dict) -> dict[str, list[int]]:
    """合并并纠正详情弹窗关闭坐标（忽略历史错误默认值）。"""
    merged = {**DEFAULT_COORDS, **(coords or {})}
    raw_close = (coords or {}).get("detail_close")
    if raw_close and len(raw_close) >= 2:
        close_xy = (int(raw_close[0]), int(raw_close[1]))
        if close_xy not in DEPRECATED_DETAIL_CLOSE:
            merged["detail_close"] = [close_xy[0], close_xy[1]]
        else:
            merged["detail_close"] = list(CALIBRATED_DETAIL_CLOSE)
    else:
        merged["detail_close"] = list(CALIBRATED_DETAIL_CLOSE)

    raw_mask = (coords or {}).get("detail_mask_tap")
    if raw_mask and len(raw_mask) >= 2:
        mask_xy = (int(raw_mask[0]), int(raw_mask[1]))
        if mask_xy not in DEPRECATED_DETAIL_MASK_TAP:
            merged["detail_mask_tap"] = [mask_xy[0], mask_xy[1]]
        else:
            merged["detail_mask_tap"] = list(CALIBRATED_DETAIL_MASK_TAP)
    else:
        merged["detail_mask_tap"] = list(CALIBRATED_DETAIL_MASK_TAP)
    return merged


def merge_task_config(cfg: dict | None) -> dict:
    raw = cfg or {}
    keep_types = [
        str(item).strip()
        for item in (
            raw.get("keep_orange_types")
            or raw.get("target_types")
            or list(KEEP_ORANGE_TYPES)
        )
        if str(item).strip() in ADMIN_TARGET_TYPES
    ]
    if not keep_types:
        keep_types = list(KEEP_ORANGE_TYPES)
    selected = [
        str(item).strip()
        for item in (raw.get("target_types") or keep_types)
        if str(item).strip() in ADMIN_TARGET_TYPES
    ]
    if not selected:
        selected = list(keep_types)

    roi = raw.get("list_roi") or list(DEFAULT_LIST_ROI)
    scroll = {**DEFAULT_SCROLL, **(raw.get("scroll") or {})}
    coords = _normalize_admin_coords(raw.get("coords") or {})
    detail_refresh_btn_roi = _as_roi(
        raw.get("detail_refresh_btn_roi"), DEFAULT_DETAIL_REFRESH_BTN_ROI
    )
    detail_icon_roi = _as_roi(raw.get("detail_icon_roi"), DEFAULT_DETAIL_ICON_ROI)
    detail_score_roi = _as_roi(raw.get("detail_score_roi"), DEFAULT_DETAIL_SCORE_ROI)
    detail_title_roi = _as_roi(raw.get("detail_title_roi"), DEFAULT_DETAIL_TITLE_ROI)
    detail_close_btn_roi = _as_roi(
        raw.get("detail_close_btn_roi"), DEFAULT_DETAIL_CLOSE_BTN_ROI
    )
    return {
        "target_types": selected,
        "score_threshold": int(raw.get("score_threshold", DEFAULT_SCORE_THRESHOLD)),
        "scan_interval": float(raw.get("scan_interval", DEFAULT_SCAN_INTERVAL)),
        "step_delay": float(raw.get("step_delay", DEFAULT_STEP_DELAY)),
        "match_threshold": float(raw.get("match_threshold", DEFAULT_MATCH_THRESHOLD)),
        "countdown_threshold": float(
            raw.get("countdown_threshold", DEFAULT_COUNTDOWN_THRESHOLD)
        ),
        "ocr_engine": str(raw.get("ocr_engine") or "auto"),
        "list_roi": _as_roi(roi, DEFAULT_LIST_ROI),
        "exclude_top_px": int(raw.get("exclude_top_px", DEFAULT_EXCLUDE_TOP_PX)),
        "column_count": int(raw.get("column_count", DEFAULT_COLUMN_COUNT)),
        "detail_refresh_btn_roi": detail_refresh_btn_roi,
        "detail_icon_roi": detail_icon_roi,
        "detail_score_roi": detail_score_roi,
        "detail_title_roi": detail_title_roi,
        "detail_close_btn_roi": detail_close_btn_roi,
        "detail_match_threshold": float(
            raw.get("detail_match_threshold", DEFAULT_DETAIL_MATCH_THRESHOLD)
        ),
        "keep_orange_types": keep_types,
        "use_score_ocr": bool(raw.get("use_score_ocr", False)),
        "scroll": scroll,
        "coords": {
            key: [int(coords[key][0]), int(coords[key][1])]
            for key in DEFAULT_COORDS
            if key in coords and len(coords[key]) >= 2
        },
    }


def _as_roi(raw, fallback) -> tuple[int, int, int, int]:
    try:
        values = [int(v) for v in raw]
        if len(values) != 4:
            raise ValueError
        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            raise ValueError
        return x1, y1, x2, y2
    except (TypeError, ValueError):
        return tuple(int(v) for v in fallback)  # type: ignore[return-value]


def _offset_roi(
    roi: tuple[int, int, int, int],
    origin: tuple[int, int],
) -> tuple[int, int, int, int]:
    ox, oy = origin
    x1, y1, x2, y2 = roi
    return x1 + ox, y1 + oy, x2 + ox, y2 + oy


def _patch_signature(patch: np.ndarray) -> tuple[int, ...]:
    if patch.size == 0:
        return ()
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (48, 24), interpolation=cv2.INTER_AREA)
    return tuple(int(v) for v in small.flatten()[::12])


def _card_content_key(screen: np.ndarray, card: AdminCard) -> tuple[int, tuple[int, ...]]:
    """按列 + 图标内容去重。去掉底部倒计时条，避免每秒跳变导致重复处理。"""
    icon = _crop(screen, card.icon_roi)
    if icon.size == 0:
        return card.column, (card.row_key,)
    height = icon.shape[0]
    # 列表卡底部常有倒计时/进度，不参与指纹
    icon = icon[: max(12, int(height * 0.70)), :]
    gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    # 量化减少截图像素抖动
    quant = (small.astype(np.int16) // 16).flatten()
    return card.column, tuple(int(v) for v in quant)


def _content_keys_similar(
    key_a: tuple[int, tuple[int, ...]],
    key_b: tuple[int, tuple[int, ...]],
    *,
    min_ratio: float = 0.82,
) -> bool:
    """同列且图标指纹足够像，视为同一张卡。"""
    if key_a[0] != key_b[0]:
        return False
    a = key_a[1]
    b = key_b[1]
    if not a or not b or len(a) != len(b):
        return False
    same = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1)
    return (same / len(a)) >= min_ratio


def _is_duplicate_content(
    content_key: tuple[int, tuple[int, ...]],
    processed: set[tuple[int, tuple[int, ...]]],
) -> bool:
    if content_key in processed:
        return True
    return any(_content_keys_similar(content_key, seen) for seen in processed)


def _prepare_detail_icon_patch(patch: np.ndarray) -> np.ndarray:
    """详情弹窗图标 ROI 去掉底部倒计时条，避免干扰模板匹配。"""
    if patch.size == 0:
        return patch
    height = patch.shape[0]
    cut = max(DETAIL_ICON_TIMER_EXCLUDE_PX, int(height * DETAIL_ICON_TIMER_RATIO))
    icon = patch[: max(40, height - cut), :]
    return icon if icon.size else patch


def classify_card_bg_color(patch: np.ndarray) -> tuple[str, dict[str, float]]:
    """根据图标卡片底色分类：橙 / 紫 / 蓝。

    采样四周边框（避开中心图案与底部倒计时），用 HSV 比例判定。
    """
    if patch.size == 0:
        return CARD_BG_UNKNOWN, {}

    icon = _prepare_detail_icon_patch(patch)
    if icon.size == 0:
        icon = patch
    height, width = icon.shape[:2]
    if height < 20 or width < 20:
        return CARD_BG_UNKNOWN, {}

    border = max(4, min(height, width) // 10)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:border, :] = 1
    mask[-border:, :] = 1
    mask[:, :border] = 1
    mask[:, -border:] = 1
    # 去掉角落过多外框，保留侧边饱和底色
    hsv = cv2.cvtColor(icon, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask == 1]
    if pixels.size == 0:
        return CARD_BG_UNKNOWN, {}

    hue = pixels[:, 0].astype(np.int16)
    sat = pixels[:, 1].astype(np.int16)
    val = pixels[:, 2].astype(np.int16)
    colorful = sat > 50
    if float(np.mean(colorful)) < 0.15:
        return CARD_BG_UNKNOWN, {}

    h = hue[colorful]
    # OpenCV H: 0-179
    ratios = {
        CARD_BG_ORANGE: float(np.mean(((h >= 5) & (h <= 28)))),
        CARD_BG_PURPLE: float(np.mean(((h >= 120) & (h <= 165)))),
        CARD_BG_BLUE: float(np.mean(((h >= 88) & (h <= 119)))),
    }
    best = max(ratios, key=ratios.get)
    if ratios[best] < 0.35:
        # 低饱和橙/蓝扩展：亮度高且偏暖视为橙
        warm = float(np.mean((hue <= 30) | (hue >= 170)))
        cool = float(np.mean((hue >= 90) & (hue <= 140)))
        if warm > 0.45 and float(np.mean(val > 160)) > 0.4:
            return CARD_BG_ORANGE, ratios
        if cool > 0.45:
            return CARD_BG_BLUE if ratios[CARD_BG_BLUE] >= ratios[CARD_BG_PURPLE] else CARD_BG_PURPLE, ratios
        return CARD_BG_UNKNOWN, ratios
    return best, ratios


def _prepare_admin_icon_patch(patch: np.ndarray) -> np.ndarray:
    """裁掉彩色卡片边框，只留中心图标区域再匹配。"""
    if patch.size == 0:
        return patch
    h, w = patch.shape[:2]
    mx = max(1, int(w * 0.16))
    my = max(1, int(h * 0.14))
    inner = patch[my : h - my, mx : w - mx]
    return inner if inner.size else patch


def _estimate_content_top(patch: np.ndarray, exclude_top_px: int) -> int:
    """跳过「可接次数」蓝条等 UI，定位首张卡片可能出现的行。"""
    floor = max(0, exclude_top_px)
    height = patch.shape[0]
    if height < floor + 40:
        return floor

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    scan_end = min(height - 20, floor + 130)
    for y in range(floor, scan_end):
        if float(np.mean(hsv[y, :, 1]) > 88):
            return max(floor, y - 10)
    return floor


def _score_strip(col_gray: np.ndarray) -> np.ndarray:
    """列内横向积分条搜索区域（去掉左右边框）。"""
    _height, width = col_gray.shape[:2]
    return col_gray[:, int(width * 0.06) : int(width * 0.94)]


def _band_strength(patch: np.ndarray) -> tuple[float, float]:
    """积分条在白/浅紫/浅橙底上白度差异大，合并两种阈值。"""
    strong = float(np.mean(patch > 198))
    soft = float(np.mean(patch > 175))
    mid = patch.shape[0] // 2
    lower = float(np.mean(patch[mid:, :] > 175)) if patch.shape[0] else 0.0
    return max(strong, soft * 0.88), lower


def _find_score_bands(col_gray: np.ndarray, *, content_top: int = 0) -> list[tuple[int, int]]:
    """扫描列内所有高白积分条，按行聚类后保留每行最佳一条。"""
    height = col_gray.shape[0]
    if height < 60:
        return []

    strip = _score_strip(col_gray)
    min_y1 = max(24, content_top + 12)
    candidates: list[tuple[int, int, float, int]] = []

    for band_h in range(26, 37):
        for y2 in range(min_y1 + band_h, height):
            y1 = y2 - band_h + 1
            if y1 < min_y1:
                continue
            patch = strip[y1 : y2 + 1]
            strength, lower = _band_strength(patch)
            if strength < 0.42 or lower < 0.45:
                continue
            center = (y1 + y2) // 2
            candidates.append((y1, y2, strength, center))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: list[tuple[int, int, float, int]] = []
    for candidate in candidates:
        _y1, _y2, _white, center = candidate
        if any(abs(center - picked[3]) < 36 for picked in selected):
            continue
        selected.append(candidate)
        if len(selected) >= 4:
            break

    selected.sort(key=lambda item: item[3])
    return [(y1, y2) for y1, y2, _white, _center in selected]


def _normalize_score_band(
    y1: int, y2: int, patch_height: int
) -> tuple[int, int]:
    """在检测到的白条基础上略向下扩展，避免把上方彩色图标算进 OCR 区。"""
    pad_top = 2
    pad_bottom = 4
    return max(0, y1 - pad_top), min(patch_height - 1, y2 + pad_bottom)


def _derive_icon_band(score_y1: int) -> tuple[int, int]:
    icon_y2 = max(0, score_y1 - 2)
    icon_y1 = max(0, icon_y2 - ADMIN_CARD_ICON_HEIGHT + 1)
    return icon_y1, icon_y2


def _score_band_is_readable(
    col_bgr: np.ndarray, y1: int, y2: int, patch_height: int
) -> bool:
    """积分条（银币+数字）需完整可见；贴 ROI 底边的完整白条仍视为可读。"""
    if y2 >= patch_height or y1 < 0:
        return False

    band = col_bgr[y1 : y2 + 1]
    if band.size == 0:
        return False

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    strength, _lower = _band_strength(gray)
    if strength < 0.40:
        return False

    text_area = gray[:, int(gray.shape[1] * 0.25) :]
    if text_area.size == 0:
        return False
    if float(text_area.max() - text_area.min()) < 25:
        return False

    if y2 >= patch_height - 6:
        tail = gray[-4:, :]
        if float(np.mean(tail > 195)) < 0.72:
            return False
    return True


def detect_admin_cards(
    screen: np.ndarray,
    *,
    list_roi: tuple[int, int, int, int] = DEFAULT_LIST_ROI,
    column_count: int = DEFAULT_COLUMN_COUNT,
    exclude_top_px: int = DEFAULT_EXCLUDE_TOP_PX,
    only_complete: bool = True,
) -> list[AdminCard]:
    """在管理员列表 ROI 内检测任务卡（跳过顶部专属任务区）。"""
    x1, y1, x2, y2 = list_roi
    patch = _crop(screen, list_roi)
    if patch.size == 0:
        return []

    height, width = patch.shape[:2]
    content_top = _estimate_content_top(patch, max(0, min(exclude_top_px, height - 80)))
    col_width = max(1, width // max(1, column_count))
    cards: list[AdminCard] = []

    for column in range(column_count):
        cx1 = column * col_width
        cx2 = (column + 1) * col_width if column < column_count - 1 else width
        col_patch = patch[:, cx1:cx2]
        col_gray = cv2.cvtColor(col_patch, cv2.COLOR_BGR2GRAY)

        for raw_y1, raw_y2 in _find_score_bands(col_gray, content_top=content_top):
            score_y1, score_y2 = _normalize_score_band(raw_y1, raw_y2, height)
            if score_y2 < content_top:
                continue

            icon_y1, icon_y2 = _derive_icon_band(score_y1)
            visible_icon_y1 = max(icon_y1, content_top)
            visible_icon_h = icon_y2 - visible_icon_y1 + 1
            if visible_icon_h < 20:
                continue

            readable = _score_band_is_readable(col_patch, score_y1, score_y2, height)
            partial_top = icon_y1 < content_top
            if only_complete:
                if partial_top:
                    complete = readable and visible_icon_h >= 20
                else:
                    complete = (
                        readable
                        and visible_icon_h >= 48
                        and icon_y2 >= content_top + 6
                        and icon_y1 >= content_top - 16
                    )
            else:
                complete = readable
            if only_complete and not complete:
                continue

            icon_roi = _offset_roi(
                (cx1 + 0, visible_icon_y1, cx2, icon_y2 + 1),
                (x1, y1),
            )
            score_roi = _offset_roi(
                (cx1 + 0, score_y1, cx2, score_y2 + 1),
                (x1, y1),
            )
            row_key = int((score_y1 + score_y2) // 2)
            cards.append(
                AdminCard(
                    column=column,
                    row_key=row_key,
                    icon_roi=icon_roi,
                    score_roi=score_roi,
                )
            )

    cards.sort(key=lambda item: (item.row_key, item.column))
    return _filter_sparse_rows(cards)


def _filter_sparse_rows(cards: list[AdminCard]) -> list[AdminCard]:
    """去掉仅单列命中、且图标过小的伪行（彩色底误检）。"""
    if len(cards) < 2:
        return cards

    ordered = sorted(cards, key=lambda item: item.row_key)
    groups: list[list[AdminCard]] = []
    for card in ordered:
        if not groups or card.row_key - groups[-1][0].row_key > 24:
            groups.append([card])
        else:
            groups[-1].append(card)

    kept: list[AdminCard] = []
    seen: set[tuple[int, int]] = set()
    for group in groups:
        columns = {card.column for card in group}
        heights = [
            card.icon_roi[3] - card.icon_roi[1] + 1 for card in group
        ]
        max_h = max(heights) if heights else 0
        if len(columns) >= 2 and max_h >= 48:
            for card in group:
                if card.key not in seen:
                    kept.append(card)
                    seen.add(card.key)
            continue
        if len(group) == 1:
            card = group[0]
            if max_h >= 48 and card.key not in seen:
                kept.append(card)
                seen.add(card.key)
    kept.sort(key=lambda item: (item.row_key, item.column))
    return kept


def _needs_scroll_down(screen: np.ndarray, list_roi: tuple[int, int, int, int]) -> bool:
    """底部有图标但未见完整积分条 → 向上滑以露出下方卡片。"""
    patch = _crop(screen, list_roi)
    if patch.size == 0:
        return False

    height = patch.shape[0]
    bottom = patch[int(height * 0.72) :, :]
    if bottom.size == 0:
        return False

    hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    colorful = float(np.mean(sat > 45))
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    white_bottom = float(np.mean(gray[int(height * 0.28) :] > 210))
    return colorful > 0.12 and white_bottom < 0.35


class AllianceMobilizationAdminSession:
    """联盟总动员管理员：滚动列表 + 练兵筛选。"""

    def __init__(
        self,
        adb: AdbClient,
        *,
        admin_cfg: dict | None = None,
        target_types: list[str] | None = None,
        score_threshold: int = DEFAULT_SCORE_THRESHOLD,
        scan_interval: float = DEFAULT_SCAN_INTERVAL,
        step_delay: float = DEFAULT_STEP_DELAY,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        countdown_threshold: float = DEFAULT_COUNTDOWN_THRESHOLD,
        ocr_engine: str = "auto",
        list_roi: tuple[int, int, int, int] | None = None,
        exclude_top_px: int = DEFAULT_EXCLUDE_TOP_PX,
        column_count: int = DEFAULT_COLUMN_COUNT,
        scroll: dict | None = None,
        coords: dict[str, list[int]] | None = None,
        on_status: StatusCallback | None = None,
    ):
        if admin_cfg is not None:
            merged = merge_task_config(admin_cfg)
        else:
            merged = merge_task_config(
                {
                    "target_types": target_types or DEFAULT_TARGET_TYPES,
                    "score_threshold": score_threshold,
                    "scan_interval": scan_interval,
                    "step_delay": step_delay,
                    "match_threshold": match_threshold,
                    "countdown_threshold": countdown_threshold,
                    "ocr_engine": ocr_engine,
                    "list_roi": list(list_roi) if list_roi else list(DEFAULT_LIST_ROI),
                    "exclude_top_px": exclude_top_px,
                    "column_count": column_count,
                    "scroll": scroll or {},
                    "coords": coords or {},
                }
            )
        self.adb = adb
        self.target_types = list(merged["target_types"])
        self.score_threshold = int(merged["score_threshold"])
        self.scan_interval = float(merged["scan_interval"])
        self.step_delay = float(merged["step_delay"])
        self.match_threshold = float(merged["match_threshold"])
        self.countdown_threshold = float(merged["countdown_threshold"])
        self.ocr_engine = resolve_ocr_engine(merged["ocr_engine"])
        self.list_roi = merged["list_roi"]
        self.exclude_top_px = int(merged["exclude_top_px"])
        self.column_count = int(merged["column_count"])
        self.detail_refresh_btn_roi = merged["detail_refresh_btn_roi"]
        self.detail_icon_roi = merged["detail_icon_roi"]
        self.detail_score_roi = merged["detail_score_roi"]
        self.detail_title_roi = merged["detail_title_roi"]
        self.detail_close_btn_roi = merged["detail_close_btn_roi"]
        self.detail_match_threshold = float(merged["detail_match_threshold"])
        self.keep_orange_types = set(merged["keep_orange_types"])
        self.use_score_ocr = bool(merged["use_score_ocr"])
        self.scroll = merged["scroll"]
        self.coords = merged["coords"]
        self.on_status = on_status
        self._stop_event = threading.Event()
        self.vision = Vision(TEMPLATE_DIR, threshold=self.match_threshold)
        mask_x, mask_y = self._detail_mask_xy()
        logger.info(
            f"[{self.name}] 详情关闭：遮罩=({mask_x},{mask_y})"
        )

    def _detail_mask_xy(self) -> tuple[int, int]:
        raw = self.coords.get("detail_mask_tap")
        if raw and len(raw) >= 2:
            xy = (int(raw[0]), int(raw[1]))
            if xy not in DEPRECATED_DETAIL_MASK_TAP:
                return xy
        return CALIBRATED_DETAIL_MASK_TAP

    @property
    def name(self) -> str:
        return "联盟管理员刷新"

    def stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

    def _interrupted(self) -> bool:
        return self._stop_event.is_set()

    def _emit(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
        if self.on_status:
            self.on_status(message)

    def _tap_xy(self, x: int, y: int, delay: float | None = None) -> None:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        self.adb.tap(x, y)
        time.sleep(delay if delay is not None else self.step_delay)

    def _sleep_interruptible(self, seconds: float) -> None:
        end = time.time() + max(0.0, seconds)
        while time.time() < end:
            if self._interrupted():
                raise InterruptedError("任务已停止")
            time.sleep(min(0.2, end - time.time()))

    def _swipe_up(self, *, mode: str = "normal") -> tuple[int, int, int, int]:
        """上滑列表。mode: small / medium / normal。慢滑减少惯性飞滑。"""
        if self._interrupted():
            raise InterruptedError("任务已停止")
        sx = int(self.scroll["swipe_x"])
        if mode == "small":
            y1 = int(self.scroll["swipe_small_y1"])
            y2 = int(self.scroll["swipe_small_y2"])
            ms = int(self.scroll.get("swipe_small_ms", 1200))
        elif mode == "medium":
            y1 = int(self.scroll.get("swipe_medium_y1", 1240))
            y2 = int(self.scroll.get("swipe_medium_y2", 1085))
            ms = int(self.scroll.get("swipe_medium_ms", 1100))
        else:
            y1 = int(self.scroll["swipe_up_y1"])
            y2 = int(self.scroll["swipe_up_y2"])
            ms = int(self.scroll["swipe_ms"])
        self.adb.swipe(sx, y1, sx, y2, duration_ms=ms)
        return sx, y1, y2, ms

    def _swipe_down(self) -> None:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        sx = int(self.scroll["swipe_x"])
        y1 = int(self.scroll["swipe_down_y1"])
        y2 = int(self.scroll["swipe_down_y2"])
        ms = int(self.scroll["swipe_ms"])
        self.adb.swipe(sx, y1, sx, y2, duration_ms=ms)

    def _screenshot_stable(self) -> np.ndarray:
        self._sleep_interruptible(float(self.scroll.get("screen_delay", 0.35)))
        return self.adb.screenshot()

    def _scroll_to_top(self) -> None:
        """每轮扫描前固定滑回列表顶部。"""
        count = int(self.scroll.get("scroll_to_top_max", 10))
        self._emit(f"滚回列表顶部：下滑 {count} 次（scroll_to_top_max={count}）…")
        settle = float(self.scroll["settle_delay"])
        for index in range(count):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            self._swipe_down()
            self._sleep_interruptible(settle)
            if index == 0 or index == count - 1 or (index + 1) % 5 == 0:
                self._emit(f"回顶下滑进度 {index + 1}/{count}")
        self._emit(f"回顶下滑完成（共 {count} 次）")

    def _match_in_roi(
        self,
        screen: np.ndarray,
        roi: tuple[int, int, int, int],
        template_name: str,
        threshold: float,
        *,
        admin_icon: bool = False,
    ) -> float:
        patch = _crop(screen, roi)
        if patch.size == 0:
            return 0.0
        if admin_icon:
            patch = _prepare_admin_icon_patch(patch)
        old = self.vision.threshold
        self.vision.threshold = threshold
        try:
            result = self.vision.match_template_multiscale(
                patch,
                template_name,
                scales=ADMIN_TRAIN_MATCH_SCALES,
            )
        finally:
            self.vision.threshold = old
        return float(result.confidence)

    def _match_template_in_patch(
        self,
        patch: np.ndarray,
        template_name: str,
        *,
        scales: tuple[float, ...] = ADMIN_TRAIN_MATCH_SCALES,
    ) -> float:
        if patch.size == 0:
            return 0.0
        old = self.vision.threshold
        self.vision.threshold = 0.0
        try:
            result = self.vision.match_template_multiscale(
                patch, template_name, scales=scales
            )
        finally:
            self.vision.threshold = old
        return float(result.confidence)

    def _admin_template_exists(self, template_name: str) -> bool:
        return (TEMPLATE_DIR / template_name).is_file()

    def _resolve_detail_templates(self, type_id: str) -> list[str]:
        """详情匹配：优先细节模板 / tight 裁切，再回落到列表模板。"""
        names: list[str] = []
        preferred = TASK_TYPE_ADMIN_DETAIL_TEMPLATES.get(type_id)
        if preferred:
            names.append(preferred)
        base = TASK_TYPE_ADMIN_TEMPLATES.get(type_id)
        if base:
            tight = base.replace(".png", "_tight.png")
            names.append(tight)
            names.append(base)
        if type_id == TASK_TYPE_TRAIN:
            names.append(TRAIN_ICON_ADMIN_INNER_TEMPLATE)
            names.append(TRAIN_ICON_TEMPLATE)
        # 去重且文件存在
        seen: set[str] = set()
        out: list[str] = []
        for name in names:
            if name in seen or not self._admin_template_exists(name):
                continue
            seen.add(name)
            out.append(name)
        return out

    def _match_admin_train_conf_in_patch(
        self,
        patch: np.ndarray,
        *,
        scales: tuple[float, ...] = ADMIN_TRAIN_DETAIL_MATCH_SCALES,
    ) -> float:
        return self._match_best_template_in_patch(
            patch, self._resolve_detail_templates(TASK_TYPE_TRAIN), scales=scales
        )

    def _match_best_template_in_patch(
        self,
        patch: np.ndarray,
        template_names: list[str],
        *,
        scales: tuple[float, ...] = ADMIN_TRAIN_DETAIL_MATCH_SCALES,
    ) -> float:
        best = 0.0
        for name in template_names:
            best = max(
                best, self._match_template_in_patch(patch, name, scales=scales)
            )
        return best

    def _score_detail_icon_types(self, patch: np.ndarray) -> dict[str, float]:
        """详情弹窗图标区：对所有管理员模板打分。"""
        scores: dict[str, float] = {}
        for type_id in ADMIN_DETAIL_TYPE_ORDER:
            names = self._resolve_detail_templates(type_id)
            if not names:
                continue
            scores[type_id] = self._match_best_template_in_patch(patch, names)
        return scores

    def _classify_detail_icon(
        self,
        screen: np.ndarray,
        *,
        list_hint: bool = False,
    ) -> tuple[str | None, dict[str, float]]:
        """详情二次确认：返回最佳类型与各类型置信度。"""
        raw = _crop(screen, self.detail_icon_roi)
        # 同时用「去倒计时」与「原 ROI」取各类型最高分，避免裁切过狠
        patches = [
            _prepare_detail_icon_patch(raw),
            raw,
        ]
        scores: dict[str, float] = {}
        for patch in patches:
            if patch.size == 0:
                continue
            for type_id, conf in self._score_detail_icon_types(patch).items():
                scores[type_id] = max(scores.get(type_id, 0.0), conf)

        if not scores:
            self._emit("详情图标校验：无可用模板")
            return None, scores

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_type, best_conf = ranked[0]
        second_conf = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_conf - second_conf
        parts = [
            f"{TASK_TYPE_LABELS.get(type_id, type_id)}={conf:.2f}"
            for type_id, conf in ranked
        ]
        best_label = TASK_TYPE_LABELS.get(best_type, best_type)
        self._emit(
            f"详情图标校验: {', '.join(parts)} → {best_label} "
            f"(领先次佳 {margin:.2f})"
        )

        threshold = max(self.detail_match_threshold, DETAIL_NOISE_FLOOR)
        if list_hint and best_type == TASK_TYPE_TRAIN:
            threshold = max(0.48, threshold - 0.05)

        # 全员 0.3x：噪音区，不能当真
        if best_conf < threshold or margin < DETAIL_MIN_CONF_MARGIN:
            if list_hint and best_conf >= 0.40 and (
                best_type == TASK_TYPE_TRAIN
                or scores.get(TASK_TYPE_TRAIN, 0.0) >= best_conf - 0.05
            ):
                self._emit(
                    f"详情置信度不足(best={best_conf:.2f}/margin={margin:.2f})，"
                    f"但列表预检为练兵，按练兵处理"
                )
                return TASK_TYPE_TRAIN, scores
            self._emit(
                f"详情无法确认(best={best_conf:.2f}/margin={margin:.2f}，"
                f"需要≥{threshold:.2f}且领先≥{DETAIL_MIN_CONF_MARGIN:.2f})，"
                f"视为非练兵"
            )
            return None, scores
        return best_type, scores

    def _is_cooldown(self, screen: np.ndarray, card: AdminCard) -> bool:
        """自己刷新后的冷却（与普通模式相同，看图标区）。"""
        icon = _crop(screen, card.icon_roi)
        conf = self._match_in_roi(
            screen, card.icon_roi, COUNTDOWN_TEMPLATE, self.countdown_threshold
        )
        if conf >= self.countdown_threshold:
            return True
        text, _ = ocr_chip_text(icon, engine=self.ocr_engine)
        return _looks_like_countdown_text(text)

    def _match_train_at_roi(
        self, screen: np.ndarray, roi: tuple[int, int, int, int]
    ) -> tuple[bool, float, float]:
        """指定 ROI 内管理员练兵模板匹配（详情弹窗二次确认）。"""
        admin_template = TASK_TYPE_ADMIN_TEMPLATES.get(TASK_TYPE_TRAIN)
        if not admin_template:
            conf = self._match_in_roi(
                screen, roi, TRAIN_ICON_TEMPLATE, self.match_threshold, admin_icon=True
            )
            return conf >= self.match_threshold, conf, conf

        admin_conf = self._match_in_roi(
            screen, roi, admin_template, self.match_threshold, admin_icon=True
        )
        legacy_conf = self._match_in_roi(
            screen, roi, TRAIN_ICON_TEMPLATE, 0.0, admin_icon=True
        )
        is_train = admin_conf >= self.match_threshold and admin_conf >= (
            legacy_conf + ADMIN_TRAIN_LEGACY_MIN_DELTA
        )
        return is_train, admin_conf, legacy_conf

    def _match_train_admin(self, screen: np.ndarray, card: AdminCard) -> tuple[bool, float, float]:
        """管理员练兵：专用模板为主，旧模板作误报抑制。"""
        return self._match_train_at_roi(screen, card.icon_roi)

    def _match_target_type(self, screen: np.ndarray, card: AdminCard) -> str | None:
        best_type = None
        best_conf = 0.0
        label = f"列{card.column + 1}/行{card.row_key}"
        for task_type in self.target_types:
            if task_type == TASK_TYPE_TRAIN:
                is_train, admin_conf, legacy_conf = self._match_train_admin(screen, card)
                conf = admin_conf
                if is_train and conf > best_conf:
                    best_type = task_type
                    best_conf = conf
                if not is_train:
                    logger.info(
                        f"[{self.name}] {label} 非练兵 "
                        f"(admin={admin_conf:.2f}, legacy={legacy_conf:.2f})"
                    )
                continue

            template = TASK_TYPE_TEMPLATES.get(task_type)
            if not template:
                continue
            conf = self._match_in_roi(
                screen,
                card.icon_roi,
                template,
                self.match_threshold,
                admin_icon=False,
            )
            if conf >= self.match_threshold and conf > best_conf:
                best_type = task_type
                best_conf = conf

        if best_type == TASK_TYPE_TRAIN:
            logger.info(f"[{self.name}] {label} 识别为练兵（conf={best_conf:.2f}）")
        return best_type

    def _read_score(self, screen: np.ndarray, card: AdminCard) -> int | None:
        patch = _crop(screen, card.score_roi)
        if patch.size == 0:
            return None
        label = f"列{card.column + 1}/行{card.row_key}"
        best_score = None
        best_text = ""
        best_engine = "unknown"
        variants = [
            _prepare_score_patch(patch, crop_left_ratio=0.15),
            _prepare_score_patch(patch, crop_left_ratio=0.22),
            _prepare_score_patch(patch, crop_left_ratio=0.25),
            _prepare_score_patch(patch, crop_left_ratio=0.0),
            cv2.resize(patch, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
            patch,
        ]
        for variant in variants:
            text, engine = ocr_chip_text(variant, engine=self.ocr_engine)
            score = _parse_score(text)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_text = text
                best_engine = engine
        logger.info(
            f"[{self.name}] {label} 分数 OCR({best_engine})={best_text!r} -> {best_score}"
        )
        return best_score

    def _card_in_content_area(self, card: AdminCard) -> bool:
        _, list_y1, _, _ = self.list_roi
        _, icon_y1, _, _ = card.icon_roi
        return icon_y1 >= list_y1 + self.exclude_top_px

    def _prepare_detail_button_ocr(self, patch: np.ndarray) -> np.ndarray:
        if patch.size == 0:
            return patch
        scaled = cv2.resize(patch, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    def _detail_refresh_button_text(self, screen: np.ndarray) -> tuple[str, str]:
        patch = _crop(screen, self.detail_refresh_btn_roi)
        if patch.size == 0:
            return "", "empty"
        variants = [self._prepare_detail_button_ocr(patch), patch]
        best_text = ""
        best_engine = "unknown"
        for variant in variants:
            text, engine = ocr_chip_text(variant, engine=self.ocr_engine)
            if len(text) > len(best_text):
                best_text = text
                best_engine = engine
        return best_text, best_engine

    def _detail_has_refresh_button(self, screen: np.ndarray) -> bool:
        text, engine = self._detail_refresh_button_text(screen)
        compact = (text or "").replace(" ", "").replace("\n", "")
        ok = "刷新任务" in compact or ("刷新" in compact and "任务" in compact)
        logger.info(
            f"[{self.name}] 详情按钮 OCR({engine})={text!r} -> "
            f"{'刷新任务' if ok else '非刷新按钮'}"
        )
        return ok

    def _read_detail_score(self, screen: np.ndarray) -> int | None:
        patch = _crop(screen, self.detail_score_roi)
        if patch.size == 0:
            return None
        best_score = None
        best_text = ""
        best_engine = "unknown"
        variants = [
            _prepare_score_patch(patch, crop_left_ratio=0.28),
            _prepare_score_patch(patch, crop_left_ratio=0.35),
            _prepare_score_patch(patch, crop_left_ratio=0.20),
            _prepare_score_patch(patch, crop_left_ratio=0.0),
            cv2.resize(patch, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
            patch,
        ]
        for variant in variants:
            text, engine = ocr_chip_text(variant, engine=self.ocr_engine)
            score = _parse_score(text)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_text = text
                best_engine = engine
        logger.info(
            f"[{self.name}] 详情分数 OCR({best_engine})={best_text!r} -> {best_score}"
        )
        return best_score

    def _open_card_detail(self, card: AdminCard) -> None:
        cx, cy = _roi_center(card.icon_roi)
        self._tap_xy(cx, cy, delay=self.step_delay)

    def _detail_popup_open(self, screen: np.ndarray) -> bool:
        """是否处于「公共任务」详情弹窗（仅认标题 OCR，避免列表背景误判）。"""
        title_patch = _crop(screen, self.detail_title_roi)
        if title_patch.size == 0:
            return False
        text, engine = ocr_chip_text(title_patch, engine=self.ocr_engine)
        compact = (text or "").replace(" ", "").replace("\n", "")
        is_open = "公共任务" in compact
        logger.info(
            f"[{self.name}] 详情弹窗检测 OCR({engine})={text!r} -> "
            f"{'打开' if is_open else '关闭'}"
        )
        return is_open

    def _close_detail_popup(self) -> bool:
        """关闭详情弹窗：点一次遮罩即可，不按返回键（易退出联盟界面）。"""
        if self._interrupted():
            raise InterruptedError("任务已停止")

        screen = self._screenshot_stable()
        if not self._detail_popup_open(screen):
            return True

        mask_x, mask_y = self._detail_mask_xy()
        self._emit(f"点遮罩关闭详情 ({mask_x},{mask_y})")
        self.adb.tap(mask_x, mask_y)
        self._sleep_interruptible(
            max(DETAIL_CLOSE_SETTLE_SEC, self.step_delay * 0.75)
        )

        if self._detail_popup_open(self._screenshot_stable()):
            logger.warning(
                f"[{self.name}] 遮罩后标题仍像弹窗，不再按返回键"
            )
        return True

    def _ensure_detail_closed(self, label: str) -> None:
        self._close_detail_popup()

    def _finish_refresh_from_detail(self) -> None:
        """弹窗已打开时：点「刷新任务」+ 确认（原三次点击的后两次）。"""
        rx, ry = self.coords["refresh_tap"]
        confirm_x, confirm_y = self.coords["refresh_confirm"]
        self._tap_xy(rx, ry, delay=self.step_delay)
        self._tap_xy(confirm_x, confirm_y, delay=self.step_delay)

    def _process_card(self, screen: np.ndarray, card: AdminCard) -> tuple[str, bool]:
        """处理单卡：点一次开详情弹窗，弹窗内确认后再决定是否刷新。"""
        label = f"列{card.column + 1}/行{card.row_key}"
        if not self._card_in_content_area(card):
            self._emit(f"{label}：位于专属任务区，跳过")
            return "skip_exclusive", False
        if self._is_cooldown(screen, card):
            self._emit(f"{label}：冷却中，跳过")
            return "cooldown", False

        list_train, list_admin, list_legacy = self._match_train_admin(screen, card)
        if list_train:
            logger.info(
                f"[{self.name}] {label} 列表预检为练兵 "
                f"(admin={list_admin:.2f}, legacy={list_legacy:.2f})"
            )

        self._emit(f"{label}：打开任务详情…")
        self._open_card_detail(card)
        self._sleep_interruptible(DETAIL_POPUP_WAIT_SEC)
        detail = self._screenshot_stable()

        if not self._detail_has_refresh_button(detail):
            btn_text, _ = self._detail_refresh_button_text(detail)
            if self._detail_popup_open(detail):
                hint = f"（识别到：{btn_text!r}）" if btn_text else ""
                self._emit(f"{label}：详情无「刷新任务」按钮{hint}，关闭")
                self._ensure_detail_closed(label)
            else:
                self._emit(f"{label}：详情未打开，跳过")
            return "skip_no_refresh", False

        detail_type, type_scores = self._classify_detail_icon(
            detail, list_hint=list_train
        )
        train_conf = type_scores.get(TASK_TYPE_TRAIN, 0.0)
        icon_patch = _crop(detail, self.detail_icon_roi)
        bg_color, bg_ratios = classify_card_bg_color(icon_patch)
        bg_label = CARD_BG_LABELS.get(bg_color, bg_color)
        self._emit(
            f"{label}：底色={bg_label} "
            f"(橙{bg_ratios.get(CARD_BG_ORANGE, 0):.2f}/"
            f"紫{bg_ratios.get(CARD_BG_PURPLE, 0):.2f}/"
            f"蓝{bg_ratios.get(CARD_BG_BLUE, 0):.2f})"
        )

        # 紫/蓝：低价值，直接刷新；未知底色也走刷新更安全
        if bg_color != CARD_BG_ORANGE:
            type_label = TASK_TYPE_LABELS.get(detail_type, detail_type or "未知")
            self._emit(
                f"{label}：{bg_label}卡（{type_label}，练兵={train_conf:.2f}），刷新"
            )
            self._finish_refresh_from_detail()
            self._sleep_interruptible(DETAIL_POPUP_WAIT_SEC)
            return "refresh", True

        # 橙色：再看是否为目标样式（现阶段仅练兵）
        keep_type = detail_type if detail_type in self.keep_orange_types else None
        if keep_type is None and TASK_TYPE_TRAIN in self.keep_orange_types:
            # 橙底 + 练兵分最高（即便未过硬阈值）或列表预检练兵 → 保留
            ranked = sorted(type_scores.items(), key=lambda item: item[1], reverse=True)
            top_type = ranked[0][0] if ranked else None
            if list_train or (
                top_type == TASK_TYPE_TRAIN and train_conf >= 0.40
            ):
                keep_type = TASK_TYPE_TRAIN
                self._emit(
                    f"{label}：橙色 + 练兵样式（"
                    f"{'列表预检' if list_train else f'练兵={train_conf:.2f}'}），保留"
                )

        if keep_type is None:
            type_label = TASK_TYPE_LABELS.get(detail_type, detail_type or "未知")
            self._emit(
                f"{label}：橙色但非目标样式（{type_label}，练兵={train_conf:.2f}），刷新"
            )
            self._finish_refresh_from_detail()
            self._sleep_interruptible(DETAIL_POPUP_WAIT_SEC)
            return "refresh", True

        type_label = TASK_TYPE_LABELS.get(keep_type, keep_type)
        if not self.use_score_ocr:
            self._emit(f"{label}：橙色{type_label}，保留（跳过分数 OCR）")
            self._ensure_detail_closed(label)
            return "keep", False

        score = self._read_detail_score(detail)
        if score is None:
            self._emit(f"{label}：橙色{type_label} 分数未识别，保留")
            self._ensure_detail_closed(label)
            return "keep_unknown_score", False

        if score >= self.score_threshold:
            self._emit(
                f"{label}：橙色{type_label} +{score} ≥ {self.score_threshold}，保留"
            )
            self._ensure_detail_closed(label)
            return "keep", False

        self._emit(
            f"{label}：橙色{type_label} +{score} < {self.score_threshold}，刷新"
        )
        self._finish_refresh_from_detail()
        self._sleep_interruptible(DETAIL_POPUP_WAIT_SEC)
        return "refresh", True

    def _scan_scrollable_once(self) -> tuple[bool, int, int]:
        """滚动扫描一轮。返回 (是否应停止, 保留数, 可处理数)。"""
        # 刷新按钮/弹窗标题仍可能用 OCR；底色+图标主流程可不依赖分数 OCR
        if self.use_score_ocr and not ocr_engine_available(self.ocr_engine):
            raise RuntimeError(
                "OCR 不可用。请安装 rapidocr-onnxruntime onnxruntime，"
                "或确保本机已安装 Tesseract"
            )
        if not ocr_engine_available(self.ocr_engine):
            raise RuntimeError(
                "OCR 不可用（刷新按钮/标题检测仍需要）。"
                "请安装 rapidocr-onnxruntime onnxruntime，或确保本机已安装 Tesseract"
            )

        keep_labels = "、".join(
            TASK_TYPE_LABELS[t] for t in self.keep_orange_types if t in TASK_TYPE_LABELS
        )
        self._emit(
            f"开始扫描管理员列表（保留：橙色{keep_labels}；紫/蓝刷新；"
            f"{'读分数' if self.use_score_ocr else '跳过分数 OCR'}）"
        )

        self._scroll_to_top()
        self._sleep_interruptible(float(self.scroll["settle_delay"]))

        processed: set[tuple[int, tuple[int, ...]]] = set()
        keep_count = 0
        actionable_count = 0
        refreshed = 0
        stale_scrolls = 0
        max_swipes = int(self.scroll["max_swipes_per_pass"])
        end_reason = "unknown"
        self._emit(
            f"扫描参数：回顶下滑={int(self.scroll.get('scroll_to_top_max', 10))} 次，"
            f"本轮最多上滑={max_swipes} 次（max_swipes_per_pass）"
        )

        for swipe_index in range(max_swipes + 1):
            if self._interrupted():
                raise InterruptedError("任务已停止")

            screen = self._screenshot_stable()
            cards = detect_admin_cards(
                screen,
                list_roi=self.list_roi,
                column_count=self.column_count,
                exclude_top_px=self.exclude_top_px,
                only_complete=True,
            )
            new_cards: list[AdminCard] = []
            for card in cards:
                # 冷却/倒计时槽（整行里另外两个灰底）直接跳过，不进入处理
                if self._is_cooldown(screen, card):
                    cool_key = _card_content_key(screen, card)
                    processed.add(cool_key)
                    continue
                content_key = _card_content_key(screen, card)
                if _is_duplicate_content(content_key, processed):
                    continue
                new_cards.append(card)
            if new_cards:
                new_cards.sort(key=lambda card: card.row_key)
                self._emit(
                    f"本屏发现 {len(new_cards)} 张待处理卡片"
                    f"（上滑进度 {swipe_index}/{max_swipes}）"
                )

            for card in new_cards:
                content_key = _card_content_key(screen, card)
                processed.add(content_key)
                actionable_count += 1
                result, did_refresh = self._process_card(screen, card)
                if did_refresh:
                    refreshed += 1
                    # 刷新后槽位是新卡，不要把新卡指纹记入 processed，本轮后续可再处理
                    screen = self._screenshot_stable()
                    continue

                # 保留/跳过：再记处理后指纹（忽略倒计时跳变）
                screen = self._screenshot_stable()
                processed.add(_card_content_key(screen, card))
                if result == "keep":
                    keep_count += 1

            if swipe_index >= max_swipes:
                end_reason = "max_swipes"
                self._emit(
                    f"结束原因：本轮上滑次数已达上限 "
                    f"{max_swipes}/{max_swipes}（max_swipes_per_pass），"
                    f"下方未扫到的卡片需增大该值"
                )
                break

            # 本屏卡片处理完（含关弹窗）后重新截图，再判断是否需要补滑
            screen = self._screenshot_stable()
            before_sig = _patch_signature(_crop(screen, self.list_roi))
            partial = detect_admin_cards(
                screen,
                list_roi=self.list_roi,
                column_count=self.column_count,
                exclude_top_px=self.exclude_top_px,
                only_complete=False,
            )
            complete_keys = {card.key for card in cards}
            has_partial = any(card.key not in complete_keys for card in partial)

            if has_partial:
                pre_delay = float(self.scroll.get("partial_pre_delay", 0.55))
                self._emit(
                    f"底部卡片未完整露出，{pre_delay:.1f}s 后小幅上滑"
                )
                self._sleep_interruptible(pre_delay)
            elif _needs_scroll_down(screen, self.list_roi):
                self._emit("底部卡片不完整，小幅上滑列表")
            else:
                self._emit("继续小幅上滑以扫描更多卡片")

            # 统一小幅慢滑：距离够翻一行，时长拉长压住惯性
            sx, y1, y2, ms = self._swipe_up(mode="small")
            self._emit(
                f"上滑[small] ({sx},{y1})→({sx},{y2}) {ms}ms "
                f"[{swipe_index + 1}/{max_swipes}]"
            )
            settle = (
                float(self.scroll.get("partial_settle_delay", 1.35))
                if has_partial
                else float(self.scroll["settle_delay"])
            )
            self._sleep_interruptible(settle)
            after_screen = self._screenshot_stable()
            after_sig = _patch_signature(_crop(after_screen, self.list_roi))

            if after_sig == before_sig:
                self._emit("列表未移动，改用中等上滑重试（慢滑）")
                sx, y1, y2, ms = self._swipe_up(mode="medium")
                self._emit(f"上滑[medium] ({sx},{y1})→({sx},{y2}) {ms}ms")
                self._sleep_interruptible(float(self.scroll["settle_delay"]))
                after_screen = self._screenshot_stable()
                after_sig = _patch_signature(_crop(after_screen, self.list_roi))

            if after_sig == before_sig and not has_partial:
                stale_scrolls += 1
                self._emit(
                    f"上滑后画面未变（连续 {stale_scrolls}/2），"
                    f"当前上滑进度 {swipe_index + 1}/{max_swipes}"
                )
                if stale_scrolls >= 2:
                    end_reason = "bottom"
                    self._emit(
                        "结束原因：连续 2 次上滑列表未移动，判定已滑到底部"
                    )
                    break
            else:
                stale_scrolls = 0

        if end_reason == "unknown":
            end_reason = "loop_finished"
            self._emit(
                f"结束原因：扫描循环正常走完"
                f"（上滑上限 {max_swipes}，实际屏次数={max_swipes + 1}）"
            )

        if actionable_count == 0:
            self._emit("未发现完整卡片，请确认已在管理员任务列表页")
            return False, 0, 0

        done = keep_count >= actionable_count and refreshed == 0
        if done:
            self._emit(
                f"目标已达成：{keep_count}/{actionable_count} 张卡均为达标任务，停止"
                f"（结束原因={end_reason}）"
            )
        else:
            reason_text = {
                "max_swipes": "上滑次数上限",
                "bottom": "判定滑到底部",
                "loop_finished": "循环结束",
            }.get(end_reason, end_reason)
            self._emit(
                f"本轮保留 {keep_count} 张，刷新 {refreshed} 张；"
                f"结束原因={reason_text}；"
                f"{int(self.scan_interval // 60)} 分钟后再次扫描"
            )
        return done, keep_count, actionable_count

    def run_until_stopped(self) -> None:
        self.reset_stop()
        try:
            while not self._interrupted():
                done, _keep, _total = self._scan_scrollable_once()
                if done:
                    return
                self._sleep_interruptible(self.scan_interval)
        except InterruptedError:
            self._emit("任务已结束")
            return
