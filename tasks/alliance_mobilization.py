"""联盟总动员自动刷新：扫描任务卡，保留高分练兵，刷新其它并等待冷却。"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from loguru import logger

from core.adb_client import AdbClient
from core.dream_memory.ocr_engine import ocr_chip_text, ocr_engine_available, resolve_ocr_engine
from core.vision import Vision

StatusCallback = Callable[[str], None]

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "assets" / "templates"
ASSET_SUBDIR = "alliance_mobilization"

TRAIN_ICON_TEMPLATE = f"{ASSET_SUBDIR}/train_icon.png"
TRAIN_ICON_ADMIN_TEMPLATE = f"{ASSET_SUBDIR}/train_icon_admin.png"
COUNTDOWN_TEMPLATE = f"{ASSET_SUBDIR}/countdown.png"

TASK_TYPE_TRAIN = "train"
TASK_TYPE_BEAST = "beast"
TASK_TYPE_SPEEDUP = "speedup"
TASK_TYPE_TECH = "tech"
TASK_TYPE_ORANGE_SHARD = "orange_shard"
TASK_TYPE_GEM = "gem"
TASK_TYPE_SMALL_MONSTER = "small_monster"
TASK_TYPE_GIFT_PACK = "gift_pack"
TASK_TYPE_BUILDING = "building"
TASK_TYPE_DIAMOND = "diamond"
TASK_TYPE_FIRE_CRYSTAL = "fire_crystal"
TASK_TYPE_ENERGY_STONE = "energy_stone"
TASK_TYPE_RESOURCE = "resource"

# 详情弹窗图标分类 / GUI 多选顺序
ADMIN_DETAIL_TYPE_ORDER: tuple[str, ...] = (
    TASK_TYPE_TRAIN,
    TASK_TYPE_BEAST,
    TASK_TYPE_SPEEDUP,
    TASK_TYPE_TECH,
    TASK_TYPE_ORANGE_SHARD,
    TASK_TYPE_GEM,
    TASK_TYPE_FIRE_CRYSTAL,
    TASK_TYPE_SMALL_MONSTER,
    TASK_TYPE_GIFT_PACK,
    TASK_TYPE_BUILDING,
    TASK_TYPE_DIAMOND,
    TASK_TYPE_ENERGY_STONE,
    TASK_TYPE_RESOURCE,
)

TASK_TYPE_LABELS: dict[str, str] = {
    TASK_TYPE_TRAIN: "练兵",
    TASK_TYPE_BEAST: "巨兽",
    TASK_TYPE_SPEEDUP: "加速",
    TASK_TYPE_TECH: "科技",
    TASK_TYPE_ORANGE_SHARD: "橙碎",
    TASK_TYPE_GEM: "宝石",
    TASK_TYPE_FIRE_CRYSTAL: "火晶",
    TASK_TYPE_SMALL_MONSTER: "小怪",
    TASK_TYPE_GIFT_PACK: "礼包",
    TASK_TYPE_BUILDING: "建造",
    TASK_TYPE_DIAMOND: "钻石",
    TASK_TYPE_ENERGY_STONE: "能源石",
    TASK_TYPE_RESOURCE: "资源",
}

# GUI / 配置可多选的全部类型
ADMIN_TARGET_TYPES: tuple[str, ...] = ADMIN_DETAIL_TYPE_ORDER

TASK_TYPE_TEMPLATES: dict[str, str] = {
    TASK_TYPE_TRAIN: TRAIN_ICON_TEMPLATE,
}
TASK_TYPE_ADMIN_TEMPLATES: dict[str, str] = {
    TASK_TYPE_TRAIN: TRAIN_ICON_ADMIN_TEMPLATE,
    TASK_TYPE_BEAST: f"{ASSET_SUBDIR}/beast_icon_admin.png",
    TASK_TYPE_SPEEDUP: f"{ASSET_SUBDIR}/speedup_icon_admin.png",
    TASK_TYPE_TECH: f"{ASSET_SUBDIR}/tech_icon_admin.png",
    TASK_TYPE_ORANGE_SHARD: f"{ASSET_SUBDIR}/orange_shard_icon_admin.png",
    TASK_TYPE_GEM: f"{ASSET_SUBDIR}/gem_icon_admin.png",
    TASK_TYPE_FIRE_CRYSTAL: f"{ASSET_SUBDIR}/fire_crystal_icon_admin.png",
    TASK_TYPE_SMALL_MONSTER: f"{ASSET_SUBDIR}/small_monster_icon_admin.png",
    TASK_TYPE_GIFT_PACK: f"{ASSET_SUBDIR}/gift_pack_icon_admin.png",
    TASK_TYPE_BUILDING: f"{ASSET_SUBDIR}/building_icon_admin.png",
    TASK_TYPE_DIAMOND: f"{ASSET_SUBDIR}/diamond_icon_admin.png",
    TASK_TYPE_ENERGY_STONE: f"{ASSET_SUBDIR}/energy_stone_icon_admin.png",
    TASK_TYPE_RESOURCE: f"{ASSET_SUBDIR}/resource_icon_admin.png",
}
# 详情弹窗优先用 *_detail / *_tight（从弹窗裁的模板底色更接近）
TASK_TYPE_ADMIN_DETAIL_TEMPLATES: dict[str, str] = {
    TASK_TYPE_TRAIN: f"{ASSET_SUBDIR}/train_icon_admin_detail.png",
}

DEFAULT_SLOTS: list[dict] = [
    {
        "name": "左卡",
        "icon_roi": [152, 696, 250, 788],
        "score_roi": [146, 832, 296, 884],
    },
    {
        "name": "中卡",
        "icon_roi": [468, 700, 572, 792],
        "score_roi": [462, 832, 612, 884],
    },
]

DEFAULT_COORDS: dict[str, list[int]] = {
    "refresh_tap": [196, 824],
    "refresh_confirm": [512, 776],
}

DEFAULT_SCAN_INTERVAL = 6 * 60
DEFAULT_SCORE_THRESHOLD = 500
DEFAULT_STEP_DELAY = 1.0
# 仅匹配练兵模板；装备等其它类型靠「未命中」来刷新，不做装备排除（易与练兵混淆）
DEFAULT_MATCH_THRESHOLD = 0.6
DEFAULT_COUNTDOWN_THRESHOLD = 0.62
DEFAULT_TARGET_TYPES: list[str] = [TASK_TYPE_TRAIN]
# 实机练兵图标比模板 PNG 大约大 35%~45%，仅搜到 1.2 会把真练兵压在 ~0.55
TRAIN_MATCH_SCALES: tuple[float, ...] = tuple(round(i / 100, 2) for i in range(85, 156, 5))


@dataclass(frozen=True)
class SlotConfig:
    name: str
    icon_roi: tuple[int, int, int, int]
    score_roi: tuple[int, int, int, int]


def available_task_types() -> list[tuple[str, str]]:
    return [(key, TASK_TYPE_LABELS[key]) for key in ADMIN_TARGET_TYPES]


def merge_task_config(cfg: dict | None) -> dict:
    raw = cfg or {}
    selected = [
        str(item).strip()
        for item in (raw.get("target_types") or DEFAULT_TARGET_TYPES)
        if str(item).strip() in ADMIN_TARGET_TYPES
    ]
    if not selected:
        selected = list(DEFAULT_TARGET_TYPES)

    slots: list[SlotConfig] = []
    for index, default in enumerate(DEFAULT_SLOTS):
        overrides = raw.get("slots") or []
        item = (
            overrides[index]
            if index < len(overrides) and isinstance(overrides[index], dict)
            else {}
        )
        icon = item.get("icon_roi", default["icon_roi"])
        score = item.get("score_roi", default["score_roi"])
        slots.append(
            SlotConfig(
                name=str(item.get("name", default["name"])),
                icon_roi=_as_roi(icon, default["icon_roi"]),
                score_roi=_as_roi(score, default["score_roi"]),
            )
        )

    coords = {**DEFAULT_COORDS, **(raw.get("coords") or {})}
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
        "slots": slots,
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


def _crop(screen: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    h, w = screen.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return screen[y1:y2, x1:x2]


def _roi_center(roi: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = roi
    return (x1 + x2) // 2, (y1 + y2) // 2


def _parse_score(text: str) -> int | None:
    """从 OCR 文本解析积分。优先 +N，否则取最大 2–4 位整数（避免把 520 读成 52）。"""
    cleaned = (
        (text or "")
        .replace(" ", "")
        .replace("＋", "+")
        .replace("，", ",")
        .replace("O", "0")
        .replace("o", "0")
        .replace("〇", "0")
        .replace("l", "1")
        .replace("I", "1")
    )
    plus_hits = [int(m) for m in re.findall(r"\+(\d{2,4})", cleaned)]
    if plus_hits:
        return max(plus_hits)
    digits = [int(m) for m in re.findall(r"\d{2,4}", cleaned)]
    if not digits:
        return None
    return max(digits)



def _looks_like_countdown_text(text: str) -> bool:
    return bool(re.search(r"\d{1,2}:\d{2}:\d{2}", text or ""))


def _prepare_score_patch(patch: np.ndarray, *, crop_left_ratio: float = 0.30) -> np.ndarray:
    """裁掉左侧积分币，放大后做对比度增强，便于 OCR。"""
    if patch.size == 0:
        return patch
    h, w = patch.shape[:2]
    x0 = int(w * crop_left_ratio)
    text_area = patch[:, x0:] if 0 < x0 < w else patch
    if text_area.size == 0:
        text_area = patch
    scaled = cv2.resize(text_area, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    # 黑字浅底：拉高对比
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)



class AllianceMobilizationSession:
    """联盟总动员：按目标任务类型刷分，达标后自动结束。"""

    def __init__(
        self,
        adb: AdbClient,
        *,
        target_types: list[str] | None = None,
        score_threshold: int = DEFAULT_SCORE_THRESHOLD,
        scan_interval: float = DEFAULT_SCAN_INTERVAL,
        step_delay: float = DEFAULT_STEP_DELAY,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        countdown_threshold: float = DEFAULT_COUNTDOWN_THRESHOLD,
        ocr_engine: str = "auto",
        slots: list[SlotConfig] | None = None,
        coords: dict[str, list[int]] | None = None,
        on_status: StatusCallback | None = None,
    ):
        merged = merge_task_config(
            {
                "target_types": target_types or DEFAULT_TARGET_TYPES,
                "score_threshold": score_threshold,
                "scan_interval": scan_interval,
                "step_delay": step_delay,
                "match_threshold": match_threshold,
                "countdown_threshold": countdown_threshold,
                "ocr_engine": ocr_engine,
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
        self.slots = list(slots or merged["slots"])
        self.coords = merged["coords"]
        self.on_status = on_status
        self._stop_event = threading.Event()
        self.vision = Vision(TEMPLATE_DIR, threshold=self.match_threshold)

    @property
    def name(self) -> str:
        return "联盟总动员"

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

    def _match_in_roi(
        self,
        screen: np.ndarray,
        roi: tuple[int, int, int, int],
        template_name: str,
        threshold: float,
    ) -> float:
        patch = _crop(screen, roi)
        if patch.size == 0:
            return 0.0
        old = self.vision.threshold
        self.vision.threshold = threshold
        try:
            result = self.vision.match_template_multiscale(
                patch,
                template_name,
                scales=TRAIN_MATCH_SCALES,
            )
        finally:
            self.vision.threshold = old
        return float(result.confidence)

    def _is_countdown(self, screen: np.ndarray, slot: SlotConfig) -> bool:
        icon = _crop(screen, slot.icon_roi)
        conf = self._match_in_roi(
            screen, slot.icon_roi, COUNTDOWN_TEMPLATE, self.countdown_threshold
        )
        if conf >= self.countdown_threshold:
            return True
        text, _ = ocr_chip_text(icon, engine=self.ocr_engine)
        if _looks_like_countdown_text(text):
            return True
        # 灰底仅作辅助：必须 OCR 到倒计时文本才认定冷却，避免把正常任务卡当冷却
        return False

    def _match_target_type(self, screen: np.ndarray, slot: SlotConfig) -> str | None:
        best_type = None
        best_conf = 0.0
        raw_conf = 0.0
        for task_type in self.target_types:
            template = TASK_TYPE_TEMPLATES.get(task_type)
            if not template:
                continue
            conf = self._match_in_roi(
                screen, slot.icon_roi, template, self.match_threshold
            )
            if task_type == TASK_TYPE_TRAIN:
                raw_conf = conf
            logger.debug(
                f"[{self.name}] {slot.name} 类型匹配 {task_type} conf={conf:.3f}"
            )
            if conf >= self.match_threshold and conf > best_conf:
                best_type = task_type
                best_conf = conf

        if best_type is None and TASK_TYPE_TRAIN in self.target_types:
            logger.info(
                f"[{self.name}] {slot.name} 练兵模板 conf={raw_conf:.2f} "
                f"< {self.match_threshold:.2f}，视为非练兵"
            )
        elif best_type == TASK_TYPE_TRAIN:
            logger.info(
                f"[{self.name}] {slot.name} 识别为练兵（conf={best_conf:.2f}）"
            )

        return best_type

    def _read_score(self, screen: np.ndarray, slot: SlotConfig) -> int | None:
        patch = _crop(screen, slot.score_roi)
        if patch.size == 0:
            return None
        best_score = None
        best_text = ""
        best_engine = "unknown"
        # 多种裁剪/原图尝试，取最大读数（避免把 520 读成 52）
        variants = [
            _prepare_score_patch(patch, crop_left_ratio=0.30),
            _prepare_score_patch(patch, crop_left_ratio=0.20),
            _prepare_score_patch(patch, crop_left_ratio=0.0),
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
            f"[{self.name}] {slot.name} 分数 OCR({best_engine})={best_text!r} -> {best_score}"
        )
        return best_score

    def _refresh_slot(self, slot: SlotConfig) -> None:
        cx, cy = _roi_center(slot.icon_roi)
        rx, ry = self.coords["refresh_tap"]
        confirm_x, confirm_y = self.coords["refresh_confirm"]
        self._emit(f"{slot.name}：刷新任务，点击区域 ({cx},{cy})")
        self._tap_xy(cx, cy, delay=self.step_delay)
        self._emit(f"{slot.name}：点击刷新 ({rx},{ry})")
        self._tap_xy(rx, ry, delay=self.step_delay)
        self._emit(f"{slot.name}：确认刷新 ({confirm_x},{confirm_y})")
        self._tap_xy(confirm_x, confirm_y, delay=self.step_delay)

    def _scan_once(self) -> bool:
        """扫描一轮。返回 True 表示目标已全部达标，应停止任务。"""
        if not ocr_engine_available(self.ocr_engine):
            raise RuntimeError(
                "OCR 不可用。请安装 rapidocr-onnxruntime onnxruntime，"
                "或确保本机已安装 Tesseract"
            )

        screen = self.adb.screenshot()
        keep_count = 0
        actionable_count = 0
        refreshed = 0

        labels = "、".join(TASK_TYPE_LABELS[t] for t in self.target_types)
        self._emit(f"开始扫描（目标：{labels}，分数 ≥ {self.score_threshold}）")

        for slot in self.slots:
            if self._interrupted():
                raise InterruptedError("任务已停止")

            if self._is_countdown(screen, slot):
                self._emit(f"{slot.name}：冷却倒计时中，跳过")
                continue

            actionable_count += 1
            task_type = self._match_target_type(screen, slot)
            if task_type is None:
                self._emit(f"{slot.name}：非目标任务，执行刷新")
                self._refresh_slot(slot)
                refreshed += 1
                screen = self.adb.screenshot()
                continue

            score = self._read_score(screen, slot)
            type_label = TASK_TYPE_LABELS.get(task_type, task_type)
            if score is None:
                # 已识别为目标类型时，分数读失败不得刷新，避免误删高分卡
                self._emit(
                    f"{slot.name}：识别到{type_label}，分数暂未识别，保留待下轮重试"
                )
                continue

            if score >= self.score_threshold:
                keep_count += 1
                self._emit(
                    f"{slot.name}：{type_label} 分数 {score} ≥ {self.score_threshold}，保留"
                )
                continue

            self._emit(
                f"{slot.name}：{type_label} 分数 {score} < {self.score_threshold}，刷新"
            )
            self._refresh_slot(slot)
            refreshed += 1
            screen = self.adb.screenshot()

        if actionable_count == 0:
            self._emit("当前可见卡均为冷却，等待下一轮")
            return False

        if keep_count >= actionable_count and refreshed == 0:
            self._emit(
                f"目标已达成：{keep_count}/{actionable_count} 张卡均为达标任务，停止"
            )
            return True

        self._emit(
            f"本轮保留 {keep_count} 张，刷新 {refreshed} 张，"
            f"{int(self.scan_interval // 60)} 分钟后再次扫描"
        )
        return False

    def run_until_stopped(self) -> None:
        self.reset_stop()
        try:
            while not self._interrupted():
                done = self._scan_once()
                if done:
                    return
                self._sleep_interruptible(self.scan_interval)
        except InterruptedError:
            self._emit("任务已结束")
            return
