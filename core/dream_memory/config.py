"""寻梦记忆配置（config.yaml dream_memory / dream_memory_pk 段）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from core.config_path import ROOT, resolve_config_path

MAPS_DIR = ROOT / "assets" / "dream_memory" / "maps"
PREVIEWS_DIR = ROOT / "assets" / "dream_memory" / "previews"
PK_MAPS_DIR = ROOT / "assets" / "dream_memory" / "maps_pk"
PK_PREVIEWS_DIR = ROOT / "assets" / "dream_memory" / "previews_pk"
CHIP_REFS_DIR = ROOT / "assets" / "dream_memory" / "chip_refs"

# 720×1280 底栏三个目标按钮 ROI (x1, y1, x2, y2)
DEFAULT_TARGET_SLOTS: tuple[tuple[int, int, int, int], ...] = (
    (38, 1140, 248, 1194),
    (250, 1142, 462, 1196),
    (468, 1136, 674, 1200),
)

DEFAULT_TARGET_BAR: tuple[int, int, int, int] = (36, 1138, 688, 1194)
DEFAULT_PK_SLOT_COUNT = 6

# PK 六槽固定 ROI (x1, y1, x2, y2)，顺序 1→6；槽位自右向左逐格消失（6→5→4…）
DEFAULT_PK_TARGET_SLOTS: tuple[tuple[int, int, int, int], ...] = (
    (26, 1120, 236, 1178),
    (246, 1118, 472, 1180),
    (484, 1116, 698, 1182),
    (22, 1188, 226, 1256),
    (240, 1186, 464, 1256),
    (476, 1188, 688, 1254),
)

DEFAULT_PK_TARGET_BAR: tuple[int, int, int, int] = (22, 1116, 698, 1256)

DEFAULT_TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
DEFAULT_OCR_ENGINE = "rapidocr"

PK_ITEM_FILTER_ALL = "all"
PK_ITEM_FILTER_ODD = "odd"
PK_ITEM_FILTER_EVEN = "even"
PK_ITEM_FILTER_CHOICES = (PK_ITEM_FILTER_ALL, PK_ITEM_FILTER_ODD, PK_ITEM_FILTER_EVEN)
PK_ITEM_FILTER_LABELS: dict[str, str] = {
    PK_ITEM_FILTER_ALL: "全部",
    PK_ITEM_FILTER_ODD: "单数",
    PK_ITEM_FILTER_EVEN: "双数",
}


def normalize_pk_item_filter(raw: str | None) -> str:
    """PK 标定点分工：all=全部，odd=第1/3/5…个，even=第2/4/6…个（按 YAML 标定顺序）。"""
    if not raw:
        return PK_ITEM_FILTER_ALL
    text = str(raw).strip()
    for key, label in PK_ITEM_FILTER_LABELS.items():
        if text == key or text == label:
            return key
    lowered = text.lower()
    if lowered in PK_ITEM_FILTER_CHOICES:
        return lowered
    return PK_ITEM_FILTER_ALL


def pk_item_filter_matches(ordinal_1based: int, filter_mode: str) -> bool:
    """ordinal_1based 为标定顺序（从 1 开始）。"""
    mode = normalize_pk_item_filter(filter_mode)
    if mode == PK_ITEM_FILTER_ALL:
        return True
    if mode == PK_ITEM_FILTER_ODD:
        return ordinal_1based % 2 == 1
    if mode == PK_ITEM_FILTER_EVEN:
        return ordinal_1based % 2 == 0
    return True


def default_pk_target_slots() -> tuple[tuple[int, int, int, int], ...]:
    """PK 六槽固定坐标（可在 config dream_memory_pk.target_slots 覆盖）。"""
    return DEFAULT_PK_TARGET_SLOTS


@dataclass
class DreamMemoryConfig:
    tesseract_cmd: Path = field(default_factory=lambda: DEFAULT_TESSERACT_CMD)
    selected_map: str = ""
    tap_delay: float = 1.2
    tap_between_delay: float = 0.55
    tap_between_delay_min: float = 0.45
    tap_between_delay_max: float = 0.7
    tap_between_delay_mode: float = 0.55
    scan_interval: float = 0.35
    chip_active_min_brightness: float = 95.0
    target_bar: tuple[int, int, int, int] = DEFAULT_TARGET_BAR
    max_target_slots: int = 4
    min_target_slots: int = 3
    target_slots: tuple[tuple[int, int, int, int], ...] = DEFAULT_TARGET_SLOTS
    chip_refs_dir: Path = field(default_factory=lambda: CHIP_REFS_DIR)
    chip_template_min_score: float = 0.88
    chip_template_min_margin: float = 0.08
    chip_fuzzy_min_ratio: float = 0.72
    ocr_engine: str = DEFAULT_OCR_ENGINE
    maps_dir: Path = field(default_factory=lambda: MAPS_DIR)
    previews_dir: Path = field(default_factory=lambda: PREVIEWS_DIR)
    enable_misclick: bool = False
    misclick_interval_min: int = 8
    misclick_interval_max: int = 12
    misclick_center_x: int = 360
    misclick_center_y: int = 640
    misclick_radius_x: int = 90
    misclick_radius_y: int = 120
    pk_mode: bool = False
    bar_refresh_min_wait: float = 0.4
    bar_refresh_poll: float = 0.08
    bar_refresh_timeout: float = 2.5
    bar_change_mean_delta: float = 8.0
    pk_item_filter: str = PK_ITEM_FILTER_ALL

    def ensure_dirs(self) -> None:
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        self.chip_refs_dir.mkdir(parents=True, exist_ok=True)


def _parse_slots(raw: list | None) -> tuple[tuple[int, int, int, int], ...] | None:
    if not raw:
        return None
    slots: list[tuple[int, int, int, int]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            slots.append(tuple(int(v) for v in item))  # type: ignore[arg-type]
    return tuple(slots) if slots else None


def _parse_bar(raw: list | None) -> tuple[int, int, int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return tuple(int(v) for v in raw)  # type: ignore[return-value]
    return DEFAULT_TARGET_BAR


def _parse_optional_path(raw: str | None, default: Path) -> Path:
    if not raw:
        return default
    path_obj = Path(str(raw))
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _build_config(raw: dict, *, pk: bool) -> DreamMemoryConfig:
    tesseract_raw = raw.get("tesseract_cmd") or DEFAULT_TESSERACT_CMD
    if pk:
        default_slots = default_pk_target_slots()
        default_maps = PK_MAPS_DIR
        default_previews = PK_PREVIEWS_DIR
        timing = dict(
            tap_between_delay=0.1,
            tap_between_delay_min=0.1,
            tap_between_delay_max=0.1,
            tap_between_delay_mode=0.1,
            scan_interval=0.3,
            bar_refresh_min_wait=0.0,
            bar_refresh_poll=0.05,
            bar_refresh_timeout=0.0,
        )
        slot_defaults = dict(
            max_target_slots=6,
            min_target_slots=1,
            target_slots=default_slots,
        )
        misclick_default = False
    else:
        default_slots = DEFAULT_TARGET_SLOTS
        default_maps = MAPS_DIR
        default_previews = PREVIEWS_DIR
        timing = dict(
            tap_between_delay=0.55,
            tap_between_delay_min=0.45,
            tap_between_delay_max=0.7,
            tap_between_delay_mode=0.55,
            scan_interval=0.35,
            bar_refresh_min_wait=0.4,
            bar_refresh_poll=0.08,
            bar_refresh_timeout=2.5,
        )
        slot_defaults = dict(
            max_target_slots=4,
            min_target_slots=3,
            target_slots=DEFAULT_TARGET_SLOTS,
        )
        misclick_default = True

    fuzzy_default = 0.9 if pk else 0.72

    cfg = DreamMemoryConfig(
        tesseract_cmd=Path(str(tesseract_raw)),
        selected_map=str(raw.get("selected_map") or ""),
        tap_delay=float(raw.get("tap_delay", 1.2)),
        tap_between_delay=float(raw.get("tap_between_delay", timing["tap_between_delay"])),
        tap_between_delay_min=float(
            raw.get("tap_between_delay_min", timing["tap_between_delay_min"])
        ),
        tap_between_delay_max=float(
            raw.get("tap_between_delay_max", timing["tap_between_delay_max"])
        ),
        tap_between_delay_mode=float(
            raw.get(
                "tap_between_delay_mode",
                raw.get("tap_between_delay", timing["tap_between_delay_mode"]),
            )
        ),
        scan_interval=float(raw.get("scan_interval", timing["scan_interval"])),
        chip_active_min_brightness=float(raw.get("chip_active_min_brightness", 95.0)),
        target_bar=_parse_bar(raw.get("target_bar")),
        max_target_slots=int(raw.get("max_target_slots", slot_defaults["max_target_slots"])),
        min_target_slots=int(raw.get("min_target_slots", slot_defaults["min_target_slots"])),
        target_slots=_parse_slots(raw.get("target_slots")) or slot_defaults["target_slots"],
        chip_refs_dir=_parse_optional_path(raw.get("chip_refs_dir"), CHIP_REFS_DIR),
        chip_template_min_score=float(raw.get("chip_template_min_score", 0.88)),
        chip_template_min_margin=float(raw.get("chip_template_min_margin", 0.08)),
        chip_fuzzy_min_ratio=float(raw.get("chip_fuzzy_min_ratio", fuzzy_default)),
        ocr_engine=str(raw.get("ocr_engine") or DEFAULT_OCR_ENGINE),
        enable_misclick=bool(raw.get("enable_misclick", misclick_default)),
        misclick_interval_min=int(raw.get("misclick_interval_min", 8)),
        misclick_interval_max=int(raw.get("misclick_interval_max", 12)),
        misclick_center_x=int(raw.get("misclick_center_x", 360)),
        misclick_center_y=int(raw.get("misclick_center_y", 640)),
        misclick_radius_x=int(raw.get("misclick_radius_x", 90)),
        misclick_radius_y=int(raw.get("misclick_radius_y", 120)),
        pk_mode=pk,
        bar_refresh_min_wait=float(
            raw.get("bar_refresh_min_wait", timing["bar_refresh_min_wait"])
        ),
        bar_refresh_poll=float(raw.get("bar_refresh_poll", timing["bar_refresh_poll"])),
        bar_refresh_timeout=float(
            raw.get("bar_refresh_timeout", timing["bar_refresh_timeout"])
        ),
        bar_change_mean_delta=float(raw.get("bar_change_mean_delta", 8.0)),
        pk_item_filter=normalize_pk_item_filter(raw.get("pk_item_filter")),
    )
    cfg.maps_dir = _parse_optional_path(raw.get("maps_dir"), default_maps)
    cfg.previews_dir = _parse_optional_path(raw.get("previews_dir"), default_previews)
    cfg.ensure_dirs()
    return cfg


def _load_section(config_path: str | Path | None, section: str, *, pk: bool) -> DreamMemoryConfig:
    path = resolve_config_path(config_path)
    raw: dict = {}
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        raw = loaded.get(section) or {}
    return _build_config(raw, pk=pk)


def load_dream_memory_config(config_path: str | Path | None = None) -> DreamMemoryConfig:
    return _load_section(config_path, "dream_memory", pk=False)


def load_dream_memory_pk_config(config_path: str | Path | None = None) -> DreamMemoryConfig:
    return _load_section(config_path, "dream_memory_pk", pk=True)


def sample_tap_between_delay(cfg: DreamMemoryConfig) -> float:
    """同批内两次点击间隔：三角分布，众数约 mode，均值接近 mode。"""
    import random

    low = min(cfg.tap_between_delay_min, cfg.tap_between_delay_max)
    high = max(cfg.tap_between_delay_min, cfg.tap_between_delay_max)
    mode = min(max(cfg.tap_between_delay_mode, low), high)
    return random.triangular(low, high, mode)


def save_selected_map(
    map_id: str,
    config_path: str | Path | None = None,
    *,
    pk: bool = False,
) -> None:
    path = resolve_config_path(config_path)
    data: dict = {}
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    section_name = "dream_memory_pk" if pk else "dream_memory"
    section = data.setdefault(section_name, {})
    section["selected_map"] = map_id
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
