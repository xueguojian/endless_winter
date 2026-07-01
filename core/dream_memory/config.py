"""寻梦记忆配置（config.yaml dream_memory 段）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from core.config_path import ROOT, resolve_config_path

MAPS_DIR = ROOT / "assets" / "dream_memory" / "maps"
CHIP_REFS_DIR = ROOT / "assets" / "dream_memory" / "chip_refs"

# 720×1280 底栏三个目标按钮 ROI (x1, y1, x2, y2)
DEFAULT_TARGET_SLOTS: tuple[tuple[int, int, int, int], ...] = (
    (38, 1140, 248, 1194),
    (250, 1142, 462, 1196),
    (468, 1136, 674, 1200),
)

# 整栏 ROI，仅在不配置 target_slots 时用于自动均分（当前默认用固定三槽）
DEFAULT_TARGET_BAR: tuple[int, int, int, int] = (36, 1138, 688, 1194)

DEFAULT_TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
DEFAULT_OCR_ENGINE = "rapidocr"


@dataclass
class DreamMemoryConfig:
    tesseract_cmd: Path = field(default_factory=lambda: DEFAULT_TESSERACT_CMD)
    selected_map: str = ""
    tap_delay: float = 1.2
    tap_between_delay: float = 0.4
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

    def ensure_dirs(self) -> None:
        self.maps_dir.mkdir(parents=True, exist_ok=True)
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


def load_dream_memory_config(config_path: str | Path | None = None) -> DreamMemoryConfig:
    path = resolve_config_path(config_path)
    raw: dict = {}
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        raw = loaded.get("dream_memory") or {}

    tesseract_raw = raw.get("tesseract_cmd") or DEFAULT_TESSERACT_CMD
    cfg = DreamMemoryConfig(
        tesseract_cmd=Path(str(tesseract_raw)),
        selected_map=str(raw.get("selected_map") or ""),
        tap_delay=float(raw.get("tap_delay", 1.2)),
        tap_between_delay=float(raw.get("tap_between_delay", 0.4)),
        scan_interval=float(raw.get("scan_interval", 0.35)),
        chip_active_min_brightness=float(raw.get("chip_active_min_brightness", 95.0)),
        target_bar=_parse_bar(raw.get("target_bar")),
        max_target_slots=int(raw.get("max_target_slots", 4)),
        min_target_slots=int(raw.get("min_target_slots", 3)),
        target_slots=_parse_slots(raw.get("target_slots")) or DEFAULT_TARGET_SLOTS,
        chip_refs_dir=_parse_optional_path(raw.get("chip_refs_dir"), CHIP_REFS_DIR),
        chip_template_min_score=float(raw.get("chip_template_min_score", 0.88)),
        chip_template_min_margin=float(raw.get("chip_template_min_margin", 0.08)),
        chip_fuzzy_min_ratio=float(raw.get("chip_fuzzy_min_ratio", 0.72)),
        ocr_engine=str(raw.get("ocr_engine") or DEFAULT_OCR_ENGINE),
    )
    maps_dir = raw.get("maps_dir")
    if maps_dir:
        path_obj = Path(str(maps_dir))
        cfg.maps_dir = path_obj if path_obj.is_absolute() else ROOT / path_obj
    cfg.ensure_dirs()
    return cfg


def save_selected_map(map_id: str, config_path: str | Path | None = None) -> None:
    path = resolve_config_path(config_path)
    data: dict = {}
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    section = data.setdefault("dream_memory", {})
    section["selected_map"] = map_id
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
