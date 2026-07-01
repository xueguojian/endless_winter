"""寻梦记忆地图配置（物品名 → 点击坐标）。"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

from core.dream_memory.config import MAPS_DIR


def _parse_coord(value) -> list[int] | None:
    """解析 YAML 坐标或误写入 aliases 的坐标字符串。"""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return None
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                try:
                    return [int(parsed[0]), int(parsed[1])]
                except (TypeError, ValueError):
                    return None
    return None


def _sanitize_aliases(
    aliases: dict[str, str],
    items: dict[str, list[int]],
) -> dict[str, str]:
    """仅保留「误识别名 → 已有物品名」别名，丢弃坐标等脏数据。"""
    clean: dict[str, str] = {}
    for alias, canonical in aliases.items():
        key = str(alias).strip()
        target = str(canonical).strip()
        if not key or not target:
            continue
        if _parse_coord(canonical) is not None:
            continue
        if target not in items:
            continue
        if key in items:
            continue
        clean[key] = target
    return clean


def _load_aliases_raw(
    aliases_raw: dict,
    items: dict[str, list[int]],
) -> dict[str, str]:
    """加载 aliases；若误把坐标写进 aliases，自动恢复到 items。"""
    aliases: dict[str, str] = {}
    if not isinstance(aliases_raw, dict):
        return aliases

    for alias, canonical in aliases_raw.items():
        key = str(alias).strip()
        if not key:
            continue

        coord = _parse_coord(canonical)
        if coord is not None:
            if key not in items:
                items[key] = coord
                logger.warning(
                    f"地图 aliases 中误存坐标，已恢复到 items: {key!r} -> {coord}"
                )
            continue

        target = str(canonical).strip()
        if target:
            aliases[key] = target

    return _sanitize_aliases(aliases, items)


@dataclass
class DreamMemoryMap:
    map_id: str
    name: str
    items: dict[str, list[int]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    source_path: Path | None = None

    def resolve_label(self, label: str) -> str:
        from core.dream_memory.vision import normalize_item_name

        key = normalize_item_name(label)
        if not key:
            return ""
        if key in self.items:
            return key
        canonical = self.aliases.get(key)
        if canonical and canonical in self.items:
            return canonical
        return key

    def lookup(self, label: str) -> tuple[int, int] | None:
        from core.dream_memory.vision import normalize_item_name

        key = normalize_item_name(label)
        if not key:
            return None

        direct = self.resolve_label(key)
        if direct in self.items:
            coord = self.items[direct]
            if len(coord) >= 2:
                return int(coord[0]), int(coord[1])

        for name, coord in self.items.items():
            norm = normalize_item_name(name)
            if norm == key:
                if len(coord) >= 2:
                    return int(coord[0]), int(coord[1])
            # 仅当 OCR 更长时允许前缀扩展（如 单筒望远镜 包含 望远镜）
            elif len(key) > len(norm) and norm in key:
                if len(coord) >= 2:
                    return int(coord[0]), int(coord[1])
        return None


def list_maps(maps_dir: Path | None = None) -> list[DreamMemoryMap]:
    root = maps_dir or MAPS_DIR
    if not root.is_dir():
        return []
    maps: list[DreamMemoryMap] = []
    for path in sorted(root.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        try:
            maps.append(load_map(path))
        except (OSError, yaml.YAMLError, ValueError):
            continue
    maps.sort(key=lambda m: m.name)
    return maps


def load_map(map_id_or_path: str | Path, *, maps_dir: Path | None = None) -> DreamMemoryMap:
    root = maps_dir or MAPS_DIR
    path = Path(map_id_or_path)
    if path.is_file():
        target = path
    else:
        map_id = str(map_id_or_path).removesuffix(".yaml")
        target = root / f"{map_id}.yaml"
        if not target.is_file():
            raise FileNotFoundError(f"地图不存在: {target}")

    with target.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    map_id = str(raw.get("id") or target.stem)
    name = str(raw.get("name") or map_id)
    items_raw = raw.get("items") or {}
    items: dict[str, list[int]] = {}
    for label, coord in items_raw.items():
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            items[str(label)] = [int(coord[0]), int(coord[1])]

    aliases_raw = raw.get("aliases") or {}
    aliases = _load_aliases_raw(aliases_raw, items)

    return DreamMemoryMap(
        map_id=map_id,
        name=name,
        items=items,
        aliases=aliases,
        source_path=target,
    )


def format_map_choice(dream_map: DreamMemoryMap) -> str:
    """下拉框展示文案：有独立显示名时「名称 (id)」，否则仅 id。"""
    if dream_map.name and dream_map.name != dream_map.map_id:
        return f"{dream_map.name} ({dream_map.map_id})"
    return dream_map.map_id


def save_map(
    dream_map: DreamMemoryMap,
    *,
    maps_dir: Path | None = None,
) -> Path:
    root = maps_dir or MAPS_DIR
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{dream_map.map_id}.yaml"
    dream_map.aliases = _sanitize_aliases(dream_map.aliases, dream_map.items)
    payload = {
        "id": dream_map.map_id,
        "name": dream_map.name,
        "items": dict(dream_map.items),
    }
    if dream_map.aliases:
        payload["aliases"] = dict(sorted(dream_map.aliases.items()))
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False, width=88)
    dream_map.source_path = out
    return out


def rename_map_name(
    map_id: str,
    new_name: str,
    *,
    maps_dir: Path | None = None,
) -> Path:
    """修改地图显示名称（YAML 内 name 字段，不改 id/文件名）。"""
    name = new_name.strip()
    if not name:
        raise ValueError("地图名称不能为空")
    dream_map = load_map(map_id, maps_dir=maps_dir)
    dream_map.name = name
    return save_map(dream_map, maps_dir=maps_dir)


def delete_map(map_id: str, *, maps_dir: Path | None = None) -> None:
    """删除整张地图配置文件。"""
    dream_map = load_map(map_id, maps_dir=maps_dir)
    path = dream_map.source_path
    if path is None or not path.is_file():
        root = maps_dir or MAPS_DIR
        path = root / f"{map_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"地图不存在: {map_id}")
    path.unlink()
