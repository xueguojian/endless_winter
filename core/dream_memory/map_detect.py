"""根据底栏目标物品反推当前寻梦地图。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from core.dream_memory.chip_match import fuzzy_match_map_key
from core.dream_memory.config import CURRENT_MAP_PERIOD, DreamMemoryConfig
from core.dream_memory.maps import DreamMemoryMap, list_maps
from core.dream_memory.ocr import clean_ocr_text, normalize_item_name_for_match
from core.dream_memory.vision import normalize_item_name, read_target_chips


@dataclass(frozen=True)
class MapDetectHit:
    dream_map: DreamMemoryMap
    matched: tuple[str, ...]
    score: int


@dataclass
class MapDetectResult:
    dream_map: DreamMemoryMap | None
    labels: tuple[str, ...] = ()
    candidates: tuple[MapDetectHit, ...] = ()
    reason: str = ""

    @property
    def unique(self) -> bool:
        return self.dream_map is not None and len(self.candidates) == 1


def _normalize_label(text: str) -> str:
    return normalize_item_name_for_match(clean_ocr_text(text or ""))


def _map_vocab(maps: list[DreamMemoryMap]) -> tuple[tuple[str, ...], dict[str, str]]:
    keys: list[str] = []
    aliases: dict[str, str] = {}
    for dream_map in maps:
        keys.extend(dream_map.items.keys())
        keys.extend(dream_map.aliases.keys())
        for alias, canonical in dream_map.aliases.items():
            aliases.setdefault(str(alias), str(canonical))
    # 保序去重
    uniq_keys = tuple(dict.fromkeys(str(k).strip() for k in keys if str(k).strip()))
    return uniq_keys, aliases


def _canonicalize_against_vocab(
    label: str,
    vocab_keys: tuple[str, ...],
    vocab_aliases: dict[str, str],
) -> str:
    """先把 OCR 结果归一到当期总词表（精确/别名/一次全局模糊）。"""
    key = _normalize_label(label) or normalize_item_name(label)
    if not key:
        return ""
    if key in vocab_aliases:
        target = str(vocab_aliases[key]).strip()
        if target in vocab_keys:
            return target
    if key in vocab_keys:
        return key
    fuzzy = fuzzy_match_map_key(key, vocab_keys, min_ratio=0.78)
    if fuzzy:
        name, _score = fuzzy
        return name
    return key


def _label_in_map(dream_map: DreamMemoryMap, label: str) -> str | None:
    """若物品属于该图，返回物品名。仅精确/别名，避免按图模糊误伤。"""
    key = _normalize_label(label) or normalize_item_name(label)
    if not key:
        return None
    if key in dream_map.items:
        return key
    canonical = dream_map.aliases.get(key)
    if canonical and canonical in dream_map.items:
        return canonical
    # resolve_label 可能返回未在 items 中的原文
    resolved = dream_map.resolve_label(key)
    if resolved in dream_map.items:
        return resolved
    return None


def match_maps_by_labels(
    labels: list[str] | tuple[str, ...],
    maps: list[DreamMemoryMap],
    *,
    min_hits: int | None = None,
    vocab_keys: tuple[str, ...] | None = None,
    vocab_aliases: dict[str, str] | None = None,
) -> list[MapDetectHit]:
    """用底栏物品反查地图；按命中数降序。"""
    if not maps:
        return []
    if vocab_keys is None or vocab_aliases is None:
        vocab_keys, vocab_aliases = _map_vocab(maps)

    cleaned: list[str] = []
    for raw in labels:
        canon = _canonicalize_against_vocab(raw, vocab_keys, vocab_aliases)
        if canon and canon not in cleaned:
            cleaned.append(canon)
    if not cleaned:
        return []

    required = min_hits
    if required is None:
        # 读到 ≥3 个：至少命中 2；读到 2 个：必须全中；1 个不够定图
        required = 2 if len(cleaned) >= 2 else 99

    hits: list[MapDetectHit] = []
    for dream_map in maps:
        matched: list[str] = []
        for label in cleaned:
            item = _label_in_map(dream_map, label)
            if item and item not in matched:
                matched.append(item)
        score = len(matched)
        if score >= required:
            hits.append(
                MapDetectHit(
                    dream_map=dream_map,
                    matched=tuple(matched),
                    score=score,
                )
            )

    hits.sort(
        key=lambda h: (
            h.score,
            # 同分优先物品更少的图（更特异）
            -len(h.dream_map.items),
            h.dream_map.name,
            h.dream_map.map_id,
        ),
        reverse=True,
    )
    return hits


def pick_best_map(hits: list[MapDetectHit]) -> DreamMemoryMap | None:
    if not hits:
        return None
    best = hits[0]
    # 最高分唯一才算锁定；同分多图视为歧义
    tops = [h for h in hits if h.score == best.score]
    if len(tops) != 1:
        return None
    return best.dream_map


def detect_map_from_labels(
    labels: list[str] | tuple[str, ...],
    maps: list[DreamMemoryMap],
    *,
    min_hits: int | None = None,
) -> MapDetectResult:
    vocab_keys, vocab_aliases = _map_vocab(maps)
    cleaned_list: list[str] = []
    for raw in labels:
        canon = _canonicalize_against_vocab(raw, vocab_keys, vocab_aliases)
        if canon and canon not in cleaned_list:
            cleaned_list.append(canon)
    cleaned = tuple(cleaned_list)

    hits = match_maps_by_labels(
        cleaned,
        maps,
        min_hits=min_hits,
        vocab_keys=vocab_keys,
        vocab_aliases=vocab_aliases,
    )
    if not cleaned:
        return MapDetectResult(None, labels=(), reason="底栏未识别到有效物品名")
    if not hits:
        return MapDetectResult(
            None,
            labels=cleaned,
            reason=f"当期地图无法匹配物品：{'、'.join(cleaned)}",
        )
    best = pick_best_map(hits)
    if best is None:
        names = " / ".join(f"{h.dream_map.name}({h.score})" for h in hits[:4])
        return MapDetectResult(
            None,
            labels=cleaned,
            candidates=tuple(hits),
            reason=f"多图歧义：{names}",
        )
    return MapDetectResult(
        best,
        labels=cleaned,
        candidates=tuple(hits),
        reason=f"匹配 {best.name}（命中 {hits[0].score}/{len(cleaned)}：{'、'.join(hits[0].matched)}）",
    )


def detect_map_from_screen(
    screen: np.ndarray,
    config: DreamMemoryConfig,
    *,
    maps: list[DreamMemoryMap] | None = None,
    period: int | None = None,
) -> MapDetectResult:
    """截图 OCR 底栏后反推地图。"""
    use_period = int(period if period is not None else config.selected_period or CURRENT_MAP_PERIOD)
    candidates = maps if maps is not None else list_maps(config.maps_dir, period=use_period)
    if not candidates:
        return MapDetectResult(None, reason=f"第 {use_period} 期没有可用地图")

    vocab_keys, vocab_aliases = _map_vocab(candidates)
    chips = read_target_chips(
        screen,
        config.target_slots,
        map_keys=vocab_keys,
        map_aliases=vocab_aliases,
        tesseract_cmd=config.tesseract_cmd,
        ocr_engine=config.ocr_engine,
        min_brightness=config.chip_active_min_brightness,
        refs_dir=config.chip_refs_dir,
        template_min_score=config.chip_template_min_score,
        template_min_margin=config.chip_template_min_margin,
        fuzzy_min_ratio=config.chip_fuzzy_min_ratio,
        pk_mode=False,
    )
    labels: list[str] = []
    for chip in chips:
        if not chip.active:
            continue
        text = (chip.text or chip.ocr_raw or "").strip()
        if text:
            labels.append(text)

    result = detect_map_from_labels(labels, candidates)
    logger.info(
        f"自动认图 period={use_period} labels={list(result.labels)} -> "
        f"{result.dream_map.map_id if result.dream_map else None} ({result.reason})"
    )
    return result
