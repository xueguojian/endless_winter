"""底栏 OCR 结果解析：别名、易混组消歧、模板辅助。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from core.dream_memory.chip_match import fuzzy_match_map_key, match_chip_template
from core.dream_memory.chip_image import normalize_chip
from core.dream_memory.config import CHIP_REFS_DIR

# 地图 YAML aliases 之外的全局 OCR 纠错（仅当 canonical 在本图 items 中且 alias 不是独立物品）
DEFAULT_OCR_ALIASES: dict[str, str] = {
    "销": "锁",
    "琐": "锁",
    "所": "锁",
    "引": "弓",
}

# 同图并存时需视觉/多路 OCR 消歧（不能单靠 OCR 字面）
CONFUSABLE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"X", "弓"}),
)


def _script_kind(text: str) -> str:
    if not text:
        return ""
    if all(c.isascii() for c in text):
        return "ascii"
    if all("\u4e00" <= c <= "\u9fff" for c in text):
        return "cjk"
    return "mixed"


def apply_ocr_aliases(
    text: str,
    map_keys: tuple[str, ...] | list[str],
    map_aliases: dict[str, str] | None = None,
) -> str:
    key = (text or "").strip()
    if not key:
        return ""

    keys_set = set(map_keys)
    merged = dict(DEFAULT_OCR_ALIASES)
    if map_aliases:
        merged.update(map_aliases)

    if key in keys_set:
        return key

    canonical = merged.get(key)
    if canonical and canonical in keys_set:
        logger.debug(f"OCR 别名: {key!r} -> {canonical!r}")
        return canonical
    return key


def confusable_peers(label: str, map_keys: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    keys_set = set(map_keys)
    for group in CONFUSABLE_GROUPS:
        if label not in group:
            continue
        peers = tuple(k for k in group if k in keys_set)
        if len(peers) >= 2:
            return peers
    return ()


def _binary_as_bgr(chip_bgr: np.ndarray) -> np.ndarray:
    gray = normalize_chip(chip_bgr)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _ocr_vote_among(
    chip_bgr: np.ndarray,
    candidates: tuple[str, ...] | list[str],
) -> str | None:
    from core.dream_memory.ocr_rapid import ocr_chip_rapid

    cand_set = set(candidates)
    votes: Counter[str] = Counter()
    variants: list[tuple[np.ndarray, float]] = [
        (chip_bgr, 2.0),
        (chip_bgr, 3.0),
        (_binary_as_bgr(chip_bgr), 2.0),
        (_binary_as_bgr(chip_bgr), 3.0),
    ]
    for image, scale in variants:
        raw = ocr_chip_rapid(image, scale=scale)
        if raw in cand_set:
            votes[raw] += 1
            continue
        alias = apply_ocr_aliases(raw, candidates)
        if alias in cand_set:
            votes[alias] += 1
    if not votes:
        return None
    best, count = votes.most_common(1)[0]
    if count >= 2 or len(votes) == 1:
        return best
    return None


def disambiguate_confusable(
    chip_bgr: np.ndarray,
    ocr_text: str,
    map_keys: tuple[str, ...] | list[str],
    *,
    refs_dir: Path | None = None,
    template_min_score: float = 0.72,
    template_min_margin: float = 0.05,
) -> str | None:
    """在易混组（如 X / 弓）内用模板 + 多路 OCR 投票选出正确项。"""
    peers = confusable_peers(ocr_text, map_keys)
    if len(peers) < 2:
        return None

    root = refs_dir or CHIP_REFS_DIR
    matched = match_chip_template(
        chip_bgr,
        peers,
        refs_dir=root,
        min_score=template_min_score,
        min_margin=template_min_margin,
    )
    if matched:
        name, score = matched
        logger.info(f"易混消歧 template: OCR={ocr_text!r} -> {name!r} ({score:.2f})")
        return name

    voted = _ocr_vote_among(chip_bgr, peers)
    if voted and voted != ocr_text:
        logger.info(f"易混消歧 vote: OCR={ocr_text!r} -> {voted!r}")
        return voted
    return None


def resolve_chip_label(
    chip_bgr: np.ndarray,
    ocr_text: str,
    map_keys: tuple[str, ...] | list[str],
    *,
    map_aliases: dict[str, str] | None = None,
    refs_dir: Path | None = None,
    fuzzy_min_ratio: float = 0.72,
    template_min_score: float = 0.72,
    template_min_margin: float = 0.05,
) -> tuple[str, str]:
    """OCR 原始文本 → 地图物品名。"""
    keys_set = set(map_keys)
    raw = (ocr_text or "").strip()
    text = apply_ocr_aliases(raw, map_keys, map_aliases)

    peers = confusable_peers(text, map_keys) if text in keys_set else ()
    if len(peers) >= 2:
        picked = disambiguate_confusable(
            chip_bgr,
            text,
            map_keys,
            refs_dir=refs_dir,
            template_min_score=template_min_score,
            template_min_margin=template_min_margin,
        )
        if picked:
            return picked, f"disambig({raw!r}->{picked})"

    if text in keys_set:
        return text, f"rapidocr({raw!r})"

    fuzzy = fuzzy_match_map_key(text, map_keys, min_ratio=fuzzy_min_ratio)
    if fuzzy:
        name, score = fuzzy
        peer_group = confusable_peers(name, map_keys)
        if len(peer_group) >= 2:
            picked = disambiguate_confusable(
                chip_bgr,
                name,
                map_keys,
                refs_dir=refs_dir,
                template_min_score=template_min_score,
                template_min_margin=template_min_margin,
            )
            if picked:
                return picked, f"disambig_fuzzy({raw!r}->{picked})"
        tag = "rapidocr_fuzzy" if name != text else "rapidocr"
        return name, f"{tag}({raw!r}->{name}, {score:.2f})"

    if len(raw) == 1 and _script_kind(raw) == "cjk":
        voted = _ocr_vote_among(
            chip_bgr, tuple(k for k in map_keys if len(k) == 1 and _script_kind(k) == "cjk")
        )
        if voted:
            return voted, f"short_vote({raw!r}->{voted})"

    if text:
        return text, f"rapidocr({raw!r})"
    return "", ""
