"""从截图裁剪灯塔 alt 模板并生成边缘图。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.coords import crop_center
from core.lighthouse_vision import _normalize_screen_for_scan, _to_edges

OUT = ROOT / "assets" / "templates" / "lighthouse"

# 中心坐标, 裁剪尺寸 (w, h)
CROPS: tuple[tuple[str, tuple[int, int], tuple[int, int]], ...] = (
    ("hero_journey_orange", (118, 252), (56, 72)),
    ("tent_blue", (648, 442), (56, 72)),
    ("tent_blue_alt", (538, 598), (56, 72)),
    ("small_monster_blue", (398, 205), (56, 72)),
    ("small_monster_purple", (112, 322), (56, 72)),
    ("small_monster_purple_alt", (200, 452), (56, 72)),
    ("small_monster_orange", (102, 472), (56, 72)),
    ("small_monster_orange_alt", (362, 532), (56, 72)),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="从游戏截图生成灯塔模板变体")
    parser.add_argument(
        "screenshot",
        type=Path,
        help="720×1280 竖屏游戏截图路径",
    )
    args = parser.parse_args()

    img_path = args.screenshot if args.screenshot.is_absolute() else ROOT / args.screenshot
    screen = _normalize_screen_for_scan(cv2.imread(str(img_path)))
    if screen is None:
        raise SystemExit(f"无法读取: {img_path}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, center, size in CROPS:
        w, h = size
        crop = crop_center(screen, center[0], center[1], w, h)
        sym_path = OUT / f"{name}.png"
        edge_path = OUT / f"{name}_edges.png"
        cv2.imwrite(str(sym_path), crop)
        cv2.imwrite(str(edge_path), _to_edges(crop))
        print(f"saved {sym_path.name} center={center}")


if __name__ == "__main__":
    main()
