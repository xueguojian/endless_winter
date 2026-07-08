"""标注联盟管理员刷新检测区域，并诊断当前屏卡片检出情况。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.adb_client import AdbClient
from core.config_path import ensure_config_file, resolve_config_path
from core.vision import Vision
from tasks.alliance_mobilization import (
    TRAIN_ICON_ADMIN_TEMPLATE,
    TRAIN_ICON_TEMPLATE,
    TEMPLATE_DIR,
    _parse_score,
    _prepare_score_patch,
)
from tasks.alliance_mobilization_admin import (
    ADMIN_TRAIN_MATCH_SCALES,
    _prepare_admin_icon_patch,
    detect_admin_cards,
    merge_task_config,
    _crop,
)
from core.dream_memory.ocr_engine import ocr_chip_text

OUT = ROOT / "assets" / "debug"
DEFAULT_CONFIG = ROOT / "config.yaml"


def draw_roi_overlay(
    screen: np.ndarray,
    cfg: dict,
    *,
    cards: list | None = None,
) -> np.ndarray:
    out = screen.copy()
    list_roi = tuple(cfg["list_roi"])
    exclude_top = int(cfg.get("exclude_top_px", 0))
    column_count = int(cfg.get("column_count", 3))
    x1, y1, x2, y2 = list_roi

    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        out,
        f"list_roi ({x1},{y1})-({x2},{y2})",
        (x1 + 4, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if exclude_top > 0:
        cv2.rectangle(out, (x1, y1), (x2, y1 + exclude_top), (0, 0, 255), 2)
        cv2.putText(
            out,
            f"exclude_top {exclude_top}px",
            (x1 + 6, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    elif y1 > 900:
        cv2.putText(
            out,
            "tip: set exclude_top_px if exclusive tasks overlap",
            (x1 + 6, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    content_y1 = y1 + exclude_top
    detect_y1 = content_y1 + (8 if exclude_top > 0 else 0)
    cv2.rectangle(out, (x1, detect_y1), (x2, y2 - 2), (0, 255, 0), 1)
    cv2.putText(
        out,
        "effective detect zone",
        (x1 + 6, detect_y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    col_w = max(1, (x2 - x1) // max(1, column_count))
    for col in range(1, column_count):
        cx = x1 + col * col_w
        cv2.line(out, (cx, y1), (cx, y2), (255, 200, 0), 1)

    scroll = cfg.get("scroll") or {}
    sx = int(scroll.get("swipe_x", 358))
    for key, color in [
        ("swipe_up_y1", (255, 128, 0)),
        ("swipe_up_y2", (255, 128, 0)),
        ("swipe_down_y1", (128, 128, 255)),
        ("swipe_down_y2", (128, 128, 255)),
    ]:
        if key in scroll:
            y = int(scroll[key])
            cv2.circle(out, (sx, y), 5, color, -1)

    vision = Vision(TEMPLATE_DIR, threshold=float(cfg.get("match_threshold", 0.55)))
    cards = cards if cards is not None else detect_admin_cards(
        screen,
        list_roi=list_roi,
        column_count=column_count,
        exclude_top_px=exclude_top,
        only_complete=True,
    )
    for card in cards:
        ix1, iy1, ix2, iy2 = card.icon_roi
        sx1, sy1, sx2, sy2 = card.score_roi
        cv2.rectangle(out, (ix1, iy1), (ix2, iy2), (255, 0, 255), 2)
        cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (0, 165, 255), 2)
        cx, cy = (ix1 + ix2) // 2, (iy1 + iy2) // 2
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
        patch = _prepare_admin_icon_patch(_crop(screen, card.icon_roi))
        admin_conf = 0.0
        legacy_conf = 0.0
        if patch.size:
            admin_conf = float(
                vision.match_template_multiscale(
                    patch, TRAIN_ICON_ADMIN_TEMPLATE, scales=ADMIN_TRAIN_MATCH_SCALES
                ).confidence
            )
            legacy_conf = float(
                vision.match_template_multiscale(
                    patch, TRAIN_ICON_TEMPLATE, scales=ADMIN_TRAIN_MATCH_SCALES
                ).confidence
            )
        label = f"c{card.column + 1} a={admin_conf:.2f} l={legacy_conf:.2f}"
        cv2.putText(
            out,
            label,
            (ix1, max(iy1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def load_admin_cfg(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return merge_task_config(raw.get("alliance_mobilization_admin", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="联盟管理员检测区域诊断")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--image", help="本地截图路径（默认 ADB 截图）")
    parser.add_argument(
        "--out",
        default=str(OUT / "admin_roi_overlay.png"),
        help="标注图输出路径",
    )
    args = parser.parse_args()

    cfg = load_admin_cfg(Path(args.config))
    OUT.mkdir(parents=True, exist_ok=True)

    if args.image:
        screen = cv2.imread(args.image)
        if screen is None:
            raise SystemExit(f"无法读取图片: {args.image}")
    else:
        adb = AdbClient()
        if not adb.wait_for_device(retries=3, interval=1.0):
            raise SystemExit("ADB 未连接，请用 --image 指定本地截图")
        screen = adb.screenshot()

    cards = detect_admin_cards(
        screen,
        list_roi=tuple(cfg["list_roi"]),
        column_count=int(cfg["column_count"]),
        exclude_top_px=int(cfg["exclude_top_px"]),
        only_complete=True,
    )
    partial = detect_admin_cards(
        screen,
        list_roi=tuple(cfg["list_roi"]),
        column_count=int(cfg["column_count"]),
        exclude_top_px=int(cfg["exclude_top_px"]),
        only_complete=False,
    )

    overlay = draw_roi_overlay(screen, cfg, cards=cards)
    out_path = Path(args.out)
    cv2.imwrite(str(out_path), overlay)

    roi = cfg["list_roi"]
    print("=== 联盟管理员检测区域 ===")
    print(f"list_roi (全区域):     ({roi[0]}, {roi[1]}) - ({roi[2]}, {roi[3]})")
    print(f"  宽×高: {roi[2] - roi[0]} × {roi[3] - roi[1]} px")
    print(f"exclude_top_px:        {cfg['exclude_top_px']} px")
    print(
        f"实际检测起点 Y:       {roi[1] + cfg['exclude_top_px']} "
        f"(专属任务区以下)"
    )
    print(f"完整卡片检出:         {len(cards)} 张")
    print(f"含不完整卡片:         {len(partial)} 张")
    print(f"标注图:               {out_path}")
    print()
    vision = Vision(TEMPLATE_DIR, threshold=0.0)
    threshold = float(cfg["match_threshold"])
    for card in cards:
        patch = _prepare_admin_icon_patch(_crop(screen, card.icon_roi))
        admin_conf = 0.0
        legacy_conf = 0.0
        if patch.size:
            admin_conf = float(
                vision.match_template_multiscale(
                    patch, TRAIN_ICON_ADMIN_TEMPLATE, scales=ADMIN_TRAIN_MATCH_SCALES
                ).confidence
            )
            legacy_conf = float(
                vision.match_template_multiscale(
                    patch, TRAIN_ICON_TEMPLATE, scales=ADMIN_TRAIN_MATCH_SCALES
                ).confidence
            )
        is_train = admin_conf >= threshold and admin_conf >= legacy_conf + 0.18
        score_patch = _crop(screen, card.score_roi)
        score = None
        if score_patch.size:
            for variant in [
                _prepare_score_patch(score_patch, crop_left_ratio=0.15),
                cv2.resize(score_patch, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
            ]:
                parsed = _parse_score(ocr_chip_text(variant)[0])
                if parsed is not None and (score is None or parsed > score):
                    score = parsed
        cx, cy = (card.icon_roi[0] + card.icon_roi[2]) // 2, (
            card.icon_roi[1] + card.icon_roi[3]
        ) // 2
        action = "保留/刷分(练兵)" if is_train else "刷新(非练兵)"
        print(
            f"  列{card.column + 1} row~{card.row_key} icon={card.icon_roi} "
            f"score={score} admin={admin_conf:.2f} legacy={legacy_conf:.2f} "
            f"-> {action} 点击({cx},{cy})"
        )


if __name__ == "__main__":
    main()
