"""纯本地：用 admin_screen_now.png 检测练兵，不连 ADB。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.vision import Vision
from tasks.alliance_mobilization import (
    TEMPLATE_DIR,
    TRAIN_ICON_ADMIN_TEMPLATE,
    TRAIN_ICON_TEMPLATE,
    _crop,
)
from tasks.alliance_mobilization_admin import (
    ADMIN_TRAIN_LEGACY_MIN_DELTA,
    DEFAULT_LIST_ROI,
    DEFAULT_MATCH_THRESHOLD,
    _prepare_admin_icon_patch,
    detect_admin_cards,
)

SCALES = tuple(round(i / 100, 2) for i in range(45, 121, 5))


def main() -> None:
    debug = ROOT / "assets" / "debug"
    screen = cv2.imread(str(debug / "admin_screen_now.png"))
    if screen is None:
        raise SystemExit("missing admin_screen_now.png")
    print("screen", screen.shape)

    tpl = cv2.imread(str(TEMPLATE_DIR / TRAIN_ICON_ADMIN_TEMPLATE))
    print("tpl", None if tpl is None else tpl.shape)

    list_roi = DEFAULT_LIST_ROI
    cards = detect_admin_cards(screen, list_roi=list_roi)
    print("cards", len(cards), "roi", list_roi)

    vision = Vision(TEMPLATE_DIR, threshold=0.0)
    annot = screen.copy()
    x1, y1, x2, y2 = list_roi
    cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 255, 255), 2)

    hits = []
    for card in cards:
        patch = _prepare_admin_icon_patch(_crop(screen, card.icon_roi))
        admin = legacy = 0.0
        if patch.size:
            admin = float(
                vision.match_template_multiscale(
                    patch, TRAIN_ICON_ADMIN_TEMPLATE, scales=SCALES
                ).confidence
            )
            legacy = float(
                vision.match_template_multiscale(
                    patch, TRAIN_ICON_TEMPLATE, scales=SCALES
                ).confidence
            )
        is_train = admin >= DEFAULT_MATCH_THRESHOLD and admin >= (
            legacy + ADMIN_TRAIN_LEGACY_MIN_DELTA
        )
        cx = (card.icon_roi[0] + card.icon_roi[2]) // 2
        cy = (card.icon_roi[1] + card.icon_roi[3]) // 2
        print(
            f"col{card.column+1} row{card.row_key} center=({cx},{cy}) "
            f"admin={admin:.3f} legacy={legacy:.3f} train={is_train}"
        )
        color = (0, 255, 0) if is_train else (180, 0, 255)
        cv2.rectangle(
            annot,
            (card.icon_roi[0], card.icon_roi[1]),
            (card.icon_roi[2], card.icon_roi[3]),
            color,
            2,
        )
        cv2.putText(
            annot,
            f"a={admin:.2f}",
            (card.icon_roi[0], max(12, card.icon_roi[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )
        if is_train:
            hits.append((cx, cy, admin))

    roi = screen[y1:y2, x1:x2]
    best = vision.match_template_multiscale(roi, TRAIN_ICON_ADMIN_TEMPLATE, scales=SCALES)
    bc = (best.center[0] + x1, best.center[1] + y1)
    print(f"whole_roi best={best.confidence:.3f} center={bc}")
    if best.confidence >= 0.45:
        cv2.circle(annot, bc, 24, (0, 0, 255), 2)

    out = debug / "admin_train_hits.png"
    cv2.imwrite(str(out), annot)
    print("hits", hits)
    print("saved", out)


if __name__ == "__main__":
    main()
