"""详情弹窗练兵图标匹配诊断。"""

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
    ADMIN_TRAIN_MATCH_SCALES,
    DEFAULT_DETAIL_ICON_ROI,
    _prepare_admin_icon_patch,
)

IMG = ROOT / (
    "assets/c__Users_Administrator_AppData_Roaming_Cursor_User_workspaceStorage_"
    "45a0255e87c8b4478d065b7c43fa2c5b_images_image-c94d9c7b-4d27-4737-"
    "98c0-d8787f07a9d1.png"
)


def main() -> None:
    screen = cv2.imread(str(IMG))
    if screen is None:
        raise SystemExit(f"no image: {IMG}")
    print("screen", screen.shape)
    patch = _crop(screen, DEFAULT_DETAIL_ICON_ROI)
    print("patch", patch.shape, "roi", DEFAULT_DETAIL_ICON_ROI)

    vision = Vision(TEMPLATE_DIR, threshold=0.0)
    thr = 0.66

    def score(prep, label: str) -> None:
        admin = vision.match_template_multiscale(
            prep, TRAIN_ICON_ADMIN_TEMPLATE, scales=ADMIN_TRAIN_MATCH_SCALES
        )
        legacy = vision.match_template_multiscale(
            prep, TRAIN_ICON_TEMPLATE, scales=ADMIN_TRAIN_MATCH_SCALES
        )
        ok = admin.confidence >= thr and admin.confidence >= (
            legacy.confidence + ADMIN_TRAIN_LEGACY_MIN_DELTA
        )
        print(
            f"{label:24s} admin={admin.confidence:.3f} "
            f"legacy={legacy.confidence:.3f} train={ok}"
        )

    score(_prepare_admin_icon_patch(patch), "full+admin_patch")
    score(patch, "full raw")

    h = patch.shape[0]
    for frac in (0.65, 0.72, 0.78, 0.85):
        top = patch[: max(1, int(h * frac)), :]
        score(_prepare_admin_icon_patch(top), f"top{frac:.2f}+patch")
        score(top, f"top{frac:.2f} raw")


if __name__ == "__main__":
    main()
