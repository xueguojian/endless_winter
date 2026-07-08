"""在管理员截图上检测练兵图标（list_roi 下方区域）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.adb_client import AdbClient
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
    DEFAULT_LIST_ROI,
    DEFAULT_MATCH_THRESHOLD,
    _prepare_admin_icon_patch,
    detect_admin_cards,
    merge_task_config,
)

# 增加更小尺度，适配裁剪后的模板
SCALES = tuple(round(i / 100, 2) for i in range(45, 121, 5))


def main() -> None:
    cfg_path = ROOT / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = merge_task_config(raw.get("alliance_mobilization_admin", {}))
    port = int((raw.get("device") or {}).get("adb_port", 5555))
    adb_path = str((raw.get("device") or {}).get("adb_path") or "")

    debug = ROOT / "assets" / "debug"
    debug.mkdir(parents=True, exist_ok=True)

    use_cache = "--cache" in sys.argv
    if use_cache:
        screen = cv2.imread(str(debug / "admin_screen_now.png"))
        print(f"CACHE screen={None if screen is None else screen.shape}")
        if screen is None:
            raise SystemExit("无可用截图")
    else:
        try:
            adb = AdbClient(port=port, adb_path=adb_path)
            adb.connect()
            screen = adb.screenshot()
            cv2.imwrite(str(debug / "admin_screen_now.png"), screen)
            print(f"LIVE screen={screen.shape} port={port}")
        except Exception as exc:
            screen = cv2.imread(str(debug / "admin_screen_now.png"))
            print(
                f"LIVE fail ({exc}); using cached admin_screen_now.png "
                f"shape={None if screen is None else screen.shape}"
            )
            if screen is None:
                raise SystemExit("无可用截图")

    list_roi = tuple(cfg["list_roi"])
    print(f"list_roi={list_roi}")
    print(f"exclude_top={cfg['exclude_top_px']} threshold={cfg['match_threshold']}")

    tpl = cv2.imread(str(TEMPLATE_DIR / TRAIN_ICON_ADMIN_TEMPLATE))
    print(f"train_icon_admin={None if tpl is None else tpl.shape}")

    cards = detect_admin_cards(
        screen,
        list_roi=list_roi,
        column_count=int(cfg["column_count"]),
        exclude_top_px=int(cfg["exclude_top_px"]),
        only_complete=True,
    )
    print(f"complete cards={len(cards)}")

    vision = Vision(TEMPLATE_DIR, threshold=0.0)
    annot = screen.copy()
    x1, y1, x2, y2 = list_roi
    cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 255, 255), 2)
    excl = int(cfg["exclude_top_px"])
    if excl > 0:
        cv2.rectangle(annot, (x1, y1), (x2, y1 + excl), (0, 0, 255), 1)

    hits = []
    for card in cards:
        patch = _prepare_admin_icon_patch(_crop(screen, card.icon_roi))
        admin = 0.0
        legacy = 0.0
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
        thr = float(cfg["match_threshold"])
        is_train = admin >= thr and admin >= legacy + ADMIN_TRAIN_LEGACY_MIN_DELTA
        cx = (card.icon_roi[0] + card.icon_roi[2]) // 2
        cy = (card.icon_roi[1] + card.icon_roi[3]) // 2
        print(
            f"  col{card.column + 1} row{card.row_key} "
            f"icon={card.icon_roi} center=({cx},{cy}) "
            f"admin={admin:.3f} legacy={legacy:.3f} train={is_train}"
        )
        color = (0, 255, 0) if is_train else (255, 0, 255)
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
            cv2.LINE_AA,
        )
        if is_train:
            hits.append((cx, cy, admin, card))

    # 整区多尺度（对照）
    roi = screen[y1:y2, x1:x2]
    best = vision.match_template_multiscale(
        roi, TRAIN_ICON_ADMIN_TEMPLATE, scales=SCALES
    )
    bc = (best.center[0] + x1, best.center[1] + y1)
    print(
        f"whole list_roi best admin conf={best.confidence:.3f} center={bc} "
        f"size={best.size}"
    )
    if best.confidence >= 0.5:
        cv2.circle(annot, bc, 22, (0, 0, 255), 2)
        cv2.putText(
            annot,
            f"best {best.confidence:.2f}",
            (bc[0] - 40, bc[1] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    out = debug / "admin_train_hits.png"
    cv2.imwrite(str(out), annot)
    print(f"train_hits={len(hits)}")
    for cx, cy, conf, card in hits:
        print(f"  TRAIN @ ({cx},{cy}) conf={conf:.3f} icon_roi={card.icon_roi}")
    print(f"saved {out}")
    print(f"scales used={SCALES[:3]}...{SCALES[-1]} (code default={ADMIN_TRAIN_MATCH_SCALES[0]}..{ADMIN_TRAIN_MATCH_SCALES[-1]})")
    print(f"code threshold={DEFAULT_MATCH_THRESHOLD} list_default={DEFAULT_LIST_ROI}")


if __name__ == "__main__":
    main()
