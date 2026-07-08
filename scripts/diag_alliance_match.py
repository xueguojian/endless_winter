"""诊断联盟总动员练兵模板匹配置信度。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.adb_client import AdbClient
from core.vision import Vision
from tasks.alliance_mobilization import DEFAULT_SLOTS, TRAIN_ICON_TEMPLATE, TEMPLATE_DIR
import yaml

SCALES = (0.75, 0.85, 0.95, 1.0, 1.1, 1.2)
OUT = ROOT / "assets" / "debug"


def crop(screen: np.ndarray, roi: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    return screen[y1:y2, x1:x2].copy()


def match_detail(patch: np.ndarray, template_path: Path) -> dict:
    tpl_bgr = cv2.imread(str(template_path))
    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    vision = Vision(TEMPLATE_DIR, threshold=0.0)
    ms = vision.match_template_multiscale(
        patch, TRAIN_ICON_TEMPLATE, scales=SCALES
    )

    per_scale: list[tuple[float, float]] = []
    best_gray = (0.0, 1.0)
    best_color = (0.0, 1.0)
    for scale in SCALES:
        if scale == 1.0:
            t_gray = tpl_gray
            t_bgr = tpl_bgr
        else:
            t_gray = cv2.resize(
                tpl_gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
            t_bgr = cv2.resize(
                tpl_bgr, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
        th, tw = t_gray.shape[:2]
        if th > patch_gray.shape[0] or tw > patch_gray.shape[1]:
            continue
        g = float(cv2.matchTemplate(patch_gray, t_gray, cv2.TM_CCOEFF_NORMED).max())
        c = float(cv2.matchTemplate(patch, t_bgr, cv2.TM_CCOEFF_NORMED).max())
        per_scale.append((scale, max(g, c)))
        if g > best_gray[0]:
            best_gray = (g, scale)
        if c > best_color[0]:
            best_color = (c, scale)

    # 仅中心图标（去掉橙底边框），看背景是否拉低分数
    h, w = patch.shape[:2]
    inner = patch[int(h * 0.08): int(h * 0.82), int(w * 0.08): int(w * 0.82)]
    inner_ms = vision.match_template_multiscale(
        inner, TRAIN_ICON_TEMPLATE, scales=SCALES
    )

    # 模板 vs 自身缩放
    self_ms = vision.match_template_multiscale(
        tpl_bgr, TRAIN_ICON_TEMPLATE, scales=SCALES
    )

    return {
        "patch_wh": (patch.shape[1], patch.shape[0]),
        "template_wh": (tpl_bgr.shape[1], tpl_bgr.shape[0]),
        "multiscale_best": ms.confidence,
        "best_gray_scale": best_gray,
        "best_color_scale": best_color,
        "per_scale": per_scale,
        "inner_roi_best": inner_ms.confidence,
        "template_self": self_ms.confidence,
        "patch_mean_bgr": [float(x) for x in patch.mean(axis=(0, 1))],
        "tpl_mean_bgr": [float(x) for x in tpl_bgr.mean(axis=(0, 1))],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = ROOT / "config_5555.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        dev = yaml.safe_load(f).get("device", {})
    adb = AdbClient(
        host=dev.get("adb_host", "127.0.0.1"),
        port=int(dev.get("adb_port", 5555)),
        adb_path=str(dev.get("adb_path", "")),
        touch_width=int(dev.get("touch_width", 720)),
        touch_height=int(dev.get("touch_height", 1280)),
    )
    screen = adb.screenshot()
    cv2.imwrite(str(OUT / "alliance_screen.png"), screen)

    tpl_path = TEMPLATE_DIR / TRAIN_ICON_TEMPLATE
    print(f"设备: {adb.address}")
    print(f"截图: {screen.shape[1]}x{screen.shape[0]}")
    print(f"模板: {tpl_path} ({cv2.imread(str(tpl_path)).shape[1]}x{cv2.imread(str(tpl_path)).shape[0]})")
    print()

    for index, slot in enumerate(DEFAULT_SLOTS):
        name = slot["name"]
        roi = slot["icon_roi"]
        patch = crop(screen, roi)
        cv2.imwrite(str(OUT / f"alliance_slot{index}_icon_roi.png"), patch)
        d = match_detail(patch, tpl_path)
        print(f"=== {name} icon_roi={roi} patch={d['patch_wh']} ===")
        print(f"  多尺度最高 conf     : {d['multiscale_best']:.4f}")
        print(f"  灰度最佳            : {d['best_gray_scale'][0]:.4f} @ scale {d['best_gray_scale'][1]}")
        print(f"  彩色最佳            : {d['best_color_scale'][0]:.4f} @ scale {d['best_color_scale'][1]}")
        print(f"  裁内框后 conf       : {d['inner_roi_best']:.4f}")
        print(f"  模板自匹配 conf     : {d['template_self']:.4f}")
        print(f"  各 scale conf       : {', '.join(f'{s}={c:.3f}' for s, c in d['per_scale'])}")
        print(f"  patch 平均 BGR      : {d['patch_mean_bgr']}")
        print(f"  模板 平均 BGR       : {d['tpl_mean_bgr']}")
        print(f"  已保存              : assets/debug/alliance_slot{index}_icon_roi.png")
        print()


if __name__ == "__main__":
    main()
