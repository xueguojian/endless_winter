"""生成 level_num 等级数字模板。

用法：
1. 打开游戏，进入野外 → 放大镜 → 选中「冰原巨兽」tab
2. 运行: python scripts/calibrate_level_num.py
3. 脚本会自动点减号到 1 级，再逐级点加号，截取白框数字保存到
   assets/templates/level_num/1.png ~ N.png

若 +/- 无效，请手动把等级调到 1 后再运行，或从 debug_level_row.png 手动裁白框数字保存为 level_num/8.png。

LEVEL_NUM_ROI 应对准 +/- 行右侧白框内数字（约 598,1041,680,1078），只裁数字区域。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.adb_client import AdbClient
from tasks.hunt_ice_beast import HuntIceBeastTask, LEVEL_NUM_ROI

NUM_DIR = ROOT / "assets" / "templates" / "level_num"
MAX_LEVEL = 30


def _crop_digit_only(patch: np.ndarray) -> np.ndarray:
    """从白框 ROI 中只保留深色数字，去掉空白圆角区域。"""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return patch
    pad = 2
    y1, y2 = max(0, ys.min() - pad), min(patch.shape[0], ys.max() + 1 + pad)
    x1, x2 = max(0, xs.min() - pad), min(patch.shape[1], xs.max() + 1 + pad)
    return patch[y1:y2, x1:x2].copy()


def main() -> None:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dev = cfg["device"]
    hunt = cfg["tasks"]["hunt_ice_beast"]
    adb = AdbClient(
        host=dev["adb_host"],
        port=dev["adb_port"],
        adb_path=dev.get("adb_path", ""),
        touch_width=dev.get("touch_width", 720),
        touch_height=dev.get("touch_height", 1280),
    )

    if not adb.wait_for_device(retries=5, interval=1.0):
        print("无法连接模拟器")
        sys.exit(1)

    NUM_DIR.mkdir(parents=True, exist_ok=True)
    task = HuntIceBeastTask(adb=adb, coords=hunt["coords"], step_delay=0.4)
    task._open_search_panel()
    task._select_ice_beast_tab()

    x1, y1, x2, y2 = LEVEL_NUM_ROI
    minus = hunt["coords"]["level_minus"]
    plus = hunt["coords"]["level_plus"]

    print("降到 1 级…")
    for _ in range(MAX_LEVEL):
        adb.tap(*minus)
        time.sleep(0.25)

    prev = None
    saved = 0
    for lv in range(1, MAX_LEVEL + 1):
        sp = adb.screenshot()
        crop = _crop_digit_only(sp[y1:y2, x1:x2])
        path = NUM_DIR / f"{lv}.png"

        if prev is not None:
            diff = float(np.abs(crop.astype(float) - prev.astype(float)).mean())
            if diff < 0.5 and lv > 1:
                print(f"等级未变化，停止于 {lv - 1} 级（请检查 +/- 坐标）")
                break

        cv2.imwrite(str(path), crop)
        prev = crop.copy()
        saved += 1
        print(f"  已保存 level_num/{lv}.png")

        if lv < MAX_LEVEL:
            adb.tap(*plus)
            time.sleep(0.3)

    print(f"完成，共保存 {saved} 个等级模板 → {NUM_DIR}")


if __name__ == "__main__":
    main()
