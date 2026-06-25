"""从模拟器竖屏截图中裁剪模板（坐标与 config 里 touch 坐标一致）。

用法:
  .venv\\Scripts\\python.exe tools/capture_template.py march_btn 560 1200 120 60
  .venv\\Scripts\\python.exe tools/capture_template.py lighthouse/tent 265 630 64 76

参数: 模板名(可含子目录)  中心x  中心y  宽  高
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.adb_client import AdbClient
from core.coords import PORTRAIT_HEIGHT, PORTRAIT_WIDTH, crop_center


def main() -> None:
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)

    name = sys.argv[1]
    cx, cy, w, h = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dev = cfg["device"]
    adb = AdbClient(
        host=dev["adb_host"],
        port=dev["adb_port"],
        adb_path=dev.get("adb_path", ""),
        touch_width=dev.get("touch_width", PORTRAIT_WIDTH),
        touch_height=dev.get("touch_height", PORTRAIT_HEIGHT),
    )
    if not adb.wait_for_device(retries=5, interval=1.0):
        print("无法连接模拟器")
        sys.exit(1)

    screen = adb.screenshot()
    sh, sw = screen.shape[:2]
    print(f"截图: {sw}×{sh}  触摸坐标系: {adb.touch_width}×{adb.touch_height}")
    if (sw, sh) != (adb.touch_width, adb.touch_height):
        print("警告：截图与 touch 尺寸不一致，裁剪坐标可能偏移")

    crop = crop_center(screen, cx, cy, w, h)
    out = ROOT / "assets" / "templates" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), crop)
    x1, y1 = cx - w // 2, cy - h // 2
    print(f"已保存: {out}  ({w}×{h}，中心 {cx},{cy}，区域 [{x1},{y1},{x1+w},{y1+h}])")


if __name__ == "__main__":
    main()
