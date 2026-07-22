"""标定「获取更多」弹窗中领主体力「使用」按钮坐标。

用法：在模拟器中手动打开体力不足弹窗，然后运行：
    .venv\\Scripts\\python.exe scripts\\calibrate_stamina.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.adb_client import AdbClient
from core.config_path import PRIMARY_CONFIG_PATH, ensure_config_file, resolve_config_path
from core.stamina_use import STAMINA_TITLE_ROI
from core.vision import Vision
from tasks.hunt_ice_beast import (
    STAMINA_GET_MORE_TITLE,
    STAMINA_TITLE_THRESHOLD,
    STAMINA_USE_BTN,
    STAMINA_USE_ROW_ROI,
    TEMPLATE_DIR,
)

OUT_DIR = ROOT / "assets" / "flow"


def main() -> None:
    config_path = ensure_config_file(resolve_config_path(PRIMARY_CONFIG_PATH))
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dev = cfg["device"]
    adb = AdbClient(
        host=dev["adb_host"],
        port=dev["adb_port"],
        adb_path=dev.get("adb_path", ""),
        touch_width=dev.get("touch_width", 720),
        touch_height=dev.get("touch_height", 1280),
    )
    if not adb.connect():
        print("无法连接模拟器")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    screen = adb.screenshot()
    cv2.imwrite(str(OUT_DIR / "stamina_calibrate.png"), screen)

    title_vision = Vision(TEMPLATE_DIR, threshold=STAMINA_TITLE_THRESHOLD)
    x1, y1, x2, y2 = STAMINA_TITLE_ROI
    title = title_vision.match_template(screen[y1:y2, x1:x2], STAMINA_GET_MORE_TITLE)
    print(f"弹窗标题匹配: found={title.found} conf={title.confidence:.3f}")

    ux1, uy1, ux2, uy2 = STAMINA_USE_ROW_ROI
    use_vision = Vision(TEMPLATE_DIR, threshold=0.58)
    use = use_vision.match_template_multiscale(screen[uy1:uy2, ux1:ux2], STAMINA_USE_BTN)
    if use.found:
        cx, cy = ux1 + use.center[0], uy1 + use.center[1]
        print(f"「使用」按钮匹配: ({cx}, {cy}) conf={use.confidence:.3f}")
    else:
        cx, cy = cfg["tasks"]["hunt_ice_beast"]["coords"].get("stamina_use", [621, 750])
        print(f"「使用」按钮未匹配，当前配置: ({cx}, {cy})")

    marked = screen.copy()
    cv2.circle(marked, (cx, cy), 10, (0, 0, 255), 2)
    cv2.imwrite(str(OUT_DIR / "stamina_calibrate_marked.png"), marked)
    print(f"已保存 assets/flow/stamina_calibrate_marked.png，红点为建议点击位置 ({cx}, {cy})")


if __name__ == "__main__":
    main()
