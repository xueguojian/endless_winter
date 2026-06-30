"""验证灯塔图钉扫描：对比检测结果与期望坐标。

用法:
  .venv\\Scripts\\python.exe tools/verify_lighthouse_scan.py [截图路径]
  .venv\\Scripts\\python.exe tools/verify_lighthouse_scan.py --adb

期望坐标（720×1280 竖屏）可通过 --expect x,y 多次指定。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.adb_client import AdbClient
from core.lighthouse_vision import (
    configure_lighthouse_scan,
    scan_mission_icons,
    _normalize_screen_for_scan,
    _refine_to_mission_pin_head,
    LIGHTHOUSE_SCAN_ROI,
)

DEFAULT_EXPECT = (
    (622, 496),
    (466, 308),
    (426, 278),
    (540, 718),
)
MATCH_RADIUS = 36


def _load_screen(path: Path | None, use_adb: bool):
    if use_adb:
        adb = AdbClient()
        if not adb.wait_for_device(retries=3, interval=1.0):
            raise RuntimeError("ADB 设备未连接")
        return adb.screenshot()
    if path is None:
        raise ValueError("请提供截图路径或使用 --adb")
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


def _nearest_distance(center: tuple[int, int], expected: tuple[int, int]) -> float:
    return max(abs(center[0] - expected[0]), abs(center[1] - expected[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="验证灯塔图钉扫描")
    parser.add_argument("image", nargs="?", type=Path, help="截图路径")
    parser.add_argument("--adb", action="store_true", help="从 ADB 截屏")
    parser.add_argument(
        "--event",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="活动期间背景（默认开启）",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="X,Y",
        help="期望任务坐标，可重复",
    )
    args = parser.parse_args()

    expected: list[tuple[int, int]] = []
    for item in args.expect or DEFAULT_EXPECT:
        if isinstance(item, tuple):
            expected.append(item)
        else:
            x_str, y_str = str(item).split(",")
            expected.append((int(x_str.strip()), int(y_str.strip())))

    configure_lighthouse_scan(event_period=args.event)
    screen = _normalize_screen_for_scan(_load_screen(args.image, args.adb))
    result = scan_mission_icons(screen)

    print(f"背景: event_period={args.event}")
    print(f"检测: {len(result.missions)} 个（差分候选 {result.candidate_locations}）")
    for i, m in enumerate(
        sorted(result.missions, key=lambda item: (item.center[1], item.center[0])), 1
    ):
        print(f"  [{i}] {m.center} conf={m.confidence:.3f}")

    x1, y1, x2, y2 = LIGHTHOUSE_SCAN_ROI
    roi_hsv = cv2.cvtColor(screen[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    print("\n期望坐标匹配:")
    matched_detected: set[int] = set()
    for exp in expected:
        head = _refine_to_mission_pin_head(roi_hsv, exp, (x1, y1))
        best_d = min(
            (_nearest_distance(m.center, exp) for m in result.missions),
            default=999,
        )
        best_i = min(
            range(len(result.missions)),
            key=lambda i: _nearest_distance(result.missions[i].center, exp),
            default=-1,
        )
        ok = best_d <= MATCH_RADIUS
        if ok and best_i >= 0:
            matched_detected.add(best_i)
        head_note = f"pin_head={head[0]} score={head[1]:.2f}" if head else "pin_head=无"
        status = "OK" if ok else "MISS"
        print(f"  {status} 期望 {exp} 最近检测 dist={best_d:.0f}  {head_note}")

    extras = [
        m.center
        for i, m in enumerate(result.missions)
        if i not in matched_detected
    ]
    if extras:
        print(f"\n多余检测（可能是晶簇误报）: {extras}")

    out = ROOT / "assets" / "debug" / "lighthouse_scan_verify.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    vis = screen.copy()
    for exp in expected:
        cv2.drawMarker(vis, exp, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    for m in result.missions:
        cv2.circle(vis, m.center, 14, (0, 0, 255), 2)
    cv2.imwrite(str(out), vis)
    print(f"\n标注图: {out}")


if __name__ == "__main__":
    main()
