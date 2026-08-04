"""截取当前灯塔情报页，标注图钉并清理无用调试图。

用法:
  .venv\\Scripts\\python.exe tools/annotate_lighthouse_scan.py
  .venv\\Scripts\\python.exe tools/annotate_lighthouse_scan.py --reuse   # 复用已有 screen.png
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.adb_client import AdbClient
from core.lighthouse_vision import (
    LIGHTHOUSE_SCAN_ROI,
    SKIP_MISSION_KINDS,
    scan_mission_icons,
    tag_scanned_missions,
)

OUT_DIR = ROOT / "assets" / "debug" / "lighthouse_scan"


def cleanup_unused_images() -> int:
    removed = 0
    keep_dirs = {OUT_DIR}
    for folder in (ROOT / "assets" / "flow" / "shop_debug", ROOT / "assets" / "debug"):
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            if path.name == ".gitkeep":
                continue
            if any(path == d or d in path.parents for d in keep_dirs):
                continue
            path.unlink(missing_ok=True)
            removed += 1
    shop_debug = ROOT / "assets" / "flow" / "shop_debug"
    if shop_debug.exists() and not any(shop_debug.iterdir()):
        shop_debug.rmdir()
    inspect = ROOT / "assets" / "debug" / "_inspect"
    if inspect.exists():
        shutil.rmtree(inspect, ignore_errors=True)
    return removed


def _near_any(pt: tuple[int, int], others: list[tuple[int, int]], dist: int = 36) -> bool:
    x, y = pt
    return any(abs(x - ox) <= dist and abs(y - oy) <= dist for ox, oy in others)


def find_excluded_marks(
    screen: np.ndarray, known: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """用正式识别逻辑找应排除的大怪 / 小飞机（仅标注观察）。"""
    from core.lighthouse_vision import (  # noqa: WPS433
        _find_mission_pin_centers,
        _is_plane_point,
        _is_super_boss_point,
        _patch_looks_like_mission_pin,
        _extract_center_patch,
        _match_plane_template_rotated,
        PLANE_ROTATE_MATCH_MIN,
    )

    x1, y1, x2, y2 = LIGHTHOUSE_SCAN_ROI
    roi = screen[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    offset = (x1, y1)
    bosses: list[tuple[int, int]] = []
    planes: list[tuple[int, int]] = []

    candidates = list(_find_mission_pin_centers(roi, offset))
    # 再扫一遍鲜艳色块中心，覆盖模板漏掉的橙光兽头 / 飞机
    vivid = cv2.inRange(hsv, (8, 90, 120), (30, 255, 255))
    vivid = cv2.morphologyEx(vivid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    for contour in cv2.findContours(vivid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(contour)
        if area < 120 or area > 12000:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"]) + x1
        cy = int(moments["m01"] / moments["m00"]) + y1
        candidates.append((cx, cy))

    for pt in candidates:
        if _near_any(pt, known, 36):
            continue
        if _is_super_boss_point(pt, roi, hsv, offset):
            if not _near_any(pt, bosses, 40):
                bosses.append(pt)
            continue
        if not _is_plane_point(pt, roi=roi, roi_offset=offset):
            continue
        # 标注观察：只保留模板分较高的真飞机，避免基地/地面误标
        local = (pt[0] - x1, pt[1] - y1)
        patch = _extract_center_patch(roi, local)
        if patch.size == 0 or _patch_looks_like_mission_pin(patch):
            continue
        if _match_plane_template_rotated(patch) < PLANE_ROTATE_MATCH_MIN:
            continue
        if not _near_any(pt, planes, 40):
            planes.append(pt)
    return bosses[:1], planes[:1]


def annotate(
    screen: np.ndarray,
    missions,
    extras: list[tuple[str, tuple[int, int], tuple[int, int, int]]],
) -> np.ndarray:
    annotated = screen.copy()
    x1, y1, x2, y2 = LIGHTHOUSE_SCAN_ROI
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 80, 80), 1)
    for index, mission in enumerate(missions, start=1):
        cx, cy = mission.center
        skip = mission.kind in SKIP_MISSION_KINDS
        color = (0, 165, 255) if skip else (0, 255, 0)
        cv2.circle(annotated, (cx, cy), 18, color, 2)
        cv2.putText(
            annotated,
            str(index),
            (cx - 8, cy - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        kind = mission.kind or "pin"
        cv2.putText(
            annotated,
            f"{kind} {mission.confidence:.2f}",
            (cx + 12, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    for label, (cx, cy), color in extras:
        cv2.circle(annotated, (cx, cy), 22, color, 2)
        cv2.putText(
            annotated,
            label,
            (cx - 20, cy - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="复用已有 screen.png")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    removed = cleanup_unused_images()
    print(f"已删除无用调试图 {removed} 个")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    screen_path = OUT_DIR / "screen.png"
    if args.reuse and screen_path.exists():
        screen = cv2.imread(str(screen_path))
        if screen is None:
            raise SystemExit(f"无法读取 {screen_path}")
        print(f"复用截图 {screen.shape[1]}x{screen.shape[0]}")
    else:
        adb = AdbClient(port=args.port)
        screen = adb.screenshot()
        cv2.imwrite(str(screen_path), screen)
        print(f"截图 {screen.shape[1]}x{screen.shape[0]} -> {screen_path}")

    result = scan_mission_icons(screen)
    missions = (
        tag_scanned_missions(screen, result.missions) if result.missions else []
    )
    known = [m.center for m in missions]
    orange_specials, planes = find_excluded_marks(screen, known)

    print(
        f"识别到 {len(missions)} 个常规任务图钉（候选 {result.candidate_locations}）；"
        f"橙光特殊 {len(orange_specials)}；小飞机 {len(planes)}"
    )
    for index, mission in enumerate(missions, start=1):
        kind = mission.kind or "pin"
        print(
            f"  #{index} ({mission.center[0]},{mission.center[1]}) "
            f"conf={mission.confidence:.3f} kind={kind}"
        )
    for pt in orange_specials:
        print(f"  [橙光] {pt}")
    for pt in planes:
        print(f"  [飞机] {pt}")

    extras: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
    for i, pt in enumerate(orange_specials, start=1):
        extras.append((f"BOSS{i}", pt, (0, 140, 255)))
    for i, pt in enumerate(planes, start=1):
        extras.append((f"PLANE{i}", pt, (255, 80, 80)))

    annotated = annotate(screen, missions, extras)
    cv2.imwrite(str(OUT_DIR / "annotated.png"), annotated)

    lines = ["# index\tx,y\tconfidence\tkind"]
    for index, mission in enumerate(missions, start=1):
        kind = mission.kind or "pin"
        lines.append(
            f"{index}\t{mission.center[0]},{mission.center[1]}\t"
            f"{mission.confidence:.3f}\t{kind}"
        )
    for i, pt in enumerate(orange_specials, start=1):
        lines.append(f"BOSS{i}\t{pt[0]},{pt[1]}\t-\torange_special")
    for i, pt in enumerate(planes, start=1):
        lines.append(f"PLANE{i}\t{pt[0]},{pt[1]}\t-\tplane_excluded")
    (OUT_DIR / "missions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"标注结果已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
