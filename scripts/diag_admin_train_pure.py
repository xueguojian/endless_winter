"""纯 OpenCV：不导入任务模块，避免 OCR 初始化卡住。"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "assets" / "debug" / "admin_screen_now.png"
TPL = ROOT / "assets" / "templates" / "alliance_mobilization" / "train_icon_admin.png"
OUT = ROOT / "assets" / "debug" / "admin_train_hits.png"
LIST_ROI = (44, 906, 714, 1264)
EXCLUDE_TOP = 72
SCALES = tuple(round(i / 100, 2) for i in range(40, 126, 5))


def match_multiscale(roi_gray, tpl_gray, scales):
    best = (-1.0, 1.0, (0, 0), (0, 0))
    for scale in scales:
        tw = max(8, int(tpl_gray.shape[1] * scale))
        th = max(8, int(tpl_gray.shape[0] * scale))
        if th >= roi_gray.shape[0] or tw >= roi_gray.shape[1]:
            continue
        scaled = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(roi_gray, scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best[0]:
            cx = max_loc[0] + tw // 2
            cy = max_loc[1] + th // 2
            best = (float(max_val), scale, (cx, cy), (tw, th))
    return best


def find_peaks(roi_gray, tpl_gray, scale, thr=0.60):
    tw = max(8, int(tpl_gray.shape[1] * scale))
    th = max(8, int(tpl_gray.shape[0] * scale))
    scaled = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(roi_gray, scaled, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= thr)
    peaks = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        v = float(res[y, x])
        cx = x + tw // 2
        cy = y + th // 2
        if any(abs(cx - px) < 50 and abs(cy - py) < 50 for px, py, _ in peaks):
            continue
        peaks.append((cx, cy, v))
    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks, (tw, th)


def main() -> None:
    print("start", flush=True)
    screen = cv2.imread(str(SCREEN))
    tpl = cv2.imread(str(TPL))
    print("screen", screen.shape, "tpl", tpl.shape, flush=True)

    x1, y1, x2, y2 = LIST_ROI
    detect_y1 = y1 + EXCLUDE_TOP
    roi = screen[detect_y1:y2, x1:x2]
    print("roi", roi.shape, "detect_y1", detect_y1, flush=True)

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

    conf, scale, center_roi, size = match_multiscale(roi_gray, tpl_gray, SCALES)
    center = (center_roi[0] + x1, center_roi[1] + detect_y1)
    print(f"BEST conf={conf:.3f} scale={scale} center={center} size={size}", flush=True)

    peaks, wh = find_peaks(roi_gray, tpl_gray, scale, thr=0.55)
    print(f"peaks>=0.55 count={len(peaks)} tpl_wh={wh}", flush=True)
    for cx, cy, v in peaks:
        print(f"  peak center=({cx + x1},{cy + detect_y1}) conf={v:.3f}", flush=True)

    peaks60, _ = find_peaks(roi_gray, tpl_gray, scale, thr=0.66)
    print(f"peaks>=0.66 count={len(peaks60)}", flush=True)
    for cx, cy, v in peaks60:
        print(f"  TRAIN center=({cx + x1},{cy + detect_y1}) conf={v:.3f}", flush=True)

    annot = screen.copy()
    cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.rectangle(annot, (x1, y1), (x2, detect_y1), (0, 0, 255), 1)
    cv2.circle(annot, center, 24, (0, 0, 255), 2)
    cv2.putText(
        annot,
        f"{conf:.2f}",
        (center[0] - 20, center[1] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    for cx, cy, v in peaks:
        pt = (cx + x1, cy + detect_y1)
        cv2.circle(annot, pt, 18, (0, 255, 0), 2)
        cv2.putText(
            annot,
            f"{v:.2f}",
            (pt[0] - 18, pt[1] - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )
    cv2.imwrite(str(OUT), annot)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
