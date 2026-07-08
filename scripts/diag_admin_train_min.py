"""Minimal train-icon match: pure OpenCV, no task imports."""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "assets" / "debug" / "admin_raw_bottom.png"
TPL = ROOT / "assets" / "templates" / "alliance_mobilization" / "train_icon_admin.png"
OUT = ROOT / "assets" / "debug" / "admin_train_hits.png"
ROI = (35, 360, 685, 1085)


def main() -> None:
    print("load screen", SCREEN)
    screen = cv2.imread(str(SCREEN))
    print("screen", None if screen is None else screen.shape)
    tpl = cv2.imread(str(TPL))
    print("tpl", None if tpl is None else tpl.shape)
    x1, y1, x2, y2 = ROI
    roi = screen[y1:y2, x1:x2]
    print("roi", roi.shape)

    gray_r = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_t = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    best = (-1.0, 1.0, (0, 0), (0, 0))
    for scale in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15):
        tw = max(8, int(gray_t.shape[1] * scale))
        th = max(8, int(gray_t.shape[0] * scale))
        scaled = cv2.resize(gray_t, (tw, th), interpolation=cv2.INTER_AREA)
        if scaled.shape[0] >= gray_r.shape[0] or scaled.shape[1] >= gray_r.shape[1]:
            continue
        res = cv2.matchTemplate(gray_r, scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        print(f"scale={scale:.2f} conf={max_val:.3f} loc={max_loc}")
        if max_val > best[0]:
            cx = x1 + max_loc[0] + tw // 2
            cy = y1 + max_loc[1] + th // 2
            best = (max_val, scale, (cx, cy), max_loc)

    conf, scale, center, loc = best
    print(f"BEST conf={conf:.3f} scale={scale} center={center} roi_tl={loc}")

    # also find all peaks above 0.55
    scaled = cv2.resize(
        gray_t,
        (
            max(8, int(gray_t.shape[1] * scale)),
            max(8, int(gray_t.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    res = cv2.matchTemplate(gray_r, scaled, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= 0.55)
    peaks = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        v = float(res[y, x])
        cx = x1 + x + scaled.shape[1] // 2
        cy = y1 + y + scaled.shape[0] // 2
        # nms
        if any(abs(cx - px) < 40 and abs(cy - py) < 40 for px, py, _ in peaks):
            continue
        peaks.append((cx, cy, v))
    peaks.sort(key=lambda p: p[2], reverse=True)
    print(f"peaks>=0.55 count={len(peaks)}")
    for p in peaks:
        print(f"  peak center=({p[0]},{p[1]}) conf={p[2]:.3f}")

    annot = screen.copy()
    cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for cx, cy, v in peaks:
        cv2.circle(annot, (cx, cy), 20, (0, 0, 255), 2)
        cv2.putText(
            annot,
            f"{v:.2f}",
            (cx - 20, cy - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
    if center != (0, 0):
        cv2.circle(annot, center, 26, (0, 255, 255), 2)
    cv2.imwrite(str(OUT), annot)
    print("saved", OUT)


if __name__ == "__main__":
    main()
