"""区域监控：本机抓雷电窗口 ROI，显示 FPS，可映射触控坐标并点击。"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

try:
    from PIL import Image, ImageTk

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from core.window_capture import (
    WindowCapture,
    WindowInfo,
    client_to_touch,
    list_ldplayer_windows,
)

PREVIEW_MAX_W = 360
PREVIEW_MAX_H = 480


class RegionMonitorWindow(tk.Toplevel):
    """Windows 区域监控原型：DXGI 优先抓取，用于快反应小游戏标定。"""

    def __init__(
        self,
        master: tk.Misc,
        *,
        touch_width: int = 720,
        touch_height: int = 1280,
        tap_cb=None,
    ):
        super().__init__(master)
        self.title("区域监控（本机抓屏）")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.touch_width = touch_width
        self.touch_height = touch_height
        self._tap_cb = tap_cb

        self._capture = WindowCapture()
        self._windows: list[WindowInfo] = []
        self._running = False
        self._worker: threading.Thread | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._last_frame: np.ndarray | None = None
        self._fps = 0.0
        self._frame_count = 0
        self._fps_t0 = time.perf_counter()

        self.var_window = tk.StringVar(value="")
        self.var_backend = tk.StringVar(value=f"后端：{self._capture.backend}")
        self.var_fps = tk.StringVar(value="FPS：—")
        self.var_roi = tk.StringVar(value="ROI：未设置（默认整窗预览）")
        self.var_hint = tk.StringVar(
            value="选择雷电窗口 → 开始监控。在预览上拖拽框选 ROI。"
        )
        self.var_x = tk.IntVar(value=0)
        self.var_y = tk.IntVar(value=0)
        self.var_w = tk.IntVar(value=0)
        self.var_h = tk.IntVar(value=0)
        self._roi: tuple[int, int, int, int] | None = None  # x,y,w,h in client px
        self._drag_start: tuple[int, int] | None = None
        self._preview_scale = 1.0
        self._preview_offset = (0, 0)
        self._client_size = (0, 0)

        self._build_ui()
        self._refresh_windows()
        self.withdraw()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(fill=tk.X)

        ttk.Label(top, text="窗口").pack(side=tk.LEFT)
        self.cmb_window = ttk.Combobox(
            top, textvariable=self.var_window, width=42, state="readonly"
        )
        self.cmb_window.pack(side=tk.LEFT, padx=(4, 6))
        ttk.Button(top, text="刷新", width=6, command=self._refresh_windows).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self.btn_start = ttk.Button(
            top, text="开始监控", width=10, command=self._toggle_monitor
        )
        self.btn_start.pack(side=tk.LEFT)

        info = ttk.Frame(self, padding=(8, 0, 8, 4))
        info.pack(fill=tk.X)
        ttk.Label(info, textvariable=self.var_backend).pack(side=tk.LEFT)
        ttk.Label(info, textvariable=self.var_fps).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(info, textvariable=self.var_roi).pack(side=tk.LEFT, padx=(12, 0))

        roi_row = ttk.Frame(self, padding=(8, 2, 8, 4))
        roi_row.pack(fill=tk.X)
        ttk.Label(roi_row, text="ROI x,y,w,h").pack(side=tk.LEFT)
        for var, width in (
            (self.var_x, 5),
            (self.var_y, 5),
            (self.var_w, 5),
            (self.var_h, 5),
        ):
            ttk.Spinbox(
                roi_row, from_=0, to=4000, textvariable=var, width=width
            ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(roi_row, text="应用ROI", width=8, command=self._apply_roi_vars).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(roi_row, text="整窗", width=6, command=self._clear_roi).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Button(
            roi_row, text="点ROI中心", width=10, command=self._tap_roi_center
        ).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(
            self, textvariable=self.var_hint, foreground="#2ec27e", padding=(8, 0)
        ).pack(anchor=tk.W)

        frame = ttk.Frame(self, padding=(8, 4, 8, 8))
        frame.pack()
        self.canvas = tk.Canvas(
            frame,
            width=PREVIEW_MAX_W,
            height=PREVIEW_MAX_H,
            bg="#1e1e1e",
            highlightthickness=1,
            highlightbackground="#666",
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)

    def show_window(self) -> None:
        self.deiconify()
        self.lift()
        self.focus_force()
        self.var_backend.set(f"后端：{self._capture.backend}")

    def _on_close(self) -> None:
        self._stop_monitor()
        self.withdraw()

    def _refresh_windows(self) -> None:
        self._windows = list_ldplayer_windows()
        labels = [w.label for w in self._windows]
        self.cmb_window["values"] = labels
        if labels:
            cur = self.var_window.get()
            if cur not in labels:
                # 优先带雷电字样的
                pick = next(
                    (
                        w.label
                        for w in self._windows
                        if any(
                            k in w.title.lower()
                            for k in ("雷电", "ldplayer", "leidian")
                        )
                    ),
                    labels[0],
                )
                self.var_window.set(pick)
        else:
            self.var_window.set("")
            self.var_hint.set("未找到窗口，请先打开雷电模拟器")

    def _selected_window(self) -> WindowInfo | None:
        label = self.var_window.get().strip()
        for w in self._windows:
            if w.label == label:
                return w
        return None

    def _toggle_monitor(self) -> None:
        if self._running:
            self._stop_monitor()
        else:
            self._start_monitor()

    def _start_monitor(self) -> None:
        if not HAS_PIL:
            messagebox.showerror("缺少依赖", "需要 Pillow")
            return
        win = self._selected_window()
        if win is None:
            messagebox.showwarning("提示", "请先选择雷电窗口")
            return
        try:
            crect = self._capture.set_window(win.hwnd)
            self._client_size = (crect.width, crect.height)
            if self._roi is None:
                self.var_w.set(crect.width)
                self.var_h.set(crect.height)
        except Exception as exc:
            messagebox.showerror("绑定窗口失败", str(exc))
            return

        self._running = True
        self._frame_count = 0
        self._fps_t0 = time.perf_counter()
        self.btn_start.configure(text="停止监控")
        self.var_hint.set(
            f"监控中（{self._capture.backend}）。在预览上拖拽框选 ROI。"
        )
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _stop_monitor(self) -> None:
        self._running = False
        if hasattr(self, "btn_start"):
            self.btn_start.configure(text="开始监控")
        self.var_fps.set("FPS：—")

    def _loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            try:
                if self._roi is not None:
                    x, y, w, h = self._roi
                    frame = self._capture.grab_region(x, y, w, h)
                else:
                    frame = self._capture.grab_client()
                    crect = self._capture.client_rect()
                    self._client_size = (crect.width, crect.height)
                self._last_frame = frame
                self._frame_count += 1
                now = time.perf_counter()
                if now - self._fps_t0 >= 0.5:
                    self._fps = self._frame_count / (now - self._fps_t0)
                    self._frame_count = 0
                    self._fps_t0 = now
                    self.after(0, lambda f=self._fps: self.var_fps.set(f"FPS：{f:.0f}"))
                self.after(0, lambda img=frame: self._show_frame(img))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_loop_error(e))
                break
            # 略让出 CPU；DXGI 本身很快，这里限制到约 120Hz 上限
            elapsed = time.perf_counter() - t0
            sleep_s = max(0.0, 0.008 - elapsed)
            if sleep_s:
                time.sleep(sleep_s)

    def _on_loop_error(self, exc: Exception) -> None:
        self._stop_monitor()
        self.var_hint.set(f"监控中断：{exc}")

    def _show_frame(self, frame: np.ndarray) -> None:
        if not self.winfo_exists() or frame is None:
            return
        h, w = frame.shape[:2]
        scale = min(PREVIEW_MAX_W / w, PREVIEW_MAX_H / h, 1.0)
        dw = max(1, int(w * scale))
        dh = max(1, int(h * scale))
        self._preview_scale = scale
        rgb = frame[:, :, ::-1]
        image = Image.fromarray(rgb).resize((dw, dh), Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        ox = (PREVIEW_MAX_W - dw) // 2
        oy = (PREVIEW_MAX_H - dh) // 2
        self._preview_offset = (ox, oy)
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self._photo)
        if self._roi is not None and self._roi[2] > 0 and self._roi[3] > 0:
            # 当前预览已是 ROI 内容时，画边框提示即可
            self.canvas.create_rectangle(
                ox + 1, oy + 1, ox + dw - 1, oy + dh - 1, outline="#ffcc00", width=2
            )

    def _canvas_to_frame(self, cx: int, cy: int) -> tuple[int, int] | None:
        if self._last_frame is None:
            return None
        ox, oy = self._preview_offset
        rel_x = cx - ox
        rel_y = cy - oy
        fh, fw = self._last_frame.shape[:2]
        if rel_x < 0 or rel_y < 0:
            return None
        fx = int(rel_x / self._preview_scale)
        fy = int(rel_y / self._preview_scale)
        if fx >= fw or fy >= fh:
            return None
        return fx, fy

    def _on_drag_start(self, event: tk.Event) -> None:
        if not self._running or self._roi is not None:
            # 整窗预览时可框选；已是 ROI 预览时用数值框改
            if self._roi is not None:
                self.var_hint.set("已在 ROI 预览中，请改数值或点「整窗」后重新框选")
                return
        pos = self._canvas_to_frame(event.x, event.y)
        if pos is None:
            return
        self._drag_start = pos

    def _on_drag_move(self, event: tk.Event) -> None:
        if self._drag_start is None or self._last_frame is None:
            return
        pos = self._canvas_to_frame(event.x, event.y)
        if pos is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = pos
        ox, oy = self._preview_offset
        s = self._preview_scale
        self.canvas.delete("sel")
        self.canvas.create_rectangle(
            ox + min(x0, x1) * s,
            oy + min(y0, y1) * s,
            ox + max(x0, x1) * s,
            oy + max(y0, y1) * s,
            outline="#00e676",
            width=2,
            tags="sel",
        )

    def _on_drag_end(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        pos = self._canvas_to_frame(event.x, event.y)
        start = self._drag_start
        self._drag_start = None
        if pos is None:
            return
        x0, y0 = start
        x1, y1 = pos
        x = min(x0, x1)
        y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        if w < 4 or h < 4:
            return
        self._set_roi(x, y, w, h)

    def _set_roi(self, x: int, y: int, w: int, h: int) -> None:
        self._roi = (int(x), int(y), int(w), int(h))
        self.var_x.set(self._roi[0])
        self.var_y.set(self._roi[1])
        self.var_w.set(self._roi[2])
        self.var_h.set(self._roi[3])
        self.var_roi.set(f"ROI：{self._roi[0]},{self._roi[1]} {self._roi[2]}x{self._roi[3]}")
        cw, ch = self._client_size
        tx, ty = client_to_touch(
            self._roi[0] + self._roi[2] / 2,
            self._roi[1] + self._roi[3] / 2,
            client_w=max(1, cw),
            client_h=max(1, ch),
            touch_w=self.touch_width,
            touch_h=self.touch_height,
        )
        self.var_hint.set(f"ROI 已设置，中心触控约 ({tx},{ty})")

    def _apply_roi_vars(self) -> None:
        x, y, w, h = (
            int(self.var_x.get()),
            int(self.var_y.get()),
            int(self.var_w.get()),
            int(self.var_h.get()),
        )
        if w <= 0 or h <= 0:
            messagebox.showwarning("提示", "宽高必须大于 0")
            return
        self._set_roi(x, y, w, h)

    def _clear_roi(self) -> None:
        self._roi = None
        self.var_roi.set("ROI：未设置（默认整窗预览）")
        self.var_hint.set("已恢复整窗预览，可重新拖拽框选 ROI")

    def _tap_roi_center(self) -> None:
        if self._tap_cb is None:
            messagebox.showinfo("提示", "未绑定点击回调")
            return
        if self._roi is None:
            messagebox.showwarning("提示", "请先框选或填写 ROI")
            return
        cw, ch = self._client_size
        if cw <= 0 or ch <= 0:
            try:
                crect = self._capture.client_rect()
                cw, ch = crect.width, crect.height
                self._client_size = (cw, ch)
            except Exception as exc:
                messagebox.showerror("失败", str(exc))
                return
        x, y, w, h = self._roi
        tx, ty = client_to_touch(
            x + w / 2,
            y + h / 2,
            client_w=cw,
            client_h=ch,
            touch_w=self.touch_width,
            touch_h=self.touch_height,
        )
        try:
            self._tap_cb(tx, ty)
            self.var_hint.set(f"已点击触控坐标 ({tx},{ty})")
        except Exception as exc:
            messagebox.showerror("点击失败", str(exc))
