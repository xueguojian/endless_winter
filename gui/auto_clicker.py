"""连点器：截取模拟器画面，点击屏幕选定连点坐标。"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

CANVAS_MAX_W = 360
CANVAS_MAX_H = 640
PAD = 8


class AutoClickerWindow(tk.Toplevel):
    """连点器窗口：截图后点击画面记录坐标，供一键连点使用。"""

    def __init__(
        self,
        master: tk.Misc,
        *,
        touch_width: int = 720,
        touch_height: int = 1280,
        screenshot_cb=None,
        on_point_changed=None,
        on_start=None,
        on_stop=None,
        is_running=None,
        initial_x: int = 360,
        initial_y: int = 640,
        initial_interval: float = 0.1,
        account_name: str = "",
        port: int | str = "",
    ):
        super().__init__(master)
        self._account_name = (account_name or "").strip()
        self._port = str(port or "").strip()
        self._update_window_title()
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.touch_width = touch_width
        self.touch_height = touch_height
        self._screenshot_cb = screenshot_cb
        self._on_point_changed = on_point_changed
        self._on_start = on_start
        self._on_stop = on_stop
        self._is_running = is_running

        self._photo: ImageTk.PhotoImage | None = None
        self._pil_image: Image.Image | None = None
        self._img_w = 0
        self._img_h = 0
        self._disp_w = 0
        self._disp_h = 0
        self._offset_x = PAD
        self._offset_y = PAD
        self._point: tuple[int, int] | None = (
            max(0, min(touch_width - 1, int(initial_x))),
            max(0, min(touch_height - 1, int(initial_y))),
        )

        self.var_coord = tk.StringVar(value=self._coord_text(*self._point))
        self.var_hint = tk.StringVar(value="截图后点击画面，选定连点位置")
        self.var_interval = tk.StringVar(value=str(float(initial_interval)))
        self.var_account_info = tk.StringVar(value=self._account_info_text())

        self._build_ui()
        self._draw_empty()
        self.withdraw()
        self._poll_running_state()

    def _account_info_text(self) -> str:
        parts: list[str] = []
        if self._account_name and self._account_name not in ("未知", "未识别"):
            parts.append(self._account_name)
        if self._port:
            parts.append(f"端口 {self._port}")
        return " · ".join(parts) if parts else "未获取账号"

    def _update_window_title(self) -> None:
        parts: list[str] = []
        if self._account_name and self._account_name not in ("未知", "未识别"):
            parts.append(self._account_name)
        if self._port:
            parts.append(str(self._port))
        parts.append("连点器")
        self.title(" · ".join(parts))

    def set_account_info(self, account_name: str = "", port: int | str = "") -> None:
        self._account_name = (account_name or "").strip()
        self._port = str(port or "").strip()
        self._update_window_title()
        if hasattr(self, "var_account_info"):
            self.var_account_info.set(self._account_info_text())

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(8, 6, 8, 0))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            textvariable=self.var_account_info,
            font=("", 10, "bold"),
            foreground="#c45c00",
        ).pack(side=tk.LEFT)

        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.pack(fill=tk.X)

        if self._screenshot_cb is not None:
            ttk.Button(
                toolbar, text="模拟器截图", command=self._capture_screen, width=10
            ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="打开图片", command=self._open_image, width=10).pack(
            side=tk.LEFT, padx=(0, 6)
        )

        ttk.Label(toolbar, text="间隔").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(
            toolbar,
            from_=0.05,
            to=10.0,
            increment=0.05,
            textvariable=self.var_interval,
            width=5,
        ).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="秒").pack(side=tk.LEFT, padx=(2, 8))

        self.btn_start = ttk.Button(
            toolbar, text="开始连点", command=self._start_click, width=9
        )
        self.btn_start.pack(side=tk.LEFT, padx=(0, 4))
        self.btn_stop = ttk.Button(
            toolbar,
            text="停止",
            command=self._stop_click,
            width=6,
            state=tk.DISABLED,
        )
        self.btn_stop.pack(side=tk.LEFT)

        info = ttk.Frame(self, padding=(10, 4, 10, 0))
        info.pack(fill=tk.X)
        ttk.Label(info, text="连点坐标", font=("", 9)).pack(side=tk.LEFT)
        ttk.Entry(
            info,
            textvariable=self.var_coord,
            font=("Consolas", 14),
            width=12,
            justify=tk.CENTER,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Label(
            info,
            textvariable=self.var_hint,
            font=("", 9),
            foreground="#2ec27e",
        ).pack(side=tk.LEFT)

        frame = ttk.Frame(self, padding=(8, 4, 8, 8))
        frame.pack()
        self.canvas = tk.Canvas(
            frame,
            width=CANVAS_MAX_W + PAD * 2,
            height=CANVAS_MAX_H + PAD * 2,
            bg="#2b2b2b",
            highlightthickness=1,
            highlightbackground="#666",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)

    def _on_close(self) -> None:
        self.withdraw()

    def show_window(self) -> None:
        self.deiconify()
        self.lift()
        self.focus_force()
        self._sync_running_buttons()

    def get_point(self) -> tuple[int, int] | None:
        return self._point

    def get_interval(self) -> float:
        try:
            return max(0.01, float(self.var_interval.get()))
        except (TypeError, ValueError):
            return 0.1

    def set_point(self, x: int, y: int, *, notify: bool = False) -> None:
        x = max(0, min(self.touch_width - 1, int(x)))
        y = max(0, min(self.touch_height - 1, int(y)))
        self._point = (x, y)
        self.var_coord.set(self._coord_text(x, y))
        self._redraw_marker()
        if notify and self._on_point_changed is not None:
            self._on_point_changed(x, y, self.get_interval())

    def _coord_text(self, tx: int, ty: int) -> str:
        return f"{tx}, {ty}"

    def _open_image(self) -> None:
        if not HAS_PIL:
            messagebox.showerror("缺少依赖", "需要安装 Pillow：pip install Pillow")
            return
        path = filedialog.askopenfilename(
            title="选择截图",
            filetypes=[
                ("图片", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self._load_image(Image.open(path).convert("RGB"))
            self.var_hint.set("点击画面选定连点位置")
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def _capture_screen(self) -> None:
        if not HAS_PIL:
            messagebox.showerror("缺少依赖", "需要安装 Pillow：pip install Pillow")
            return
        if self._screenshot_cb is None:
            return

        self.var_hint.set("正在截取模拟器画面…")

        def work() -> None:
            try:
                rgb = self._screenshot_cb()
                image = Image.fromarray(rgb).convert("RGB")

                def done() -> None:
                    self._load_image(image)
                    self.var_hint.set("截图已加载，点击画面选定连点位置")

                self.after(0, done)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("截图失败", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _load_image(self, image: Image.Image) -> None:
        self._pil_image = image
        self._img_w, self._img_h = image.size
        self._calc_display_size()
        self._redraw()

    def _calc_display_size(self) -> None:
        if self._img_w <= 0 or self._img_h <= 0:
            self._disp_w = 0
            self._disp_h = 0
            return
        scale = min(CANVAS_MAX_W / self._img_w, CANVAS_MAX_H / self._img_h)
        self._disp_w = max(1, int(self._img_w * scale))
        self._disp_h = max(1, int(self._img_h * scale))

    def _touch_to_canvas(self, tx: float, ty: float) -> tuple[float, float]:
        cx = self._offset_x + tx / self.touch_width * self._disp_w
        cy = self._offset_y + ty / self.touch_height * self._disp_h
        return cx, cy

    def _canvas_to_touch(self, cx: float, cy: float) -> tuple[int, int] | None:
        if self._disp_w <= 0 or self._disp_h <= 0:
            return None
        rel_x = cx - self._offset_x
        rel_y = cy - self._offset_y
        if rel_x < 0 or rel_y < 0 or rel_x > self._disp_w or rel_y > self._disp_h:
            return None
        tx = int(round(rel_x / self._disp_w * self.touch_width))
        ty = int(round(rel_y / self._disp_h * self.touch_height))
        tx = max(0, min(tx, self.touch_width - 1))
        ty = max(0, min(ty, self.touch_height - 1))
        return tx, ty

    def _draw_empty(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            PAD + CANVAS_MAX_W // 2,
            PAD + CANVAS_MAX_H // 2,
            text="请截取模拟器画面，然后点击选定连点位置",
            fill="#aaa",
            font=("", 11),
            width=CANVAS_MAX_W - 20,
        )

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._pil_image is None:
            self._draw_empty()
            return
        resized = self._pil_image.resize(
            (self._disp_w, self._disp_h), Image.Resampling.LANCZOS
        )
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(
            self._offset_x, self._offset_y, anchor=tk.NW, image=self._photo
        )
        self._redraw_marker()

    def _redraw_marker(self) -> None:
        self.canvas.delete("marker")
        if self._point is None or self._pil_image is None:
            return
        tx, ty = self._point
        cx, cy = self._touch_to_canvas(tx, ty)
        r = 7
        self.canvas.create_oval(
            cx - r,
            cy - r,
            cx + r,
            cy + r,
            outline="#ff4444",
            fill="#ff4444",
            width=1,
            tags="marker",
        )
        arm = 12
        self.canvas.create_line(
            cx - arm, cy, cx + arm, cy, fill="#ffff00", width=2, tags="marker"
        )
        self.canvas.create_line(
            cx, cy - arm, cx, cy + arm, fill="#ffff00", width=2, tags="marker"
        )

    def _on_motion(self, event: tk.Event) -> None:
        if self._is_busy():
            return
        pos = self._canvas_to_touch(event.x, event.y)
        if pos is None:
            return
        self.var_hint.set(f"悬停：{pos[0]}, {pos[1]}（点击选定）")

    def _on_click(self, event: tk.Event) -> None:
        if self._is_busy():
            return
        pos = self._canvas_to_touch(event.x, event.y)
        if pos is None:
            return
        tx, ty = pos
        self.set_point(tx, ty, notify=True)
        self.var_hint.set(f"已选定连点：{tx}, {ty}")

    def _is_busy(self) -> bool:
        if self._is_running is None:
            return False
        try:
            return bool(self._is_running())
        except Exception:
            return False

    def _start_click(self) -> None:
        if self._point is None:
            messagebox.showwarning("提示", "请先点击画面选定连点坐标")
            return
        if self._on_start is None:
            return
        self._on_start(self._point[0], self._point[1], self.get_interval())
        self._sync_running_buttons()

    def _stop_click(self) -> None:
        if self._on_stop is not None:
            self._on_stop()
        self._sync_running_buttons()

    def _sync_running_buttons(self) -> None:
        running = self._is_busy()
        if hasattr(self, "btn_start"):
            self.btn_start.configure(state=tk.DISABLED if running else tk.NORMAL)
        if hasattr(self, "btn_stop"):
            self.btn_stop.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _poll_running_state(self) -> None:
        if self.winfo_exists():
            self._sync_running_buttons()
            self.after(400, self._poll_running_state)
