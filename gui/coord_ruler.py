"""截图坐标标尺：加载图片后显示 720×1280 触控坐标网格。"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

RULER_MARGIN = 44
GRID_STEP_DEFAULT = 50
CANVAS_MAX_W = 420
CANVAS_MAX_H = 640


class CoordRulerWindow(tk.Toplevel):
    """坐标标尺窗口（默认由主界面按钮打开）。"""

    def __init__(
        self,
        master: tk.Misc,
        touch_width: int = 720,
        touch_height: int = 1280,
        screenshot_cb=None,
    ):
        super().__init__(master)
        self.title("坐标标尺")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.touch_width = touch_width
        self.touch_height = touch_height
        self._screenshot_cb = screenshot_cb

        self._photo: ImageTk.PhotoImage | None = None
        self._pil_image: Image.Image | None = None
        self._img_w = 0
        self._img_h = 0
        self._disp_w = 0
        self._disp_h = 0
        self._offset_x = RULER_MARGIN
        self._offset_y = RULER_MARGIN
        self._markers: list[int] = []

        self.var_grid = tk.IntVar(value=GRID_STEP_DEFAULT)
        self.var_coord = tk.StringVar(value="—")
        self.var_hint = tk.StringVar(value="点击图片自动复制坐标到剪贴板")

        self._build_ui()
        self._draw_empty()
        self.withdraw()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="打开图片", command=self._open_image, width=10).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        if self._screenshot_cb is not None:
            ttk.Button(toolbar, text="模拟器截图", command=self._capture_screen, width=10).pack(
                side=tk.LEFT, padx=(0, 6)
            )

        ttk.Label(toolbar, text="网格间隔").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Spinbox(
            toolbar,
            from_=10,
            to=200,
            increment=10,
            textvariable=self.var_grid,
            width=5,
            command=self._redraw,
        ).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="刷新网格", command=self._redraw, width=8).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(toolbar, text="清除标记", command=self._clear_markers, width=8).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        coord_frame = ttk.Frame(self, padding=(10, 6, 10, 0))
        coord_frame.pack(fill=tk.X)

        ttk.Label(coord_frame, text="触控坐标", font=("", 9)).pack(side=tk.LEFT)
        self.entry_coord = ttk.Entry(
            coord_frame,
            textvariable=self.var_coord,
            font=("Consolas", 16),
            width=14,
            justify=tk.CENTER,
            state="readonly",
        )
        self.entry_coord.pack(side=tk.LEFT, padx=(8, 6))
        ttk.Button(coord_frame, text="复制", command=self._copy_current_coord, width=6).pack(
            side=tk.LEFT
        )
        ttk.Label(
            coord_frame,
            textvariable=self.var_hint,
            font=("", 9),
            foreground="#2ec27e",
        ).pack(side=tk.LEFT, padx=(10, 0))

        frame = ttk.Frame(self, padding=(8, 0, 8, 8))
        frame.pack()

        self.canvas = tk.Canvas(
            frame,
            width=CANVAS_MAX_W + RULER_MARGIN,
            height=CANVAS_MAX_H + RULER_MARGIN,
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
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def _capture_screen(self) -> None:
        if not HAS_PIL:
            messagebox.showerror("缺少依赖", "需要安装 Pillow：pip install Pillow")
            return
        if self._screenshot_cb is None:
            return

        self.var_coord.set("正在截取模拟器画面…")

        def work() -> None:
            try:
                rgb = self._screenshot_cb()
                image = Image.fromarray(rgb).convert("RGB")

                def done() -> None:
                    self._load_image(image)
                    self.var_coord.set("—")
                    self.var_hint.set("截图已加载，点击图片复制坐标")

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
            self._offset_x + CANVAS_MAX_W // 2,
            self._offset_y + CANVAS_MAX_H // 2,
            text="请打开图片或截取模拟器画面",
            fill="#aaa",
            font=("", 11),
        )
        self._draw_rulers()

    def _draw_rulers(self) -> None:
        step = max(10, int(self.var_grid.get() or GRID_STEP_DEFAULT))
        self.canvas.delete("ruler")

        for tx in range(0, self.touch_width + 1, step):
            cx, _ = self._touch_to_canvas(tx, 0)
            self.canvas.create_line(
                cx,
                0,
                cx,
                RULER_MARGIN - 4,
                fill="#888",
                tags="ruler",
            )
            if tx % (step * 2) == 0 or step >= 100:
                self.canvas.create_text(
                    cx,
                    RULER_MARGIN - 14,
                    text=str(tx),
                    fill="#ccc",
                    font=("Consolas", 7),
                    tags="ruler",
                )

        for ty in range(0, self.touch_height + 1, step):
            _, cy = self._touch_to_canvas(0, ty)
            self.canvas.create_line(
                0,
                cy,
                RULER_MARGIN - 4,
                cy,
                fill="#888",
                tags="ruler",
            )
            if ty % (step * 2) == 0 or step >= 100:
                self.canvas.create_text(
                    RULER_MARGIN - 10,
                    cy,
                    text=str(ty),
                    fill="#ccc",
                    font=("Consolas", 7),
                    angle=0,
                    tags="ruler",
                )

    def _draw_grid(self) -> None:
        step = max(10, int(self.var_grid.get() or GRID_STEP_DEFAULT))
        self.canvas.delete("grid")

        x_right = self._offset_x + self._disp_w
        y_bottom = self._offset_y + self._disp_h

        for tx in range(0, self.touch_width + 1, step):
            cx, _ = self._touch_to_canvas(tx, 0)
            color = "#ffcc00" if tx % 100 == 0 else "#555"
            width = 2 if tx % 100 == 0 else 1
            self.canvas.create_line(
                cx,
                self._offset_y,
                cx,
                y_bottom,
                fill=color,
                width=width,
                tags="grid",
            )

        for ty in range(0, self.touch_height + 1, step):
            _, cy = self._touch_to_canvas(0, ty)
            color = "#ffcc00" if ty % 100 == 0 else "#555"
            width = 2 if ty % 100 == 0 else 1
            self.canvas.create_line(
                self._offset_x,
                cy,
                x_right,
                cy,
                fill=color,
                width=width,
                tags="grid",
            )

    def _redraw(self) -> None:
        self.canvas.delete("all")
        self._draw_rulers()

        if self._pil_image is None:
            self.canvas.create_text(
                self._offset_x + CANVAS_MAX_W // 2,
                self._offset_y + CANVAS_MAX_H // 2,
                text="请打开图片或截取模拟器画面",
                fill="#aaa",
                font=("", 11),
            )
            return

        resized = self._pil_image.resize((self._disp_w, self._disp_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self._offset_x, self._offset_y, anchor=tk.NW, image=self._photo)
        self._draw_grid()
        self._redraw_markers()

    def _redraw_markers(self) -> None:
        self.canvas.delete("marker")
        for i in range(0, len(self._markers), 2):
            tx, ty = self._markers[i], self._markers[i + 1]
            cx, cy = self._touch_to_canvas(tx, ty)
            r = 6
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
            arm = 10
            self.canvas.create_line(
                cx - arm, cy, cx + arm, cy, fill="#ffff00", width=2, tags="marker"
            )
            self.canvas.create_line(
                cx, cy - arm, cx, cy + arm, fill="#ffff00", width=2, tags="marker"
            )

    def _clear_markers(self) -> None:
        self._markers.clear()
        self.canvas.delete("marker")

    def _coord_text(self, tx: int, ty: int) -> str:
        return f"{tx}, {ty}"

    def _copy_to_clipboard(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_idletasks()

    def _copy_current_coord(self) -> None:
        text = self.var_coord.get().strip()
        if not text or text == "—":
            return
        self._copy_to_clipboard(text)
        self.var_hint.set("已复制到剪贴板")

    def _set_coord_text(self, tx: int, ty: int, *, copied: bool = False) -> None:
        self.var_coord.set(self._coord_text(tx, ty))
        if copied:
            self.var_hint.set("已复制到剪贴板")
        else:
            self.var_hint.set("点击图片自动复制")

    def _on_motion(self, event: tk.Event) -> None:
        pos = self._canvas_to_touch(event.x, event.y)
        if pos is None:
            return
        self._set_coord_text(pos[0], pos[1])

    def _on_click(self, event: tk.Event) -> None:
        pos = self._canvas_to_touch(event.x, event.y)
        if pos is None:
            return
        tx, ty = pos
        self._markers.extend([tx, ty])
        self._redraw_markers()
        text = self._coord_text(tx, ty)
        self._set_coord_text(tx, ty, copied=True)
        self._copy_to_clipboard(text)
