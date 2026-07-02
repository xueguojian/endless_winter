"""寻梦记忆地图标定窗口：截图 → 点击取坐标 → 一键写入 YAML。"""

from __future__ import annotations

import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

try:
    from PIL import Image, ImageTk

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from core.dream_memory.config import load_dream_memory_config
from core.dream_memory.maps import DreamMemoryMap, delete_map, load_map, rename_map_name, save_map
from gui.coord_ruler import CANVAS_MAX_H, CANVAS_MAX_W, GRID_STEP_DEFAULT, RULER_MARGIN

PANEL_WIDTH = 228


class DreamMemoryCalibratorWindow(tk.Toplevel):
    """在截图上点击物品位置，填写名称后写入当前地图配置。"""

    def __init__(
        self,
        master: tk.Misc,
        *,
        config_path: Path,
        get_map_id: callable,
        screenshot_cb=None,
        on_saved: callable | None = None,
        touch_width: int = 720,
        touch_height: int = 1280,
        pk_mode: bool = False,
    ):
        super().__init__(master)
        self.title("寻梦记忆PK — 地图标定" if pk_mode else "寻梦记忆 — 地图标定")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.config_path = Path(config_path)
        self._get_map_id = get_map_id
        self._screenshot_cb = screenshot_cb
        self._on_saved = on_saved
        self.touch_width = touch_width
        self.touch_height = touch_height
        self._pk_mode = pk_mode

        self._photo: ImageTk.PhotoImage | None = None
        self._pil_image: Image.Image | None = None
        self._img_w = 0
        self._img_h = 0
        self._disp_w = 0
        self._disp_h = 0
        self._offset_x = RULER_MARGIN
        self._offset_y = RULER_MARGIN

        self.var_grid = tk.IntVar(value=GRID_STEP_DEFAULT)
        self.var_coord = tk.StringVar(value="—")
        self.var_item_name = tk.StringVar(value="")
        self.var_map_label = tk.StringVar(value="")
        self.var_hint = tk.StringVar(value="先截图，再点击物品中心")

        self._pending_tx = 0
        self._pending_ty = 0
        self._saved_markers: dict[str, tuple[int, int]] = {}

        self._build_ui()
        self._refresh_map_label()
        self._reload_item_list()
        self.withdraw()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(fill=tk.X)
        ttk.Label(top, text="当前地图:").pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.var_map_label, font=("", 9, "bold")).pack(
            side=tk.LEFT, padx=(4, 12)
        )
        ttk.Button(top, text="刷新列表", command=self._reload_item_list, width=8).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(top, text="新建地图", command=self._create_map_dialog, width=8).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(top, text="重命名", command=self._rename_map_dialog, width=7).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(top, text="删除地图", command=self._delete_map_dialog, width=8).pack(
            side=tk.LEFT
        )

        toolbar = ttk.Frame(self, padding=(8, 0))
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="打开图片", command=self._open_image, width=10).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        if self._screenshot_cb is not None:
            ttk.Button(toolbar, text="模拟器截图", command=self._capture_screen, width=10).pack(
                side=tk.LEFT, padx=(0, 6)
            )
        ttk.Button(toolbar, text="清除十字", command=self._clear_pending_marker, width=8).pack(
            side=tk.LEFT
        )

        body = ttk.Frame(self, padding=(8, 6))
        body.pack()

        canvas_frame = ttk.Frame(body)
        canvas_frame.pack(side=tk.LEFT)
        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_MAX_W + RULER_MARGIN,
            height=CANVAS_MAX_H + RULER_MARGIN,
            bg="#2b2b2b",
            highlightthickness=1,
            highlightbackground="#666",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)

        panel = ttk.LabelFrame(body, text="写入地图", padding=8, width=PANEL_WIDTH)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        panel.pack_propagate(False)

        ttk.Label(panel, text="触控坐标").pack(anchor=tk.W)
        ttk.Entry(
            panel,
            textvariable=self.var_coord,
            font=("Consolas", 12),
            state="readonly",
            justify=tk.CENTER,
        ).pack(fill=tk.X, pady=(2, 8))

        ttk.Label(panel, text="物品名（与底栏文字一致）").pack(anchor=tk.W)
        ttk.Entry(panel, textvariable=self.var_item_name).pack(fill=tk.X, pady=(2, 8))

        ttk.Button(
            panel,
            text="写入地图",
            command=self._save_item_to_map,
            width=16,
        ).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            panel,
            textvariable=self.var_hint,
            font=("", 8),
            foreground="#2ec27e",
            wraplength=PANEL_WIDTH - 16,
        ).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(panel, text="已标定物品").pack(anchor=tk.W)
        list_frame = ttk.Frame(panel)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 6))
        self.list_items = tk.Listbox(list_frame, height=14, font=("Consolas", 9))
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.list_items.yview)
        self.list_items.configure(yscrollcommand=scroll.set)
        self.list_items.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.list_items.bind("<<ListboxSelect>>", self._on_list_select)

        ttk.Button(panel, text="删除选中", command=self._delete_selected_item, width=16).pack(
            fill=tk.X
        )

        ttk.Label(
            self,
            text="绿点=已保存；黄十字=当前点击。写入后绿点会更新。",
            font=("", 8),
            foreground="gray",
        ).pack(fill=tk.X, padx=10, pady=(0, 8))

        self._draw_empty()

    def _on_close(self) -> None:
        self.withdraw()

    def show_window(self) -> None:
        self._refresh_map_label()
        self._reload_item_list()
        if self._pil_image is None:
            self._draw_empty()
            n = len(self._saved_markers)
            self.var_hint.set(
                f"请先点「模拟器截图」或「打开图片」才能在图上点击标定。"
                f" 右侧列表已有 {n} 个物品，开始游戏不依赖本窗口是否显示图片。"
            )
        self.deiconify()
        self.lift()
        self.focus_force()

    def _dm_cfg(self):
        if self._pk_mode:
            from core.dream_memory.config import load_dream_memory_pk_config

            return load_dream_memory_pk_config(self.config_path)
        return load_dream_memory_config(self.config_path)

    def _current_map_id(self) -> str:
        return str(self._get_map_id() or "").strip()

    def _refresh_map_label(self) -> None:
        map_id = self._current_map_id()
        if not map_id:
            self.var_map_label.set("（未选择）")
            return
        try:
            dream_map = load_map(map_id, maps_dir=self._dm_cfg().maps_dir)
            self.var_map_label.set(f"{dream_map.name} ({map_id})")
        except FileNotFoundError:
            self.var_map_label.set(f"{map_id}（文件不存在）")

    def _reload_item_list(self) -> None:
        self._refresh_map_label()
        map_id = self._current_map_id()
        self.list_items.delete(0, tk.END)
        self._saved_markers.clear()
        if not map_id:
            self._redraw_markers()
            return
        try:
            dream_map = load_map(map_id, maps_dir=self._dm_cfg().maps_dir)
        except FileNotFoundError:
            self._redraw_markers()
            return
        for index, (name, coord) in enumerate(dream_map.items.items(), start=1):
            if len(coord) >= 2:
                x, y = int(coord[0]), int(coord[1])
                self._saved_markers[name] = (x, y)
                self.list_items.insert(tk.END, f"{index}. {name}  ({x}, {y})")
        self._redraw_markers()

    def _create_map_dialog(self) -> None:
        map_id = simpledialog.askstring(
            "新建地图",
            "地图 ID（英文/数字，如 gazebo_snow）:",
            parent=self,
        )
        if not map_id:
            return
        map_id = re.sub(r"[^\w\-]+", "_", map_id.strip()).strip("_")
        if not map_id:
            messagebox.showwarning("无效 ID", "地图 ID 不能为空", parent=self)
            return
        name = simpledialog.askstring(
            "新建地图",
            "显示名称（如 凉亭雪景）:",
            parent=self,
            initialvalue=map_id,
        )
        if not name:
            return
        path = save_map(
            DreamMemoryMap(map_id=map_id, name=name.strip(), items={}),
            maps_dir=self._dm_cfg().maps_dir,
        )
        if self._on_saved:
            self._on_saved(map_id)
        self._refresh_map_label()
        self._reload_item_list()
        messagebox.showinfo("已创建", f"地图已创建:\n{path}", parent=self)

    def _rename_map_dialog(self) -> None:
        map_id = self._current_map_id()
        if not map_id:
            messagebox.showwarning("未选择地图", "请先在主界面选择地图", parent=self)
            return
        try:
            dream_map = load_map(map_id, maps_dir=self._dm_cfg().maps_dir)
        except FileNotFoundError:
            messagebox.showerror("错误", f"地图不存在: {map_id}", parent=self)
            return
        new_name = simpledialog.askstring(
            "重命名地图",
            f"地图 ID: {map_id}\n新的显示名称:",
            parent=self,
            initialvalue=dream_map.name,
        )
        if not new_name or new_name.strip() == dream_map.name:
            return
        try:
            rename_map_name(map_id, new_name.strip(), maps_dir=self._dm_cfg().maps_dir)
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("重命名失败", str(exc), parent=self)
            return
        self._refresh_map_label()
        if self._on_saved:
            self._on_saved(map_id)
        messagebox.showinfo("完成", f"已重命名为「{new_name.strip()}」", parent=self)

    def _delete_map_dialog(self) -> None:
        map_id = self._current_map_id()
        if not map_id:
            messagebox.showwarning("未选择地图", "请先在主界面选择地图", parent=self)
            return
        try:
            dream_map = load_map(map_id, maps_dir=self._dm_cfg().maps_dir)
        except FileNotFoundError:
            messagebox.showerror("错误", f"地图不存在: {map_id}", parent=self)
            return
        item_count = len(dream_map.items)
        if not messagebox.askyesno(
            "删除地图",
            f"确定删除地图「{dream_map.name}」({map_id})？\n"
            f"含 {item_count} 个标定物品，此操作不可恢复。",
            parent=self,
        ):
            return
        try:
            delete_map(map_id, maps_dir=self._dm_cfg().maps_dir)
        except FileNotFoundError as exc:
            messagebox.showerror("删除失败", str(exc), parent=self)
            return
        self._saved_markers.clear()
        self._reload_item_list()
        if self._on_saved:
            self._on_saved(None)
        messagebox.showinfo("已删除", f"地图「{dream_map.name}」已删除", parent=self)

    def _save_item_to_map(self) -> None:
        map_id = self._current_map_id()
        if not map_id:
            messagebox.showwarning("未选择地图", "请先在主界面「寻梦记忆」页选择或新建地图", parent=self)
            return
        name = self.var_item_name.get().strip()
        if not name:
            messagebox.showwarning("缺少名称", "请填写物品名", parent=self)
            return
        if self.var_coord.get().strip() in ("", "—"):
            messagebox.showwarning("缺少坐标", "请先在截图上点击物品位置", parent=self)
            return

        try:
            dream_map = load_map(map_id, maps_dir=self._dm_cfg().maps_dir)
        except FileNotFoundError:
            messagebox.showerror("地图不存在", f"找不到地图: {map_id}", parent=self)
            return

        dream_map.items[name] = [self._pending_tx, self._pending_ty]
        try:
            save_map(dream_map, maps_dir=self._dm_cfg().maps_dir)
        except PermissionError:
            messagebox.showerror(
                "无法保存",
                f"没有写入权限: {map_id}.yaml\n\n"
                "若该文件是从别处复制来的，可能带有「只读」属性。"
                "请在资源管理器中右键文件 → 属性 → 取消「只读」，或删除后重新标定。",
                parent=self,
            )
            return
        self.var_hint.set(f"已保存「{name}」→ ({self._pending_tx}, {self._pending_ty})")
        self.var_item_name.set("")
        self._reload_item_list()
        if self._on_saved:
            self._on_saved(map_id)

    def _item_name_from_list_line(self, line: str) -> str:
        name = line.split("  (", 1)[0].strip()
        prefix, sep, rest = name.partition(". ")
        if sep and prefix.isdigit():
            return rest
        return name

    def _delete_selected_item(self) -> None:
        selection = self.list_items.curselection()
        if not selection:
            messagebox.showinfo("提示", "请先在列表中选择一项", parent=self)
            return
        line = self.list_items.get(selection[0])
        name = self._item_name_from_list_line(line)
        map_id = self._current_map_id()
        if not map_id:
            return
        if not messagebox.askyesno("确认删除", f"删除「{name}」？", parent=self):
            return
        try:
            dream_map = load_map(map_id, maps_dir=self._dm_cfg().maps_dir)
        except FileNotFoundError:
            return
        dream_map.items.pop(name, None)
        save_map(dream_map, maps_dir=self._dm_cfg().maps_dir)
        self._reload_item_list()
        if self._on_saved:
            self._on_saved(map_id)

    def _on_list_select(self, _event=None) -> None:
        selection = self.list_items.curselection()
        if not selection:
            return
        line = self.list_items.get(selection[0])
        name = self._item_name_from_list_line(line)
        self.var_item_name.set(name)
        if name in self._saved_markers:
            x, y = self._saved_markers[name]
            self._pending_tx, self._pending_ty = x, y
            self.var_coord.set(f"{x}, {y}")
            self._redraw_markers()

    # --- 截图与画布（与坐标标尺相同逻辑）---

    def _open_image(self) -> None:
        if not HAS_PIL:
            messagebox.showerror("缺少依赖", "需要安装 Pillow", parent=self)
            return
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="选择寻梦记忆截图",
            filetypes=[("图片", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            self._load_image(Image.open(path).convert("RGB"))
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc), parent=self)

    def _capture_screen(self) -> None:
        if not HAS_PIL or self._screenshot_cb is None:
            return
        self.var_coord.set("截图中…")

        def work() -> None:
            try:
                rgb = self._screenshot_cb()
                image = Image.fromarray(rgb).convert("RGB")

                def done() -> None:
                    self._load_image(image)
                    self.var_coord.set("—")
                    self.var_hint.set("截图已加载，点击物品中心")

                self.after(0, done)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("截图失败", str(exc), parent=self))

        threading.Thread(target=work, daemon=True).start()

    def _load_image(self, image: Image.Image) -> None:
        self._pil_image = image
        self._img_w, self._img_h = image.size
        self._calc_display_size()
        self._redraw()

    def _calc_display_size(self) -> None:
        if self._img_w <= 0 or self._img_h <= 0:
            self._disp_w = self._disp_h = 0
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
        return max(0, min(tx, self.touch_width - 1)), max(0, min(ty, self.touch_height - 1))

    def _draw_empty(self) -> None:
        self.canvas.delete("all")
        cx = self._offset_x + CANVAS_MAX_W // 2
        cy = self._offset_y + CANVAS_MAX_H // 2
        self.canvas.create_text(
            cx,
            cy - 16,
            text="暂无背景图",
            fill="#ccc",
            font=("", 12, "bold"),
        )
        self.canvas.create_text(
            cx,
            cy + 14,
            text="请点上方「模拟器截图」或「打开图片」",
            fill="#888",
            font=("", 10),
        )

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._pil_image is None:
            self._draw_empty()
            self._redraw_markers()
            return
        resized = self._pil_image.resize((self._disp_w, self._disp_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self._offset_x, self._offset_y, anchor=tk.NW, image=self._photo)
        self._redraw_markers()

    def _clear_pending_marker(self) -> None:
        self.var_coord.set("—")
        self.var_hint.set("已清除当前点击")
        self._redraw_markers()

    def _redraw_markers(self) -> None:
        self.canvas.delete("marker")
        for name, (tx, ty) in self._saved_markers.items():
            cx, cy = self._touch_to_canvas(tx, ty)
            r = 5
            self.canvas.create_oval(
                cx - r,
                cy - r,
                cx + r,
                cy + r,
                outline="#44cc66",
                fill="#44cc66",
                tags="marker",
            )
            self.canvas.create_text(
                cx,
                cy - 12,
                text=name,
                fill="#a6e3a1",
                font=("", 8),
                tags="marker",
            )

        if self.var_coord.get().strip() not in ("", "—"):
            cx, cy = self._touch_to_canvas(self._pending_tx, self._pending_ty)
            arm = 10
            self.canvas.create_line(
                cx - arm, cy, cx + arm, cy, fill="#ffff00", width=2, tags="marker"
            )
            self.canvas.create_line(
                cx, cy - arm, cx, cy + arm, fill="#ffff00", width=2, tags="marker"
            )

    def _on_motion(self, event: tk.Event) -> None:
        pos = self._canvas_to_touch(event.x, event.y)
        if pos is None:
            return
        self.var_coord.set(f"{pos[0]}, {pos[1]}")

    def _on_click(self, event: tk.Event) -> None:
        pos = self._canvas_to_touch(event.x, event.y)
        if pos is None:
            return
        self._pending_tx, self._pending_ty = pos
        self.var_coord.set(f"{pos[0]}, {pos[1]}")
        self.var_hint.set("填写物品名后点「写入地图」")
        self._redraw_markers()
