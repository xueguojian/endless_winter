"""寻梦记忆 / 寻梦记忆PK Tab 共用逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from core.dream_memory.config import (
    CURRENT_MAP_PERIOD,
    TAP_INTERVAL_LABELS,
    load_dream_memory_config,
    load_dream_memory_pk_config,
    normalize_tap_between_interval,
)
from core.dream_memory.maps import (
    find_map_preview,
    format_map_choice,
    format_period_choice,
    has_map_preview,
    list_map_periods,
    list_maps,
    load_map,
    parse_period_choice,
    save_map_preview,
)

if TYPE_CHECKING:
    from gui.app import EndlessWinterApp


@dataclass
class DreamTabWidgets:
    pk: bool
    var_period: tk.StringVar
    cmb_period: ttk.Combobox
    var_map: tk.StringVar
    cmb_map: ttk.Combobox
    lbl_summary: ttk.Label
    preview_btn_frame: ttk.Frame
    lbl_ocr: ttk.Label
    btn_start: ttk.Button
    btn_stop: ttk.Button
    var_tap_interval: tk.StringVar | None = None
    cmb_tap_interval: ttk.Combobox | None = None
    config_section: str = ""
    label_to_id: dict[str, str] = field(default_factory=dict)
    id_to_label: dict[str, str] = field(default_factory=dict)


@dataclass
class TurboPkControls:
    """极速寻梦PK：雷电窗口 + 底栏 ROI（客户区 x,y,w,h）。"""

    var_window: tk.StringVar
    cmb_window: ttk.Combobox
    var_x: tk.IntVar
    var_y: tk.IntVar
    var_w: tk.IntVar
    var_h: tk.IntVar
    windows: list = field(default_factory=list)

    def selected_hwnd(self) -> int | None:
        label = self.var_window.get().strip()
        for win in self.windows:
            if win.label == label:
                return int(win.hwnd)
        return None

    def bar_roi(self) -> tuple[int, int, int, int]:
        return (
            int(self.var_x.get()),
            int(self.var_y.get()),
            int(self.var_w.get()),
            int(self.var_h.get()),
        )


def load_dream_cfg(app: EndlessWinterApp, pk: bool):
    if pk:
        return load_dream_memory_pk_config(app.config_path)
    return load_dream_memory_config(app.config_path)


def rebuild_map_index(maps) -> tuple[dict[str, str], dict[str, str]]:
    label_to_id = {format_map_choice(m): m.map_id for m in maps}
    id_to_label = {m.map_id: format_map_choice(m) for m in maps}
    return label_to_id, id_to_label


def get_selected_map_id(widgets: DreamTabWidgets) -> str:
    sel = widgets.var_map.get().strip()
    if sel in widgets.label_to_id:
        return widgets.label_to_id[sel]
    return sel


def get_selected_period(widgets: DreamTabWidgets) -> int:
    return parse_period_choice(
        widgets.var_period.get(),
        default=CURRENT_MAP_PERIOD,
    )


def get_tap_interval_mode(widgets: DreamTabWidgets) -> str:
    if widgets.var_tap_interval is None:
        return normalize_tap_between_interval(None)
    label = widgets.var_tap_interval.get().strip()
    for key, text in TAP_INTERVAL_LABELS.items():
        if label == text:
            return key
    return normalize_tap_between_interval(label)


def set_map_selection(widgets: DreamTabWidgets, map_id: str) -> None:
    label = widgets.id_to_label.get(map_id, map_id)
    widgets.var_map.set(label)


def set_period_selection(widgets: DreamTabWidgets, period: int) -> None:
    widgets.var_period.set(format_period_choice(period))


def update_summary(app: EndlessWinterApp, widgets: DreamTabWidgets) -> None:
    map_id = get_selected_map_id(widgets)
    if not map_id:
        widgets.lbl_summary.configure(text="暂无地图")
        update_preview_buttons(app, widgets)
        return
    try:
        dm_cfg = load_dream_cfg(app, widgets.pk)
        dream_map = load_map(map_id, maps_dir=dm_cfg.maps_dir)
        slot_hint = f"{len(dm_cfg.target_slots)} 槽"
        preview_hint = (
            " · 已上传预览图"
            if has_map_preview(map_id, previews_dir=dm_cfg.previews_dir)
            else ""
        )
        widgets.lbl_summary.configure(
            text=(
                f"第 {dream_map.period} 期 · 显示名：{dream_map.name}"
                f" · 已标定 {len(dream_map.items)} 个物品"
                f" · 识别区 {slot_hint}{preview_hint}"
            )
        )
    except FileNotFoundError:
        widgets.lbl_summary.configure(text=f"地图文件不存在: {map_id}")
    update_preview_buttons(app, widgets)


def update_preview_buttons(app: EndlessWinterApp, widgets: DreamTabWidgets) -> None:
    for child in widgets.preview_btn_frame.winfo_children():
        child.destroy()

    map_id = get_selected_map_id(widgets)
    if not map_id:
        return

    dm_cfg = load_dream_cfg(app, widgets.pk)
    if has_map_preview(map_id, previews_dir=dm_cfg.previews_dir):
        ttk.Button(
            widgets.preview_btn_frame,
            text="查看预览图",
            command=lambda: view_preview(app, widgets),
            width=10,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            widgets.preview_btn_frame,
            text="替换预览图",
            command=lambda: upload_preview(app, widgets),
            width=10,
        ).pack(side=tk.LEFT, padx=(0, 6))
    else:
        ttk.Button(
            widgets.preview_btn_frame,
            text="上传预览图",
            command=lambda: upload_preview(app, widgets),
            width=10,
        ).pack(side=tk.LEFT, padx=(0, 6))


def upload_preview(app: EndlessWinterApp, widgets: DreamTabWidgets) -> None:
    map_id = get_selected_map_id(widgets)
    if not map_id:
        messagebox.showwarning("提示", "请先选择地图")
        return
    path = filedialog.askopenfilename(
        title="选择地图预览图",
        filetypes=[
            ("图片", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"),
            ("所有文件", "*.*"),
        ],
    )
    if not path:
        return
    dm_cfg = load_dream_cfg(app, widgets.pk)
    try:
        out = save_map_preview(map_id, path, previews_dir=dm_cfg.previews_dir)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        messagebox.showerror("上传失败", str(exc))
        return
    update_summary(app, widgets)
    messagebox.showinfo("完成", f"预览图已保存:\n{out.name}")


def view_preview(app: EndlessWinterApp, widgets: DreamTabWidgets) -> None:
    map_id = get_selected_map_id(widgets)
    if not map_id:
        messagebox.showwarning("提示", "请先选择地图")
        return
    dm_cfg = load_dream_cfg(app, widgets.pk)
    preview_path = find_map_preview(map_id, previews_dir=dm_cfg.previews_dir)
    if preview_path is None:
        messagebox.showinfo("提示", "当前地图尚无预览图")
        update_preview_buttons(app, widgets)
        return
    try:
        from PIL import Image, ImageTk

        dream_map = load_map(map_id, maps_dir=dm_cfg.maps_dir)
    except FileNotFoundError:
        messagebox.showerror("错误", f"地图不存在: {map_id}")
        return
    except ImportError:
        messagebox.showerror("缺少依赖", "需要安装 Pillow")
        return

    win_attr = "_dream_pk_preview_window" if widgets.pk else "_dream_preview_window"
    old = getattr(app, win_attr, None)
    if old is not None and old.winfo_exists():
        old.destroy()

    win = tk.Toplevel(app)
    win.title(f"预览图 — {dream_map.name}")
    setattr(app, win_attr, win)

    try:
        image = Image.open(preview_path).convert("RGB")
    except OSError as exc:
        messagebox.showerror("打开失败", str(exc), parent=win)
        win.destroy()
        return

    max_w, max_h = 520, 780
    scale = min(max_w / image.width, max_h / image.height, 1.0)
    if scale < 1.0:
        disp_w = max(1, int(image.width * scale))
        disp_h = max(1, int(image.height * scale))
        image = image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)

    photo = ImageTk.PhotoImage(image)
    label = ttk.Label(win, image=photo)
    label.image = photo  # type: ignore[attr-defined]
    label.pack(padx=8, pady=8)

    def on_close() -> None:
        setattr(app, win_attr, None)
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)


def build_dream_tab(
    app: EndlessWinterApp,
    parent: ttk.Frame,
    *,
    pk: bool,
    on_map_changed: Callable[[], None],
    on_period_changed: Callable[[], None],
    on_start: Callable[[], None],
    on_stop: Callable[[], None],
    on_refresh: Callable[[], None],
    on_calibrate: Callable[[], None],
    on_rename: Callable[[], None],
    on_delete: Callable[[], None],
    config_section: str | None = None,
    show_map_tools: bool = True,
    show_summary: bool = True,
) -> DreamTabWidgets:
    dm_cfg = load_dream_cfg(app, pk)
    periods = list_map_periods(dm_cfg.maps_dir)
    period_labels = [format_period_choice(p) for p in periods]
    section = config_section or ("dream_memory_pk" if pk else "dream_memory")
    section_cfg = app.config.get(section, {}) if isinstance(app.config.get(section), dict) else {}
    saved_period = int(
        section_cfg.get("selected_period")
        or getattr(dm_cfg, "selected_period", CURRENT_MAP_PERIOD)
        or CURRENT_MAP_PERIOD
    )
    if saved_period not in periods:
        periods = sorted({*periods, saved_period})
        period_labels = [format_period_choice(p) for p in periods]
    if saved_period < 1:
        saved_period = CURRENT_MAP_PERIOD

    maps = list_maps(dm_cfg.maps_dir, period=saved_period)
    label_to_id, id_to_label = rebuild_map_index(maps)
    map_ids = [m.map_id for m in maps]
    labels = list(label_to_id.keys())

    saved = str(section_cfg.get("selected_map") or "")
    if saved not in map_ids and map_ids:
        saved = map_ids[0]
    if saved not in map_ids:
        saved = ""
    saved_label = id_to_label.get(saved, saved)

    var_period = tk.StringVar(value=format_period_choice(saved_period))
    var_map = tk.StringVar(value=saved_label)

    row0 = ttk.Frame(parent)
    row0.pack(fill=tk.X, pady=(0, 6))
    ttk.Label(row0, text="期数").pack(side=tk.LEFT)
    cmb_period = ttk.Combobox(
        row0,
        textvariable=var_period,
        values=period_labels,
        state="readonly",
        width=8,
    )
    cmb_period.pack(side=tk.LEFT, padx=(8, 12))
    ttk.Label(row0, text="地图").pack(side=tk.LEFT)
    cmb_map = ttk.Combobox(
        row0,
        textvariable=var_map,
        values=labels,
        state="readonly" if labels else "disabled",
        width=24,
    )
    cmb_map.pack(side=tk.LEFT, padx=(8, 8))

    lbl_summary = ttk.Label(
        parent,
        text="",
        foreground="#555",
        wraplength=480,
        justify=tk.LEFT,
    )
    if show_summary:
        lbl_summary.pack(anchor=tk.W, pady=(0, 4))

    interval_key = normalize_tap_between_interval(
        str(section_cfg.get("tap_between_delay_interval") or dm_cfg.tap_between_interval)
    )
    var_tap_interval = tk.StringVar(value=TAP_INTERVAL_LABELS[interval_key])
    row_interval = ttk.Frame(parent)
    row_interval.pack(fill=tk.X, pady=(0, 6))
    ttk.Label(row_interval, text="连点间隔").pack(side=tk.LEFT)
    cmb_tap_interval = ttk.Combobox(
        row_interval,
        textvariable=var_tap_interval,
        values=list(TAP_INTERVAL_LABELS.values()),
        state="readonly",
        width=8,
    )
    cmb_tap_interval.pack(side=tk.LEFT, padx=(8, 8))
    if section == "dream_memory_turbo_pk":
        from core.dream_memory.config import TURBO_PK_TAP_BETWEEN

        tap_min = float(section_cfg.get("tap_between_delay_min", TURBO_PK_TAP_BETWEEN))
        tap_max = float(section_cfg.get("tap_between_delay_max", TURBO_PK_TAP_BETWEEN))
    else:
        tap_min = dm_cfg.tap_between_delay_min
        tap_max = dm_cfg.tap_between_delay_max
    ttk.Label(
        row_interval,
        text=(
            f"固定=min({tap_min:g}s)，"
            f"随机={tap_min:g}~{tap_max:g}s"
        ),
        foreground="#555",
    ).pack(side=tk.LEFT)

    preview_row = ttk.Frame(parent)
    preview_row.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))
    preview_btn_frame = ttk.Frame(preview_row)
    preview_btn_frame.pack(side=tk.LEFT)

    widgets = DreamTabWidgets(
        pk=pk,
        var_period=var_period,
        cmb_period=cmb_period,
        var_map=var_map,
        cmb_map=cmb_map,
        lbl_summary=lbl_summary,
        preview_btn_frame=preview_btn_frame,
        lbl_ocr=ttk.Label(preview_row, text=""),
        btn_start=ttk.Button(parent, text="开始游戏"),
        btn_stop=ttk.Button(parent, text="结束"),
        var_tap_interval=var_tap_interval,
        cmb_tap_interval=cmb_tap_interval,
        config_section=section,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )

    update_summary(app, widgets)

    ttk.Button(row0, text="刷新", command=on_refresh, width=6).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    if show_map_tools:
        ttk.Button(row0, text="标定地图", command=on_calibrate, width=8).pack(
            side=tk.LEFT
        )

    # 重命名 / 删除 放到预览图同一行
    if show_map_tools:
        ttk.Button(preview_row, text="重命名", command=on_rename, width=7).pack(
            side=tk.LEFT, padx=(8, 6)
        )
        ttk.Button(preview_row, text="删除地图", command=on_delete, width=8).pack(
            side=tk.LEFT, padx=(0, 6)
        )

    from core.dream_memory.ocr_engine import ocr_engine_available, resolve_ocr_engine

    ocr_engine = resolve_ocr_engine(dm_cfg.ocr_engine)
    ocr_ok = ocr_engine_available(dm_cfg.ocr_engine)
    if ocr_engine == "rapidocr":
        ocr_text = "RapidOCR: 已就绪" if ocr_ok else "RapidOCR: 未安装"
    else:
        ocr_text = (
            f"Tesseract: 已就绪"
            if ocr_ok
            else f"Tesseract: 未找到"
        )
    widgets.lbl_ocr.configure(
        text=ocr_text,
        foreground="green" if ocr_ok else "red",
    )
    widgets.lbl_ocr.pack(side=tk.LEFT, padx=(10, 0))

    # 开始/结束统一在主界面运行区，此处按钮仅作内部状态兼容，不展示
    widgets.btn_start.configure(text="开始游戏", command=on_start, width=10)
    widgets.btn_stop.configure(text="结束", command=on_stop, width=10, state=tk.DISABLED)

    cmb_period.bind("<<ComboboxSelected>>", lambda _e: on_period_changed())
    cmb_map.bind("<<ComboboxSelected>>", lambda _e: on_map_changed())
    return widgets


def build_turbo_pk_controls(
    app: EndlessWinterApp,
    parent: ttk.Frame,
) -> TurboPkControls:
    """极速 PK：选雷电窗口，一次性框住整条六字栏（客户区 ROI）。"""
    from core.dream_memory.config import DEFAULT_PK_TARGET_BAR
    from core.window_capture import (
        WindowCapture,
        list_ldplayer_windows,
        touch_roi_to_client_xywh,
    )

    turbo_cfg = app.config.get("dream_memory_turbo_pk", {})
    if not isinstance(turbo_cfg, dict):
        turbo_cfg = {}

    tip = ttk.Label(
        parent,
        text="盯底栏六字 ROI；默认底栏换算；未见字 0.1s / 见字后 0.5s；勿遮挡画面。",
        foreground="#555",
    )
    tip.pack(anchor=tk.W, pady=(0, 4))

    row = ttk.Frame(parent)
    row.pack(fill=tk.X, pady=(0, 4))
    ttk.Label(row, text="窗口").pack(side=tk.LEFT)
    var_window = tk.StringVar(value="")
    cmb_window = ttk.Combobox(row, textvariable=var_window, width=36, state="readonly")
    cmb_window.pack(side=tk.LEFT, padx=(8, 6))

    saved_roi = turbo_cfg.get("bar_roi")
    if isinstance(saved_roi, (list, tuple)) and len(saved_roi) == 4:
        rx, ry, rw, rh = (int(v) for v in saved_roi)
    else:
        rx, ry, rw, rh = 0, 0, 0, 0

    var_x = tk.IntVar(value=rx)
    var_y = tk.IntVar(value=ry)
    var_w = tk.IntVar(value=rw)
    var_h = tk.IntVar(value=rh)
    controls = TurboPkControls(
        var_window=var_window,
        cmb_window=cmb_window,
        var_x=var_x,
        var_y=var_y,
        var_w=var_w,
        var_h=var_h,
    )

    def refresh_windows() -> None:
        controls.windows = list_ldplayer_windows()
        labels = [w.label for w in controls.windows]
        cmb_window.configure(values=labels)
        saved_title = str(turbo_cfg.get("window_title") or "")
        chosen = ""
        if saved_title:
            for win in controls.windows:
                if win.title == saved_title or saved_title in win.title:
                    chosen = win.label
                    break
        if not chosen and labels:
            chosen = labels[0]
        var_window.set(chosen)

    def apply_default_bar(*, silent: bool = False) -> None:
        hwnd = controls.selected_hwnd()
        if hwnd is None:
            if not silent:
                messagebox.showwarning("提示", "请先选择雷电窗口")
            return
        try:
            cap = WindowCapture()
            crect = cap.set_window(hwnd)
            cap.close()
            x, y, w, h = touch_roi_to_client_xywh(
                DEFAULT_PK_TARGET_BAR,
                client_w=crect.width,
                client_h=crect.height,
            )
        except Exception as exc:
            if not silent:
                messagebox.showerror("换算失败", str(exc))
            return
        var_x.set(x)
        var_y.set(y)
        var_w.set(w)
        var_h.set(h)
        if h < 40 or w < 200:
            if not silent:
                messagebox.showwarning(
                    "ROI 异常偏小",
                    f"当前换算结果约 {w}×{h}，客户区约 {crect.width}×{crect.height}。\n"
                    "多半选错了窗口（要选游戏画面那扇，不要选很小的子窗口）。\n"
                    "请换窗口后再点「默认底栏」。",
                )
        if not silent:
            try:
                app._save_config()
            except Exception:
                pass

    ttk.Button(row, text="刷新", width=6, command=refresh_windows).pack(side=tk.LEFT)
    ttk.Button(row, text="默认底栏", width=8, command=apply_default_bar).pack(
        side=tk.LEFT, padx=(6, 0)
    )

    roi_row = ttk.Frame(parent)
    roi_row.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(roi_row, text="底栏ROI x,y,w,h").pack(side=tk.LEFT)
    for var, width in (
        (var_x, 5),
        (var_y, 5),
        (var_w, 5),
        (var_h, 5),
    ):
        ttk.Spinbox(
            roi_row, from_=0, to=4000, textvariable=var, width=width
        ).pack(side=tk.LEFT, padx=(4, 0))

    refresh_windows()
    if var_w.get() <= 0 or var_h.get() <= 0:
        apply_default_bar(silent=True)
    return controls
