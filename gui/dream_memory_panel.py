"""寻梦记忆 / 寻梦记忆PK Tab 共用逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from core.dream_memory.config import (
    PK_ITEM_FILTER_LABELS,
    load_dream_memory_config,
    load_dream_memory_pk_config,
    normalize_pk_item_filter,
)
from core.dream_memory.maps import (
    find_map_preview,
    format_map_choice,
    has_map_preview,
    list_maps,
    load_map,
    save_map_preview,
)

if TYPE_CHECKING:
    from gui.app import EndlessWinterApp


@dataclass
class DreamTabWidgets:
    pk: bool
    var_map: tk.StringVar
    cmb_map: ttk.Combobox
    lbl_summary: ttk.Label
    preview_btn_frame: ttk.Frame
    lbl_ocr: ttk.Label
    btn_start: ttk.Button
    btn_stop: ttk.Button
    var_item_filter: tk.StringVar | None = None
    label_to_id: dict[str, str] = field(default_factory=dict)
    id_to_label: dict[str, str] = field(default_factory=dict)


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


def set_map_selection(widgets: DreamTabWidgets, map_id: str) -> None:
    label = widgets.id_to_label.get(map_id, map_id)
    widgets.var_map.set(label)


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
        filter_hint = ""
        if widgets.pk and widgets.var_item_filter is not None:
            mode = normalize_pk_item_filter(widgets.var_item_filter.get())
            if mode != "all":
                filter_hint = f" · 分工 {PK_ITEM_FILTER_LABELS.get(mode, mode)}"
        widgets.lbl_summary.configure(
            text=(
                f"显示名：{dream_map.name} · 已标定 {len(dream_map.items)} 个物品"
                f" · 识别区 {slot_hint}{preview_hint}{filter_hint}"
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
        ).pack(side=tk.LEFT)
    else:
        ttk.Button(
            widgets.preview_btn_frame,
            text="上传预览图",
            command=lambda: upload_preview(app, widgets),
            width=10,
        ).pack(side=tk.LEFT)


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
    on_start: Callable[[], None],
    on_stop: Callable[[], None],
    on_refresh: Callable[[], None],
    on_calibrate: Callable[[], None],
    on_rename: Callable[[], None],
    on_delete: Callable[[], None],
) -> DreamTabWidgets:
    dm_cfg = load_dream_cfg(app, pk)
    maps = list_maps(dm_cfg.maps_dir)
    label_to_id, id_to_label = rebuild_map_index(maps)
    map_ids = [m.map_id for m in maps]
    labels = list(label_to_id.keys())

    section = "dream_memory_pk" if pk else "dream_memory"
    saved = str(app.config.get(section, {}).get("selected_map") or "")
    if saved not in map_ids and map_ids:
        saved = map_ids[0]
    saved_label = id_to_label.get(saved, saved)

    var_map = tk.StringVar(value=saved_label)

    row0 = ttk.Frame(parent)
    row0.pack(fill=tk.X, pady=(0, 6))
    ttk.Label(row0, text="地图").pack(side=tk.LEFT)
    cmb_map = ttk.Combobox(
        row0,
        textvariable=var_map,
        values=labels,
        state="readonly" if labels else "disabled",
        width=28,
    )
    cmb_map.pack(side=tk.LEFT, padx=(8, 8))

    lbl_summary = ttk.Label(
        parent,
        text="",
        foreground="#555",
        wraplength=480,
        justify=tk.LEFT,
    )
    lbl_summary.pack(anchor=tk.W, pady=(0, 4))

    preview_btn_frame = ttk.Frame(parent)
    preview_btn_frame.pack(anchor=tk.W, pady=(0, 6))

    widgets = DreamTabWidgets(
        pk=pk,
        var_map=var_map,
        cmb_map=cmb_map,
        lbl_summary=lbl_summary,
        preview_btn_frame=preview_btn_frame,
        lbl_ocr=ttk.Label(parent, text=""),
        btn_start=ttk.Button(parent, text="开始游戏"),
        btn_stop=ttk.Button(parent, text="结束"),
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )

    update_summary(app, widgets)

    ttk.Button(row0, text="刷新", command=on_refresh, width=6).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    ttk.Button(row0, text="标定地图", command=on_calibrate, width=8).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    ttk.Button(row0, text="重命名", command=on_rename, width=7).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    ttk.Button(row0, text="删除地图", command=on_delete, width=8).pack(side=tk.LEFT)

    from core.dream_memory.ocr_engine import ocr_engine_available, resolve_ocr_engine

    ocr_engine = resolve_ocr_engine(dm_cfg.ocr_engine)
    ocr_ok = ocr_engine_available(dm_cfg.ocr_engine)
    if ocr_engine == "rapidocr":
        ocr_text = "RapidOCR: 已就绪（推荐，中文游戏字体）" if ocr_ok else (
            "RapidOCR: 未安装，请运行 pip install rapidocr-onnxruntime onnxruntime"
        )
    else:
        ocr_text = (
            f"Tesseract: 已就绪 ({dm_cfg.tesseract_cmd})"
            if ocr_ok
            else f"Tesseract: 未找到 ({dm_cfg.tesseract_cmd})"
        )
    widgets.lbl_ocr.configure(
        text=ocr_text,
        foreground="green" if ocr_ok else "red",
        wraplength=480,
        justify=tk.LEFT,
    )
    widgets.lbl_ocr.pack(anchor=tk.W, pady=(0, 6))

    if pk:
        filter_row = ttk.Frame(parent)
        filter_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(filter_row, text="标定点分工").pack(side=tk.LEFT)
        saved_filter = normalize_pk_item_filter(
            str(app.config.get(section, {}).get("pk_item_filter") or "all")
        )
        var_item_filter = tk.StringVar(
            value=PK_ITEM_FILTER_LABELS.get(saved_filter, "全部")
        )
        cmb_filter = ttk.Combobox(
            filter_row,
            textvariable=var_item_filter,
            values=[PK_ITEM_FILTER_LABELS[k] for k in ("all", "odd", "even")],
            state="readonly",
            width=10,
        )
        cmb_filter.pack(side=tk.LEFT, padx=(8, 0))
        cmb_filter.bind("<<ComboboxSelected>>", lambda _e: on_map_changed())
        widgets.var_item_filter = var_item_filter
        ttk.Label(
            filter_row,
            text="（按标定顺序：单数=第1/3/5…，双数=第2/4/6…，双开各选一项）",
            foreground="gray",
        ).pack(side=tk.LEFT, padx=(8, 0))

    if pk:
        usage = (
            "PK 模式：扫描与点击解耦。每个物品整局只入队一次；"
            "双开时一个选「单数」、另一个选「双数」即可各点各的。"
        )
    else:
        usage = (
            "用法：模拟器内手动进入寻梦记忆关卡 → 本页选地图 → 点「开始游戏」"
            " → 脚本 OCR 底栏并点击 → 关卡结束点「结束」。"
            "约每 8–12 次正常点击会随机误点一次屏幕中心附近。"
        )
    ttk.Label(
        parent,
        text=usage,
        wraplength=480,
        justify=tk.LEFT,
        foreground="gray",
    ).pack(anchor=tk.W, pady=(0, 8))

    btn_row = ttk.Frame(parent)
    btn_row.pack(fill=tk.X)
    widgets.btn_start.configure(text="开始游戏", command=on_start, width=10)
    widgets.btn_start.pack(side=tk.LEFT, padx=(0, 8))
    widgets.btn_stop.configure(text="结束", command=on_stop, width=10, state=tk.DISABLED)
    widgets.btn_stop.pack(side=tk.LEFT)

    cal_hint = "标定：选地图 → 点「标定地图」→ 模拟器截图 → 点击物品 → 填名称 → 写入地图"
    ttk.Label(
        parent,
        text=cal_hint,
        font=("", 8),
        foreground="gray",
        wraplength=480,
        justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=(8, 0))

    cmb_map.bind("<<ComboboxSelected>>", lambda _e: on_map_changed())
    return widgets
