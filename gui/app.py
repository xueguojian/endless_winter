"""无尽冬日 — 图形界面。"""

from __future__ import annotations

import queue
import random
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, simpledialog, ttk

import cv2
import yaml

from core.adb_client import AdbClient
from core.common_task_opts import apply_common_options, resolve_common_options
from core.config_path import (
    default_instance_name,
    ensure_config_file,
    resolve_config_path,
)
from core.dream_memory.config import load_dream_memory_config, load_dream_memory_pk_config
from core.dream_memory.maps import delete_map, list_maps, load_map, rename_map_name
from core.dream_memory.ocr_engine import (
    ocr_chip_text,
    ocr_engine_available,
    resolve_ocr_engine,
    warmup_ocr,
)
from core.navigation import WildernessNavigator, return_to_main_screen
from gui.coord_ruler import CoordRulerWindow
from gui.dream_memory_calibrator import DreamMemoryCalibratorWindow
from gui.dream_memory_panel import (
    DreamTabWidgets,
    build_dream_tab,
    get_selected_map_id,
    get_tap_interval_mode,
    load_dream_cfg,
    rebuild_map_index,
    set_map_selection,
    update_summary,
)
from gui.task_registry import TaskEntry, HOSTING_TASK_IDS, loop_tasks, once_tasks, TASK_ENTRIES
from tasks.alliance_mobilization import (
    CARD_BG_LABELS,
    CARD_BG_ORANGE,
    CARD_BG_PURPLE,
    GUI_ALLIANCE_TYPE_ORDER,
    KEEPABLE_BG_COLORS,
    TASK_TYPE_LABELS,
    AllianceMobilizationSession,
    merge_task_config as merge_alliance_config,
    normalize_type_keep_rules,
)
from tasks.alliance_mobilization_admin import (
    AllianceMobilizationAdminSession,
    CALIBRATED_DETAIL_MASK_TAP,
    merge_task_config as merge_alliance_admin_config,
)
from tasks.auto_lighthouse import AutoLighthouseTask, merge_task_config as merge_lighthouse_config
from tasks.auto_mining import AutoMiningTask, merge_task_config as merge_mining_config
from tasks.auto_train_troops import (
    AutoTrainTroopsTask,
    merge_task_config as merge_train_config,
)
from tasks.collect_commander_supplies import (
    CollectCommanderSuppliesTask,
    merge_task_config as merge_commander_config,
)
from tasks.auto_shop_exchange import (
    AutoShopExchangeTask,
    merge_task_config as merge_shop_exchange_config,
)
from tasks.collect_pet_supplies import (
    CollectPetSuppliesTask,
    merge_task_config as merge_pet_supplies_config,
)
from tasks.collect_supplies import CollectSuppliesTask, merge_task_config as merge_collect_config
from tasks.donate_alliance_supplies import (
    DonateAllianceSuppliesTask,
    merge_task_config as merge_donate_config,
)
from tasks.dream_memory import DreamMemorySession
from tasks.hunt_ice_beast import HuntIceBeastTask
from tasks.hunt_monster import HuntMonsterTask

ROOT = Path(__file__).parent.parent

# 冰原巨兽 / 打野 与其余循环任务互斥；捐献/采集/练兵/领取探险物资 可同时勾选
LOOP_COMBAT_EXCLUSIVE_TASK_IDS = frozenset({"hunt_ice_beast", "hunt_monster"})

# 运行任务 Tab → 功能参数 Tab
TASK_TAB_TO_PARAM_TAB: dict[str, str] = {
    "循环任务": "冰原巨兽",
    "一次性任务": "灯塔任务",
    "联盟总动员自动刷新": "联盟总动员",
    "联盟管理员刷新": "联盟总动员",
}

MAIN_WIDTH = 580
MAIN_HEIGHT = 700
LOG_WIDTH = 300
ONCE_TASK_GAP_SEC = 3
ONCE_TASK_COLUMNS = 2
LOOP_TASK_COLUMNS = 2

# 启动时随机挑选一个窗口标题
WINDOW_TITLE_POOL: tuple[str, ...] = (
    "永冬裁决 · 极寒统御者",
    "霜原统帅 · 无尽寒锋",
    "极寒主宰 · 永夜猎杀令",
    "永冬猎手 · 霜刃裁决台",
    "无尽寒锋 · 冰原指挥官",
    "霜月裁决 · 永冬核心引擎",
    "极地统御 · 寒域主宰者",
    "永冬引擎 · 霜原猎杀者",
    "寒锋破晓 · 无尽冬日统帅",
    "霜刃永夜 · 极寒裁决者",
    "永冬裂隙 · 冰原征服者",
    "寒域敕令 · 霜原铁腕",
)

# 个人主页：头像入口与昵称 OCR 区域（720×1280）
ACCOUNT_PROFILE_TAP = (42, 52)
ACCOUNT_NAME_ROI = (284, 878, 606, 926)

FORMATION_SLOT_MIN = 1
FORMATION_SLOT_MAX = 8

# 功能参数 Tab：标签列最小宽度 + 输入框左间距
FORM_LABEL_COL_MINSIZE = 120
FORM_INPUT_PADX = (14, 0)

MINING_LEVEL_MIN = 1
MINING_LEVEL_MAX = 8
MINING_LEVEL_DEFAULT_MIN = 8
MINING_LEVEL_DEFAULT_MAX = 8


def _normalize_formation_slot(raw) -> int:
    """将配置中的编队槽位规范为 1~8。"""
    try:
        slot = int(str(raw).strip())
    except (TypeError, ValueError):
        slot = 7
    return max(FORMATION_SLOT_MIN, min(FORMATION_SLOT_MAX, slot))


def _normalize_mining_level(raw, default: int = MINING_LEVEL_MIN) -> int:
    """将采矿等级规范为 1~8。"""
    try:
        level = int(str(raw).strip())
    except (TypeError, ValueError):
        level = default
    return max(MINING_LEVEL_MIN, min(MINING_LEVEL_MAX, level))


def _normalize_mining_range(level_min, level_max) -> tuple[int, int]:
    """规范采矿范围，允许 min == max。"""
    low = _normalize_mining_level(level_min, MINING_LEVEL_DEFAULT_MIN)
    high = _normalize_mining_level(level_max, MINING_LEVEL_DEFAULT_MAX)
    if high < low:
        low, high = high, low
    return low, high


def _configure_param_tab_grid(tab: ttk.Frame) -> None:
    tab.grid_columnconfigure(0, minsize=FORM_LABEL_COL_MINSIZE)


def _bind_mousewheel_recursive(widget: tk.Misc, handler) -> None:
    widget.bind("<MouseWheel>", handler)
    widget.bind("<Button-4>", handler)
    widget.bind("<Button-5>", handler)
    for child in widget.winfo_children():
        _bind_mousewheel_recursive(child, handler)


def _create_scrollable_frame(
    parent: tk.Misc,
    *,
    height: int = 240,
) -> tuple[ttk.Frame, ttk.Frame, tk.Canvas]:
    """返回 (inner, container, canvas)，支持滚动条 + 鼠标滚轮。"""
    container = ttk.Frame(parent)
    canvas = tk.Canvas(
        container,
        height=height,
        highlightthickness=1,
        highlightbackground="#cccccc",
    )
    scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
    inner = ttk.Frame(canvas)
    window_id = canvas.create_window((0, 0), window=inner, anchor=tk.NW)

    def _sync_scrollregion(_event=None) -> None:
        canvas.update_idletasks()
        bbox = canvas.bbox("all")
        if bbox:
            canvas.configure(scrollregion=bbox)

    def _sync_width(event) -> None:
        canvas.itemconfigure(window_id, width=event.width)

    def _on_mousewheel(event) -> None:
        if getattr(event, "delta", 0):
            canvas.yview_scroll(int(-event.delta / 120), "units")
        elif getattr(event, "num", None) == 4:
            canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(3, "units")

    inner.bind("<Configure>", _sync_scrollregion)
    canvas.bind("<Configure>", _sync_width)
    _bind_mousewheel_recursive(container, _on_mousewheel)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    return inner, container, canvas


class EndlessWinterApp(tk.Tk):
    def __init__(self, config_path: Path | None = None):
        super().__init__()
        self.config_path = ensure_config_file(resolve_config_path(config_path))
        self.config = self._load_config()

        self._update_window_title()
        self.geometry(f"{MAIN_WIDTH}x{MAIN_HEIGHT}")
        self.resizable(False, False)

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._loop_worker: threading.Thread | None = None
        self._once_worker: threading.Thread | None = None
        self._loop_stop_event = threading.Event()
        self._once_stop_event = threading.Event()
        self._loop_tasks: list = []
        self._once_tasks: list = []
        self._task_vars: dict[str, tk.BooleanVar] = {}
        self._adb: AdbClient | None = None
        self._log_visible = True
        self._coord_ruler_window: CoordRulerWindow | None = None
        self._dream_memory_calibrator: DreamMemoryCalibratorWindow | None = None
        self._dream_pk_calibrator: DreamMemoryCalibratorWindow | None = None
        self._dream_widgets: DreamTabWidgets | None = None
        self._dream_pk_widgets: DreamTabWidgets | None = None
        self._dream_memory_worker: threading.Thread | None = None
        self._dream_pk_worker: threading.Thread | None = None
        self._dream_memory_stop_event = threading.Event()
        self._dream_pk_stop_event = threading.Event()
        self._dream_memory_session: DreamMemorySession | None = None
        self._dream_pk_session: DreamMemorySession | None = None
        self._dream_preview_window: tk.Toplevel | None = None
        self._dream_pk_preview_window: tk.Toplevel | None = None
        self._alliance_worker: threading.Thread | None = None
        self._alliance_session: AllianceMobilizationSession | None = None
        self._alliance_admin_worker: threading.Thread | None = None
        self._alliance_admin_session: AllianceMobilizationAdminSession | None = None
        self._alliance_type_bg_vars: dict[str, dict[str, tk.BooleanVar]] = {}
        self._account_check_worker: threading.Thread | None = None
        self._account_check_stop_event = threading.Event()

        self._build_ui()
        self._poll_log_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(300, lambda: self._refresh_devices(probe=False))

    def _load_config(self) -> dict:
        with open(self.config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _account_name_value(self) -> str:
        gui = self.config.get("gui", {})
        name = str(gui.get("account_name", "")).strip()
        return name

    def _update_window_title(self) -> None:
        self.title(random.choice(WINDOW_TITLE_POOL))

    def _set_account_name_display(self, name: str) -> None:
        text = name.strip() if name and name.strip() else "未知"
        if hasattr(self, "lbl_account_name"):
            self.lbl_account_name.configure(text=text)

    def _task_enabled_in_config(self, entry: TaskEntry) -> bool:
        task_cfg = self.config.get("tasks", {}).get(entry.config_key, {})
        if entry.task_id == "hunt_ice_beast":
            return bool(task_cfg.get("enabled", True))
        if entry.task_id == "hunt_monster":
            return bool(task_cfg.get("enabled", False))
        if entry.task_id == "donate_alliance_supplies":
            return bool(task_cfg.get("enabled", False))
        return bool(task_cfg.get("enabled", False))

    def _collect_alliance_type_keep_rules(self) -> dict[str, list[str]]:
        rules: dict[str, list[str]] = {}
        if not self._alliance_type_bg_vars:
            return rules
        for type_id in GUI_ALLIANCE_TYPE_ORDER:
            bg_vars = self._alliance_type_bg_vars.get(type_id, {})
            colors = [
                color_id
                for color_id in KEEPABLE_BG_COLORS
                if bg_vars.get(color_id) is not None and bool(bg_vars[color_id].get())
            ]
            if colors:
                rules[type_id] = colors
        return rules

    def _format_type_keep_rules(self, rules: dict[str, list[str]]) -> str:
        parts: list[str] = []
        for type_id in GUI_ALLIANCE_TYPE_ORDER:
            colors = rules.get(type_id)
            if not colors:
                continue
            bg_text = "、".join(
                CARD_BG_LABELS[c] for c in KEEPABLE_BG_COLORS if c in colors
            )
            parts.append(f"{TASK_TYPE_LABELS[type_id]}[{bg_text}]")
        return "；".join(parts) if parts else "无"

    def _save_config(self) -> None:
        cfg = self.config
        tasks = cfg.setdefault("tasks", {})

        common_opts = {
            "use_formation": bool(self.var_common_use_formation.get()),
            "adjust_level": bool(self.var_common_adjust_level.get()),
            "use_stamina": bool(self.var_common_use_stamina.get()),
            "stamina_can_limit": int(self.var_common_stamina_can_limit.get()),
        }
        apply_common_options(tasks, common_opts)

        ice = tasks.setdefault("hunt_ice_beast", {})
        ice["enabled"] = bool(self._task_vars["hunt_ice_beast"].get())
        ice["interval"] = int(self.var_interval.get()) * 60
        ice["beast_level"] = int(self.var_level.get())
        ice["formation_name"] = str(_normalize_formation_slot(self.var_formation_slot.get()))
        ice["rally_duration_minutes"] = int(self.var_rally_duration.get())

        lighthouse = tasks.setdefault("auto_lighthouse", {})
        lighthouse["enabled"] = bool(self._task_vars["auto_lighthouse"].get())
        lighthouse["interval"] = int(self.var_lighthouse_interval.get()) * 60
        lighthouse["formation_slot"] = int(
            _normalize_formation_slot(self.var_lighthouse_formation_slot.get())
        )
        lighthouse["event_period"] = bool(self.var_lighthouse_event_period.get())
        lighthouse["monster_cooldown"] = int(self.var_lighthouse_monster_cooldown.get()) * 60
        merged_lighthouse = merge_lighthouse_config(lighthouse)
        lighthouse["step_delay"] = merged_lighthouse["step_delay"]
        lighthouse["coords"] = merged_lighthouse["coords"]

        monster = tasks.setdefault("hunt_monster", {})
        monster["enabled"] = bool(self._task_vars["hunt_monster"].get())
        monster["interval"] = int(self.var_monster_interval.get()) * 60
        monster["monster_level"] = int(self.var_monster_level.get())
        monster["formation_name"] = str(
            _normalize_formation_slot(self.var_monster_formation_slot.get())
        )
        donate = tasks.setdefault("donate_alliance_supplies", {})
        donate["enabled"] = bool(self._task_vars["donate_alliance_supplies"].get())
        donate["interval"] = int(self.var_donate_interval.get()) * 60
        donate["donate_times"] = int(self.var_donate_times.get())
        merged_donate = merge_donate_config(donate)
        donate.setdefault("step_delay", merged_donate["step_delay"])
        donate["coords"] = merged_donate["coords"]

        mining = tasks.setdefault("auto_mining", {})
        level_min = _normalize_mining_level(self.var_mining_level_min.get())
        level_max = _normalize_mining_level(self.var_mining_level_max.get())
        if level_max < level_min:
            raise ValueError("采矿范围无效：最高等级不能低于最低等级")
        mining["enabled"] = bool(self._task_vars["auto_mining"].get())
        mining["interval"] = int(self.var_mining_interval.get()) * 60
        mining["use_mining_hero"] = bool(self.var_use_mining_hero.get())
        mining["level_min"] = level_min
        mining["level_max"] = level_max
        merged_mining = merge_mining_config(mining)
        mining["step_delay"] = merged_mining["step_delay"]
        mining["hero_roi"] = list(merged_mining["hero_roi"])
        mining["hero_match_threshold"] = merged_mining["hero_match_threshold"]
        mining["coords"] = merged_mining["coords"]

        collect = tasks.setdefault("collect_supplies", {})
        collect["enabled"] = bool(self._task_vars["collect_supplies"].get())
        collect["interval"] = int(self.var_collect_interval.get()) * 3600
        merged_collect = merge_collect_config(collect)
        collect["step_delay"] = merged_collect["step_delay"]
        collect["coords"] = merged_collect["coords"]

        train = tasks.setdefault("auto_train_troops", {})
        train["enabled"] = bool(self._task_vars["auto_train_troops"].get())
        train["interval"] = int(self.var_train_interval.get()) * 3600
        merged_train = merge_train_config(train)
        train["step_delay"] = merged_train["step_delay"]
        train["train_ready_threshold"] = merged_train["train_ready_threshold"]
        train["coords"] = merged_train["coords"]

        for entry in once_tasks():
            section = tasks.setdefault(entry.config_key, {})
            if entry.task_id in self._task_vars:
                section["enabled"] = bool(self._task_vars[entry.task_id].get())
            if entry.task_id == "collect_commander_supplies":
                merged = merge_commander_config(section)
                section["step_delay"] = merged["step_delay"]
                section["double_tap_delay"] = merged["double_tap_delay"]
                section["coords"] = merged["coords"]
            if entry.task_id == "collect_pet_supplies":
                merged = merge_pet_supplies_config(section)
                section["step_delay"] = merged["step_delay"]
                section["coords"] = merged["coords"]
            if entry.task_id == "auto_shop_exchange":
                merged = merge_shop_exchange_config(section)
                section["step_delay"] = merged["step_delay"]
                section["coords"] = merged["coords"]

        gui = cfg.setdefault("gui", {})
        gui["show_console"] = bool(self.var_show_console.get())
        gui["log_visible"] = self._log_visible
        gui["instance_name"] = default_instance_name(cfg, self.config_path)
        account = self._account_name_value()
        if hasattr(self, "lbl_account_name"):
            shown = self.lbl_account_name.cget("text").strip()
            if shown and shown not in ("未知", "未识别"):
                account = shown
        if account:
            gui["account_name"] = account
        elif "account_name" not in gui:
            gui["account_name"] = ""

        if hasattr(self, "var_device_serial"):
            serial = self.var_device_serial.get().strip()
            if serial:
                host, port = AdbClient.parse_address(serial)
                dev = cfg.setdefault("device", {})
                dev["adb_host"] = host
                dev["adb_port"] = int(port)

        if self._dream_widgets is not None:
            dm = cfg.setdefault("dream_memory", {})
            dm["selected_map"] = get_selected_map_id(self._dream_widgets)
            dm["tap_between_delay_interval"] = get_tap_interval_mode(self._dream_widgets)
        if self._dream_pk_widgets is not None:
            pk = cfg.setdefault("dream_memory_pk", {})
            pk["selected_map"] = get_selected_map_id(self._dream_pk_widgets)
            pk["tap_between_delay_interval"] = get_tap_interval_mode(self._dream_pk_widgets)

        alliance = cfg.setdefault("alliance_mobilization", {})
        type_keep_rules = self._collect_alliance_type_keep_rules()
        selected_types = list(type_keep_rules.keys())
        if not selected_types:
            type_keep_rules = normalize_type_keep_rules({})
            selected_types = list(type_keep_rules.keys())
        alliance["type_keep_rules"] = {
            tid: {"bg_colors": colors} for tid, colors in type_keep_rules.items()
        }
        alliance["target_types"] = selected_types
        if hasattr(self, "var_alliance_scan_minutes"):
            alliance["scan_interval"] = float(self.var_alliance_scan_minutes.get()) * 60.0
        merged_alliance = merge_alliance_config(alliance)
        alliance["scan_interval"] = merged_alliance["scan_interval"]
        alliance["step_delay"] = merged_alliance["step_delay"]
        alliance["type_keep_rules"] = {
            tid: {"bg_colors": colors}
            for tid, colors in merged_alliance["type_keep_rules"].items()
        }
        alliance["target_types"] = list(merged_alliance["target_types"])
        alliance["match_threshold"] = merged_alliance["match_threshold"]
        alliance["countdown_threshold"] = merged_alliance["countdown_threshold"]
        alliance["ocr_engine"] = merged_alliance["ocr_engine"]
        alliance["coords"] = merged_alliance["coords"]
        # slots 使用代码默认值，不写入配置，避免 yaml 序列化 Python 对象
        alliance.pop("slots", None)
        # 分数阈值不再由 GUI 配置，保留文件内已有值或默认

        alliance_admin = cfg.setdefault("alliance_mobilization_admin", {})
        alliance_admin["type_keep_rules"] = dict(alliance["type_keep_rules"])
        alliance_admin["target_types"] = list(selected_types)
        alliance_admin["keep_orange_types"] = list(selected_types)
        if hasattr(self, "var_alliance_admin_scan_minutes"):
            alliance_admin["scan_interval"] = (
                float(self.var_alliance_admin_scan_minutes.get()) * 60.0
            )
        merged_admin = merge_alliance_admin_config(alliance_admin)
        alliance_admin["scan_interval"] = merged_admin["scan_interval"]
        alliance_admin["step_delay"] = merged_admin["step_delay"]
        alliance_admin["match_threshold"] = merged_admin["match_threshold"]
        alliance_admin["countdown_threshold"] = merged_admin["countdown_threshold"]
        alliance_admin["ocr_engine"] = merged_admin["ocr_engine"]
        alliance_admin["coords"] = merged_admin["coords"]
        alliance_admin["list_roi"] = list(merged_admin["list_roi"])
        alliance_admin["exclude_top_px"] = merged_admin["exclude_top_px"]
        alliance_admin["column_count"] = merged_admin["column_count"]
        alliance_admin["scroll"] = merged_admin["scroll"]
        alliance_admin["detail_refresh_btn_roi"] = list(
            merged_admin["detail_refresh_btn_roi"]
        )
        alliance_admin["detail_icon_roi"] = list(merged_admin["detail_icon_roi"])
        alliance_admin["detail_score_roi"] = list(merged_admin["detail_score_roi"])
        alliance_admin["detail_title_roi"] = list(merged_admin["detail_title_roi"])
        alliance_admin["detail_close_btn_roi"] = list(
            merged_admin["detail_close_btn_roi"]
        )
        alliance_admin["detail_match_threshold"] = merged_admin[
            "detail_match_threshold"
        ]
        alliance_admin["keep_orange_types"] = list(merged_admin["keep_orange_types"])
        alliance_admin["type_keep_rules"] = {
            tid: {"bg_colors": colors}
            for tid, colors in merged_admin["type_keep_rules"].items()
        }
        alliance_admin.pop("keep_bg_colors", None)
        alliance_admin["use_score_ocr"] = False
        alliance_admin["target_types"] = list(merged_admin.get("target_types") or selected_types)

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    def _update_formation_spinboxes_state(self) -> None:
        enabled = bool(self.var_common_use_formation.get())
        for spin in (
            getattr(self, "_ice_formation_spinbox", None),
            getattr(self, "_monster_formation_spinbox", None),
            getattr(self, "_lighthouse_formation_spinbox", None),
        ):
            if spin is None:
                continue
            spin.state(["!disabled"] if enabled else ["disabled"])

    def _update_common_stamina_limit_state(self) -> None:
        self._set_stamina_limit_spin_state(
            getattr(self, "spn_common_stamina_can_limit", None),
            bool(self.var_common_use_stamina.get()),
        )

    def _on_common_use_formation_changed(self) -> None:
        self._update_formation_spinboxes_state()
        try:
            self._save_config()
        except ValueError:
            pass

    def _on_common_use_stamina_changed(self) -> None:
        self._update_common_stamina_limit_state()
        self._save_config()

    def _update_ice_formation_state(self) -> None:
        self._update_formation_spinboxes_state()

    def _set_stamina_limit_spin_state(self, spinbox: ttk.Spinbox | None, enabled: bool) -> None:
        if spinbox is None:
            return
        spinbox.state(["!disabled"] if enabled else ["disabled"])

    def _update_ice_stamina_limit_state(self) -> None:
        self._update_common_stamina_limit_state()

    def _update_monster_stamina_limit_state(self) -> None:
        self._update_common_stamina_limit_state()

    def _update_lighthouse_stamina_limit_state(self) -> None:
        self._update_common_stamina_limit_state()

    def _on_ice_use_stamina_changed(self) -> None:
        self._on_common_use_stamina_changed()

    def _on_monster_use_stamina_changed(self) -> None:
        self._on_common_use_stamina_changed()

    def _on_lighthouse_use_stamina_changed(self) -> None:
        self._on_common_use_stamina_changed()

    def _update_monster_formation_state(self) -> None:
        self._update_formation_spinboxes_state()

    def _on_use_formation_changed(self) -> None:
        self._on_common_use_formation_changed()

    def _on_monster_use_formation_changed(self) -> None:
        self._on_common_use_formation_changed()

    def _select_param_tab_by_name(self, tab_name: str) -> None:
        notebook = getattr(self, "_param_notebook", None)
        if notebook is None:
            return
        for tab_id in notebook.tabs():
            if notebook.tab(tab_id, "text") == tab_name:
                notebook.select(tab_id)
                return

    def _on_task_notebook_changed(self, _event=None) -> None:
        notebook = getattr(self, "_task_notebook", None)
        if notebook is None:
            return
        try:
            tab_id = notebook.select()
            task_tab_name = notebook.tab(tab_id, "text")
        except tk.TclError:
            return
        param_tab = TASK_TAB_TO_PARAM_TAB.get(task_tab_name)
        if param_tab:
            self._select_param_tab_by_name(param_tab)

    def _on_loop_checkbox_changed(self, task_id: str) -> None:
        """冰原巨兽/打野与其他循环任务互斥；其余四个可同时勾选。"""
        if not self._task_vars[task_id].get():
            return
        all_loop_ids = [entry.task_id for entry in loop_tasks()]
        if task_id in LOOP_COMBAT_EXCLUSIVE_TASK_IDS:
            for other_id in all_loop_ids:
                if other_id != task_id and other_id in self._task_vars:
                    self._task_vars[other_id].set(False)
        else:
            for combat_id in LOOP_COMBAT_EXCLUSIVE_TASK_IDS:
                if combat_id in self._task_vars:
                    self._task_vars[combat_id].set(False)
        try:
            self._save_config()
        except ValueError:
            pass

    def _build_task_checkboxes(
        self,
        parent: ttk.Frame,
        entries: list[TaskEntry],
        *,
        columns: int = 1,
        show_hint: bool = True,
    ) -> None:
        row_frame: ttk.Frame | None = None
        col = 0

        for entry in entries:
            if col == 0:
                row_frame = ttk.Frame(parent)
                row_frame.pack(fill=tk.X, pady=1)

            var = tk.BooleanVar(value=self._task_enabled_in_config(entry))
            self._task_vars[entry.task_id] = var

            label = entry.label
            if not entry.available:
                label = f"{entry.label}（敬请期待）"

            cell = ttk.Frame(row_frame)
            cell.pack(side=tk.LEFT, padx=(0, 16))

            cb_kwargs: dict = {"text": label, "variable": var}
            if entry.kind == "loop":
                cb_kwargs["command"] = lambda tid=entry.task_id: self._on_loop_checkbox_changed(tid)

            cb = ttk.Checkbutton(cell, **cb_kwargs)
            cb.pack(side=tk.LEFT)
            if not entry.available:
                cb.state(["disabled"])

            if show_hint and entry.hint:
                ttk.Label(cell, text=entry.hint, font=("", 8), foreground="gray").pack(
                    side=tk.LEFT, padx=(8, 0)
                )

            col += 1
            if col >= columns:
                col = 0

    def _dream_widgets_for(self, pk: bool) -> DreamTabWidgets:
        widgets = self._dream_pk_widgets if pk else self._dream_widgets
        if widgets is None:
            raise RuntimeError("寻梦记忆 Tab 尚未初始化")
        return widgets

    def _refresh_dream_maps(self, pk: bool = False) -> None:
        widgets = self._dream_widgets_for(pk)
        dm_cfg = load_dream_cfg(self, pk)
        maps = list_maps(dm_cfg.maps_dir)
        label_to_id, id_to_label = rebuild_map_index(maps)
        widgets.label_to_id = label_to_id
        widgets.id_to_label = id_to_label
        map_ids = [m.map_id for m in maps]
        labels = list(label_to_id.keys())
        widgets.cmb_map.configure(
            values=labels,
            state="readonly" if labels else "disabled",
        )
        current_id = get_selected_map_id(widgets)
        if current_id not in map_ids:
            current_id = map_ids[0] if map_ids else ""
        if current_id:
            set_map_selection(widgets, current_id)
        else:
            widgets.var_map.set("")
        update_summary(self, widgets)

        from core.dream_memory.ocr_engine import ocr_engine_available, resolve_ocr_engine

        ocr_engine = resolve_ocr_engine(dm_cfg.ocr_engine)
        ocr_ok = ocr_engine_available(dm_cfg.ocr_engine)
        if ocr_engine == "rapidocr":
            ocr_text = "RapidOCR: 已就绪" if ocr_ok else "RapidOCR: 未安装"
        else:
            ocr_text = "Tesseract: 已就绪" if ocr_ok else "Tesseract: 未找到"
        widgets.lbl_ocr.configure(
            text=ocr_text,
            foreground="green" if ocr_ok else "red",
        )
        if not map_ids:
            title = "寻梦记忆PK" if pk else "寻梦记忆"
            messagebox.showinfo(
                title,
                "暂无地图配置。\n"
                "请点「标定地图」新建，或运行:\n"
                "  tools/calibrate_dream_memory_map.py --create 地图ID --name 显示名",
            )

    def _build_dream_memory_tab(self, parent: ttk.Frame) -> None:
        self._dream_widgets = build_dream_tab(
            self,
            parent,
            pk=False,
            on_map_changed=lambda: (update_summary(self, self._dream_widgets), self._save_config()),
            on_start=lambda: self._start_dream_session(pk=False),
            on_stop=lambda: self._stop_dream_session(pk=False),
            on_refresh=lambda: self._refresh_dream_maps(pk=False),
            on_calibrate=lambda: self._open_dream_calibrator(pk=False),
            on_rename=lambda: self._rename_dream_map(pk=False),
            on_delete=lambda: self._delete_dream_map(pk=False),
        )
        self.var_dream_memory_map = self._dream_widgets.var_map
        self.btn_dream_start = self._dream_widgets.btn_start
        self.btn_dream_stop = self._dream_widgets.btn_stop
        dm_cfg = load_dream_memory_config(self.config_path)
        if resolve_ocr_engine(dm_cfg.ocr_engine) == "rapidocr" and ocr_engine_available(
            dm_cfg.ocr_engine
        ):
            threading.Thread(
                target=lambda: warmup_ocr(dm_cfg.ocr_engine),
                daemon=True,
            ).start()

    def _build_dream_pk_tab(self, parent: ttk.Frame) -> None:
        self._dream_pk_widgets = build_dream_tab(
            self,
            parent,
            pk=True,
            on_map_changed=lambda: (
                update_summary(self, self._dream_pk_widgets),
                self._save_config(),
            ),
            on_start=lambda: self._start_dream_session(pk=True),
            on_stop=lambda: self._stop_dream_session(pk=True),
            on_refresh=lambda: self._refresh_dream_maps(pk=True),
            on_calibrate=lambda: self._open_dream_calibrator(pk=True),
            on_rename=lambda: self._rename_dream_map(pk=True),
            on_delete=lambda: self._delete_dream_map(pk=True),
        )
        self.btn_dream_pk_start = self._dream_pk_widgets.btn_start
        self.btn_dream_pk_stop = self._dream_pk_widgets.btn_stop

    def _refresh_dream_memory_maps(self) -> None:
        self._refresh_dream_maps(pk=False)

    def _build_ui(self) -> None:
        lighthouse_cfg = self.config.get("tasks", {}).get("auto_lighthouse", {})
        hunt_cfg = self.config.get("tasks", {}).get("hunt_ice_beast", {})
        monster_cfg = self.config.get("tasks", {}).get("hunt_monster", {})
        donate_cfg = self.config.get("tasks", {}).get("donate_alliance_supplies", {})
        mining_cfg = self.config.get("tasks", {}).get("auto_mining", {})
        collect_cfg = self.config.get("tasks", {}).get("collect_supplies", {})
        train_cfg = self.config.get("tasks", {}).get("auto_train_troops", {})
        gui_cfg = self.config.get("gui", {})
        dev_cfg = self.config.get("device", {})
        default_device = AdbClient.format_address(
            dev_cfg.get("adb_host", "127.0.0.1"),
            int(dev_cfg.get("adb_port", 5555)),
        )

        self._log_visible = bool(gui_cfg.get("log_visible", True))
        self.var_show_console = tk.BooleanVar(value=bool(gui_cfg.get("show_console", True)))
        self._build_menu_bar()

        self._container = ttk.Frame(self)
        self._container.pack(fill=tk.BOTH, expand=True)

        pad = {"padx": 10, "pady": 3}
        self._left = ttk.Frame(self._container, width=MAIN_WIDTH)
        self._left.pack(side=tk.LEFT, fill=tk.BOTH)
        self._left.pack_propagate(False)

        # 底部固定：运行控制 + 设备；运行控制向上吃掉与「运行任务」之间的空隙
        bottom = ttk.Frame(self._left, padding=(10, 2, 10, 8))
        bottom.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)

        run_frame = ttk.LabelFrame(bottom, text="运行控制", padding=6)
        run_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        account_row = ttk.Frame(run_frame)
        account_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(
            account_row,
            text="账号名称",
            font=("", 11, "bold"),
            foreground="#c45c00",
        ).pack(side=tk.LEFT)
        self.lbl_account_name = tk.Label(
            account_row,
            text="未知",
            font=("", 16, "bold"),
            foreground="#e67e22",
            bg="#f0f0f0",
        )
        self.lbl_account_name.pack(side=tk.LEFT, padx=(10, 0))

        run_btn_row = ttk.Frame(run_frame)
        run_btn_row.pack(fill=tk.X)
        self.btn_start = ttk.Button(
            run_btn_row, text="开始", command=self._start_from_current_tab, width=8
        )
        self.btn_start.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_stop = ttk.Button(
            run_btn_row,
            text="结束",
            command=self._stop_running_tasks,
            width=8,
            state=tk.DISABLED,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_hosting = ttk.Button(
            run_btn_row, text="一键辅助", command=self._run_hosting_batch, width=8
        )
        self.btn_hosting.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_quick_monster = ttk.Button(
            run_btn_row,
            text="一键打野",
            command=lambda: self._start_quick_loop_task("hunt_monster"),
            width=8,
        )
        self.btn_quick_monster.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_quick_ice = ttk.Button(
            run_btn_row,
            text="一键集结巨兽",
            command=lambda: self._start_quick_loop_task("hunt_ice_beast"),
            width=12,
        )
        self.btn_quick_ice.pack(side=tk.LEFT)

        status_row = ttk.Frame(run_frame)
        status_row.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self._status_lines: list[str] = ["待命"]
        self.txt_status = tk.Text(
            status_row,
            height=2,
            wrap=tk.WORD,
            font=("", 9),
            foreground="#555555",
            bg="#f0f0f0",
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=2,
            pady=2,
        )
        self.txt_status.pack(fill=tk.BOTH, expand=True)
        self.txt_status.insert("1.0", "状态：待命")
        self.txt_status.configure(state=tk.DISABLED)
        # 兼容旧引用
        self.lbl_status = self.txt_status

        settings_frame = ttk.LabelFrame(bottom, text="设备", padding=6)
        settings_frame.pack(fill=tk.X)

        dev_row = ttk.Frame(settings_frame)
        dev_row.pack(fill=tk.X)

        ttk.Label(dev_row, text="模拟器设备").pack(side=tk.LEFT)
        self.var_device_serial = tk.StringVar(value=default_device)
        self.cmb_device = ttk.Combobox(
            dev_row,
            textvariable=self.var_device_serial,
            width=24,
            state="readonly",
        )
        self.cmb_device.pack(side=tk.LEFT, padx=(6, 0))
        self.cmb_device.bind("<<ComboboxSelected>>", lambda _event: self._on_device_selected())

        ttk.Button(
            dev_row,
            text="刷新设备",
            width=8,
            command=lambda: self._refresh_devices(probe=True),
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            dev_row,
            text="坐标标尺",
            width=8,
            command=self._open_coord_ruler,
        ).pack(side=tk.LEFT, padx=(6, 0))
        self.btn_check_account = ttk.Button(
            dev_row,
            text="检查账号名称",
            command=self._start_check_account_name,
            width=12,
        )
        self.btn_check_account.pack(side=tk.LEFT, padx=(6, 0))

        sys_row = ttk.Frame(settings_frame)
        sys_row.pack(fill=tk.X, pady=(6, 0))

        self.lbl_conn = ttk.Label(sys_row, text="设备：未连接")
        self.lbl_conn.pack(side=tk.LEFT)
        ttk.Label(
            sys_row,
            text=f"配置：{self.config_path.name}",
            font=("", 8),
            foreground="gray",
        ).pack(side=tk.LEFT, padx=(12, 0))

        param_outer = ttk.LabelFrame(self._left, text="功能参数", padding=6)
        param_outer.pack(fill=tk.X, **pad)

        common_opts = resolve_common_options(self.config.get("tasks", {}))

        self.var_common_use_formation = tk.BooleanVar(
            value=bool(common_opts["use_formation"])
        )
        self.var_common_adjust_level = tk.BooleanVar(
            value=bool(common_opts["adjust_level"])
        )
        self.var_common_use_stamina = tk.BooleanVar(
            value=bool(common_opts["use_stamina"])
        )
        self.var_common_stamina_can_limit = tk.IntVar(
            value=int(common_opts["stamina_can_limit"])
        )

        common_frame = ttk.LabelFrame(param_outer, text="通用出征 / 搜索", padding=6)
        common_frame.pack(fill=tk.X, pady=(0, 6))

        common_row1 = ttk.Frame(common_frame)
        common_row1.pack(fill=tk.X)
        ttk.Checkbutton(
            common_row1,
            text="启用编队",
            variable=self.var_common_use_formation,
            command=self._on_common_use_formation_changed,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(
            common_row1,
            text="修改等级",
            variable=self.var_common_adjust_level,
            command=self._save_config,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(
            common_row1,
            text="自动使用罐头",
            variable=self.var_common_use_stamina,
            command=self._on_common_use_stamina_changed,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(common_row1, text="罐头数量").pack(side=tk.LEFT)
        self.spn_common_stamina_can_limit = ttk.Spinbox(
            common_row1,
            from_=1,
            to=99999,
            textvariable=self.var_common_stamina_can_limit,
            width=8,
            command=self._save_config,
        )
        self.spn_common_stamina_can_limit.pack(side=tk.LEFT, padx=(6, 0))
        self._update_common_stamina_limit_state()

        self.var_interval = tk.IntVar(value=hunt_cfg.get("interval", 120) // 60)
        self.var_lighthouse_interval = tk.IntVar(
            value=max(1, int(lighthouse_cfg.get("interval", 60) // 60))
        )
        self.var_lighthouse_formation_slot = tk.IntVar(
            value=_normalize_formation_slot(lighthouse_cfg.get("formation_slot", 8))
        )
        self.var_lighthouse_event_period = tk.BooleanVar(
            value=bool(lighthouse_cfg.get("event_period", False))
        )
        self.var_lighthouse_monster_cooldown = tk.IntVar(
            value=max(1, int(lighthouse_cfg.get("monster_cooldown", 120) // 60))
        )
        self.var_level = tk.IntVar(value=hunt_cfg.get("beast_level", 8))
        self.var_formation_slot = tk.IntVar(
            value=_normalize_formation_slot(hunt_cfg.get("formation_name", 7))
        )
        self.var_rally_duration = tk.StringVar(
            value=str(hunt_cfg.get("rally_duration_minutes", 5))
        )
        self.var_monster_interval = tk.IntVar(value=monster_cfg.get("interval", 60) // 60)
        self.var_monster_level = tk.IntVar(value=monster_cfg.get("monster_level", 30))
        self.var_monster_formation_slot = tk.IntVar(
            value=_normalize_formation_slot(monster_cfg.get("formation_name", 8))
        )
        self.var_donate_interval = tk.IntVar(value=donate_cfg.get("interval", 3600) // 60)
        self.var_donate_times = tk.IntVar(value=donate_cfg.get("donate_times", 25))
        self.var_collect_interval = tk.IntVar(
            value=max(1, collect_cfg.get("interval", 5 * 3600) // 3600)
        )
        self.var_train_interval = tk.IntVar(
            value=max(1, train_cfg.get("interval", 3 * 3600) // 3600)
        )

        mining_level_min, mining_level_max = _normalize_mining_range(
            mining_cfg.get("level_min", MINING_LEVEL_DEFAULT_MIN),
            mining_cfg.get("level_max", MINING_LEVEL_DEFAULT_MAX),
        )
        self.var_mining_interval = tk.IntVar(value=mining_cfg.get("interval", 3600) // 60)
        self.var_use_mining_hero = tk.BooleanVar(
            value=bool(mining_cfg.get("use_mining_hero", True))
        )
        self.var_mining_level_min = tk.IntVar(value=mining_level_min)
        self.var_mining_level_max = tk.IntVar(value=mining_level_max)

        self._param_notebook = ttk.Notebook(param_outer)
        self._param_notebook.pack(fill=tk.X)
        notebook = self._param_notebook

        tab_ice = ttk.Frame(notebook, padding=6)
        notebook.add(tab_ice, text="冰原巨兽")
        _configure_param_tab_grid(tab_ice)

        row = 0
        ttk.Label(tab_ice, text="集结间隔（分钟）").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_ice, from_=5, to=120, textvariable=self.var_interval, width=8).grid(
            row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX
        )
        row += 1

        ttk.Label(tab_ice, text="巨兽等级").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(tab_ice, from_=1, to=30, textvariable=self.var_level, width=8).grid(
            row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX
        )
        row += 1

        ttk.Label(tab_ice, text="编队槽位").grid(row=row, column=0, sticky=tk.W, pady=2)
        self._ice_formation_spinbox = ttk.Spinbox(
            tab_ice,
            from_=FORMATION_SLOT_MIN,
            to=FORMATION_SLOT_MAX,
            textvariable=self.var_formation_slot,
            width=8,
        )
        self._ice_formation_spinbox.grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        self._ice_formation_hint = ttk.Label(
            tab_ice, text="启用编队时生效（1~8）", font=("", 8)
        )
        self._ice_formation_hint.grid(row=row, column=2, sticky=tk.W, padx=6)
        row += 1

        ttk.Label(tab_ice, text="集结等待（分钟）").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(
            tab_ice,
            textvariable=self.var_rally_duration,
            values=("5", "15", "30", "60"),
            width=6,
            state="readonly",
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        self._update_formation_spinboxes_state()

        tab_lighthouse = ttk.Frame(notebook, padding=6)
        notebook.add(tab_lighthouse, text="灯塔任务")
        _configure_param_tab_grid(tab_lighthouse)

        row = 0
        ttk.Label(tab_lighthouse, text="扫描间隔（分钟）").grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        ttk.Spinbox(
            tab_lighthouse, from_=1, to=1440, textvariable=self.var_lighthouse_interval, width=8
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        ttk.Label(tab_lighthouse, text="（一次性任务，此项预留）", font=("", 8)).grid(
            row=row, column=2, sticky=tk.W, padx=(4, 0)
        )
        row += 1

        ttk.Label(tab_lighthouse, text="编队槽位").grid(row=row, column=0, sticky=tk.W, pady=2)
        self._lighthouse_formation_spinbox = ttk.Spinbox(
            tab_lighthouse,
            from_=FORMATION_SLOT_MIN,
            to=FORMATION_SLOT_MAX,
            textvariable=self.var_lighthouse_formation_slot,
            width=8,
        )
        self._lighthouse_formation_spinbox.grid(
            row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX
        )
        ttk.Label(tab_lighthouse, text="启用编队时生效（1~8）", font=("", 8)).grid(
            row=row, column=2, sticky=tk.W, padx=6
        )
        row += 1

        ttk.Label(tab_lighthouse, text="小怪打怪间隔（分钟）").grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        ttk.Spinbox(
            tab_lighthouse, from_=1, to=60, textvariable=self.var_lighthouse_monster_cooldown, width=8
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        ttk.Label(tab_lighthouse, text="出征后冷却，期间可打帐篷/英雄之旅", font=("", 8)).grid(
            row=row, column=2, sticky=tk.W, padx=6
        )
        row += 1

        ttk.Checkbutton(
            tab_lighthouse,
            text="活动期间（使用含红色晶簇的活动背景图扫描）",
            variable=self.var_lighthouse_event_period,
            command=self._save_config,
        ).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)

        tab_mining = ttk.Frame(notebook, padding=6)
        notebook.add(tab_mining, text="自动采集")
        _configure_param_tab_grid(tab_mining)

        row = 0
        ttk.Label(tab_mining, text="采矿间隔（分钟）").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            tab_mining, from_=1, to=999, textvariable=self.var_mining_interval, width=8
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        row += 1

        ttk.Checkbutton(
            tab_mining,
            text="采矿英雄采矿",
            variable=self.var_use_mining_hero,
            command=self._save_config,
        ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(tab_mining, text="采矿范围").grid(row=row, column=0, sticky=tk.W, pady=2)
        range_row = ttk.Frame(tab_mining)
        range_row.grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        ttk.Spinbox(
            range_row,
            from_=MINING_LEVEL_MIN,
            to=MINING_LEVEL_MAX,
            textvariable=self.var_mining_level_min,
            width=5,
        ).pack(side=tk.LEFT)
        ttk.Label(range_row, text=" — ").pack(side=tk.LEFT)
        ttk.Spinbox(
            range_row,
            from_=MINING_LEVEL_MIN,
            to=MINING_LEVEL_MAX,
            textvariable=self.var_mining_level_max,
            width=5,
        ).pack(side=tk.LEFT)

        tab_monster = ttk.Frame(notebook, padding=6)
        notebook.add(tab_monster, text="打野怪")
        _configure_param_tab_grid(tab_monster)

        row = 0
        ttk.Label(tab_monster, text="攻击间隔（分钟）").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            tab_monster, from_=1, to=120, textvariable=self.var_monster_interval, width=8
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        row += 1

        ttk.Label(tab_monster, text="野怪等级").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            tab_monster, from_=1, to=30, textvariable=self.var_monster_level, width=8
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        row += 1

        ttk.Label(tab_monster, text="编队槽位").grid(row=row, column=0, sticky=tk.W, pady=2)
        self._monster_formation_spinbox = ttk.Spinbox(
            tab_monster,
            from_=FORMATION_SLOT_MIN,
            to=FORMATION_SLOT_MAX,
            textvariable=self.var_monster_formation_slot,
            width=8,
        )
        self._monster_formation_spinbox.grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        ttk.Label(
            tab_monster, text="启用编队时生效（1~8）", font=("", 8)
        ).grid(row=row, column=2, sticky=tk.W, padx=6)
        row += 1
        self._update_formation_spinboxes_state()

        tab_donate = ttk.Frame(notebook, padding=6)
        notebook.add(tab_donate, text="捐献物资")
        _configure_param_tab_grid(tab_donate)

        row = 0
        ttk.Label(tab_donate, text="捐献间隔（分钟）").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            tab_donate, from_=5, to=1440, textvariable=self.var_donate_interval, width=8
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        ttk.Label(
            tab_donate,
            text="暂时只能捐献联盟永续",
            font=("", 8),
            foreground="gray",
        ).grid(row=row, column=2, sticky=tk.W, padx=6)
        row += 1

        ttk.Label(tab_donate, text="每次捐献次数").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            tab_donate, from_=1, to=25, textvariable=self.var_donate_times, width=8
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)

        tab_collect = ttk.Frame(notebook, padding=6)
        notebook.add(tab_collect, text="探险物资")
        _configure_param_tab_grid(tab_collect)

        row = 0
        ttk.Label(tab_collect, text="领取间隔（小时）").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            tab_collect, from_=1, to=48, textvariable=self.var_collect_interval, width=8
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)

        tab_train = ttk.Frame(notebook, padding=6)
        notebook.add(tab_train, text="自动练兵")
        _configure_param_tab_grid(tab_train)

        row = 0
        ttk.Label(tab_train, text="扫描间隔（小时）").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            tab_train, from_=1, to=48, textvariable=self.var_train_interval, width=8
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)

        alliance_cfg = self.config.get("alliance_mobilization", {})
        alliance_merged = merge_alliance_config(alliance_cfg)
        admin_cfg = self.config.get("alliance_mobilization_admin", {})
        admin_merged = merge_alliance_admin_config(admin_cfg)
        type_keep_rules = (
            admin_merged.get("type_keep_rules")
            or alliance_merged.get("type_keep_rules")
            or normalize_type_keep_rules({})
        )
        self.var_alliance_scan_minutes = tk.IntVar(
            value=max(1, int(round(float(alliance_merged.get("scan_interval", 360)) / 60)))
        )
        self.var_alliance_admin_scan_minutes = tk.IntVar(
            value=max(1, int(round(float(admin_merged.get("scan_interval", 360)) / 60)))
        )

        for type_id in GUI_ALLIANCE_TYPE_ORDER:
            colors = set(type_keep_rules.get(type_id, []))
            self._alliance_type_bg_vars[type_id] = {
                CARD_BG_ORANGE: tk.BooleanVar(value=CARD_BG_ORANGE in colors),
                CARD_BG_PURPLE: tk.BooleanVar(value=CARD_BG_PURPLE in colors),
            }
        if not type_keep_rules:
            self._alliance_type_bg_vars["train"][CARD_BG_ORANGE].set(True)
            self._alliance_type_bg_vars["train"][CARD_BG_PURPLE].set(True)

        tab_alliance = ttk.Frame(notebook, padding=4)
        notebook.add(tab_alliance, text="联盟总动员")
        _configure_param_tab_grid(tab_alliance)

        row = 0
        cycle_row = ttk.Frame(tab_alliance)
        cycle_row.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 2))
        ttk.Label(cycle_row, text="普通循环").pack(side=tk.LEFT)
        ttk.Spinbox(
            cycle_row,
            from_=1,
            to=720,
            textvariable=self.var_alliance_scan_minutes,
            width=5,
            command=self._save_config,
        ).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(cycle_row, text="管理员循环").pack(side=tk.LEFT)
        ttk.Spinbox(
            cycle_row,
            from_=1,
            to=720,
            textvariable=self.var_alliance_admin_scan_minutes,
            width=5,
            command=self._save_config,
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(cycle_row, text="（分钟）", font=("", 8), foreground="gray").pack(
            side=tk.LEFT, padx=(4, 0)
        )
        row += 1

        ttk.Label(tab_alliance, text="保留规则").grid(
            row=row, column=0, sticky=tk.NW, pady=1
        )
        type_inner, scroll_wrap, type_canvas = _create_scrollable_frame(
            tab_alliance, height=88
        )
        scroll_wrap.grid(row=row, column=1, columnspan=2, sticky="new", padx=FORM_INPUT_PADX)

        _ALLIANCE_CELL_COLS = 3  # 类型名 + 橙 + 紫
        for index, type_id in enumerate(GUI_ALLIANCE_TYPE_ORDER):
            grid_row = index // 2
            cell = index % 2
            base_col = cell * _ALLIANCE_CELL_COLS
            pad_left = 0 if cell == 0 else 16
            ttk.Label(
                type_inner,
                text=TASK_TYPE_LABELS[type_id],
                width=5,
            ).grid(row=grid_row, column=base_col, sticky=tk.W, padx=(pad_left, 2), pady=0)
            ttk.Checkbutton(
                type_inner,
                text="橙",
                variable=self._alliance_type_bg_vars[type_id][CARD_BG_ORANGE],
                command=self._save_config,
                width=3,
            ).grid(row=grid_row, column=base_col + 1, sticky=tk.W)
            ttk.Checkbutton(
                type_inner,
                text="紫",
                variable=self._alliance_type_bg_vars[type_id][CARD_BG_PURPLE],
                command=self._save_config,
                width=3,
            ).grid(row=grid_row, column=base_col + 2, sticky=tk.W)
        type_inner.update_idletasks()
        bbox = type_canvas.bbox("all")
        if bbox:
            type_canvas.configure(scrollregion=bbox)

        def _alliance_rules_wheel(event) -> None:
            if getattr(event, "delta", 0):
                type_canvas.yview_scroll(int(-event.delta / 120), "units")
            elif getattr(event, "num", None) == 4:
                type_canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                type_canvas.yview_scroll(3, "units")

        _bind_mousewheel_recursive(type_inner, _alliance_rules_wheel)

        tab_more = ttk.Frame(notebook, padding=8)
        notebook.add(tab_more, text="更多")
        ttk.Label(tab_more, text="更多任务参数将在此添加", foreground="gray").pack(
            anchor=tk.W
        )

        task_frame = ttk.LabelFrame(self._left, text="运行任务", padding=4)
        task_frame.pack(fill=tk.X, **pad)

        self._task_notebook = ttk.Notebook(task_frame)
        self._task_notebook.pack(fill=tk.X)
        task_notebook = self._task_notebook

        tab_loop = ttk.Frame(task_notebook, padding=6)
        task_notebook.add(tab_loop, text="循环任务")
        self._build_task_checkboxes(
            tab_loop, loop_tasks(), columns=LOOP_TASK_COLUMNS, show_hint=False
        )

        tab_once = ttk.Frame(task_notebook, padding=6)
        task_notebook.add(tab_once, text="一次性任务")
        self._build_task_checkboxes(
            tab_once, once_tasks(), columns=ONCE_TASK_COLUMNS, show_hint=False
        )

        tab_alliance_run = ttk.Frame(task_notebook, padding=6)
        task_notebook.add(tab_alliance_run, text="联盟总动员自动刷新")

        tab_alliance_admin_run = ttk.Frame(task_notebook, padding=6)
        task_notebook.add(tab_alliance_admin_run, text="联盟管理员刷新")

        tab_dream = ttk.Frame(task_notebook, padding=6)
        task_notebook.add(tab_dream, text="寻梦记忆")
        self._build_dream_memory_tab(tab_dream)

        tab_dream_pk = ttk.Frame(task_notebook, padding=6)
        task_notebook.add(tab_dream_pk, text="寻梦记忆PK")
        self._build_dream_pk_tab(tab_dream_pk)

        self._task_notebook.bind("<<NotebookTabChanged>>", self._on_task_notebook_changed)

        self._log_frame = ttk.LabelFrame(self._container, text="运行日志", padding=8)
        self.log_text = scrolledtext.ScrolledText(
            self._log_frame, width=36, height=28, state=tk.DISABLED, font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        if self._log_visible:
            self._show_log_panel(save_config=False)

    def _build_menu_bar(self) -> None:
        menubar = tk.Menu(self)
        super().config(menu=menubar)

        system_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="系统", menu=system_menu)

        self._menu_log_hidden = tk.BooleanVar(value=not self._log_visible)
        system_menu.add_checkbutton(
            label="隐藏日志",
            variable=self._menu_log_hidden,
            command=self._on_menu_log_hidden_toggled,
        )
        system_menu.add_checkbutton(
            label="启动时显示 CMD 窗口",
            variable=self.var_show_console,
            command=self._save_config,
        )
        system_menu.add_separator()
        system_menu.add_command(label="测试连接", command=self._test_connection_with_dialog)

    def _sync_log_menu_check(self) -> None:
        if hasattr(self, "_menu_log_hidden"):
            self._menu_log_hidden.set(not self._log_visible)

    def _on_menu_log_hidden_toggled(self) -> None:
        if self._menu_log_hidden.get():
            if self._log_visible:
                self._hide_log_panel()
        elif not self._log_visible:
            self._show_log_panel()

    def _toggle_log(self) -> None:
        if self._log_visible:
            self._hide_log_panel()
        else:
            self._show_log_panel()

    def _show_log_panel(self, save_config: bool = True) -> None:
        self._log_visible = True
        self._log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.geometry(f"{MAIN_WIDTH + LOG_WIDTH}x{MAIN_HEIGHT}")
        self._sync_log_menu_check()
        if save_config:
            self._save_config()

    def _hide_log_panel(self) -> None:
        self._log_visible = False
        self._log_frame.pack_forget()
        self.geometry(f"{MAIN_WIDTH}x{MAIN_HEIGHT}")
        self._sync_log_menu_check()
        self._save_config()

    def _test_connection_with_dialog(self) -> None:
        self._test_connection(show_dialog=True)

    def _set_status_text(self, message: str) -> None:
        """运行控制区状态：保留最近两行。"""
        text = (message or "").strip()
        if text.startswith("状态："):
            text = text[3:].strip()
        if not text:
            return
        lines = getattr(self, "_status_lines", None)
        if lines is None:
            self._status_lines = []
            lines = self._status_lines
        if lines and lines[-1] == text:
            return
        lines.append(text)
        del lines[:-2]
        body = "\n".join(f"状态：{line}" for line in lines)
        widget = getattr(self, "txt_status", None) or getattr(self, "lbl_status", None)
        if widget is None:
            return
        if isinstance(widget, tk.Text):
            widget.configure(state=tk.NORMAL)
            widget.delete("1.0", tk.END)
            widget.insert("1.0", body)
            widget.configure(state=tk.DISABLED)
        else:
            widget.configure(text=body.replace("\n", " | "))

    def _append_log(self, message: str) -> None:
        from datetime import datetime

        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{ts}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _poll_log_queue(self) -> None:
        while True:
            try:
                msg = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(msg)
            self._set_status_text(msg)
        self.after(200, self._poll_log_queue)

    def _on_status(self, message: str) -> None:
        self._log_queue.put(message)

    def _apply_device_serial(self, serial: str, *, save: bool) -> None:
        serial = AdbClient.normalize_device_serial(serial)
        if not serial:
            return
        host, port = AdbClient.parse_address(serial)
        dev = self.config.setdefault("device", {})
        dev["adb_host"] = host
        dev["adb_port"] = int(port)
        self.var_device_serial.set(serial)
        self._reset_adb()
        if save:
            try:
                self._save_config()
            except ValueError:
                dev = self.config.setdefault("device", {})
                dev["adb_host"] = host
                dev["adb_port"] = int(port)
                with open(self.config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=False)

    def _reset_adb(self) -> None:
        self._adb = None

    def _on_device_selected(self) -> None:
        serial = self.var_device_serial.get().strip()
        if not serial:
            return
        try:
            self._apply_device_serial(serial, save=True)
            self.lbl_conn.configure(text=f"设备：{serial}", foreground="green")
            self._on_status(f"已切换模拟器：{serial}")
        except ValueError as exc:
            messagebox.showerror("设备地址无效", str(exc))

    def _refresh_devices(self, *, probe: bool = True) -> None:
        self.cmb_device.configure(state="disabled")
        current = self.var_device_serial.get().strip()
        adb_path = self.config.get("device", {}).get("adb_path", "")

        def work():
            try:
                devices = AdbClient.list_connected_devices(
                    adb_path, probe_ldplayer=probe
                )
                msg = ""
            except Exception as exc:
                devices = []
                msg = str(exc)

            def finish():
                self.cmb_device.configure(state="readonly")
                if devices:
                    self.cmb_device["values"] = devices
                    if current in devices:
                        pick = current
                    else:
                        pick = devices[0]
                    self._apply_device_serial(pick, save=(pick != current))
                    self.lbl_conn.configure(
                        text=f"设备：{pick}（{len(devices)} 个已连接）",
                        foreground="green",
                    )
                    if pick != current:
                        self._on_status(f"已自动选择模拟器：{pick}")
                else:
                    self.cmb_device["values"] = (
                        [current] if current else ["127.0.0.1:5555"]
                    )
                    hint = "点「刷新」扫描多开模拟器" if not probe else "请先启动模拟器后点刷新"
                    text = f"未发现已连接设备，{hint}"
                    if msg:
                        text = f"{text}（{msg}）"
                    self.lbl_conn.configure(text=text, foreground="red")

            self.after(0, finish)

        threading.Thread(target=work, daemon=True).start()

    def _get_adb(self) -> AdbClient:
        dev = self.config["device"]
        host = dev.get("adb_host", "127.0.0.1")
        port = int(dev.get("adb_port", 5555))
        address = AdbClient.format_address(host, port)

        if hasattr(self, "var_device_serial"):
            serial = self.var_device_serial.get().strip()
            if serial:
                try:
                    host, port = AdbClient.parse_address(serial)
                    address = serial
                except ValueError:
                    pass

        if self._adb is None or self._adb.address != address:
            self._adb = AdbClient(
                host=host,
                port=port,
                adb_path=dev.get("adb_path", ""),
                touch_width=dev.get("touch_width", 720),
                touch_height=dev.get("touch_height", 1280),
            )
        return self._adb

    def _test_connection(self, *, show_dialog: bool = False) -> None:
        self.lbl_conn.configure(text="连接中…")
        self.update_idletasks()
        serial = self.var_device_serial.get().strip()
        dev_cfg = self.config.get("device", {})
        adb_path = dev_cfg.get("adb_path", "")

        def work():
            ok = False
            try:
                if serial:
                    host, port = AdbClient.parse_address(serial)
                else:
                    host = dev_cfg.get("adb_host", "127.0.0.1")
                    port = int(dev_cfg.get("adb_port", 5555))
                adb = AdbClient(
                    host=host,
                    port=port,
                    adb_path=adb_path,
                    touch_width=dev_cfg.get("touch_width", 720),
                    touch_height=dev_cfg.get("touch_height", 1280),
                )
                ok = adb.wait_for_device(retries=5, interval=1.0)
                msg = f"已连接 {adb.address}" if ok else "连接失败"
                color = "green" if ok else "red"
                connected_address = adb.address if ok else ""
            except Exception as exc:
                msg = f"错误：{exc}"
                color = "red"
                connected_address = ""

            def finish():
                self.lbl_conn.configure(text=msg, foreground=color)
                if connected_address:
                    self._adb = adb
                    self._apply_device_serial(connected_address, save=True)
                if show_dialog:
                    if ok:
                        messagebox.showinfo("测试连接", msg)
                    else:
                        messagebox.showerror("测试连接", msg)

            self.after(0, finish)

        threading.Thread(target=work, daemon=True).start()

    def _capture_for_coord_ruler(self):
        """供坐标标尺截取模拟器画面（RGB 数组）。"""
        adb = self._get_adb()
        if not adb.wait_for_device(retries=5, interval=1.0):
            raise RuntimeError("无法连接模拟器，请先启动雷电并测试连接")
        screen = adb.screenshot()
        return screen[:, :, ::-1]

    def _open_coord_ruler(self) -> None:
        dev = self.config.get("device", {})
        touch_w = int(dev.get("touch_width", 720))
        touch_h = int(dev.get("touch_height", 1280))

        if self._coord_ruler_window is None or not self._coord_ruler_window.winfo_exists():
            self._coord_ruler_window = CoordRulerWindow(
                self,
                touch_width=touch_w,
                touch_height=touch_h,
                screenshot_cb=self._capture_for_coord_ruler,
            )
        self._coord_ruler_window.show_window()

    def _on_dream_map_saved(self, map_id: str | None, *, pk: bool) -> None:
        self._refresh_dream_maps(pk=pk)
        if map_id:
            set_map_selection(self._dream_widgets_for(pk), map_id)
            try:
                self._save_config()
            except ValueError:
                pass

    def _rename_dream_map(self, pk: bool = False) -> None:
        widgets = self._dream_widgets_for(pk)
        map_id = get_selected_map_id(widgets)
        if not map_id:
            messagebox.showwarning("提示", "请先选择要重命名的地图")
            return
        dm_cfg = load_dream_cfg(self, pk)
        try:
            dream_map = load_map(map_id, maps_dir=dm_cfg.maps_dir)
        except FileNotFoundError:
            messagebox.showerror("错误", f"地图不存在: {map_id}")
            return
        new_name = simpledialog.askstring(
            "重命名地图",
            f"地图 ID: {map_id}\n新的显示名称:",
            initialvalue=dream_map.name,
        )
        if not new_name or new_name.strip() == dream_map.name:
            return
        try:
            rename_map_name(map_id, new_name.strip(), maps_dir=dm_cfg.maps_dir)
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("重命名失败", str(exc))
            return
        self._refresh_dream_maps(pk=pk)
        set_map_selection(widgets, map_id)
        messagebox.showinfo("完成", f"已重命名为「{new_name.strip()}」")

    def _delete_dream_map(self, pk: bool = False) -> None:
        widgets = self._dream_widgets_for(pk)
        map_id = get_selected_map_id(widgets)
        if not map_id:
            messagebox.showwarning("提示", "请先选择要删除的地图")
            return
        dm_cfg = load_dream_cfg(self, pk)
        try:
            dream_map = load_map(map_id, maps_dir=dm_cfg.maps_dir)
        except FileNotFoundError:
            messagebox.showerror("错误", f"地图不存在: {map_id}")
            return
        if not messagebox.askyesno(
            "删除地图",
            f"确定删除「{dream_map.name}」({map_id})？\n"
            f"含 {len(dream_map.items)} 个标定物品，不可恢复。",
        ):
            return
        try:
            delete_map(
                map_id,
                maps_dir=dm_cfg.maps_dir,
                previews_dir=dm_cfg.previews_dir,
            )
        except FileNotFoundError as exc:
            messagebox.showerror("删除失败", str(exc))
            return
        self._refresh_dream_maps(pk=pk)
        messagebox.showinfo("已删除", f"地图「{dream_map.name}」已删除")

    def _open_dream_calibrator(self, pk: bool = False) -> None:
        dev = self.config.get("device", {})
        touch_w = int(dev.get("touch_width", 720))
        touch_h = int(dev.get("touch_height", 1280))
        widgets = self._dream_widgets_for(pk)
        calibrator_attr = "_dream_pk_calibrator" if pk else "_dream_memory_calibrator"
        calibrator = getattr(self, calibrator_attr)
        if calibrator is None or not calibrator.winfo_exists():
            calibrator = DreamMemoryCalibratorWindow(
                self,
                config_path=self.config_path,
                get_map_id=lambda: get_selected_map_id(widgets),
                screenshot_cb=self._capture_for_coord_ruler,
                on_saved=lambda mid, p=pk: self._on_dream_map_saved(mid, pk=p),
                touch_width=touch_w,
                touch_height=touch_h,
                pk_mode=pk,
            )
            setattr(self, calibrator_attr, calibrator)
        else:
            calibrator._reload_item_list()
        calibrator.show_window()

    def _build_task_instance(self, entry: TaskEntry):
        if not entry.available:
            return None
        if entry.task_id == "auto_lighthouse":
            return self._build_auto_lighthouse_task()
        if entry.task_id == "hunt_ice_beast":
            return self._build_ice_beast_task()
        if entry.task_id == "hunt_monster":
            return self._build_monster_task()
        if entry.task_id == "donate_alliance_supplies":
            return self._build_donate_alliance_supplies_task()
        if entry.task_id == "auto_mining":
            return self._build_auto_mining_task()
        if entry.task_id == "auto_train_troops":
            return self._build_auto_train_troops_task()
        if entry.task_id == "collect_supplies":
            return self._build_collect_supplies_task()
        if entry.task_id == "collect_commander_supplies":
            return self._build_collect_commander_supplies_task()
        if entry.task_id == "collect_pet_supplies":
            return self._build_collect_pet_supplies_task()
        if entry.task_id == "auto_shop_exchange":
            return self._build_auto_shop_exchange_task()
        return None

    def _build_auto_lighthouse_task(self) -> AutoLighthouseTask:
        cfg = self.config.get("tasks", {}).get("auto_lighthouse", {})
        merged = merge_lighthouse_config(cfg)
        return AutoLighthouseTask(
            adb=self._get_adb(),
            coords=merged["coords"],
            interval=float(self.var_lighthouse_interval.get() * 60),
            formation_slot=int(
                _normalize_formation_slot(self.var_lighthouse_formation_slot.get())
            ),
            use_stamina=bool(self.var_common_use_stamina.get()),
            stamina_can_limit=int(self.var_common_stamina_can_limit.get()),
            use_formation=bool(self.var_common_use_formation.get()),
            event_period=bool(self.var_lighthouse_event_period.get()),
            monster_cooldown=float(self.var_lighthouse_monster_cooldown.get() * 60),
            step_delay=merged["step_delay"],
            on_status=self._on_status,
        )

    def _build_ice_beast_task(self) -> HuntIceBeastTask:
        hunt_cfg = self.config.get("tasks", {}).get("hunt_ice_beast", {})
        return HuntIceBeastTask(
            adb=self._get_adb(),
            coords=hunt_cfg.get("coords", {}),
            interval=float(self.var_interval.get() * 60),
            beast_level=int(self.var_level.get()),
            default_beast_level=hunt_cfg.get("default_beast_level", 1),
            formation_name=str(_normalize_formation_slot(self.var_formation_slot.get())),
            rally_duration_minutes=int(self.var_rally_duration.get()),
            skip_hour=hunt_cfg.get("skip_hour", 21),
            step_delay=hunt_cfg.get("step_delay", 1.5),
            use_stamina=bool(self.var_common_use_stamina.get()),
            stamina_can_limit=int(self.var_common_stamina_can_limit.get()),
            use_formation=bool(self.var_common_use_formation.get()),
            adjust_level=bool(self.var_common_adjust_level.get()),
            on_status=self._on_status,
        )

    def _build_monster_task(self) -> HuntMonsterTask:
        monster_cfg = self.config.get("tasks", {}).get("hunt_monster", {})
        return HuntMonsterTask(
            adb=self._get_adb(),
            coords=monster_cfg.get("coords", {}),
            interval=float(self.var_monster_interval.get() * 60),
            monster_level=int(self.var_monster_level.get()),
            formation_name=str(_normalize_formation_slot(self.var_monster_formation_slot.get())),
            skip_hour=monster_cfg.get("skip_hour", 21),
            step_delay=monster_cfg.get("step_delay", 1.5),
            use_stamina=bool(self.var_common_use_stamina.get()),
            stamina_can_limit=int(self.var_common_stamina_can_limit.get()),
            use_formation=bool(self.var_common_use_formation.get()),
            adjust_level=bool(self.var_common_adjust_level.get()),
            on_status=self._on_status,
        )

    def _build_auto_mining_task(self) -> AutoMiningTask:
        mining_cfg = self.config.get("tasks", {}).get("auto_mining", {})
        merged = merge_mining_config(mining_cfg)
        level_min = _normalize_mining_level(self.var_mining_level_min.get())
        level_max = _normalize_mining_level(self.var_mining_level_max.get())
        return AutoMiningTask(
            adb=self._get_adb(),
            coords=merged["coords"],
            interval=float(self.var_mining_interval.get() * 60),
            level_min=level_min,
            level_max=level_max,
            use_mining_hero=bool(self.var_use_mining_hero.get()),
            adjust_level=bool(self.var_common_adjust_level.get()),
            skip_hour=mining_cfg.get("skip_hour", -1),
            step_delay=merged["step_delay"],
            hero_match_threshold=merged["hero_match_threshold"],
            on_status=self._on_status,
        )

    def _build_donate_alliance_supplies_task(self) -> DonateAllianceSuppliesTask:
        donate_cfg = self.config.get("tasks", {}).get("donate_alliance_supplies", {})
        merged = merge_donate_config(donate_cfg)
        return DonateAllianceSuppliesTask(
            adb=self._get_adb(),
            coords=merged["coords"],
            interval=float(self.var_donate_interval.get() * 60),
            donate_times=int(self.var_donate_times.get()),
            donate_click_delay=merged["donate_click_delay"],
            skip_hour=donate_cfg.get("skip_hour", -1),
            step_delay=merged["step_delay"],
            on_status=self._on_status,
        )

    def _build_collect_commander_supplies_task(self) -> CollectCommanderSuppliesTask:
        cfg = self.config.get("tasks", {}).get("collect_commander_supplies", {})
        merged = merge_commander_config(cfg)
        return CollectCommanderSuppliesTask(
            adb=self._get_adb(),
            coords=merged["coords"],
            step_delay=merged["step_delay"],
            double_tap_delay=merged["double_tap_delay"],
            on_status=self._on_status,
        )

    def _build_collect_pet_supplies_task(self) -> CollectPetSuppliesTask:
        cfg = self.config.get("tasks", {}).get("collect_pet_supplies", {})
        merged = merge_pet_supplies_config(cfg)
        return CollectPetSuppliesTask(
            adb=self._get_adb(),
            coords=merged["coords"],
            step_delay=merged["step_delay"],
            on_status=self._on_status,
        )

    def _build_auto_shop_exchange_task(self) -> AutoShopExchangeTask:
        cfg = self.config.get("tasks", {}).get("auto_shop_exchange", {})
        merged = merge_shop_exchange_config(cfg)
        return AutoShopExchangeTask(
            adb=self._get_adb(),
            coords=merged["coords"],
            step_delay=merged["step_delay"],
            on_status=self._on_status,
        )

    def _build_auto_train_troops_task(self) -> AutoTrainTroopsTask:
        cfg = self.config.get("tasks", {}).get("auto_train_troops", {})
        merged = merge_train_config(cfg)
        return AutoTrainTroopsTask(
            adb=self._get_adb(),
            coords=merged["coords"],
            interval=float(self.var_train_interval.get() * 3600),
            step_delay=merged["step_delay"],
            train_ready_threshold=merged["train_ready_threshold"],
            on_status=self._on_status,
        )

    def _build_collect_supplies_task(self) -> CollectSuppliesTask:
        cfg = self.config.get("tasks", {}).get("collect_supplies", {})
        merged = merge_collect_config(cfg)
        return CollectSuppliesTask(
            adb=self._get_adb(),
            coords=merged["coords"],
            interval=float(self.var_collect_interval.get() * 3600),
            step_delay=merged["step_delay"],
            on_status=self._on_status,
        )

    def _collect_loop_tasks(self) -> list:
        tasks: list = []
        for entry in loop_tasks():
            if self._task_vars[entry.task_id].get():
                task = self._build_task_instance(entry)
                if task is not None:
                    tasks.append(task)
        return tasks

    def _collect_once_tasks(self) -> list:
        tasks: list = []
        for entry in once_tasks():
            if not entry.available:
                continue
            if self._task_vars[entry.task_id].get():
                task = self._build_task_instance(entry)
                if task is not None:
                    tasks.append(task)
        return tasks

    def _any_loop_selected(self) -> bool:
        return any(self._task_vars[e.task_id].get() for e in loop_tasks())

    def _any_once_selected(self) -> bool:
        return any(
            self._task_vars[e.task_id].get() for e in once_tasks() if e.available
        )

    def _is_loop_running(self) -> bool:
        return self._loop_worker is not None and self._loop_worker.is_alive()

    def _is_once_running(self) -> bool:
        return self._once_worker is not None and self._once_worker.is_alive()

    def _is_dream_memory_running(self) -> bool:
        return (
            self._dream_memory_worker is not None and self._dream_memory_worker.is_alive()
        )

    def _is_dream_pk_running(self) -> bool:
        return self._dream_pk_worker is not None and self._dream_pk_worker.is_alive()

    def _is_any_dream_running(self) -> bool:
        return self._is_dream_memory_running() or self._is_dream_pk_running()

    def _is_alliance_regular_running(self) -> bool:
        return self._alliance_worker is not None and self._alliance_worker.is_alive()

    def _is_alliance_admin_running(self) -> bool:
        return (
            self._alliance_admin_worker is not None
            and self._alliance_admin_worker.is_alive()
        )

    def _is_alliance_running(self) -> bool:
        return self._is_alliance_regular_running() or self._is_alliance_admin_running()

    def _is_account_check_running(self) -> bool:
        return (
            self._account_check_worker is not None
            and self._account_check_worker.is_alive()
        )

    def _is_any_running(self) -> bool:
        return (
            self._is_loop_running()
            or self._is_once_running()
            or self._is_any_dream_running()
            or self._is_alliance_running()
            or self._is_account_check_running()
        )

    def _configure_once_action_buttons(self, *, enabled: bool) -> None:
        _ = enabled
        self._update_run_control_buttons()

    def _build_hosting_tasks(self) -> list:
        tasks: list = []
        for task_id in HOSTING_TASK_IDS:
            entry = next((e for e in TASK_ENTRIES if e.task_id == task_id), None)
            if entry is None or not entry.available:
                continue
            task = self._build_task_instance(entry)
            if task is not None:
                tasks.append(task)
        return tasks

    def _update_run_control_buttons(self, *, running: bool | None = None) -> None:
        """统一刷新开始/结束/一键辅助/打野/巨兽/检查账号按钮状态。"""
        if running is None:
            running = self._is_any_running()
        if hasattr(self, "btn_start"):
            self.btn_start.configure(state=tk.DISABLED if running else tk.NORMAL)
        if hasattr(self, "btn_stop"):
            self.btn_stop.configure(state=tk.NORMAL if running else tk.DISABLED)
        if hasattr(self, "btn_hosting"):
            self.btn_hosting.configure(state=tk.DISABLED if running else tk.NORMAL)
        if hasattr(self, "btn_quick_monster"):
            self.btn_quick_monster.configure(
                state=tk.DISABLED if running else tk.NORMAL
            )
        if hasattr(self, "btn_quick_ice"):
            self.btn_quick_ice.configure(state=tk.DISABLED if running else tk.NORMAL)
        if hasattr(self, "btn_check_account"):
            self.btn_check_account.configure(
                state=tk.DISABLED if running else tk.NORMAL
            )
        for widgets in (self._dream_widgets, self._dream_pk_widgets):
            if widgets is None:
                continue
            widgets.btn_start.configure(state=tk.DISABLED if running else tk.NORMAL)
            widgets.btn_stop.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _set_loop_buttons(self, running: bool) -> None:
        self._update_run_control_buttons(running=running)

    def _set_once_buttons(self, running: bool) -> None:
        self._update_run_control_buttons(running=running)

    def _set_alliance_buttons(self, running: bool) -> None:
        self._update_run_control_buttons(running=running)

    def _set_alliance_admin_buttons(self, running: bool) -> None:
        self._update_run_control_buttons(running=running)

    def _set_dream_buttons(self, running: bool, *, pk: bool) -> None:
        _ = pk
        self._update_run_control_buttons(running=running)

    def _set_dream_memory_buttons(self, running: bool) -> None:
        self._set_dream_buttons(running, pk=False)

    def _set_dream_pk_buttons(self, running: bool) -> None:
        self._set_dream_buttons(running, pk=True)

    def _current_task_tab_name(self) -> str:
        notebook = getattr(self, "_task_notebook", None)
        if notebook is None:
            return ""
        try:
            return notebook.tab(notebook.select(), "text")
        except tk.TclError:
            return ""

    def _start_from_current_tab(self) -> None:
        """根据当前运行任务 Tab 启动对应内容。"""
        if self._is_account_check_running():
            messagebox.showwarning("提示", "正在检查账号名称，请先点「结束」")
            return
        tab_name = self._current_task_tab_name()
        dispatch = {
            "循环任务": self._start_loop,
            "一次性任务": self._run_once_batch,
            "联盟总动员自动刷新": self._start_alliance_mobilization,
            "联盟管理员刷新": self._start_alliance_admin,
            "寻梦记忆": lambda: self._start_dream_session(pk=False),
            "寻梦记忆PK": lambda: self._start_dream_session(pk=True),
        }
        handler = dispatch.get(tab_name)
        if handler is None:
            messagebox.showwarning("提示", f"当前 Tab「{tab_name or '未知'}」无法启动")
            return
        handler()

    def _stop_running_tasks(self) -> None:
        """按当前实际运行类型结束，不看 Tab。"""
        if not self._is_any_running():
            self._on_status("当前没有运行中的任务")
            self._update_run_control_buttons()
            return
        if self._is_loop_running():
            self._stop_loop()
        if self._is_once_running():
            self._stop_once()
        if self._is_alliance_regular_running():
            self._stop_alliance_mobilization()
        if self._is_alliance_admin_running():
            self._stop_alliance_admin()
        if self._is_dream_memory_running():
            self._stop_dream_session(pk=False)
        if self._is_dream_pk_running():
            self._stop_dream_session(pk=True)
        if self._is_account_check_running():
            self._stop_check_account_name()

    def _read_account_name_from_screen(self, screen) -> str:
        x1, y1, x2, y2 = ACCOUNT_NAME_ROI
        h, w = screen.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return ""
        crop = screen[y1:y2, x1:x2]
        big = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        text, engine = ocr_chip_text(big)
        cleaned = re.sub(r"\s+", "", (text or "").strip())
        # 去掉常见 OCR 噪点符号，保留中文/字母/数字/下划线
        cleaned = re.sub(r"[^\w\u4e00-\u9fff·•\-]", "", cleaned)
        self._on_status(f"账号名称 OCR({engine})：{cleaned or '空'}")
        return cleaned

    def _start_check_account_name(self) -> None:
        if self._is_any_running():
            messagebox.showwarning("提示", "已有任务在运行，请先点「结束」")
            return
        if not self._ensure_device():
            return

        self._account_check_stop_event.clear()
        self._update_run_control_buttons(running=True)
        self._on_status("开始检查账号名称…")

        def work():
            try:
                adb = self._get_adb()
                nav = WildernessNavigator(
                    adb,
                    on_status=self._on_status,
                    interrupted=self._account_check_stop_event.is_set,
                )
                self._on_status("确保在野外主界面…")
                nav.ensure_wilderness()
                if self._account_check_stop_event.is_set():
                    raise InterruptedError("任务已停止")

                px, py = ACCOUNT_PROFILE_TAP
                self._on_status(f"打开用户页 @ ({px},{py})")
                adb.tap(px, py)
                time.sleep(2.0)
                if self._account_check_stop_event.is_set():
                    raise InterruptedError("任务已停止")

                name = self._read_account_name_from_screen(adb.screenshot())
                if not name:
                    raise RuntimeError("未能识别账号名称，请确认已打开用户页")

                def apply():
                    self.config.setdefault("gui", {})["account_name"] = name
                    self._set_account_name_display(name)
                    try:
                        self._save_config()
                    except ValueError:
                        pass
                    self._on_status(f"账号名称：{name}")
                    messagebox.showinfo("检查账号名称", f"当前账号：{name}")

                self.after(0, apply)
                try:
                    adb.back()
                    time.sleep(0.8)
                    nav.try_return_to_wilderness()
                except Exception:
                    pass
            except InterruptedError:
                self.after(0, lambda: self._on_status("检查账号名称已停止"))
            except Exception as exc:
                err = str(exc)

                def fail(msg=err):
                    self._on_status(f"检查账号名称失败：{msg}")
                    messagebox.showerror("检查账号名称", msg)

                self.after(0, fail)
            finally:
                self.after(0, self._on_account_check_done)

        self._account_check_worker = threading.Thread(target=work, daemon=True)
        self._account_check_worker.start()

    def _stop_check_account_name(self) -> None:
        self._account_check_stop_event.set()
        self._on_status("正在停止检查账号名称…")

    def _on_account_check_done(self) -> None:
        self._account_check_worker = None
        self._update_run_control_buttons(running=False)

    def _ensure_device(self) -> bool:
        adb = self._get_adb()
        if adb.wait_for_device(retries=10, interval=2.0):
            self._on_status(f"设备已连接：{adb.address}")
            return True
        self._on_status("无法连接模拟器，请确认雷电已启动")
        return False

    def _start_loop(self) -> None:
        if self._is_account_check_running():
            messagebox.showwarning("提示", "正在检查账号名称，请先点「结束」")
            return
        if self._is_loop_running():
            messagebox.showwarning("提示", "循环任务已在运行中")
            return
        if self._is_once_running():
            messagebox.showwarning("提示", "一次性任务正在执行，请稍候或先停止")
            return
        if self._is_any_dream_running():
            messagebox.showwarning("提示", "寻梦记忆正在运行，请先点「结束」")
            return
        if self._is_alliance_running():
            messagebox.showwarning("提示", "联盟总动员正在运行，请先点「结束」")
            return
        if not self._any_loop_selected():
            messagebox.showwarning("提示", "请勾选要运行的循环任务")
            return

        try:
            self._save_config()
            self.config = self._load_config()
            self._loop_tasks = self._collect_loop_tasks()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        if not self._loop_tasks:
            messagebox.showwarning("提示", "没有可启动的循环任务")
            return

        names = "、".join(t.name for t in self._loop_tasks)
        self._launch_loop_worker(status=f"启动循环任务：{names}")

    def _start_quick_loop_task(self, task_id: str) -> None:
        """一键打野 / 一键集结巨兽：直接启动对应循环任务（不依赖勾选）。"""
        labels = {
            "hunt_monster": "一键打野",
            "hunt_ice_beast": "一键集结巨兽",
        }
        label = labels.get(task_id, task_id)
        if self._is_account_check_running():
            messagebox.showwarning("提示", "正在检查账号名称，请先点「结束」")
            return
        if self._is_loop_running():
            messagebox.showwarning("提示", "循环任务已在运行中")
            return
        if self._is_once_running():
            messagebox.showwarning("提示", "一次性任务正在执行，请稍候或先停止")
            return
        if self._is_any_dream_running():
            messagebox.showwarning("提示", "寻梦记忆正在运行，请先点「结束」")
            return
        if self._is_alliance_running():
            messagebox.showwarning("提示", "联盟总动员正在运行，请先点「结束」")
            return

        entry = next((e for e in loop_tasks() if e.task_id == task_id), None)
        if entry is None or not entry.available:
            messagebox.showwarning("提示", f"「{label}」任务不可用")
            return

        try:
            self._save_config()
            self.config = self._load_config()
            task = self._build_task_instance(entry)
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        if task is None:
            messagebox.showwarning("提示", f"「{label}」任务不可用")
            return

        self._loop_tasks = [task]
        self._launch_loop_worker(status=f"{label}：{task.name}")

    def _launch_loop_worker(self, *, status: str) -> None:
        self._loop_stop_event.clear()
        self._set_loop_buttons(running=True)
        self._on_status(status)

        def work():
            try:
                if not self._ensure_device():
                    return
                self._run_loop_scheduler()
            except Exception as exc:
                self._on_status(f"循环任务异常：{exc}")
            finally:
                self.after(0, self._on_loop_worker_done)

        self._loop_worker = threading.Thread(target=work, daemon=True)
        self._loop_worker.start()

    def _run_once_batch(self) -> None:
        if self._is_once_running():
            messagebox.showwarning("提示", "一次性任务正在执行中")
            return
        if self._is_loop_running():
            messagebox.showwarning("提示", "循环任务运行中，请先停止后再执行一次性任务")
            return
        if self._is_any_dream_running():
            messagebox.showwarning("提示", "寻梦记忆正在运行，请先点「结束」")
            return
        if self._is_alliance_running():
            messagebox.showwarning("提示", "联盟总动员正在运行，请先点「结束」")
            return
        if not self._any_once_selected():
            messagebox.showwarning("提示", "请勾选要执行的一次性任务")
            return

        try:
            self._save_config()
            self.config = self._load_config()
            self._once_tasks = self._collect_once_tasks()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        if not self._once_tasks:
            messagebox.showwarning("提示", "所选一次性任务尚未开放")
            return

        self._once_stop_event.clear()
        self._set_once_buttons(running=True)
        names = "、".join(t.name for t in self._once_tasks)
        self._on_status(f"执行一次性任务：{names}")

        def work():
            try:
                if not self._ensure_device():
                    return
                for index, task in enumerate(self._once_tasks):
                    if self._once_stop_event.is_set():
                        self._on_status("一次性任务已取消")
                        return
                    self._execute_once(task)
                    if index < len(self._once_tasks) - 1:
                        self._on_status(f"等待 {ONCE_TASK_GAP_SEC} 秒后执行下一项…")
                        for _ in range(ONCE_TASK_GAP_SEC * 10):
                            if self._once_stop_event.is_set():
                                self._on_status("一次性任务已取消")
                                return
                            time.sleep(0.1)
                self._on_status("一次性任务全部完成")
            except InterruptedError:
                self._on_status("一次性任务已停止")
            except Exception as exc:
                self._on_status(f"一次性任务异常：{exc}")
            finally:
                self.after(0, self._on_once_worker_done)

        self._once_worker = threading.Thread(target=work, daemon=True)
        self._once_worker.start()

    def _run_hosting_batch(self) -> None:
        if self._is_account_check_running():
            messagebox.showwarning("提示", "正在检查账号名称，请先点「结束」")
            return
        if self._is_once_running():
            messagebox.showwarning("提示", "任务正在执行中")
            return
        if self._is_loop_running():
            messagebox.showwarning("提示", "循环任务运行中，请先停止")
            return
        if self._is_any_dream_running():
            messagebox.showwarning("提示", "寻梦记忆正在运行，请先点「结束」")
            return
        if self._is_alliance_running():
            messagebox.showwarning("提示", "联盟总动员正在运行，请先点「结束」")
            return

        try:
            self._save_config()
            self.config = self._load_config()
            self._once_tasks = self._build_hosting_tasks()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        if not self._once_tasks:
            messagebox.showwarning("提示", "辅助任务不可用")
            return

        self._once_stop_event.clear()
        self._set_once_buttons(running=True)
        names = " → ".join(t.name for t in self._once_tasks)
        self._on_status(f"一键辅助：{names}")

        def work():
            try:
                if not self._ensure_device():
                    return
                for index, task in enumerate(self._once_tasks):
                    if self._once_stop_event.is_set():
                        self._on_status("一键辅助已取消")
                        return
                    self._execute_hosting_task(task)
                    if index < len(self._once_tasks) - 1:
                        self._on_status(f"等待 {ONCE_TASK_GAP_SEC} 秒后执行下一项…")
                        for _ in range(ONCE_TASK_GAP_SEC * 10):
                            if self._once_stop_event.is_set():
                                self._on_status("一键辅助已取消")
                                return
                            time.sleep(0.1)
                self._on_status("一键辅助全部完成")
            except InterruptedError:
                self._on_status("一键辅助已停止")
            except Exception as exc:
                self._on_status(f"一键辅助异常：{exc}")
            finally:
                self.after(0, self._on_once_worker_done)

        self._once_worker = threading.Thread(target=work, daemon=True)
        self._once_worker.start()

    def _execute_hosting_task(self, task) -> None:
        self._on_status(f"▶ 辅助：{task.name}")
        try:
            if hasattr(task, "reset_stop"):
                task.reset_stop()
            if hasattr(task, "run_once"):
                task.run_once(force=True)
            elif hasattr(task, "execute"):
                task.execute()
            self._on_status(f"✓ 完成：{task.name}")
        except InterruptedError:
            raise
        except Exception as exc:
            self._on_status(f"✗ [{task.name}] 失败：{exc}")
        finally:
            try:
                return_to_main_screen(self._get_adb(), on_status=self._on_status)
            except Exception as exc:
                self._on_status(f"返回主界面失败：{exc}")

    def _execute_once(self, task) -> None:
        self._on_status(f"▶ 一次性：{task.name}")
        try:
            if hasattr(task, "run_hunt_cycle"):
                task.run_hunt_cycle()
            elif hasattr(task, "execute"):
                task.execute()
            else:
                task.run_once()
            self._on_status(f"✓ 完成：{task.name}")
        except InterruptedError:
            raise
        except Exception as exc:
            self._on_status(f"✗ [{task.name}] 失败：{exc}")
        finally:
            try:
                return_to_main_screen(self._get_adb(), on_status=self._on_status)
            except Exception as exc:
                self._on_status(f"返回主界面失败：{exc}")

    def _run_loop_scheduler(self) -> None:
        for task in self._loop_tasks:
            if hasattr(task, "reset_stop"):
                task.reset_stop()

        self._on_status("循环调度已开始，首次将全部执行一遍…")
        for index, task in enumerate(self._loop_tasks):
            if self._loop_stop_event.is_set():
                return
            self._on_status(f"▶ 首次：{task.name}")
            try:
                task.run_once(force=True)
            except InterruptedError:
                self._on_status("循环任务已停止")
                return
            except Exception as exc:
                self._on_status(f"[{task.name}] 异常：{exc}")
            if index < len(self._loop_tasks) - 1:
                for _ in range(20):
                    if self._loop_stop_event.is_set():
                        return
                    time.sleep(0.1)

        self._on_status("首次执行完成，进入循环等待…")

        while not self._loop_stop_event.is_set():
            ready_tasks = [
                task
                for task in self._loop_tasks
                if not self._loop_stop_event.is_set() and task.should_run()
            ]

            if not ready_tasks:
                for _ in range(20):
                    if self._loop_stop_event.is_set():
                        break
                    time.sleep(0.1)
                continue

            if len(ready_tasks) > 1:
                waiting_names = "、".join(t.name for t in ready_tasks[1:])
                self._on_status(f"{ready_tasks[0].name} 执行中，{waiting_names} 排队等待")

            task = ready_tasks[0]
            try:
                task.run_once()
            except InterruptedError:
                self._on_status("循环任务已停止")
                return
            except Exception as exc:
                self._on_status(f"[{task.name}] 异常：{exc}")

            for _ in range(20):
                if self._loop_stop_event.is_set():
                    break
                time.sleep(0.1)

        self._on_status("循环任务已停止")

    def _stop_loop(self) -> None:
        self._loop_stop_event.set()
        for task in self._loop_tasks:
            if hasattr(task, "stop"):
                task.stop()
        self._on_status("正在停止循环任务…")

    def _stop_once(self) -> None:
        self._once_stop_event.set()
        for task in self._once_tasks:
            if hasattr(task, "stop"):
                task.stop()
        self._on_status("正在停止一次性任务…")

    def _start_dream_session(self, pk: bool = False) -> None:
        if self._is_dream_memory_running() if not pk else self._is_dream_pk_running():
            title = "寻梦记忆PK" if pk else "寻梦记忆"
            messagebox.showwarning("提示", f"{title}已在运行")
            return
        if (
            self._is_loop_running()
            or self._is_once_running()
            or self._is_any_dream_running()
            or self._is_alliance_running()
        ):
            messagebox.showwarning("提示", "请先停止其他任务")
            return

        widgets = self._dream_widgets_for(pk)
        map_id = get_selected_map_id(widgets)
        if not map_id:
            messagebox.showwarning("提示", "请先选择或创建地图")
            return

        try:
            self._save_config()
            dm_cfg = load_dream_cfg(self, pk)
            game_map = load_map(map_id, maps_dir=dm_cfg.maps_dir)
        except FileNotFoundError as exc:
            messagebox.showerror("地图错误", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        if not ocr_engine_available(dm_cfg.ocr_engine):
            engine = resolve_ocr_engine(dm_cfg.ocr_engine)
            if engine == "rapidocr":
                messagebox.showerror(
                    "RapidOCR 未安装",
                    "请运行:\n"
                    "  .venv\\Scripts\\pip.exe install rapidocr-onnxruntime onnxruntime",
                )
            else:
                messagebox.showerror(
                    "Tesseract 未安装",
                    f"未找到:\n{dm_cfg.tesseract_cmd}",
                )
            return

        title = "寻梦记忆PK" if pk else "寻梦记忆"
        self._set_dream_buttons(running=True, pk=pk)
        self._on_status(f"{title}：{game_map.name}")

        def work():
            try:
                if not self._ensure_device():
                    return
                session = DreamMemorySession(
                    self._get_adb(),
                    game_map,
                    config=dm_cfg,
                    on_status=self._on_status,
                )
                if pk:
                    self._dream_pk_session = session
                else:
                    self._dream_memory_session = session
                session.reset_stop()
                session.run_until_stopped()
            except Exception as exc:
                self._on_status(f"{title}异常：{exc}")
            finally:
                if pk:
                    self._dream_pk_session = None
                    self.after(0, lambda: self._on_dream_session_done(pk=True))
                else:
                    self._dream_memory_session = None
                    self.after(0, lambda: self._on_dream_session_done(pk=False))

        worker = threading.Thread(target=work, daemon=True)
        if pk:
            self._dream_pk_worker = worker
        else:
            self._dream_memory_worker = worker
        worker.start()

    def _stop_dream_session(self, pk: bool = False) -> None:
        title = "寻梦记忆PK" if pk else "寻梦记忆"
        session = self._dream_pk_session if pk else self._dream_memory_session
        if session is not None:
            session.stop()
        self._on_status(f"正在结束{title}…")

    def _on_dream_session_done(self, pk: bool) -> None:
        self._set_dream_buttons(running=False, pk=pk)
        if pk:
            self._dream_pk_worker = None
        else:
            self._dream_memory_worker = None

    def _start_dream_memory(self) -> None:
        self._start_dream_session(pk=False)

    def _stop_dream_memory(self) -> None:
        self._stop_dream_session(pk=False)

    def _on_dream_memory_worker_done(self) -> None:
        self._on_dream_session_done(pk=False)

    def _start_alliance_mobilization(self) -> None:
        if self._is_alliance_running():
            messagebox.showwarning("提示", "联盟总动员已在运行")
            return
        if (
            self._is_loop_running()
            or self._is_once_running()
            or self._is_any_dream_running()
        ):
            messagebox.showwarning("提示", "请先停止其他任务")
            return

        type_keep_rules = self._collect_alliance_type_keep_rules()
        if not type_keep_rules:
            messagebox.showwarning(
                "提示",
                "请先在「功能参数 → 联盟总动员」勾选类型并指定保留底色",
            )
            return

        try:
            self._save_config()
            self.config = self._load_config()
            alliance_cfg = merge_alliance_config(
                self.config.get("alliance_mobilization", {})
            )
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        rules_text = self._format_type_keep_rules(alliance_cfg["type_keep_rules"])
        self._set_alliance_buttons(running=True)
        self._on_status(
            f"联盟总动员：保留 {rules_text}，"
            f"循环 {int(alliance_cfg['scan_interval'] // 60)} 分钟"
        )

        def work():
            try:
                if not self._ensure_device():
                    return
                session = AllianceMobilizationSession(
                    self._get_adb(),
                    target_types=list(alliance_cfg["target_types"]),
                    score_threshold=int(alliance_cfg["score_threshold"]),
                    scan_interval=float(alliance_cfg["scan_interval"]),
                    step_delay=float(alliance_cfg["step_delay"]),
                    match_threshold=float(alliance_cfg["match_threshold"]),
                    countdown_threshold=float(alliance_cfg["countdown_threshold"]),
                    ocr_engine=str(alliance_cfg["ocr_engine"]),
                    coords=alliance_cfg["coords"],
                    slots=alliance_cfg["slots"],
                    on_status=self._on_status,
                )
                self._alliance_session = session
                session.run_until_stopped()
            except Exception as exc:
                self._on_status(f"联盟总动员异常：{exc}")
            finally:
                self._alliance_session = None
                self.after(0, self._on_alliance_worker_done)

        self._alliance_worker = threading.Thread(target=work, daemon=True)
        self._alliance_worker.start()

    def _stop_alliance_mobilization(self) -> None:
        session = self._alliance_session
        if session is not None:
            session.stop()
        self._on_status("正在结束联盟总动员…")

    def _on_alliance_worker_done(self) -> None:
        self._set_alliance_buttons(running=False)
        self._alliance_worker = None

    def _start_alliance_admin(self) -> None:
        if self._is_alliance_running():
            messagebox.showwarning("提示", "联盟总动员已在运行")
            return
        if (
            self._is_loop_running()
            or self._is_once_running()
            or self._is_any_dream_running()
        ):
            messagebox.showwarning("提示", "请先停止其他任务")
            return

        type_keep_rules = self._collect_alliance_type_keep_rules()
        if not type_keep_rules:
            messagebox.showwarning(
                "提示",
                "请先在「功能参数 → 联盟总动员」勾选类型并指定保留底色",
            )
            return

        try:
            self.config = self._load_config()
            self._save_config()
            self.config = self._load_config()
            admin_cfg = merge_alliance_admin_config(
                self.config.get("alliance_mobilization_admin", {})
            )
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        rules_text = self._format_type_keep_rules(admin_cfg["type_keep_rules"])
        self._set_alliance_admin_buttons(running=True)
        mask_xy = admin_cfg["coords"].get(
            "detail_mask_tap", list(CALIBRATED_DETAIL_MASK_TAP)
        )
        self._on_status(
            f"联盟管理员刷新：保留 {rules_text}，"
            f"循环 {int(admin_cfg['scan_interval'] // 60)} 分钟，"
            f"遮罩 ({mask_xy[0]},{mask_xy[1]})"
        )

        def work():
            try:
                if not self._ensure_device():
                    return
                run_cfg = dict(admin_cfg)
                run_cfg["use_score_ocr"] = False
                session = AllianceMobilizationAdminSession(
                    self._get_adb(),
                    admin_cfg=run_cfg,
                    on_status=self._on_status,
                )
                self._alliance_admin_session = session
                session.run_until_stopped()
            except Exception as exc:
                self._on_status(f"联盟管理员刷新异常：{exc}")
            finally:
                self._alliance_admin_session = None
                self.after(0, self._on_alliance_admin_worker_done)

        self._alliance_admin_worker = threading.Thread(target=work, daemon=True)
        self._alliance_admin_worker.start()

    def _stop_alliance_admin(self) -> None:
        session = self._alliance_admin_session
        if session is not None:
            session.stop()
        self._on_status("正在结束联盟管理员刷新…")

    def _on_alliance_admin_worker_done(self) -> None:
        self._set_alliance_admin_buttons(running=False)
        self._alliance_admin_worker = None

    def _on_loop_worker_done(self) -> None:
        self._set_loop_buttons(running=False)
        self._loop_tasks.clear()
        self._loop_worker = None

    def _on_once_worker_done(self) -> None:
        self._set_once_buttons(running=False)
        self._once_tasks.clear()
        self._once_worker = None

    def _on_close(self) -> None:
        if self._is_any_running():
            if messagebox.askokcancel("退出", "任务正在运行，确定停止并退出？"):
                self._stop_loop()
                self._stop_once()
                self._stop_dream_memory()
                self._stop_dream_session(pk=True)
                self._stop_alliance_mobilization()
                self._stop_alliance_admin()
                self._stop_check_account_name()
                self.destroy()
        else:
            try:
                self._save_config()
            except ValueError as exc:
                messagebox.showerror("参数错误", str(exc))
                return
            self.destroy()


IceBeastApp = EndlessWinterApp


def main(argv: list[str] | None = None) -> None:
    import argparse

    from core.config_path import add_config_arg, parse_config_from_args

    parser = argparse.ArgumentParser(description="无尽冬日 — 自动脚本 GUI")
    add_config_arg(parser)
    args, _unknown = parser.parse_known_args(argv)
    config_path = ensure_config_file(resolve_config_path(args.config))
    app = EndlessWinterApp(config_path=config_path)
    app.mainloop()
