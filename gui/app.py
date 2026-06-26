"""无尽冬日 — 图形界面。"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

import yaml

from core.adb_client import AdbClient
from core.config_path import (
    DEFAULT_CONFIG_PATH,
    default_instance_name,
    ensure_config_file,
    resolve_config_path,
)
from core.navigation import return_to_main_screen
from gui.coord_ruler import CoordRulerWindow
from gui.task_registry import TaskEntry, loop_tasks, once_tasks
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
from tasks.collect_supplies import CollectSuppliesTask, merge_task_config as merge_collect_config
from tasks.donate_alliance_supplies import (
    DonateAllianceSuppliesTask,
    merge_task_config as merge_donate_config,
)
from tasks.hunt_ice_beast import HuntIceBeastTask
from tasks.hunt_monster import HuntMonsterTask

ROOT = Path(__file__).parent.parent

# 循环任务互斥组：同组内只能勾选一个
LOOP_EXCLUSIVE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"hunt_ice_beast", "hunt_monster"}),
)

MAIN_WIDTH = 580
MAIN_HEIGHT = 680
LOG_WIDTH = 300
ONCE_TASK_GAP_SEC = 3
ONCE_TASK_COLUMNS = 2
LOOP_TASK_COLUMNS = 2

FORMATION_SLOT_MIN = 1
FORMATION_SLOT_MAX = 8

# 功能参数 Tab：标签列最小宽度 + 输入框左间距
FORM_LABEL_COL_MINSIZE = 120
FORM_INPUT_PADX = (14, 0)

MINING_LEVEL_MIN = 1
MINING_LEVEL_MAX = 8
MINING_LEVEL_DEFAULT_MIN = 5
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
    """规范采矿范围，保证 min < max。"""
    low = _normalize_mining_level(level_min, MINING_LEVEL_MIN)
    high = _normalize_mining_level(level_max, MINING_LEVEL_MAX)
    if high <= low:
        high = min(MINING_LEVEL_MAX, low + 1)
    if high <= low:
        low = max(MINING_LEVEL_MIN, high - 1)
    return low, high


def _configure_param_tab_grid(tab: ttk.Frame) -> None:
    tab.grid_columnconfigure(0, minsize=FORM_LABEL_COL_MINSIZE)


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
        self._log_visible = False
        self._coord_ruler_window: CoordRulerWindow | None = None

        self._build_ui()
        self._poll_log_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(300, lambda: self._refresh_devices(probe=False))

    def _load_config(self) -> dict:
        with open(self.config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _instance_name_value(self) -> str:
        if hasattr(self, "var_instance_name"):
            name = self.var_instance_name.get().strip()
            if name:
                return name
        return default_instance_name(self.config, self.config_path)

    def _update_window_title(self) -> None:
        self.title(f"无尽冬日 — 自动脚本 【{self._instance_name_value()}】")

    def _on_instance_name_changed(self, *_args) -> None:
        name = self.var_instance_name.get().strip()
        if not name:
            self.var_instance_name.set(default_instance_name(self.config, self.config_path))
        self._update_window_title()
        try:
            self._save_config()
        except ValueError:
            pass

    def _task_enabled_in_config(self, entry: TaskEntry) -> bool:
        task_cfg = self.config.get("tasks", {}).get(entry.config_key, {})
        if entry.task_id == "hunt_ice_beast":
            return bool(task_cfg.get("enabled", True))
        if entry.task_id == "hunt_monster":
            return bool(task_cfg.get("enabled", False))
        if entry.task_id == "donate_alliance_supplies":
            return bool(task_cfg.get("enabled", False))
        return bool(task_cfg.get("enabled", False))

    def _save_config(self) -> None:
        cfg = self.config
        tasks = cfg.setdefault("tasks", {})

        ice = tasks.setdefault("hunt_ice_beast", {})
        ice["enabled"] = bool(self._task_vars["hunt_ice_beast"].get())
        ice["interval"] = int(self.var_interval.get()) * 60
        ice["beast_level"] = int(self.var_level.get())
        ice["formation_name"] = str(_normalize_formation_slot(self.var_formation_slot.get()))
        ice["check_march_heroes"] = bool(self.var_check_march_heroes.get())
        ice["use_formation"] = bool(self.var_use_formation.get())
        ice["rally_duration_minutes"] = int(self.var_rally_duration.get())
        ice["use_stamina"] = bool(self.var_use_stamina.get())

        lighthouse = tasks.setdefault("auto_lighthouse", {})
        lighthouse["enabled"] = bool(self._task_vars["auto_lighthouse"].get())
        lighthouse["interval"] = int(self.var_lighthouse_interval.get()) * 60
        lighthouse["formation_slot"] = int(
            _normalize_formation_slot(self.var_lighthouse_formation_slot.get())
        )
        lighthouse["use_stamina"] = bool(self.var_lighthouse_use_stamina.get())
        lighthouse["monster_cooldown"] = int(self.var_lighthouse_monster_cooldown.get()) * 60
        merged_lighthouse = merge_lighthouse_config(lighthouse)
        lighthouse["step_delay"] = merged_lighthouse["step_delay"]
        lighthouse["coords"] = merged_lighthouse["coords"]

        monster = tasks.setdefault("hunt_monster", {})
        monster["enabled"] = bool(self._task_vars["hunt_monster"].get())
        monster["interval"] = int(self.var_monster_interval.get()) * 60
        monster["monster_level"] = int(self.var_monster_level.get())

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
        if level_max <= level_min:
            raise ValueError("采矿范围无效：后面的等级必须大于前面的等级")
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

        gui = cfg.setdefault("gui", {})
        gui["show_console"] = bool(self.var_show_console.get())
        gui["log_visible"] = self._log_visible
        if hasattr(self, "var_instance_name"):
            name = self.var_instance_name.get().strip()
            gui["instance_name"] = name or default_instance_name(cfg, self.config_path)

        if hasattr(self, "var_device_serial"):
            serial = self.var_device_serial.get().strip()
            if serial:
                host, port = AdbClient.parse_address(serial)
                dev = cfg.setdefault("device", {})
                dev["adb_host"] = host
                dev["adb_port"] = int(port)

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    def _update_ice_formation_state(self) -> None:
        if not hasattr(self, "_ice_formation_spinbox"):
            return
        if bool(self.var_use_formation.get()):
            self._ice_formation_spinbox.state(["!disabled"])
        else:
            self._ice_formation_spinbox.state(["disabled"])

    def _on_use_formation_changed(self) -> None:
        self._update_ice_formation_state()
        try:
            self._save_config()
        except ValueError:
            pass

    def _on_loop_checkbox_changed(self, task_id: str) -> None:
        """循环任务勾选互斥：冰原巨兽与自动打野只能二选一。"""
        if not self._task_vars[task_id].get():
            return
        for group in LOOP_EXCLUSIVE_GROUPS:
            if task_id not in group:
                continue
            for other_id in group:
                if other_id != task_id and other_id in self._task_vars:
                    self._task_vars[other_id].set(False)
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

        self._log_visible = bool(gui_cfg.get("log_visible", False))

        self._container = ttk.Frame(self)
        self._container.pack(fill=tk.BOTH, expand=True)

        pad = {"padx": 10, "pady": 4}
        self._left = ttk.Frame(self._container, width=MAIN_WIDTH)
        self._left.pack(side=tk.LEFT, fill=tk.BOTH)
        self._left.pack_propagate(False)

        # 底部系统区先 pack，避免被上方内容挤出可视区域
        bottom = ttk.Frame(self._left, padding=(10, 4, 10, 10))
        bottom.pack(fill=tk.X, side=tk.BOTTOM)

        settings_frame = ttk.LabelFrame(bottom, text="系统", padding=8)
        settings_frame.pack(fill=tk.X)

        name_row = ttk.Frame(settings_frame)
        name_row.pack(fill=tk.X)

        ttk.Label(name_row, text="实例名称").pack(side=tk.LEFT)
        self.var_instance_name = tk.StringVar(
            value=default_instance_name(self.config, self.config_path)
        )
        instance_entry = ttk.Entry(name_row, textvariable=self.var_instance_name, width=22)
        instance_entry.pack(side=tk.LEFT, padx=(6, 0))
        instance_entry.bind("<FocusOut>", self._on_instance_name_changed)
        instance_entry.bind("<Return>", self._on_instance_name_changed)

        ttk.Label(
            settings_frame,
            text=f"配置文件：{self.config_path.name}（标题显示实例名称，便于多开区分账号）",
            font=("", 8),
            foreground="gray",
        ).pack(anchor=tk.W, pady=(2, 0))

        self.var_show_console = tk.BooleanVar(value=bool(gui_cfg.get("show_console", False)))
        ttk.Checkbutton(
            settings_frame,
            text="启动时显示 CMD 窗口（调试用，下次启动 GUI 生效）",
            variable=self.var_show_console,
            command=self._save_config,
        ).pack(anchor=tk.W)

        dev_row = ttk.Frame(settings_frame)
        dev_row.pack(fill=tk.X, pady=(6, 0))

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

        ttk.Button(dev_row, text="刷新", width=6, command=lambda: self._refresh_devices(probe=True)).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        sys_row = ttk.Frame(settings_frame)
        sys_row.pack(fill=tk.X, pady=(6, 0))

        self.lbl_conn = ttk.Label(sys_row, text="设备：未连接")
        self.lbl_conn.pack(side=tk.LEFT)

        self.btn_toggle_log = ttk.Button(
            sys_row, text="显示日志 ▸", width=12, command=self._toggle_log
        )
        self.btn_toggle_log.pack(side=tk.RIGHT)

        ttk.Button(sys_row, text="测试连接", command=self._test_connection).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

        ttk.Button(sys_row, text="坐标标尺", command=self._open_coord_ruler, width=10).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

        param_outer = ttk.LabelFrame(self._left, text="功能参数", padding=8)
        param_outer.pack(fill=tk.X, **pad)

        self.var_interval = tk.IntVar(value=hunt_cfg.get("interval", 900) // 60)
        self.var_lighthouse_interval = tk.IntVar(
            value=lighthouse_cfg.get("interval", 3600) // 60
        )
        self.var_lighthouse_formation_slot = tk.IntVar(
            value=_normalize_formation_slot(lighthouse_cfg.get("formation_slot", 7))
        )
        self.var_lighthouse_use_stamina = tk.BooleanVar(
            value=bool(lighthouse_cfg.get("use_stamina", True))
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
        self.var_use_stamina = tk.BooleanVar(value=bool(hunt_cfg.get("use_stamina", True)))
        self.var_check_march_heroes = tk.BooleanVar(
            value=bool(hunt_cfg.get("check_march_heroes", True))
        )
        self.var_use_formation = tk.BooleanVar(value=bool(hunt_cfg.get("use_formation", True)))
        self.var_monster_interval = tk.IntVar(value=monster_cfg.get("interval", 300) // 60)
        self.var_monster_level = tk.IntVar(value=monster_cfg.get("monster_level", 30))
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

        notebook = ttk.Notebook(param_outer)
        notebook.pack(fill=tk.X)

        tab_ice = ttk.Frame(notebook, padding=6)
        notebook.add(tab_ice, text="冰原巨兽")
        _configure_param_tab_grid(tab_ice)

        row = 0
        ice_opts_row = ttk.Frame(tab_ice)
        ice_opts_row.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        ttk.Checkbutton(
            ice_opts_row,
            text="检查出征英雄",
            variable=self.var_check_march_heroes,
            command=self._save_config,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(
            ice_opts_row,
            text="自动使用体力道具",
            variable=self.var_use_stamina,
            command=self._save_config,
        ).pack(side=tk.LEFT)
        row += 1

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

        ttk.Checkbutton(
            tab_ice,
            text="启用编队",
            variable=self.var_use_formation,
            command=self._on_use_formation_changed,
        ).grid(row=row, column=0, sticky=tk.W, pady=2)
        self._ice_formation_spinbox = ttk.Spinbox(
            tab_ice,
            from_=FORMATION_SLOT_MIN,
            to=FORMATION_SLOT_MAX,
            textvariable=self.var_formation_slot,
            width=8,
        )
        self._ice_formation_spinbox.grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        self._ice_formation_hint = ttk.Label(
            tab_ice, text="出征界面左起第几个编队（1~8）", font=("", 8)
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
        self._update_ice_formation_state()

        tab_lighthouse = ttk.Frame(notebook, padding=6)
        notebook.add(tab_lighthouse, text="灯塔任务")
        _configure_param_tab_grid(tab_lighthouse)

        row = 0
        ttk.Label(tab_lighthouse, text="扫描间隔（分钟）").grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        ttk.Spinbox(
            tab_lighthouse, from_=5, to=1440, textvariable=self.var_lighthouse_interval, width=8
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        ttk.Label(tab_lighthouse, text="（已改为一次性任务，此项不再生效）", font=("", 8)).grid(
            row=row, column=2, sticky=tk.W, padx=(4, 0)
        )
        row += 1

        ttk.Label(tab_lighthouse, text="编队槽位").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            tab_lighthouse,
            from_=FORMATION_SLOT_MIN,
            to=FORMATION_SLOT_MAX,
            textvariable=self.var_lighthouse_formation_slot,
            width=8,
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        ttk.Label(tab_lighthouse, text="小怪出征用（1~8）", font=("", 8)).grid(
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
            text="使用体力（不足时自动使用领主体力道具，否则结束任务）",
            variable=self.var_lighthouse_use_stamina,
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
        ttk.Label(tab_monster, text="打野间隔（分钟）").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            tab_monster, from_=1, to=120, textvariable=self.var_monster_interval, width=8
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)
        row += 1

        ttk.Label(tab_monster, text="野怪等级").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            tab_monster, from_=1, to=30, textvariable=self.var_monster_level, width=8
        ).grid(row=row, column=1, sticky=tk.W, padx=FORM_INPUT_PADX)

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

        tab_more = ttk.Frame(notebook, padding=12)
        notebook.add(tab_more, text="更多")
        ttk.Label(tab_more, text="更多任务参数将在此添加", foreground="gray").pack(
            anchor=tk.W
        )

        task_frame = ttk.LabelFrame(self._left, text="运行任务", padding=8)
        task_frame.pack(fill=tk.X, **pad)

        task_notebook = ttk.Notebook(task_frame)
        task_notebook.pack(fill=tk.X)

        tab_loop = ttk.Frame(task_notebook, padding=6)
        task_notebook.add(tab_loop, text="循环任务")
        self._build_task_checkboxes(
            tab_loop, loop_tasks(), columns=LOOP_TASK_COLUMNS, show_hint=False
        )

        loop_btn_row = ttk.Frame(tab_loop)
        loop_btn_row.pack(fill=tk.X, pady=(8, 0))
        self.btn_start_loop = ttk.Button(
            loop_btn_row, text="启动", command=self._start_loop, width=10
        )
        self.btn_start_loop.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_stop_loop = ttk.Button(
            loop_btn_row, text="停止", command=self._stop_loop, width=10, state=tk.DISABLED
        )
        self.btn_stop_loop.pack(side=tk.LEFT)

        tab_once = ttk.Frame(task_notebook, padding=6)
        task_notebook.add(tab_once, text="一次性任务")
        self._build_task_checkboxes(
            tab_once, once_tasks(), columns=ONCE_TASK_COLUMNS, show_hint=False
        )

        once_btn_row = ttk.Frame(tab_once)
        once_btn_row.pack(fill=tk.X, pady=(8, 0))
        self.btn_run_once = ttk.Button(
            once_btn_row, text="执行", command=self._run_once_batch, width=10
        )
        self.btn_run_once.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_stop_once = ttk.Button(
            once_btn_row, text="停止", command=self._stop_once, width=10, state=tk.DISABLED
        )
        self.btn_stop_once.pack(side=tk.LEFT)

        self.lbl_status = ttk.Label(task_frame, text="状态：待命", foreground="gray")
        self.lbl_status.pack(anchor=tk.W, pady=(8, 0))

        self._log_frame = ttk.LabelFrame(self._container, text="运行日志", padding=8)
        self.log_text = scrolledtext.ScrolledText(
            self._log_frame, width=36, height=28, state=tk.DISABLED, font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        if self._log_visible:
            self._show_log_panel(save_config=False)

    def _toggle_log(self) -> None:
        if self._log_visible:
            self._hide_log_panel()
        else:
            self._show_log_panel()

    def _show_log_panel(self, save_config: bool = True) -> None:
        self._log_visible = True
        self._log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.geometry(f"{MAIN_WIDTH + LOG_WIDTH}x{MAIN_HEIGHT}")
        self.btn_toggle_log.configure(text="隐藏日志 ◂")
        if save_config:
            self._save_config()

    def _hide_log_panel(self) -> None:
        self._log_visible = False
        self._log_frame.pack_forget()
        self.geometry(f"{MAIN_WIDTH}x{MAIN_HEIGHT}")
        self.btn_toggle_log.configure(text="显示日志 ▸")
        self._save_config()

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
            self.lbl_status.configure(text=f"状态：{msg}")
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
                    yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)

    def _reset_adb(self) -> None:
        self._adb = None

    def _on_device_selected(self) -> None:
        serial = self.var_device_serial.get().strip()
        if not serial:
            return
        try:
            self._apply_device_serial(serial, save=True)
            self.lbl_conn.configure(text=f"设备：{serial}", foreground="")
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
                        foreground="",
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

    def _test_connection(self) -> None:
        self.lbl_conn.configure(text="连接中…")
        self.update_idletasks()
        serial = self.var_device_serial.get().strip()
        dev_cfg = self.config.get("device", {})
        adb_path = dev_cfg.get("adb_path", "")

        def work():
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
            use_stamina=bool(self.var_lighthouse_use_stamina.get()),
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
            use_stamina=bool(self.var_use_stamina.get()),
            check_march_heroes=bool(self.var_check_march_heroes.get()),
            use_formation=bool(self.var_use_formation.get()),
            on_status=self._on_status,
        )

    def _build_monster_task(self) -> HuntMonsterTask:
        monster_cfg = self.config.get("tasks", {}).get("hunt_monster", {})
        return HuntMonsterTask(
            adb=self._get_adb(),
            coords=monster_cfg.get("coords", {}),
            interval=float(self.var_monster_interval.get() * 60),
            monster_level=int(self.var_monster_level.get()),
            max_monster_level=monster_cfg.get("max_monster_level", 30),
            skip_hour=monster_cfg.get("skip_hour", 21),
            step_delay=monster_cfg.get("step_delay", 1.5),
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

    def _is_any_running(self) -> bool:
        return self._is_loop_running() or self._is_once_running()

    def _set_loop_buttons(self, running: bool) -> None:
        self.btn_start_loop.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.btn_stop_loop.configure(state=tk.NORMAL if running else tk.DISABLED)
        if not self._is_once_running():
            self.btn_run_once.configure(state=tk.DISABLED if running else tk.NORMAL)

    def _set_once_buttons(self, running: bool) -> None:
        self.btn_run_once.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.btn_stop_once.configure(state=tk.NORMAL if running else tk.DISABLED)
        if not self._is_loop_running():
            self.btn_start_loop.configure(state=tk.DISABLED if running else tk.NORMAL)

    def _ensure_device(self) -> bool:
        adb = self._get_adb()
        if adb.wait_for_device(retries=10, interval=2.0):
            self._on_status(f"设备已连接：{adb.address}")
            return True
        self._on_status("无法连接模拟器，请确认雷电已启动")
        return False

    def _start_loop(self) -> None:
        if self._is_loop_running():
            messagebox.showwarning("提示", "循环任务已在运行中")
            return
        if self._is_once_running():
            messagebox.showwarning("提示", "一次性任务正在执行，请稍候或先停止")
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

        self._loop_stop_event.clear()
        self._set_loop_buttons(running=True)
        names = "、".join(t.name for t in self._loop_tasks)
        self._on_status(f"启动循环任务：{names}")

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
