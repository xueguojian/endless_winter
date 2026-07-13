"""GUI 任务注册表：定义可选任务及其类型（循环 / 一次性）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TaskKind = Literal["loop", "once"]


@dataclass(frozen=True)
class TaskEntry:
    task_id: str
    label: str
    kind: TaskKind
    config_key: str
    available: bool = True
    hint: str = ""


# available=False 的任务在界面显示为灰色「敬请期待」
TASK_ENTRIES: tuple[TaskEntry, ...] = (
    TaskEntry(
        task_id="auto_lighthouse",
        label="自动灯塔任务",
        kind="once",
        config_key="auto_lighthouse",
        available=True,
    ),
    TaskEntry(
        task_id="hunt_ice_beast",
        label="冰原巨兽集结",
        kind="loop",
        config_key="hunt_ice_beast",
        available=True,
    ),
    TaskEntry(
        task_id="hunt_monster",
        label="自动打野怪",
        kind="loop",
        config_key="hunt_monster",
        available=True,
    ),
    TaskEntry(
        task_id="donate_alliance_supplies",
        label="捐献联盟物资",
        kind="loop",
        config_key="donate_alliance_supplies",
        available=True,
    ),
    TaskEntry(
        task_id="auto_mining",
        label="自动采集",
        kind="loop",
        config_key="auto_mining",
        available=True,
    ),
    TaskEntry(
        task_id="traverse_island",
        label="一键遍历海岛",
        kind="once",
        config_key="traverse_island",
        available=False,
    ),
    TaskEntry(
        task_id="auto_train_troops",
        label="自动练兵",
        kind="loop",
        config_key="auto_train_troops",
        available=True,
    ),
    TaskEntry(
        task_id="collect_supplies",
        label="一键领取探险物资",
        kind="loop",
        config_key="collect_supplies",
        available=True,
    ),
    TaskEntry(
        task_id="collect_commander_supplies",
        label="一键领取统帅物资",
        kind="once",
        config_key="collect_commander_supplies",
        available=True,
    ),
    TaskEntry(
        task_id="collect_pet_supplies",
        label="领取宠物物资",
        kind="once",
        config_key="collect_pet_supplies",
        available=True,
    ),
)


def loop_tasks() -> list[TaskEntry]:
    return [e for e in TASK_ENTRIES if e.kind == "loop"]


def once_tasks() -> list[TaskEntry]:
    return [e for e in TASK_ENTRIES if e.kind == "once"]


# 一键托管：按顺序各执行一次（与勾选无关）
HOSTING_TASK_IDS: tuple[str, ...] = (
    "auto_mining",
    "donate_alliance_supplies",
    "auto_train_troops",
    "collect_supplies",
    "auto_lighthouse",
    "collect_commander_supplies",
    "collect_pet_supplies",
)
