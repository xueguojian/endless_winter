# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

无尽冬日（Endless Winter）游戏自动化脚本。通过 ADB 控制雷电模拟器（LDPlayer），使用 OpenCV 模板匹配识别游戏界面，自动执行采集、打怪、捐献等重复性任务。提供 Tkinter GUI 用于任务配置和多开管理。

## 常用命令

```bash
# 安装环境（首次）
setup.bat

# 启动 GUI（默认配置 config.yaml）
.venv\Scripts\python.exe launch_gui.py

# 多开（不同模拟器实例用不同配置文件）
.venv\Scripts\python.exe launch_gui.py --config config_5557.yaml

# 直接启动（跳过 launch_gui 的窗口模式判断）
.venv\Scripts\python.exe gui_main.py

# 裁剪模板图片
.venv\Scripts\python.exe tools/capture_template.py <模板名> <中心x> <中心y> <宽> <高>

# 生成灯塔模板变体
.venv\Scripts\python.exe tools/gen_lighthouse_templates.py

# 标定脚本
.venv\Scripts\python.exe scripts/calibrate_stamina.py
.venv\Scripts\python.exe scripts/calibrate_level_num.py
```

## 核心架构

```
core/          基础层（无游戏逻辑）
  adb_client.py    ADB 连接、截图、触控操作
  vision.py        OpenCV 模板匹配（单/多尺度）
  navigation.py    返回主界面（右下角城镇/野外按钮识别）
  deploy_march.py  出征界面：编队选择、出征按钮、体力弹窗处理
  lighthouse_vision.py  灯塔任务图标识别（颜色分割+模板匹配）
  coords.py        720×1280 竖屏坐标系约定
  config_path.py   配置文件自动创建、端口推断、多开支持

gui/           Tkinter 图形界面
  app.py           主窗口：任务勾选、参数配置、启停控制
  task_registry.py 任务注册表（loop/once 两种类型）
  coord_ruler.py   坐标标尺工具（截图→标注坐标）

tasks/        游戏任务实现
  每个任务文件包含 merge_task_config()（合并默认配置）和任务类（run_once/should_run/stop）
  auto_lighthouse.py      灯塔任务（英雄之旅/帐篷/小怪）
  auto_mining.py          自动采集
  hunt_ice_beast.py       冰原巨兽集结
  hunt_monster.py          自动打野怪
  donate_alliance_supplies.py  捐献联盟物资
  collect_supplies.py     探险物资领取
  collect_commander_supplies.py  统帅物资领取
  auto_train_troops.py    自动练兵

assets/templates/   OpenCV 模板图片（PNG），含 lighthouse/ 子目录

tools/          开发辅助
  capture_template.py     从模拟器截图裁剪模板
  gen_lighthouse_templates.py  生成灯塔图标边缘/变体模板

scripts/        标定脚本
```

## 关键设计约定

### 坐标系

所有坐标是 **720×1280 竖屏触控坐标**。ADB 截图和 `input tap` 使用同一套坐标系。`core/coords.py` 定义常量 `PORTRAIT_WIDTH = 720`, `PORTRAIT_HEIGHT = 1280`。雷电模拟器的 `wm size` 可能是横屏 1280×720，但截图和触控始终以竖屏坐标系为准——**不要做横竖屏转换**。

### 配置文件

- `config.yaml` 是模板配置，`config_5555.yaml` / `config_5557.yaml` 等是多开实例配置
- 实例配置文件不存在时会从 `config.yaml` 自动复制并根据文件名推断端口
- `device.adb_port` 决定连接哪个模拟器实例
- `tasks.<name>.coords` 的各坐标点对应游戏界面的触控位置

### 任务系统

- 任务分两种：`loop`（循环任务，按 interval 定时执行）和 `once`（一次性任务，执行完即止）
- 循环任务之间是互斥的（同一时间只运行一个），冰原巨兽和打野怪在 GUI 上只能二选一
- 每个任务类实现：`run_once()`（执行一次）、`should_run()`（判断是否到时间）、`stop()`（中断）
- `merge_task_config(cfg)` 函数合并用户配置和默认坐标，GUI save 时调用

### 图像识别

- 主要使用 `Vision.match_template()` 做单尺度模板匹配，`match_template_multiscale()` 处理尺寸偏差
- 灯塔任务更复杂：先用 HSV 颜色分割找图钉位置，再在周围搜索最佳模板匹配来分类（`lighthouse_vision.py`）
- 模板匹配阈值通常为 0.80（默认），灯塔任务降到 0.55（图标小且多变）

### ADB 连接

- 支持自动探测 LDPlayer 多开端口（5555/5557/5559 等奇数端口）
- `AdbClient.list_connected_devices()` 会主动尝试 connect 未连接的端口
- GUI 启动时自动选择已连接的设备，也可手动刷新
