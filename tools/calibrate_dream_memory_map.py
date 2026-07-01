"""寻梦记忆地图标定工具。

用法:
  # 列出地图
  .venv\\Scripts\\python.exe tools/calibrate_dream_memory_map.py --list

  # 新建地图
  .venv\\Scripts\\python.exe tools/calibrate_dream_memory_map.py --create gazebo_snow --name "凉亭雪景"

  # 添加/更新物品坐标
  .venv\\Scripts\\python.exe tools/calibrate_dream_memory_map.py --map gazebo_snow --add 梯子 380 520
  .venv\\Scripts\\python.exe tools/calibrate_dream_memory_map.py --map gazebo_snow --add 礼物盒 210 680

  # 从 adb 截图 + 坐标标尺：先截图保存，再用 GUI「坐标标尺」点位置，最后 --add

地图文件: assets/dream_memory/maps/<id>.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.dream_memory.maps import DreamMemoryMap, delete_map, list_maps, load_map, rename_map_name, save_map


def main() -> None:
    parser = argparse.ArgumentParser(description="寻梦记忆地图标定")
    parser.add_argument("--list", action="store_true", help="列出已有地图")
    parser.add_argument("--create", metavar="MAP_ID", help="新建地图 ID")
    parser.add_argument("--name", default="", help="地图显示名称（配合 --create）")
    parser.add_argument("--map", metavar="MAP_ID", help="要编辑的地图 ID")
    parser.add_argument(
        "--add",
        nargs=3,
        metavar=("物品名", "X", "Y"),
        help="添加或更新物品坐标",
    )
    parser.add_argument("--remove", metavar="物品名", help="删除物品")
    parser.add_argument("--rename", metavar="新名称", help="修改地图显示名称（配合 --map）")
    parser.add_argument("--delete-map", action="store_true", help="删除整张地图（配合 --map）")
    args = parser.parse_args()

    if args.list:
        maps = list_maps()
        if not maps:
            print("（暂无地图，使用 --create 新建）")
            return
        print(f"{'ID':<20} {'名称':<20} 物品数")
        print("-" * 52)
        for item in maps:
            print(f"{item.map_id:<20} {item.name:<20} {len(item.items)}")
        return

    if args.create:
        map_id = args.create.strip()
        if not map_id:
            parser.error("--create 需要非空 ID")
        name = args.name.strip() or map_id
        path = save_map(DreamMemoryMap(map_id=map_id, name=name, items={}))
        print(f"已创建: {path}")
        return

    if not args.map:
        parser.error("请指定 --map，或使用 --list / --create")

    try:
        dream_map = load_map(args.map)
    except FileNotFoundError:
        print(f"地图不存在: {args.map}，可用 --create {args.map} 新建", file=sys.stderr)
        sys.exit(1)

    if args.add:
        label, x_raw, y_raw = args.add
        dream_map.items[label] = [int(x_raw), int(y_raw)]
        path = save_map(dream_map)
        print(f"已更新「{label}」→ [{x_raw}, {y_raw}]  文件: {path}")
        return

    if args.remove:
        label = args.remove
        if label in dream_map.items:
            del dream_map.items[label]
            path = save_map(dream_map)
            print(f"已删除「{label}」  文件: {path}")
        else:
            print(f"物品不存在: {label}", file=sys.stderr)
            sys.exit(1)
        return

    if args.rename:
        try:
            path = rename_map_name(args.map, args.rename)
            print(f"已重命名为「{args.rename.strip()}」  文件: {path}")
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        return

    if args.delete_map:
        try:
            answer = input(f"确认删除地图 {args.map}? 输入 yes: ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "yes":
            print("已取消")
            return
        delete_map(args.map)
        print(f"已删除地图: {args.map}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
