"""将 config.example.yaml 中新增的配置项合并进实例 config_555x.yaml，不覆盖已有值。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config_path import EXAMPLE_CONFIG_PATH, PRIMARY_CONFIG_PATH, list_instance_config_paths


def deep_merge_missing(local: dict, example: dict) -> dict:
    """保留 local 已有值，仅补充 example 中缺失的键。"""
    merged = dict(local)
    for key, example_val in example.items():
        if key not in merged:
            merged[key] = example_val
        elif isinstance(example_val, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_missing(merged[key], example_val)
    return merged


def sync_config(
    local_path: Path,
    example_path: Path = EXAMPLE_CONFIG_PATH,
    dry_run: bool = False,
) -> list[str]:
    if not example_path.is_file():
        raise FileNotFoundError(f"缺少模板：{example_path}")
    if not local_path.is_file():
        raise FileNotFoundError(
            f"缺少 {local_path.name}，请先运行对应 run_gui_555x.bat 自动生成，"
            f"或 copy config.example.yaml {local_path.name}"
        )

    with open(example_path, encoding="utf-8") as f:
        example = yaml.safe_load(f) or {}
    with open(local_path, encoding="utf-8") as f:
        local = yaml.safe_load(f) or {}

    merged = deep_merge_missing(local, example)
    added: list[str] = []

    def collect_added(path: str, before, after) -> None:
        if isinstance(after, dict):
            before = before if isinstance(before, dict) else {}
            for k, v in after.items():
                sub = f"{path}.{k}" if path else k
                if k not in before:
                    added.append(sub)
                elif isinstance(v, dict):
                    collect_added(sub, before.get(k), v)
        elif path and path not in added:
            added.append(path)

    collect_added("", local, merged)

    if merged == local:
        return []

    if not dry_run:
        with open(local_path, "w", encoding="utf-8") as f:
            yaml.dump(merged, f, allow_unicode=True, sort_keys=False)

    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从 config.example.yaml 合并新增配置项到实例 yaml（不覆盖已有值）"
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help=f"实例配置文件（默认 {PRIMARY_CONFIG_PATH.name}）",
    )
    parser.add_argument(
        "--example",
        default=str(EXAMPLE_CONFIG_PATH),
        help="模板配置文件路径（默认 config.example.yaml）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="合并本机全部 config_555*.yaml",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="仅显示将新增的键，不写文件",
    )
    args = parser.parse_args(argv)

    if args.all:
        targets = list_instance_config_paths()
        if not targets:
            print("未找到 config_555*.yaml，请先运行 run_gui_5555.bat 等生成。", file=sys.stderr)
            return 1
    else:
        config = args.config or str(PRIMARY_CONFIG_PATH)
        targets = [Path(config)]

    exit_code = 0
    for target in targets:
        try:
            added = sync_config(
                target,
                Path(args.example),
                dry_run=args.dry_run,
            )
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            exit_code = 1
            continue

        if args.all:
            print(f"--- {target.name} ---")
        if not added:
            print("配置已是最新，无需合并。")
            continue

        action = "将新增" if args.dry_run else "已新增"
        print(f"{action} {len(added)} 项：")
        for key in added:
            print(f"  + {key}")
        if args.dry_run:
            print("（dry-run，未写入文件）")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
