"""将 config.example.yaml 中新增的配置项合并进本机 config.yaml，不覆盖已有值。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = ROOT / "config.example.yaml"
LOCAL_PATH = ROOT / "config.yaml"


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
    local_path: Path = LOCAL_PATH,
    example_path: Path = EXAMPLE_PATH,
    dry_run: bool = False,
) -> list[str]:
    if not example_path.is_file():
        raise FileNotFoundError(f"缺少模板：{example_path}")
    if not local_path.is_file():
        raise FileNotFoundError(
            f"缺少 {local_path.name}，请先复制：copy config.example.yaml config.yaml"
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
        description="从 config.example.yaml 合并新增配置项到 config.yaml（不覆盖已有值）"
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(LOCAL_PATH),
        help="本机配置文件路径（默认 config.yaml）",
    )
    parser.add_argument(
        "--example",
        default=str(EXAMPLE_PATH),
        help="模板配置文件路径（默认 config.example.yaml）",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="仅显示将新增的键，不写文件",
    )
    args = parser.parse_args(argv)

    try:
        added = sync_config(
            Path(args.config),
            Path(args.example),
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not added:
        print("配置已是最新，无需合并。")
        return 0

    action = "将新增" if args.dry_run else "已新增"
    print(f"{action} {len(added)} 项：")
    for key in added:
        print(f"  + {key}")
    if args.dry_run:
        print("（dry-run，未写入文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
