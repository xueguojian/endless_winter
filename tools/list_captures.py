"""列出已保存的抓包记录。

用法:
  .venv\\Scripts\\python.exe tools/list_captures.py
  .venv\\Scripts\\python.exe tools/list_captures.py --host api
  .venv\\Scripts\\python.exe tools/list_captures.py --path login
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.network.capture_store import CaptureStore


def main() -> None:
    parser = argparse.ArgumentParser(description="列出抓包记录")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--host", default=None, help="host 子串过滤")
    parser.add_argument("--path", default=None, help="path 子串过滤")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    store = CaptureStore.from_config(args.config)
    paths = store.iter_captures(
        host_contains=args.host,
        path_contains=args.path,
        newest_first=True,
    )[: args.limit]

    if not paths:
        print("（无记录）")
        return

    print(f"{'ID':<14} {'方法':<6} {'状态':<5} {'Host':<28} Path")
    print("-" * 90)
    for path in paths:
        ex = store.load(path)
        status = ex.response.status_code if ex.response else "-"
        host = (ex.host or "")[:28]
        req_path = (ex.path or "")[:40]
        print(f"{ex.id:<14} {ex.request.method:<6} {status!s:<5} {host:<28} {req_path}")
        print(f"  文件: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
