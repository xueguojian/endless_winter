"""重放已保存的抓包请求。

用法:
  .venv\\Scripts\\python.exe tools/replay_capture.py assets/captures/2026-01-01/xxx.json
  .venv\\Scripts\\python.exe tools/replay_capture.py --latest
  .venv\\Scripts\\python.exe tools/replay_capture.py --latest --host api --save-session
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.network.client import GameHttpClient
from core.network.capture_store import CaptureStore


def main() -> None:
    parser = argparse.ArgumentParser(description="重放抓包请求")
    parser.add_argument("capture", nargs="?", type=Path, help="抓包 JSON 路径")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--latest", action="store_true", help="重放最新一条")
    parser.add_argument("--host", default=None, help="配合 --latest：host 过滤")
    parser.add_argument("--path", default=None, help="配合 --latest：path 过滤")
    parser.add_argument(
        "--save-session",
        action="store_true",
        help="从重放用的抓包更新 session.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印请求，不实际发送",
    )
    args = parser.parse_args()

    store = CaptureStore.from_config(args.config)
    client = GameHttpClient(config=store.config)

    if args.latest:
        exchange = store.latest(host_contains=args.host, path_contains=args.path)
        if exchange is None:
            print("未找到匹配抓包", file=sys.stderr)
            sys.exit(1)
        capture_path = None
    elif args.capture is not None:
        exchange = store.load(args.capture)
        capture_path = args.capture
    else:
        parser.error("请指定 capture 文件或使用 --latest")

    if args.save_session:
        client.update_session_from_capture(exchange)
        print(f"会话已写入 {store.config.session_file}")

    req = exchange.request
    print(f"请求: {req.method} {req.url}")
    if args.dry_run:
        print(json.dumps(req.to_httpx_kwargs(), ensure_ascii=False, indent=2, default=str))
        return

    result = client.replay_capture(exchange)
    print(f"响应: HTTP {result.status_code}")
    preview = result.text[:2000]
    if len(result.text) > 2000:
        preview += "\n... (truncated)"
    print(preview)
    if capture_path:
        print(f"\n来源: {capture_path}")


if __name__ == "__main__":
    main()
