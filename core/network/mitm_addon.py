"""mitmproxy 插件：过滤并保存游戏 HTTP 流量。

由 tools/run_mitm_proxy.py 加载，勿直接运行。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from core.network.capture_store import CaptureStore

_CONFIG_PATH = os.environ.get("EW_CONFIG_PATH")
_store = CaptureStore.from_config(_CONFIG_PATH)


class EndlessWinterCaptureAddon:
    def response(self, flow) -> None:
        try:
            saved = _store.save_flow(flow)
            if saved is not None:
                req = flow.request
                logger.info(f"[capture] {req.method} {req.host}{req.path} -> {saved.name}")
        except Exception as exc:
            logger.warning(f"保存抓包失败: {exc}")


addons = [EndlessWinterCaptureAddon()]
