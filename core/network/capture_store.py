"""抓包记录的读写与索引。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from core.network.config import NetworkConfig, load_network_config
from core.network.models import (
    CapturedExchange,
    RequestSnapshot,
    ResponseSnapshot,
    encode_body,
)

try:
    from mitmproxy import http as mitm_http
except ImportError:  # pragma: no cover - 仅 mitm 代理进程需要
    mitm_http = None  # type: ignore[assignment,misc]


def _safe_slug(text: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip()).strip("_")
    return (slug or "request")[:max_len]


class CaptureStore:
    """将 mitmproxy 流量或手工构造的请求保存为 JSON。"""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.config.ensure_dirs()

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> CaptureStore:
        return cls(load_network_config(config_path))

    @property
    def root(self) -> Path:
        return self.config.capture_dir

    def should_capture_host(self, host: str) -> bool:
        host_lower = host.lower()
        for pattern in self.config.capture.exclude_hosts:
            if pattern.lower() in host_lower:
                return False
        include = self.config.capture.include_hosts
        if not include:
            return True
        return any(pattern.lower() in host_lower for pattern in include)

    def _daily_dir(self) -> Path:
        day = datetime.now().strftime("%Y-%m-%d")
        path = self.root / day
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_exchange(self, exchange: CapturedExchange) -> Path:
        host_slug = _safe_slug(exchange.host or "unknown")
        path_slug = _safe_slug(exchange.path or exchange.request.url)
        stamp = datetime.now().strftime("%H%M%S")
        filename = f"{stamp}_{exchange.id}_{host_slug}_{path_slug}.json"
        out = self._daily_dir() / filename
        with out.open("w", encoding="utf-8") as fh:
            json.dump(exchange.to_dict(), fh, ensure_ascii=False, indent=2)
        logger.info(f"已保存抓包: {out}")
        return out

    def save_flow(self, flow: Any) -> Path | None:
        if mitm_http is None:
            raise RuntimeError(
                "mitmproxy 未安装，请运行: pip install -r requirements-network.txt"
            )

        request = flow.request
        host = request.host or urlparse(request.pretty_url).hostname or ""
        if not self.should_capture_host(host):
            return None

        req_enc, req_body = encode_body(
            self._clip_body(request.raw_content or b"")
        )
        req_headers = {
            k: v
            for k, v in request.headers.items(multi=True)
            if k.lower() not in {"content-length"}
        }

        response_snapshot: ResponseSnapshot | None = None
        if flow.response is not None and self.config.capture.save_response:
            resp = flow.response
            resp_enc, resp_body = encode_body(
                self._clip_body(resp.raw_content or b"")
            )
            resp_headers = {
                k: v
                for k, v in resp.headers.items(multi=True)
                if k.lower() not in {"content-length"}
            }
            response_snapshot = ResponseSnapshot(
                status_code=int(resp.status_code),
                headers=resp_headers,
                body_encoding=resp_enc,
                body=resp_body,
            )

        exchange = CapturedExchange(
            request=RequestSnapshot(
                method=str(request.method),
                url=str(request.pretty_url),
                headers=req_headers,
                body_encoding=req_enc,
                body=req_body,
            ),
            response=response_snapshot,
            host=host,
            path=request.path or "/",
        )
        return self.save_exchange(exchange)

    def _clip_body(self, raw: bytes) -> bytes:
        limit = self.config.capture.max_body_bytes
        if len(raw) <= limit:
            return raw
        return raw[:limit]

    def iter_captures(
        self,
        *,
        host_contains: str | None = None,
        path_contains: str | None = None,
        newest_first: bool = True,
    ) -> list[Path]:
        files = sorted(self.root.rglob("*.json"))
        if newest_first:
            files.reverse()

        results: list[Path] = []
        for path in files:
            if path.name == "session.json":
                continue
            if host_contains or path_contains:
                try:
                    exchange = self.load(path)
                except (json.JSONDecodeError, OSError):
                    continue
                if host_contains and host_contains.lower() not in exchange.host.lower():
                    continue
                if path_contains and path_contains.lower() not in exchange.path.lower():
                    continue
            results.append(path)
        return results

    def load(self, path: str | Path) -> CapturedExchange:
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        return CapturedExchange.from_dict(data)

    def latest(
        self,
        *,
        host_contains: str | None = None,
        path_contains: str | None = None,
    ) -> CapturedExchange | None:
        paths = self.iter_captures(
            host_contains=host_contains,
            path_contains=path_contains,
            newest_first=True,
        )
        if not paths:
            return None
        return self.load(paths[0])
