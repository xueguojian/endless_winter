"""HTTP 重放与发包客户端。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from core.network.capture_store import CaptureStore
from core.network.config import NetworkConfig, load_network_config
from core.network.models import CapturedExchange, RequestSnapshot
from core.network.session import GameSession, load_session


@dataclass
class ReplayResult:
    status_code: int
    headers: dict[str, str]
    text: str
    content: bytes

    @classmethod
    def from_response(cls, response: httpx.Response) -> ReplayResult:
        return cls(
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items()},
            text=response.text,
            content=response.content,
        )


class GameHttpClient:
    """基于 httpx 的发包客户端，支持从抓包记录重放。"""

    def __init__(
        self,
        config: NetworkConfig | None = None,
        session: GameSession | None = None,
    ):
        self.config = config or load_network_config()
        self.session = session if session is not None else load_session(self.config)
        self.store = CaptureStore(self.config)

    def _build_client(self) -> httpx.Client:
        client_cfg = self.config.client
        return httpx.Client(
            timeout=client_cfg.timeout,
            verify=client_cfg.verify_ssl,
            proxy=client_cfg.proxy_url,
            follow_redirects=True,
        )

    def send(
        self,
        request: RequestSnapshot,
        *,
        extra_headers: dict[str, str] | None = None,
        override_body: bytes | None = None,
    ) -> ReplayResult:
        headers = self.session.apply_to_headers(dict(request.headers))
        if extra_headers:
            headers.update(extra_headers)

        kwargs = request.to_httpx_kwargs()
        kwargs["headers"] = headers
        if override_body is not None:
            kwargs["content"] = override_body

        logger.debug(f"发包 {kwargs['method']} {kwargs['url']}")
        with self._build_client() as client:
            response = client.request(**kwargs)
        return ReplayResult.from_response(response)

    def replay_capture(
        self,
        capture: CapturedExchange | str | Path,
        *,
        extra_headers: dict[str, str] | None = None,
        override_body: bytes | None = None,
    ) -> ReplayResult:
        if not isinstance(capture, CapturedExchange):
            capture = self.store.load(capture)
        return self.send(
            capture.request,
            extra_headers=extra_headers,
            override_body=override_body,
        )

    def replay_latest(
        self,
        *,
        host_contains: str | None = None,
        path_contains: str | None = None,
        **kwargs: Any,
    ) -> ReplayResult:
        latest = self.store.latest(
            host_contains=host_contains,
            path_contains=path_contains,
        )
        if latest is None:
            raise FileNotFoundError("未找到匹配的抓包记录")
        return self.replay_capture(latest, **kwargs)

    def update_session_from_capture(self, capture: CapturedExchange) -> GameSession:
        self.session = GameSession.from_capture(capture)
        self.session.save(self.config.session_file)
        return self.session
