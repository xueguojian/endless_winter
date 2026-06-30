"""从抓包或配置中维护登录态（Cookie / Token 等）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from core.network.config import NetworkConfig, load_network_config
from core.network.models import CapturedExchange


@dataclass
class GameSession:
    """游戏 HTTP 会话。字段需根据逆向结果自行补充。"""

    headers: dict[str, str] = field(default_factory=dict)
    cookies: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "headers": dict(self.headers),
            "cookies": dict(self.cookies),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameSession:
        return cls(
            headers={str(k): str(v) for k, v in (data.get("headers") or {}).items()},
            cookies={str(k): str(v) for k, v in (data.get("cookies") or {}).items()},
            extra=dict(data.get("extra") or {}),
        )

    def apply_to_headers(self, headers: dict[str, str]) -> dict[str, str]:
        merged = dict(headers)
        merged.update(self.headers)
        if self.cookies:
            cookie_header = "; ".join(f"{k}={v}" for k, v in self.cookies.items())
            merged["Cookie"] = cookie_header
        return merged

    @classmethod
    def from_capture(
        cls,
        exchange: CapturedExchange,
        *,
        header_keys: list[str] | None = None,
    ) -> GameSession:
        """从一次抓包提取常见鉴权头（可按游戏协议定制 header_keys）。"""
        keys = header_keys or [
            "authorization",
            "token",
            "x-token",
            "x-auth-token",
            "x-session-id",
            "x-device-id",
            "x-request-id",
            "x-sign",
            "x-timestamp",
            "x-nonce",
            "user-agent",
        ]
        headers: dict[str, str] = {}
        for key, value in exchange.request.headers.items():
            if key.lower() in keys:
                headers[key] = value

        cookies: dict[str, str] = {}
        cookie_header = exchange.request.headers.get("Cookie") or exchange.request.headers.get(
            "cookie"
        )
        if cookie_header:
            for part in cookie_header.split(";"):
                part = part.strip()
                if "=" in part:
                    name, val = part.split("=", 1)
                    cookies[name.strip()] = val.strip()

        return cls(headers=headers, cookies=cookies)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        logger.info(f"会话已保存: {path}")

    @classmethod
    def load(cls, path: str | Path) -> GameSession:
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)


def load_session(config: NetworkConfig | None = None) -> GameSession:
    cfg = config or load_network_config()
    path = cfg.session_file
    if not path.is_file():
        return GameSession()
    return GameSession.load(path)
