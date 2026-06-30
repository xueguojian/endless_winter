"""抓包记录的数据结构。"""

from __future__ import annotations

import base64
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

BodyEncoding = Literal["empty", "text", "base64"]


def encode_body(raw: bytes | None) -> tuple[BodyEncoding, str]:
    if not raw:
        return "empty", ""
    try:
        return "text", raw.decode("utf-8")
    except UnicodeDecodeError:
        return "base64", base64.b64encode(raw).decode("ascii")


def decode_body(encoding: BodyEncoding, data: str) -> bytes:
    if encoding == "empty" or not data:
        return b""
    if encoding == "text":
        return data.encode("utf-8")
    if encoding == "base64":
        return base64.b64decode(data)
    raise ValueError(f"未知 body 编码: {encoding}")


@dataclass
class RequestSnapshot:
    method: str
    url: str
    headers: dict[str, str]
    body_encoding: BodyEncoding = "empty"
    body: str = ""

    @property
    def body_bytes(self) -> bytes:
        return decode_body(self.body_encoding, self.body)

    def to_httpx_kwargs(self) -> dict[str, Any]:
        headers = dict(self.headers)
        content = self.body_bytes
        return {
            "method": self.method.upper(),
            "url": self.url,
            "headers": headers,
            "content": content,
        }


@dataclass
class ResponseSnapshot:
    status_code: int
    headers: dict[str, str]
    body_encoding: BodyEncoding = "empty"
    body: str = ""

    @property
    def body_bytes(self) -> bytes:
        return decode_body(self.body_encoding, self.body)

    @property
    def text(self) -> str:
        if self.body_encoding == "text":
            return self.body
        raw = self.body_bytes
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace")


@dataclass
class CapturedExchange:
    """一次完整的请求-响应记录，可序列化到 JSON 并重放。"""

    request: RequestSnapshot
    response: ResponseSnapshot | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    captured_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    host: str = ""
    path: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapturedExchange:
        req = data.get("request") or {}
        resp = data.get("response")
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            captured_at=str(data.get("captured_at") or ""),
            host=str(data.get("host") or ""),
            path=str(data.get("path") or ""),
            note=str(data.get("note") or ""),
            request=RequestSnapshot(
                method=str(req.get("method") or "GET"),
                url=str(req.get("url") or ""),
                headers={str(k): str(v) for k, v in (req.get("headers") or {}).items()},
                body_encoding=req.get("body_encoding") or "empty",
                body=str(req.get("body") or ""),
            ),
            response=(
                None
                if resp is None
                else ResponseSnapshot(
                    status_code=int(resp.get("status_code") or 0),
                    headers={
                        str(k): str(v) for k, v in (resp.get("headers") or {}).items()
                    },
                    body_encoding=resp.get("body_encoding") or "empty",
                    body=str(resp.get("body") or ""),
                )
            ),
        )
