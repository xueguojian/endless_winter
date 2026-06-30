"""网络抓包 / 发包配置（读取 config.yaml 的 network 段）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from core.config_path import ROOT, resolve_config_path


@dataclass
class ProxyConfig:
    listen_host: str = "0.0.0.0"
    listen_port: int = 8080


@dataclass
class CaptureConfig:
    include_hosts: list[str] = field(default_factory=list)
    exclude_hosts: list[str] = field(
        default_factory=lambda: [
            "google",
            "gstatic",
            "googleapis",
            "facebook",
            "crashlytics",
            "firebase",
        ]
    )
    save_response: bool = True
    max_body_bytes: int = 1_048_576


@dataclass
class ClientConfig:
    timeout: float = 30.0
    verify_ssl: bool = True
    proxy_url: str | None = None


@dataclass
class NetworkConfig:
    capture_dir: Path = field(default_factory=lambda: ROOT / "assets" / "captures")
    session_file: Path = field(
        default_factory=lambda: ROOT / "assets" / "captures" / "session.json"
    )
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    client: ClientConfig = field(default_factory=ClientConfig)

    def ensure_dirs(self) -> None:
        self.capture_dir.mkdir(parents=True, exist_ok=True)


def _as_path(value: str | Path | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_network_config(config_path: str | Path | None = None) -> NetworkConfig:
    path = resolve_config_path(config_path)
    raw: dict = {}
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        raw = loaded.get("network") or {}

    proxy_raw = raw.get("proxy") or {}
    capture_raw = raw.get("capture") or {}
    client_raw = raw.get("client") or {}

    cfg = NetworkConfig(
        capture_dir=_as_path(raw.get("capture_dir"), ROOT / "assets" / "captures"),
        session_file=_as_path(
            raw.get("session_file"), ROOT / "assets" / "captures" / "session.json"
        ),
        proxy=ProxyConfig(
            listen_host=str(proxy_raw.get("listen_host", "0.0.0.0")),
            listen_port=int(proxy_raw.get("listen_port", 8080)),
        ),
        capture=CaptureConfig(
            include_hosts=[str(x) for x in capture_raw.get("include_hosts") or []],
            exclude_hosts=[
                str(x)
                for x in (
                    capture_raw.get("exclude_hosts")
                    or CaptureConfig().exclude_hosts
                )
            ],
            save_response=bool(capture_raw.get("save_response", True)),
            max_body_bytes=int(capture_raw.get("max_body_bytes", 1_048_576)),
        ),
        client=ClientConfig(
            timeout=float(client_raw.get("timeout", 30.0)),
            verify_ssl=bool(client_raw.get("verify_ssl", True)),
            proxy_url=client_raw.get("proxy_url") or None,
        ),
    )
    cfg.ensure_dirs()
    return cfg
