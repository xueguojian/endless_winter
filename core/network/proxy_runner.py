"""mitmproxy 抓包代理的启动与停止（GUI / CLI 共用）。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from core.config_path import ROOT
from core.network.config import NetworkConfig, load_network_config

MITMWEB_DEFAULT_PORT = 8081


class MitmNotInstalledError(RuntimeError):
    """未安装 mitmproxy。"""


class MitmAlreadyRunningError(RuntimeError):
    """抓包代理已在运行。"""


@dataclass(frozen=True)
class MitmProxyStatus:
    listen_host: str
    listen_port: int
    web_url: str
    capture_dir: Path


def resolve_mitm_binary(*, web: bool = True) -> Path | None:
    name = "mitmweb" if web else "mitmdump"
    found = shutil.which(name)
    if found:
        return Path(found)
    candidate = ROOT / ".venv" / "Scripts" / f"{name}.exe"
    return candidate if candidate.is_file() else None


def mitm_is_installed(*, web: bool = True) -> bool:
    return resolve_mitm_binary(web=web) is not None


def build_mitm_command(
    config: NetworkConfig,
    *,
    config_path: str | Path | None = None,
    web: bool = True,
    web_host: str = "127.0.0.1",
    web_port: int = MITMWEB_DEFAULT_PORT,
) -> list[str]:
    mitm_bin = resolve_mitm_binary(web=web)
    if mitm_bin is None:
        raise MitmNotInstalledError(
            "未找到 mitmweb/mitmdump，请运行: .venv\\Scripts\\pip.exe install -r requirements-network.txt"
        )

    addon = ROOT / "core" / "network" / "mitm_addon.py"
    if not addon.is_file():
        raise FileNotFoundError(addon)

    cmd = [
        str(mitm_bin),
        "-s",
        str(addon),
        "--listen-host",
        config.proxy.listen_host,
        "--listen-port",
        str(config.proxy.listen_port),
        "--set",
        "block_global=false",
    ]
    if web:
        cmd.extend(["--web-host", web_host, "--web-port", str(web_port)])

    return cmd


def mitm_proxy_env(config_path: str | Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if config_path is not None:
        env["EW_CONFIG_PATH"] = str(Path(config_path).resolve())
    return env


class MitmProxyRunner:
    """管理 mitmproxy 子进程生命周期。"""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._status: MitmProxyStatus | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def status(self) -> MitmProxyStatus | None:
        if not self.running:
            return None
        return self._status

    def start(
        self,
        config_path: str | Path | None = None,
        *,
        web: bool = True,
        web_host: str = "127.0.0.1",
        web_port: int = MITMWEB_DEFAULT_PORT,
    ) -> MitmProxyStatus:
        if self.running:
            raise MitmAlreadyRunningError("抓包代理已在运行")

        cfg = load_network_config(config_path)
        cfg.ensure_dirs()
        cmd = build_mitm_command(
            cfg,
            config_path=config_path,
            web=web,
            web_host=web_host,
            web_port=web_port,
        )
        env = mitm_proxy_env(config_path)

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            creationflags=creationflags,
        )
        self._status = MitmProxyStatus(
            listen_host=cfg.proxy.listen_host,
            listen_port=cfg.proxy.listen_port,
            web_url=f"http://{web_host}:{web_port}" if web else "",
            capture_dir=cfg.capture_dir,
        )
        return self._status

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        self._status = None
        if proc is None:
            return
        if proc.poll() is not None:
            return

        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
            return

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
