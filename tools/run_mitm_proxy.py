"""启动 mitmproxy 抓包代理。

用法:
  .venv\\Scripts\\python.exe tools/run_mitm_proxy.py
  .venv\\Scripts\\python.exe tools/run_mitm_proxy.py --config config_5555.yaml
  .venv\\Scripts\\python.exe tools/run_mitm_proxy.py --web

前置步骤（首次）:
  1. pip install -r requirements-network.txt
  2. 雷电模拟器 WiFi 代理 -> 电脑 IP:8080（端口见 config network.proxy）
  3. 浏览器访问 http://mitm.it 安装证书到模拟器
  4. 若 HTTPS 仍无法解密，需 Frida 等方式绕过 SSL Pinning

抓包文件保存在 assets/captures/ 按日期分目录。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.network.config import load_network_config


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 mitmproxy 抓包代理")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="配置文件路径（默认 config.yaml）",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="使用 mitmweb 图形界面（默认 mitmdump 仅控制台）",
    )
    args = parser.parse_args()

    cfg = load_network_config(args.config)
    cfg.ensure_dirs()

    addon = ROOT / "core" / "network" / "mitm_addon.py"
    if not addon.is_file():
        raise FileNotFoundError(addon)

    mitm_bin = shutil.which("mitmdump" if not args.web else "mitmweb")
    if mitm_bin is None:
        venv_scripts = ROOT / ".venv" / "Scripts"
        name = "mitmweb.exe" if args.web else "mitmdump.exe"
        candidate = venv_scripts / name
        mitm_bin = str(candidate) if candidate.is_file() else None
    if not mitm_bin:
        print(
            "未找到 mitmdump/mitmweb。请先安装:\n"
            "  .venv\\Scripts\\pip.exe install -r requirements-network.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    env = os.environ.copy()
    if args.config is not None:
        env["EW_CONFIG_PATH"] = str(args.config.resolve())

    cmd = [
        mitm_bin,
        "-s",
        str(addon),
        "--listen-host",
        cfg.proxy.listen_host,
        "--listen-port",
        str(cfg.proxy.listen_port),
        "--set",
        "block_global=false",
    ]
    if args.web:
        cmd.extend(["--web-host", "127.0.0.1"])

    print(f"监听 {cfg.proxy.listen_host}:{cfg.proxy.listen_port}")
    print(f"抓包目录 {cfg.capture_dir}")
    if cfg.capture.include_hosts:
        print(f"仅保存 host 含: {cfg.capture.include_hosts}")
    print("Ctrl+C 停止\n")

    subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)


if __name__ == "__main__":
    main()
