"""打印雷电模拟器 + mitmproxy 抓包 HTTPS 的配置步骤。

用法:
  .venv\\Scripts\\python.exe tools/mitm_setup_help.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.network.config import load_network_config


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def _mitm_installed() -> bool:
    venv = ROOT / ".venv" / "Scripts" / "mitmdump.exe"
    if venv.is_file():
        return True
    try:
        subprocess.run(
            ["mitmdump", "--version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def main() -> None:
    cfg = load_network_config()
    ip = _lan_ip()
    port = cfg.proxy.listen_port
    mitm_ok = _mitm_installed()

    print("=" * 60)
    print("  无尽冬日 - HTTPS 抓包配置向导（雷电模拟器）")
    print("=" * 60)
    print()
    print(f"本机局域网 IP（代理主机名填这个）: {ip}")
    print(f"代理端口: {port}")
    print(f"mitmproxy 已安装: {'是' if mitm_ok else '否'}")
    print()

    if not mitm_ok:
        print("【0】安装抓包依赖")
        print("  .venv\\Scripts\\pip.exe install -r requirements-network.txt")
        print()

    print("【1】Windows 防火墙")
    print("  首次运行代理时，若弹出防火墙提示，请允许「专用网络」访问。")
    print("  若模拟器连不上代理，可在「高级防火墙」里放行 mitmdump/python。")
    print()

    print("【2】启动抓包代理（保持窗口不要关）")
    print("  .venv\\Scripts\\python.exe tools/run_mitm_proxy.py --web")
    print("  浏览器打开 http://127.0.0.1:8081 可看实时流量（--web 模式）")
    print()

    print("【3】雷电模拟器设置 WiFi 代理")
    print("  设置 -> WLAN -> 长按已连接 WiFi -> 修改网络")
    print("  代理: 手动")
    print(f"  主机名: {ip}")
    print(f"  端口: {port}")
    print("  保存。若雷电没有 WLAN 菜单，见下方「ADB 设代理」。")
    print()

    print("【4】ADB 设代理（雷电 UI 找不到时用，需 adb 已连接）")
    adb = "D:\\leidian\\LDPlayer9\\adb.exe"
    print(f"  {adb} shell settings put global http_proxy {ip}:{port}")
    print("  取消代理:")
    print(f"  {adb} shell settings put global http_proxy :0")
    print()

    print("【5】安装 mitm 根证书（解密 HTTPS 的关键）")
    print("  模拟器里打开浏览器，访问: http://mitm.it")
    print("  点 Android 图标下载证书，按提示安装。")
    print("  Android 7+ 用户证书默认不被 App 信任，若游戏仍无 HTTPS 明文：")
    print("    - 雷电设置里开启 Root")
    print("    - 或把证书装到系统区（需 root / Magisk 模块）")
    print("    - 或下一节 SSL Pinning 绕过")
    print()

    print("【6】验证 HTTPS 是否解密成功")
    print("  模拟器浏览器打开 https://www.baidu.com")
    print("  mitmweb 里应能看到明文请求；若只有 CONNECT 或证书错误，说明未解密。")
    print()

    print("【7】抓游戏包")
    print("  打开无尽冬日，手动点一次「领取 xx」")
    print("  抓包保存在 assets/captures/ 目录")
    print("  查看: .venv\\Scripts\\python.exe tools/list_captures.py")
    print()

    print("【8】若游戏有 SSL Pinning（常见）")
    print("  现象: 其他 App/浏览器能解密，唯独游戏连不上或仍无明文")
    print("  需要 Frida + frida-server（雷电开 Root）hook 证书校验")
    print("  或先用 Charles + JustTrustMe 等方案，原理相同")
    print("  这一步较折腾，确认 Pinning 后再做；需要时可再问我。")
    print()

    print("【9】抓完后记得关代理")
    print("  模拟器 WiFi 改回「无代理」，或 adb settings put global http_proxy :0")
    print("  否则游戏可能上不了线。")
    print()


if __name__ == "__main__":
    main()
