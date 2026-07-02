"""ADB 设备连接与操作封装。"""

from __future__ import annotations

import io
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

_WIN_SUBPROCESS_FLAGS = 0
if sys.platform == "win32":
    _WIN_SUBPROCESS_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class AdbClient:
    """通过 ADB 控制雷电模拟器。"""

    LDPLAYER_PROBE_PORTS = tuple(range(5555, 5571, 2))

    LDPLAYER_ADB_CANDIDATES = [
        r"C:\leidian\LDPlayer9\adb.exe",
        r"D:\leidian\LDPlayer9\adb.exe",
        r"C:\leidian\LDPlayer4\adb.exe",
        r"D:\leidian\LDPlayer4\adb.exe",
        r"C:\Program Files\LDPlayer\LDPlayer9\adb.exe",
        r"C:\Program Files\LDPlayer\LDPlayer9\adb.exe",
    ]

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        adb_path: str = "",
        touch_width: int = 720,
        touch_height: int = 1280,
    ):
        self.address = f"{host}:{port}"
        self.adb = self._resolve_adb(adb_path)
        self.touch_width = touch_width
        self.touch_height = touch_height
        self._io_lock = threading.Lock()

    @classmethod
    def resolve_adb_path(cls, adb_path: str = "") -> str:
        if adb_path and Path(adb_path).is_file():
            return adb_path

        for candidate in cls.LDPLAYER_ADB_CANDIDATES:
            if Path(candidate).is_file():
                logger.info(f"找到雷电 ADB: {candidate}")
                return candidate

        raise FileNotFoundError(
            "未找到 adb.exe。请在 config.yaml 的 device.adb_path 中填写雷电安装目录下的 adb.exe 路径。"
        )

    @staticmethod
    def parse_address(serial: str) -> tuple[str, int]:
        """解析 adb devices 中的序列号，如 127.0.0.1:5555、emulator-5554。"""
        serial = serial.strip()
        if not serial:
            raise ValueError("设备地址不能为空")

        if serial.startswith("emulator-"):
            port = int(serial.rsplit("-", 1)[1])
            return "127.0.0.1", port

        if ":" in serial:
            host, port_text = serial.rsplit(":", 1)
            return host, int(port_text)

        raise ValueError(f"无法解析设备地址：{serial}")

    @classmethod
    def format_address(cls, host: str, port: int) -> str:
        return f"{host}:{int(port)}"

    def _resolve_adb(self, adb_path: str) -> str:
        return self.resolve_adb_path(adb_path)

    def _parse_devices_output(self, output: str) -> list[str]:
        devices: list[str] = []
        for line in output.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("list of devices"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        return devices

    @staticmethod
    def normalize_device_serial(serial: str) -> str:
        """统一设备显示地址，合并 emulator-5554 与 127.0.0.1:5555 等等价项。"""
        serial = serial.strip()
        if serial.startswith("emulator-"):
            console_port = int(serial.rsplit("-", 1)[1])
            return f"127.0.0.1:{console_port + 1}"

        host, port = AdbClient.parse_address(serial)
        if host in {"127.0.0.1", "localhost"}:
            return f"127.0.0.1:{port}"
        return f"{host}:{port}"

    @classmethod
    def dedupe_device_serials(cls, devices: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for serial in devices:
            try:
                normalized = cls.normalize_device_serial(serial)
            except ValueError:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @classmethod
    def list_connected_devices(
        cls,
        adb_path: str = "",
        *,
        probe_ldplayer: bool = True,
    ) -> list[str]:
        """列出当前已连接的 ADB 设备；probe_ldplayer 时会尝试连接雷电多开端口。"""
        adb_bin = cls.resolve_adb_path(adb_path)
        runner = cls(host="127.0.0.1", port=5555, adb_path=adb_bin)

        result = runner._run("devices")
        devices = runner._parse_devices_output(result.stdout)

        if probe_ldplayer:
            known = {cls.normalize_device_serial(item) for item in devices}
            for port in cls.LDPLAYER_PROBE_PORTS:
                address = f"127.0.0.1:{port}"
                if address in known:
                    continue
                connect = runner._run("connect", address, timeout=4)
                output = (connect.stdout + connect.stderr).lower()
                if "connected" in output and "cannot" not in output:
                    check = runner._run("devices")
                    devices = runner._parse_devices_output(check.stdout)
                    known = {cls.normalize_device_serial(item) for item in devices}

        devices = cls.dedupe_device_serials(devices)
        logger.info(f"ADB 已连接设备: {devices or '无'}")
        return devices

    def _run(self, *args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        cmd = [self.adb, *args]
        logger.debug(f"ADB: {' '.join(cmd)}")
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            creationflags=_WIN_SUBPROCESS_FLAGS,
        )

    def connect(self) -> bool:
        result = self._run("connect", self.address)
        output = (result.stdout + result.stderr).strip()
        logger.info(output or "ADB connect 完成")
        return "connected" in output.lower() or self.address in self._run("devices").stdout

    def wait_for_device(self, retries: int = 30, interval: float = 2.0) -> bool:
        quick = self._run("-s", self.address, "shell", "echo", "ok", timeout=5)
        if quick.returncode == 0 and "ok" in quick.stdout:
            logger.debug(f"设备已就绪(快速): {self.address}")
            return True
        for attempt in range(1, retries + 1):
            if self.connect():
                check = self._run("-s", self.address, "shell", "echo", "ok")
                if check.returncode == 0 and "ok" in check.stdout:
                    logger.info(f"设备已就绪: {self.address}")
                    return True
            logger.warning(f"等待设备连接 ({attempt}/{retries})...")
            time.sleep(interval)
        return False

    def shell(self, command: str) -> str:
        result = self._run("-s", self.address, "shell", command)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ADB shell 命令失败")
        return result.stdout.strip()

    def get_screen_size(self) -> tuple[int, int]:
        """返回截图宽高 (width, height)，应与 touch_width/touch_height 一致。"""
        img = self.screenshot()
        h, w = img.shape[:2]
        return w, h

    def screenshot(self) -> np.ndarray:
        """截取屏幕，返回 BGR 数组，shape=(height, width)，与 input tap 同一竖屏坐标系。"""
        with self._io_lock:
            proc = subprocess.run(
                [self.adb, "-s", self.address, "exec-out", "screencap", "-p"],
                capture_output=True,
                timeout=15,
                creationflags=_WIN_SUBPROCESS_FLAGS,
            )
            if proc.returncode != 0:
                raise RuntimeError("截图失败，请确认模拟器已启动且 ADB 已连接")

            image = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
            rgb = np.array(image)
            return rgb[:, :, ::-1].copy()

    def tap(self, x: int, y: int) -> None:
        x = max(0, min(x, self.touch_width - 1))
        y = max(0, min(y, self.touch_height - 1))
        with self._io_lock:
            self._run("-s", self.address, "shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self._run(
            "-s",
            self.address,
            "shell",
            "input",
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            str(duration_ms),
        )

    def back(self) -> None:
        self._run("-s", self.address, "shell", "input", "keyevent", "4")

    def escape(self) -> None:
        """发送 ESC 键（Android KEYCODE_ESCAPE）。"""
        self._run("-s", self.address, "shell", "input", "keyevent", "111")

    def launch_game(self, package: str, activity: str) -> None:
        self._run("-s", self.address, "shell", "am", "start", "-n", activity)
        logger.info(f"已启动游戏: {package}")

    def get_touch_size(self) -> tuple[int, int]:
        return self.touch_width, self.touch_height
