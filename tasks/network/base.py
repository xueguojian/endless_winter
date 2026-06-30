"""基于 HTTP 发包的简单任务基类（后续简单功能继承此类）。"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Callable

from loguru import logger

from core.network.client import GameHttpClient, ReplayResult
from core.network.config import NetworkConfig, load_network_config

StatusCallback = Callable[[str], None]


class NetworkTaskBase(ABC):
    """协议任务基类：子类实现 run_once() 内的发包逻辑。

    示例::

        class MySimpleTask(NetworkTaskBase):
            name = "示例任务"

            def run_once(self) -> bool:
                result = self.client.replay_latest(path_contains="/reward")
                return result.status_code == 200
    """

    name: str = "网络任务"

    def __init__(
        self,
        *,
        config: NetworkConfig | None = None,
        client: GameHttpClient | None = None,
        interval: float = 3600.0,
        on_status: StatusCallback | None = None,
    ):
        self.network_config = config or load_network_config()
        self.client = client or GameHttpClient(config=self.network_config)
        self.interval = interval
        self.on_status = on_status
        self._last_run = 0.0
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

    def _emit(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
        if self.on_status:
            self.on_status(message)

    def _interrupted(self) -> bool:
        return self._stop_event.is_set()

    def should_run(self) -> bool:
        return time.time() - self._last_run >= self.interval

    def _check_ok(self, result: ReplayResult, *, context: str = "") -> bool:
        if 200 <= result.status_code < 300:
            return True
        suffix = f" ({context})" if context else ""
        self._emit(f"请求失败 HTTP {result.status_code}{suffix}")
        return False

    @abstractmethod
    def run_once(self) -> bool:
        """执行一次任务。返回 True 表示成功。"""

    def execute(self) -> bool:
        if self._interrupted():
            return False
        self._emit("开始")
        try:
            ok = self.run_once()
            self._last_run = time.time()
            self._emit("完成" if ok else "未完成")
            return ok
        except Exception as exc:
            self._emit(f"异常: {exc}")
            logger.exception(exc)
            return False
