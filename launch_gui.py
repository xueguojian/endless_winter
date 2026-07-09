"""根据实例 config 的 gui.show_console 决定如何启动 GUI。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from core.config_path import ensure_config_file, parse_config_from_args, resolve_config_path

ROOT = Path(__file__).parent
GUI_MAIN = ROOT / "gui_main.py"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHONW = ROOT / ".venv" / "Scripts" / "pythonw.exe"

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def _load_show_console(config_path: Path) -> bool:
    if not config_path.is_file():
        return False
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return bool(cfg.get("gui", {}).get("show_console", False))


def _is_pythonw() -> bool:
    return Path(sys.executable).name.lower() == "pythonw.exe"


def _gui_argv(exe: Path, config_arg: str | None) -> list[str]:
    argv = [str(exe), str(GUI_MAIN)]
    if config_arg:
        argv.extend(["--config", config_arg])
    return argv


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    if not GUI_MAIN.is_file():
        print(f"未找到 {GUI_MAIN}", file=sys.stderr)
        sys.exit(1)

    args, _unknown = parse_config_from_args()
    config_path = ensure_config_file(resolve_config_path(args.config))
    config_arg = args.config.strip() if args.config else None

    exe = PYTHON if PYTHON.is_file() else Path(sys.executable)
    exew = PYTHONW if PYTHONW.is_file() else exe
    gui_argv = _gui_argv(exe, config_arg)

    if _load_show_console(config_path):
        os.execv(str(exe), gui_argv)

    if _is_pythonw():
        os.execv(str(exew), _gui_argv(exew, config_arg))

    subprocess.Popen(
        _gui_argv(exew, config_arg),
        cwd=ROOT,
        creationflags=CREATE_NO_WINDOW,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
