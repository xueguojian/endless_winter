"""无尽冬日 — 自动脚本 GUI 入口。"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from gui.app import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
