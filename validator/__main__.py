"""Точка входа для запуска пакета командой `python -m validator`."""

import sys
from validator.cli import main

if __name__ == "__main__":
    sys.exit(main())
