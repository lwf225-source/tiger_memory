#!/usr/bin/env python3
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent, HERE.parent / "lib"):
    if (candidate / "tiger_memory").exists():
        sys.path.insert(0, str(candidate))
        break

from tiger_memory.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
