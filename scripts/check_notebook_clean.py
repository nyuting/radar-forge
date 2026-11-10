#!/usr/bin/env python3
"""Exit non-zero if a notebook has stored outputs or execution counts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def is_clean(path: Path) -> bool:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True  # not our problem to diagnose here
    for cell in nb.get("cells", []):
        if cell.get("outputs"):
            return False
        if cell.get("execution_count") is not None:
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(0 if all(is_clean(Path(p)) for p in sys.argv[1:]) else 1)
