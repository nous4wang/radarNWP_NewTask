"""pytest bootstrap.

Puts the repository root on ``sys.path`` so that ``import radar_nwp_bench``
resolves when the suite is collected by pytest, and makes the helper modules
in this directory importable.

The suite is also runnable without pytest via ``tests/run_tests.py``, which
performs the same path setup itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

for _path in (str(REPO_ROOT), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
