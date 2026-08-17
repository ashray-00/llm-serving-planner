"""Test path setup for running bench tests from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_DIR.parent

sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO_ROOT / "kvcalc" / "src"))
