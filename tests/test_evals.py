"""The golden eval pack must pass as shipped (offline, hermetic)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_golden_eval_pack_passes():
    proc = subprocess.run(
        [sys.executable, "evals/run_evals.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "47/47 passed" in proc.stdout
    assert "gate: PASS" in proc.stdout
