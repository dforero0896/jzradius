"""Shared tier runner for PLAN.md 18.1.

Each tier has a hard wall-clock cap and tails at most 60 output lines so a
failure cascade never floods the context (PLAN.md 2.4).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIERS = {
    "smoke": ("smoke", 60),
    "unit": ("not gpu and not slow", 300),
    "integration": ("integration", 900),
}


def run(tier: str, extra: list[str] | None = None) -> int:
    marker, cap = TIERS[tier]
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    cmd = [
        sys.executable, "-m", "pytest", "-q", "--maxfail=3",
        "-m", marker, *(extra or []),
    ]
    print(f"[{tier}] cap={cap}s cmd={' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=cap
        )
    except subprocess.TimeoutExpired as exc:
        tail = (exc.stdout or "")[-4000:]
        if isinstance(tail, bytes):
            tail = tail.decode(errors="replace")
        print(tail, file=sys.stderr)
        print(f"[{tier}] TIMEOUT after {cap}s", file=sys.stderr)
        return 124

    output = (proc.stdout or "") + (proc.stderr or "")
    print("\n".join(output.splitlines()[-60:]))
    return proc.returncode


def main(tier: str) -> int:
    return run(tier, sys.argv[1:])
