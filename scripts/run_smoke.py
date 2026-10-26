#!/usr/bin/env python3
"""Smoke tier: N <= 32, 60 s cap (PLAN.md 18.1)."""

from _runner import main

if __name__ == "__main__":
    raise SystemExit(main("smoke"))
