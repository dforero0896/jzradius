#!/usr/bin/env python3
"""Budget ledger for PLAN.md 2.2/2.3.

The harness has no dollar metering, so costs are proxy numbers
(turns / tool calls / tokens / wall seconds) and ``cost_usd`` stays null.

Usage:
    python scripts/check_budget.py status
    python scripts/check_budget.py record --milestone M1 --turns 12 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = ROOT / "budget.json"
LEDGER_PATH = ROOT / "budget_ledger.jsonl"


def _load_budget() -> dict:
    return json.loads(BUDGET_PATH.read_text())


def _load_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    entries = []
    for line_no, line in enumerate(LEDGER_PATH.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"budget_ledger.jsonl:{line_no}: invalid JSON: {exc}")
    return entries


def _sum(entries: list[dict], key: str) -> float:
    return sum(float(e.get(key) or 0.0) for e in entries)


def _per_milestone(entries: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for entry in entries:
        milestone = entry.get("milestone", "?")
        acc = out.setdefault(milestone, {})
        for key in ("turns", "tool_calls", "network_fetches",
                    "approx_input_tokens", "approx_output_tokens",
                    "build_seconds", "test_seconds"):
            acc[key] = acc.get(key, 0.0) + float(entry.get(key) or 0.0)
    return out


def cmd_status(_args: argparse.Namespace) -> int:
    budget = _load_budget()
    entries = _load_ledger()
    total_turns = _sum(entries, "turns")
    total_tool_calls = _sum(entries, "tool_calls")
    total_fetches = _sum(entries, "network_fetches")

    print(f"ledger: {LEDGER_PATH} ({len(entries)} entries)")
    print(f"turns: {total_turns:.0f}/{budget['max_agent_turns']}")
    print(f"tool calls: {total_tool_calls:.0f} (no hard cap)")
    print(f"network fetches: {total_fetches:.0f}/{budget['max_network_fetches']}")
    print("cost: proxy only (cost_usd=null, no dollar metering)")

    failures = []
    if total_turns > budget["max_agent_turns"]:
        failures.append("total turns exceed max_agent_turns")
    if total_fetches > budget["max_network_fetches"]:
        failures.append("network fetches exceed max_network_fetches")

    for milestone, acc in sorted(_per_milestone(entries).items()):
        print(
            f"  {milestone}: turns={acc['turns']:.0f} "
            f"tool_calls={acc['tool_calls']:.0f} "
            f"fetches={acc['network_fetches']:.0f} "
            f"build={acc['build_seconds']:.0f}s test={acc['test_seconds']:.0f}s"
        )
        if acc["turns"] > budget["max_turns_per_milestone"]:
            failures.append(f"{milestone}: turns exceed max_turns_per_milestone")
        if acc["test_seconds"] > budget["max_wall_seconds_per_test_command"]:
            failures.append(f"{milestone}: single test command over wall cap")
        if acc["build_seconds"] > budget["max_wall_seconds_per_build"]:
            failures.append(f"{milestone}: single build over wall cap")

    if failures:
        print("\nBUDGET VIOLATION(S):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("\nbudget OK (proxy accounting)")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    entry = {
        "milestone": args.milestone,
        "turns": args.turns,
        "tool_calls": args.tool_calls,
        "network_fetches": args.network_fetches,
        "approx_input_tokens": args.approx_input_tokens,
        "approx_output_tokens": args.approx_output_tokens,
        "build_seconds": args.build_seconds,
        "test_seconds": args.test_seconds,
        "cost_usd": None,
        "cost_status": "unknown_or_estimated",
        "note": args.note,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with LEDGER_PATH.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"appended {args.milestone} to {LEDGER_PATH.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="check the ledger against budget.json")

    rec = sub.add_parser("record", help="append one ledger entry")
    rec.add_argument("--milestone", required=True)
    rec.add_argument("--turns", type=float, default=0)
    rec.add_argument("--tool-calls", type=float, default=0)
    rec.add_argument("--network-fetches", type=float, default=0)
    rec.add_argument("--approx-input-tokens", type=float, default=0)
    rec.add_argument("--approx-output-tokens", type=float, default=0)
    rec.add_argument("--build-seconds", type=float, default=0)
    rec.add_argument("--test-seconds", type=float, default=0)
    rec.add_argument("--note", default="")

    args = parser.parse_args()
    if args.command == "status":
        return cmd_status(args)
    return cmd_record(args)


if __name__ == "__main__":
    raise SystemExit(main())
