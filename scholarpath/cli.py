"""Command line interface: `python -m scholarpath run|demo`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from scholarpath.config import PROJECT_ROOT, STRONG_MATCH_SCORE
from scholarpath.graph import build_graph, run_pipeline


def describe(node: str, update: dict[str, Any]) -> str:
    if node == "reader":
        p = update["profile"]
        missing = ", ".join(p.missing_fields) or "none"
        return (
            f"[reader]   profile: level={p.education_level}, field={p.field_of_study}, "
            f"confidence={p.confidence:.2f}, missing: {missing}"
        )
    if node == "clarify":
        return "[clarify]  added the student's answer to the profile text"
    if node == "searcher":
        log = update["search_log"]
        attempt = update["search_attempts"]
        calls = sum(1 for q in log.queries if q.attempt == attempt)
        strong = sum(1 for o in update["opportunities"] if o.match_score >= STRONG_MATCH_SCORE)
        mode = " (broadened)" if attempt > 1 else ""
        return (
            f"[searcher] attempt {attempt}{mode}: {calls} tool calls, "
            f"{len(update['opportunities'])} ranked, {strong} strong matches"
        )
    if node == "reporter":
        return f"[reporter] report written ({len(update['report'].splitlines())} lines)"
    return f"[{node}]"


def ask_user(question: str) -> str:
    print(f"\n{question}")
    return input("> ").strip() or "No additional details."


def run_one(text: str, interactive: bool, out: str | None, graph: Any = None) -> int:
    try:
        state = run_pipeline(
            text,
            graph=graph,
            interactive=interactive,
            output_dir=out,
            ask=ask_user if interactive else None,
            on_update=lambda node, update: print(describe(node, update)),
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for err in state.get("errors", []):
        print(f"warning: {err}")
    print(f"Report: {state['report_path']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="scholarpath", description="Find educational opportunities.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show agent logs")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="analyze one student and write a report")
    run.add_argument("--input", required=True, help="path to a .txt/.md file, or - for stdin")
    run.add_argument("--interactive", action="store_true", help="ask follow-up questions if needed")
    run.add_argument("--out", help="report directory (default: reports/)")

    demo = sub.add_parser("demo", help="run every example in examples/")
    demo.add_argument("--out", help="report directory (default: reports/)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    if args.command == "run":
        text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        return run_one(text, args.interactive, args.out)

    graph = build_graph()
    failures = 0
    for path in sorted((PROJECT_ROOT / "examples").glob("*.txt")):
        print(f"\n=== {path.name} ===")
        failures += run_one(path.read_text(encoding="utf-8"), False, args.out, graph)
    return 1 if failures else 0
