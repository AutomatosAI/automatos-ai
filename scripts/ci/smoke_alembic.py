#!/usr/bin/env python3
"""One migration run per backend boot, checked from the smoke's log (PRD-262 US-029).

The entrypoint is the only owner of migrations: before a server command it runs the
migration step once, and the image CMD runs no alembic. ``smoke-fresh-clone.sh``
calls this after its last backend boot, with the records ``smoke_boot.py`` appended
and the backend's ``docker logs -t``. It cuts the log at each boot's start
(``smoke_report.split_log``) and counts each boot's alembic runs the PRD-262 §8 way
(``smoke_report.alembic_runs``: the ``Context impl PostgresqlImpl`` lines after
``init_fresh_db: done``, or all of them on a boot where init_fresh_db did not run).
Every backend boot must show:

* the entrypoint's ``🚀 Starting Backend Application`` banner: the boot steps ran
  before the server command, so the stage's CMD was recognised as one;
* no ``Context impl PostgresqlImpl`` line after that banner: the CMD ran no alembic;
* at most one alembic run, and at most one ``Running database migrations`` line.

``--expect LABEL=N``: that boot counts exactly N alembic runs and N ``Running
database migrations`` lines. The smoke's ``behind-head`` boot, on a database stamped
one revision behind head (``smoke_alembic_behind.py``), expects 1.

The standard library only, like the other smoke tools. Exit 0 = every boot passed.

    python3 scripts/ci/smoke_alembic.py --results "$OUT_DIR/boots.jsonl" \\
        --backend-log "$OUT_DIR/backend.log" --expect behind-head=1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


smoke_report = _sibling("smoke_report")

BANNER_LINE = "Starting Backend Application"
MIGRATION_STEP_LINE = "Running database migrations"
MAX_RUNS_PER_BOOT = 1
TITLE = "One migration run per backend boot (PRD-262 US-029)"


def boot_facts(lines: Sequence[str]) -> dict:
    """One boot's alembic runs (§8), its banner, and the alembic lines after the banner."""
    banners = [index for index, line in enumerate(lines) if BANNER_LINE in line]
    after = lines[banners[-1] + 1:] if banners else []
    return {
        **smoke_report.alembic_runs(lines),
        "banner": bool(banners),
        "context_impl_after_banner": sum(smoke_report.ALEMBIC_RUN_LINE in line for line in after),
        "migration_steps": sum(MIGRATION_STEP_LINE in line for line in lines),
    }


def boot_problems(label: str, facts: dict, expected: Optional[int]) -> List[str]:
    """What one boot's facts break; empty when it passes."""
    runs, steps = facts["runs"], facts["migration_steps"]
    problems = []
    if not facts["banner"]:
        problems.append(f"{label}: no `{BANNER_LINE}` banner, so the entrypoint ran no boot step before the server")
    if facts["context_impl_after_banner"]:
        problems.append(f"{label}: {facts['context_impl_after_banner']} `Context impl` line(s) after the banner, "
                        "so the CMD ran alembic")
    if max(runs, steps) > MAX_RUNS_PER_BOOT:
        problems.append(f"{label}: {runs} alembic run(s) and {steps} `{MIGRATION_STEP_LINE}` line(s); "
                        f"at most {MAX_RUNS_PER_BOOT} per boot")
    if expected is not None and (runs, steps) != (expected, expected):
        problems.append(f"{label}: expected {expected} alembic run(s) and {expected} `{MIGRATION_STEP_LINE}` "
                        f"line(s), found {runs} and {steps}")
    return problems


def check(records: Sequence[dict], log_lines: Sequence[str], expected: Dict[str, int], container: str) -> dict:
    """Every backend boot's facts and the problems they show."""
    boots = sorted((record for record in records if record.get("container") == container),
                   key=lambda record: smoke_report.parse_timestamp(record["started_at"]))
    starts = [smoke_report.parse_timestamp(record["started_at"]) for record in boots]
    rows = [{"label": record["label"], **boot_facts(lines)}
            for record, lines in zip(boots, smoke_report.split_log(log_lines, starts))]
    failures = [problem for row in rows for problem in boot_problems(row["label"], row, expected.get(row["label"]))]
    failures += [f"{label}: no such backend boot was timed"
                 for label in sorted(set(expected) - {row["label"] for row in rows})]
    if not rows:
        failures.append("no backend boot was timed")
    return {"boots": rows, "expected": dict(expected), "failures": failures}


def summary_markdown(result: dict) -> str:
    verdict = "passed" if not result["failures"] else "FAILED"
    lines = [
        f"### {TITLE}: {verdict}",
        "",
        f"| Boot | alembic runs | `{MIGRATION_STEP_LINE}` | `Context impl` after the banner | banner |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {boot['label']} | {boot['runs']} | {boot['migration_steps']} | {boot['context_impl_after_banner']} "
              f"| {'yes' if boot['banner'] else 'no'} |" for boot in result["boots"]]
    lines += [""] + [f"- **FAIL**: {failure}" for failure in result["failures"]]
    return "\n".join(lines) + "\n\n"


def parse_expect(text: str) -> Tuple[str, int]:
    """``LABEL=N`` as given to ``--expect``."""
    label, sep, count = text.partition("=")
    if not (label and sep and count.isdigit()):
        raise argparse.ArgumentTypeError(f"--expect takes LABEL=N, not {text!r}")
    return label, int(count)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--results", type=Path, required=True, help="the records smoke_boot.py appended")
    parser.add_argument("--backend-log", type=Path, required=True, help="docker logs -t output of the backend")
    parser.add_argument("--backend-container", default=smoke_report.BACKEND_CONTAINER)
    parser.add_argument("--expect", type=parse_expect, action="append", default=[], metavar="LABEL=N",
                        help="that boot's exact alembic runs and `Running database migrations` lines")
    parser.add_argument("--out", type=Path, help="where the JSON record goes")
    parser.add_argument("--summary", default="/dev/stdout", help="the Markdown is appended here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log_lines = args.backend_log.read_text(encoding="utf-8", errors="replace").splitlines()
    result = check(smoke_report.read_json_lines(args.results), log_lines, dict(args.expect), args.backend_container)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with open(args.summary, "a", encoding="utf-8") as handle:
        handle.write(summary_markdown(result))
    for failure in result["failures"]:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
