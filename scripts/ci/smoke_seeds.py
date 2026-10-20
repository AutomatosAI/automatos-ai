#!/usr/bin/env python3
"""A warm restart redoes no seed work, checked from the smoke's log and database.

PRD-262 US-031: each boot seed stage stores the fingerprint of its inputs once it ran
clean, and a boot whose stored fingerprint matches skips the stage:

* the entrypoint's seed loader (``seed_fingerprint.loader``): the entrypoint asks before
  it starts the loader process, and logs ``📥 Loading seed data`` only when it starts it;
* the lifespan leader's seed block and starters pass (``seed_fingerprint.leader``).

Each skip logs ``seed_fingerprint.<stage>: seeds unchanged (fingerprint …) — skipped``.
``smoke-fresh-clone.sh`` calls this after its last backend boot, with the records
``smoke_boot.py`` appended and the backend's ``docker logs -t``, and cuts the log at
each boot's start (``smoke_report.split_log``):

* ``--expect-skipped LABEL``: that boot logs both stages' skip lines and starts no seed
  loader. The smoke names its warm restarts.
* ``--expect-loaded LABEL``: that boot starts the seed loader. The smoke names its fresh
  boot, where nothing is stored yet, so a check that always skipped would show.
* ``--credential-types BEFORE AFTER``: two snapshots of ``credential_types`` (the row
  count and a digest of every row's ``updated_at``), taken before and after the warm
  restarts, are the same: no warm restart rewrote a credential type.

The standard library only, like the other smoke tools. Exit 0 = every check passed.

    python3 scripts/ci/smoke_seeds.py --results "$OUT_DIR/boots.jsonl" \\
        --backend-log "$OUT_DIR/backend.log" --expect-loaded fresh --expect-skipped warm-1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


smoke_report = _sibling("smoke_report")

STAGES = ("loader", "leader")
SKIP_TEXT = "seeds unchanged (fingerprint"
LOADER_LINE = "Loading seed data"
TITLE = "No seed work on a warm restart (PRD-262 US-031)"


def skip_marker(stage: str) -> str:
    """The text a stage's skip line holds (core/seeds/fingerprint.py's skipped_line)."""
    return f"seed_fingerprint.{stage}: {SKIP_TEXT}"


def boot_facts(lines: Sequence[str]) -> dict:
    """One boot's seed-stage skip lines and seed-loader starts."""
    facts = {f"{stage}_skipped": sum(skip_marker(stage) in line for line in lines) for stage in STAGES}
    facts["loader_started"] = sum(LOADER_LINE in line for line in lines)
    return facts


def boot_problems(label: str, facts: dict, skipped: bool, loaded: bool) -> List[str]:
    """What one boot's facts break against what was expected of it; empty when it passes."""
    problems = []
    if skipped:
        problems += [f"{label}: no `{skip_marker(stage)}` line, so its {stage} stage ran again"
                     for stage in STAGES if not facts[f"{stage}_skipped"]]
        if facts["loader_started"]:
            problems.append(f"{label}: `{LOADER_LINE}` logged, so a seed-loader process started")
    if loaded and not facts["loader_started"]:
        problems.append(f"{label}: no `{LOADER_LINE}` line; the first boot must run the seed loader")
    return problems


def check_boots(records: Sequence[dict], log_lines: Sequence[str], skipped: Sequence[str],
                loaded: Sequence[str], container: str) -> dict:
    """Every backend boot's facts and the problems they show."""
    boots = sorted((record for record in records if record.get("container") == container),
                   key=lambda record: smoke_report.parse_timestamp(record["started_at"]))
    starts = [smoke_report.parse_timestamp(record["started_at"]) for record in boots]
    rows = [{"label": record["label"], **boot_facts(lines)}
            for record, lines in zip(boots, smoke_report.split_log(log_lines, starts))]
    failures = [problem for row in rows
                for problem in boot_problems(row["label"], row, row["label"] in skipped, row["label"] in loaded)]
    failures += [f"{label}: no such backend boot was timed"
                 for label in sorted((set(skipped) | set(loaded)) - {row["label"] for row in rows})]
    return {"boots": rows, "failures": failures}


def credential_types_problems(before: str, after: str) -> List[str]:
    """``count digest`` snapshots of credential_types: present, non-empty and unchanged."""
    count = before.split(" ", 1)[0]
    if not before or not count.isdigit() or int(count) == 0:
        return [f"the credential_types snapshot before the warm restarts is empty or unreadable: {before!r}"]
    if after != before:
        return [f"credential_types changed across the warm restarts: {before!r} became {after!r}"]
    return []


def summary_markdown(result: dict) -> str:
    verdict = "passed" if not result["failures"] else "FAILED"
    lines = [
        f"### {TITLE}: {verdict}",
        "",
        "| Boot | loader skipped | leader skipped | seed loader started |",
        "|---|---|---|---|",
    ]
    lines += [f"| {boot['label']} | {boot['loader_skipped']} | {boot['leader_skipped']} | {boot['loader_started']} |"
              for boot in result["boots"]]
    snapshot = result.get("credential_types")
    if snapshot:
        lines += ["", f"credential_types before / after the warm restarts: `{snapshot['before']}` / `{snapshot['after']}`"]
    lines += [""] + [f"- **FAIL**: {failure}" for failure in result["failures"]]
    return "\n".join(lines) + "\n\n"


def _read_snapshot(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--results", type=Path, required=True, help="the records smoke_boot.py appended")
    parser.add_argument("--backend-log", type=Path, required=True, help="docker logs -t output of the backend")
    parser.add_argument("--backend-container", default=smoke_report.BACKEND_CONTAINER)
    parser.add_argument("--expect-skipped", action="append", default=[], metavar="LABEL",
                        help="a boot that must skip both seed stages and start no seed loader")
    parser.add_argument("--expect-loaded", action="append", default=[], metavar="LABEL",
                        help="a boot that must start the seed loader")
    parser.add_argument("--credential-types", nargs=2, type=Path, metavar=("BEFORE", "AFTER"),
                        help="credential_types snapshots taken before and after the warm restarts")
    parser.add_argument("--out", type=Path, help="where the JSON record goes")
    parser.add_argument("--summary", default="/dev/stdout", help="the Markdown is appended here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log_lines = args.backend_log.read_text(encoding="utf-8", errors="replace").splitlines()
    result = check_boots(smoke_report.read_json_lines(args.results), log_lines,
                         args.expect_skipped, args.expect_loaded, args.backend_container)
    if args.credential_types:
        before, after = (_read_snapshot(path) for path in args.credential_types)
        result["credential_types"] = {"before": before, "after": after}
        result["failures"] += credential_types_problems(before, after)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with open(args.summary, "a", encoding="utf-8") as handle:
        handle.write(summary_markdown(result))
    for failure in result["failures"]:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
