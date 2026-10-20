#!/usr/bin/env python3
"""Whether a source rebuild reused the dependency installs (PRD-262 US-039).

``make up``, ``make build`` and ``make dev`` end with ``make clean``, which used to run
``docker builder prune -f``. Without ``--all``, BuildKit's prune deletes every cache
record that no image holds, and the two dependency installs are in no image: the
frontend's ``npm install`` (``npm ci`` once its lock is used) runs in a stage the
production image only copies the build output from, and the backend's ``pip install``
in the ``pybuild`` stage it copies ``/install`` from. So the next ``make build`` or
``make dev`` after a ``git pull`` installed every dependency again. ``make clean`` now
keeps a week of cache (``docker builder prune -f --filter until=168h``).

``scripts/ci/build-cache-experiment.sh`` builds the backend and the frontend, prunes,
changes one line of each app's source, rebuilds, and runs this on each rebuild's log,
in BuildKit's plain progress (a piped ``docker compose build`` prints it):

    build_cache.py LOG --arm NAME [--expect-cached] --out JSON [--summary MD]

It finds the frontend's ``RUN npm ci|install`` step and the backend's ``RUN … pip
install`` step and says whether each was ``CACHED``, ran (and for how long), failed,
or is not in the log. With ``--expect-cached`` it exits 1 unless both were CACHED;
without it the arm is recorded only. The layer cache only: no Railway-built Dockerfile
has a BuildKit cache mount (PRD-262 §10). The standard library only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
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


build_context = _sibling("build_context")  # the plain-progress step reader (PRD-262 US-017)

# (install, compose service, what its RUN runs). Any stage: npm ci moves to the base
# stage with the frontend's lock (PRD-262 US-047), and pip may run as `uv pip install`.
INSTALLS = (
    ("npm install", "frontend", re.compile(r"^RUN .*\bnpm (?:ci|install)\b")),
    ("pip install", "backend", re.compile(r"^RUN .*\bpip install\b")),
)
# `[frontend builder 4/8] RUN npm install --legacy-peer-deps`: the service, then the step.
STEP_NAME = re.compile(r"^\[(?P<service>[\w.-]+) [^\]]*\] (?P<instruction>.+)$")
DONE = re.compile(r"^DONE (\d+(?:\.\d+)?)s$")
CACHED, RAN, FAILED, MISSING = "cached", "ran", "failed", "missing"
SHOWN_CHARS = 90  # a step name in the summary; pip's RUN is a whole shell script


def install_named(text: str) -> Optional[str]:
    """The install a step's name is (``npm install`` or ``pip install``), or None."""
    name = STEP_NAME.match(text)
    if name is None:
        return None
    for install, service, pattern in INSTALLS:
        if name.group("service") == service and pattern.search(name.group("instruction")):
            return install
    return None


def step_status(text: str) -> Optional[Tuple[str, Optional[float]]]:
    """``(state, seconds)`` when a step line is its status, else None."""
    if text == "CACHED":
        return CACHED, None
    done = DONE.match(text)
    if done:
        return RAN, float(done.group(1))
    if text.startswith("ERROR"):
        return FAILED, None
    return None


def install_states(log: str) -> Dict[str, dict]:
    """Each install's step name, state and seconds in ``log``; ``missing`` when not there."""
    states = {install: {"step": None, "state": MISSING, "seconds": None} for install, _, _ in INSTALLS}
    ids: Dict[str, str] = {}
    for step, text in build_context.step_lines(log):
        install = install_named(text)
        if install:
            ids = {**ids, step: install}
            states = {**states, install: {**states[install], "step": text}}
            continue
        status = step_status(text)
        if status and step in ids:
            state, seconds = status
            states = {**states, ids[step]: {**states[ids[step]], "state": state, "seconds": seconds}}
    return states


def _described(state: dict) -> str:
    if state["state"] == RAN:
        return f"ran ({state['seconds']:.1f} s)"
    if state["state"] == MISSING:
        return "not in the log (BuildKit neither ran it nor listed it CACHED, or the log is not plain progress)"
    return "CACHED" if state["state"] == CACHED else "failed"


def check(log: str, arm: str, expect_cached: bool) -> dict:
    """The arm's record: each install's state and, when CACHED is expected, the failures."""
    states = install_states(log)
    failures: List[str] = []
    if expect_cached:
        failures = [f"{install} after {arm}: {_described(state)}, not CACHED"
                    for install, state in states.items() if state["state"] != CACHED]
    return {"arm": arm, "expect_cached": expect_cached, "installs": states, "failures": failures}


# --------------------------------------------------------------------------- Markdown


def _shown(step: Optional[str]) -> str:
    if step is None:
        return "—"
    return f"`{step if len(step) <= SHOWN_CHARS else step[:SHOWN_CHARS] + '…'}`"


def render_markdown(record: dict) -> str:
    """The step summary's section for one arm."""
    lines = [f"### Source rebuild after `{record['arm']}` (PRD-262 US-039)", "",
             "| Install | Step | Result |", "|---|---|---|"]
    lines += [f"| {install} | {_shown(state['step'])} | {_described(state)} |"
              for install, state in record["installs"].items()]
    if not record["expect_cached"]:
        verdict = "Recorded only: what this prune leaves a rebuild after a one-line change."
    elif record["failures"]:
        verdict = "**FAIL**: " + "; ".join(record["failures"]) + "."
    else:
        verdict = "Both installs CACHED: pass."
    return "\n".join(lines + ["", verdict, ""]) + "\n"


# --------------------------------------------------------------------------- command


def build_parser() -> argparse.ArgumentParser:
    """The rebuild's log, the arm's name, whether CACHED is required, and the outputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path, help="the rebuild's log, in BuildKit's plain progress")
    parser.add_argument("--arm", required=True, help="what ran before the rebuild, e.g. make clean")
    parser.add_argument("--expect-cached", action="store_true", help="fail unless both installs were CACHED")
    parser.add_argument("--out", type=Path, required=True, help="where the JSON record goes")
    parser.add_argument("--summary", default="/dev/stdout", help="the Markdown is appended here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    record = check(args.log.read_text(encoding="utf-8", errors="replace"), args.arm, args.expect_cached)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with open(args.summary, "a", encoding="utf-8") as summary:
        summary.write(render_markdown(record))
    for failure in record["failures"]:
        print(f"::error title=source build cache::{failure}")
    for install, state in record["installs"].items():
        print(f"{install} after {args.arm}: {_described(state)}")
    return 1 if record["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
