#!/usr/bin/env python3
"""What the app bytecode buys a new API container, and what it costs an update (PRD-262 US-032, §9-Q7).

A one-off experiment for the W4 pull request, run by
.github/workflows/bytecode-experiment.yml. The job builds the API's production stage
twice: as committed (``with``: the compile stage compiles the boot modules) and with
the compile step removed in the job's workspace (``without``). It builds each again
after a code-only change (``-edit``). All four are pushed to the job's own registry.
Then scripts/ci/bytecode-experiment.sh boots ``with`` and ``without`` in new
containers against one seeded database. Commands:

* ``strip-compile DOCKERFILE`` removes the compile stage's RUN in place, so the next
  build ships the code with no bytecode, as before US-032. It fails unless that
  stage has exactly one RUN and it runs compileall.
* ``layers`` reads the four builds back from the registry. Per build it records the
  compressed size of the final app layer and of the image, and the code-only update
  delta from the build to its ``-edit``, judged against US-016's bar in
  .github/image-budgets.json.
* ``boot`` times one new container from its own start (``State.StartedAt``) to the
  first ``/health`` 200, polling every ``POLL_SECONDS``.
* ``report`` writes the step summary and a JSON record: the median boot of each
  build, the gain, the bytecode's share of the app layer, the update deltas, and the
  §9-Q7 outcome. The bytecode stays only if a new container reaches /health 200 at
  least ``GAIN_BAR_SECONDS`` sooner with it and the code-only update still meets
  US-016's bar. If the gain is under the bar, W5's push removes the compile step. If
  the gain holds but the update misses, Gerard chooses in that push between removing
  the step and raising the bar.

The numbers are a report, not a gate: the job fails only when a build, a read or a
boot fails, so that no number is missing. Standard library only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


layers = _sibling("image_layers")
registry = _sibling("image_registry")
smoke_boot = _sibling("smoke_boot")

VARIANTS = ("with", "without")
EDIT_SUFFIX = "-edit"
ARCH = "amd64"
IMAGE_NAME = "api"  # the update bar's image in .github/image-budgets.json
GAIN_BAR_SECONDS = 2.0  # PRD-262 §9-Q7
POLL_SECONDS = 0.25
NS_PER_SECOND = 1_000_000_000
COMPILE_STAGE = re.compile(r"^FROM\s+\S+\s+as\s+compile\s*$", re.I)
_CONTINUED = re.compile(r"\\[ \t]*$")


# --------------------------------------------------------------------------- strip-compile


def _instruction_end(lines: Sequence[str], start: int) -> int:
    """The index after the instruction that starts at ``start`` (its continuation lines included)."""
    end = start
    while end < len(lines) and _CONTINUED.search(lines[end]):
        end += 1
    return end + 1


def without_compile(text: str) -> str:
    """The Dockerfile with the compile stage's one RUN removed. ValueError when that stage has no such RUN."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if COMPILE_STAGE.match(line.strip())), None)
    if start is None:
        raise ValueError("the Dockerfile has no compile stage")
    stop = next((i for i in range(start + 1, len(lines)) if lines[i].upper().startswith("FROM ")), len(lines))
    runs = [(i, _instruction_end(lines, i)) for i in range(start + 1, stop) if lines[i].startswith("RUN ")]
    if len(runs) != 1 or "compileall" not in "".join(lines[runs[0][0]:runs[0][1]]):
        raise ValueError(f"the compile stage has {len(runs)} RUN instruction(s), not the one compileall step")
    first, end = runs[0]
    return "".join(lines[:first] + lines[end:])


def strip_compile(args: argparse.Namespace) -> int:
    path = Path(args.dockerfile)
    path.write_text(without_compile(path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"removed the compile step from {path}")
    return 0


# --------------------------------------------------------------------------- layers


def build_layers(manifest: dict, config: dict, edited_manifest: dict, edited_config: dict, budgets: dict) -> dict:
    """One build's app layer, image size and code-only update delta, with US-016's verdict on the delta."""
    rows = layers.layer_rows(manifest, config)
    edited = layers.layer_rows(edited_manifest, edited_config)
    delta = layers.update_delta(edited, (layer["digest"] for layer in manifest["layers"]))
    measured = {"name": IMAGE_NAME, "arch": ARCH, "bytes": sum(row["size"] for row in edited),
                "layers": edited, "delta": delta}
    verdict = layers.update_verdict(measured, {**budgets, "mode": "fail"})
    return {
        "image_bytes": sum(row["size"] for row in rows),
        "app_layer_bytes": rows[-1]["size"],
        "app_layer_instruction": rows[-1]["instruction"],
        "delta_bytes": delta["bytes"],
        "delta_change": delta["change"],
        "verdict": verdict.level,
        "verdict_message": verdict.message,
    }


def read_layers(args: argparse.Namespace) -> int:
    budgets = json.loads(Path(args.budgets).read_text(encoding="utf-8"))
    found = {}
    for variant in VARIANTS:
        try:
            manifest, config, _ = registry.image(f"{args.repo}:{variant}", ARCH)
            edited_manifest, edited_config, _ = registry.image(f"{args.repo}:{variant}{EDIT_SUFFIX}", ARCH)
        except registry.READ_ERRORS as error:
            print(f"::error title=Bytecode experiment::could not read the {variant} builds: {error}")
            return 1
        found[variant] = build_layers(manifest, config, edited_manifest, edited_config, budgets)
        print(f"{variant}: app layer {layers.mb(found[variant]['app_layer_bytes'])}, "
              f"code-only update {layers.mb(found[variant]['delta_bytes'])} ({found[variant]['verdict']})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(found, indent=1), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- boot


def seconds_to_200(url: str, started_ns: int, timeout_s: float,
                   probe_fn: Callable[[str], Tuple[int, str]] = smoke_boot.probe,
                   clock: Callable[[], int] = time.time_ns,
                   sleep: Callable[[float], None] = time.sleep) -> Optional[float]:
    """Seconds from ``started_ns`` to the first response from ``url`` that carried 200; None on timeout."""
    deadline = clock() + int(timeout_s * NS_PER_SECOND)
    while clock() < deadline:
        status, _ = probe_fn(url)
        if status == 200:
            return round((clock() - started_ns) / NS_PER_SECOND, 3)
        sleep(POLL_SECONDS)
    return None


def boot(args: argparse.Namespace) -> int:
    try:
        started_at = smoke_boot.container_started_at(args.container)
        started_ns = smoke_boot.parse_timestamp(started_at)
    except (subprocess.CalledProcessError, OSError, ValueError) as error:
        print(f"::error title=Bytecode experiment::{args.label}: cannot read {args.container}'s start time: {error}")
        return 1
    seconds = seconds_to_200(args.url, started_ns, args.timeout)
    record = {"label": args.label, "variant": args.variant, "started_at": started_at, "seconds": seconds}
    with open(args.out, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    if seconds is None:
        print(f"::error title=Bytecode experiment::{args.label}: {args.url} never answered 200 in {args.timeout:.0f} s")
        return 1
    print(f"{args.label}: /health 200 {seconds:.2f} s after the container started")
    return 0


# --------------------------------------------------------------------------- report


def medians(boots: Sequence[dict]) -> Dict[str, Optional[float]]:
    """Each variant's median seconds to /health 200; None when it has no boot or one never answered."""
    found: Dict[str, Optional[float]] = {}
    for variant in VARIANTS:
        seconds = [record["seconds"] for record in boots if record.get("variant") == variant]
        found[variant] = statistics.median(seconds) if seconds and None not in seconds else None
    return found


def outcome(gain: Optional[float], with_build: dict) -> str:
    """PRD-262 §9-Q7's outcome for the measured gain and the committed build's update verdict."""
    if gain is None:
        return "no outcome: a build has no boot time"
    if gain < GAIN_BAR_SECONDS:
        return (f"W5's push removes the compile step: the bytecode takes {gain:.2f} s off a new container's "
                f"boot, under the {GAIN_BAR_SECONDS:.0f} s bar")
    if with_build["verdict"] == "ok":
        return (f"the bytecode stays: a new container reaches /health 200 {gain:.2f} s sooner, and the "
                "code-only update is within US-016's bar")
    return (f"Gerard chooses in W5's push: the bytecode takes {gain:.2f} s off a new container's boot, but the "
            "code-only update misses US-016's bar, so either the compile step goes or the bar in US-016 and §8 "
            "is raised")


def summarise(found: Dict[str, dict], boots: Sequence[dict]) -> dict:
    """The record the report writes: boots, medians, gain, the bytecode's share and the outcome."""
    median = medians(boots)
    gain = (round(median["without"] - median["with"], 3)
            if median["with"] is not None and median["without"] is not None else None)
    return {
        "boots": list(boots),
        "median_seconds": median,
        "gain_seconds": gain,
        "bytecode_bytes": found["with"]["app_layer_bytes"] - found["without"]["app_layer_bytes"],
        "builds": found,
        "outcome": outcome(gain, found["with"]),
    }


def report_markdown(summary: dict) -> str:
    builds, median = summary["builds"], summary["median_seconds"]

    def seconds(variant: str) -> str:
        each = ", ".join(f"{record['seconds']:.2f}" if record["seconds"] is not None else "none"
                         for record in summary["boots"] if record.get("variant") == variant)
        return f"{median[variant]:.2f} s ({each})" if median[variant] is not None else f"none ({each})"

    rows = [
        ("New container to /health 200, median", seconds("with"), seconds("without")),
        ("Final app layer, compressed", layers.mb(builds["with"]["app_layer_bytes"]),
         layers.mb(builds["without"]["app_layer_bytes"])),
        ("Image, compressed", layers.mb(builds["with"]["image_bytes"]), layers.mb(builds["without"]["image_bytes"])),
        ("Code-only update", f"{layers.mb(builds['with']['delta_bytes'])}: {builds['with']['verdict_message']}",
         f"{layers.mb(builds['without']['delta_bytes'])}: {builds['without']['verdict_message']}"),
    ]
    gain = summary["gain_seconds"]
    lines = ["### App bytecode experiment (PRD-262 US-032, §9-Q7)", "",
             "| | With the boot modules' bytecode (committed) | Without (compile step removed) |", "|---|---|---|"]
    lines += [f"| {name} | {with_value} | {without_value} |" for name, with_value, without_value in rows]
    lines += ["", f"Gain: **{gain:.2f} s**." if gain is not None else "Gain: not measured.",
              f"The bytecode adds {layers.mb(summary['bytecode_bytes'])} to the app layer (the boot set only; "
              "US-014 already compiles the stdlib).", "", f"**Outcome:** {summary['outcome']}."]
    return "\n".join(lines) + "\n"


def report(args: argparse.Namespace) -> int:
    missing = [path for path in (args.layers, args.boots) if not Path(path).is_file()]
    if missing:
        print(f"::error title=Bytecode experiment::no report: {', '.join(missing)} not written (a step before failed)")
        return 1
    found = json.loads(Path(args.layers).read_text(encoding="utf-8"))
    boots = [json.loads(line) for line in Path(args.boots).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarise(found, boots)
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(report_markdown(summary) + "\n")
    print(f"outcome: {summary['outcome']}")
    return 0 if summary["gain_seconds"] is not None else 1


# --------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The app bytecode experiment (PRD-262 US-032, §9-Q7).")
    commands = parser.add_subparsers(dest="command", required=True)
    strip = commands.add_parser("strip-compile", help="remove the compile stage's RUN, in place")
    strip.add_argument("dockerfile")
    strip.set_defaults(run=strip_compile)
    read = commands.add_parser("layers", help="the four builds' app layers and update deltas")
    read.add_argument("--repo", required=True, help="the job registry's repository, e.g. localhost:5000/automatos-api")
    read.add_argument("--budgets", required=True)
    read.add_argument("--out", required=True)
    read.set_defaults(run=read_layers)
    timed = commands.add_parser("boot", help="time one new container to /health 200")
    timed.add_argument("--label", required=True)
    timed.add_argument("--variant", default="", help="with or without; empty for a boot that is not measured")
    timed.add_argument("--container", required=True)
    timed.add_argument("--url", required=True)
    timed.add_argument("--timeout", type=float, default=600)
    timed.add_argument("--out", required=True)
    timed.set_defaults(run=boot)
    summary = commands.add_parser("report", help="the step summary and the §9-Q7 outcome")
    summary.add_argument("--layers", required=True)
    summary.add_argument("--boots", required=True)
    summary.add_argument("--out", required=True)
    summary.add_argument("--summary")
    summary.set_defaults(run=report)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
