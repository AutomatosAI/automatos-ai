"""The published-image smoke's step-summary section and JSON (PRD-262 US-003).

``smoke_images.py report`` reads the records the run left behind and hands them here:
the pull (``pull.json``), the backend's processes (``processes.json``), the app
images' checks (``images.json``) and the boot records ``smoke_boot.py`` appended. A
record is None when the run stopped before its step, and the section says so, so a
failed run still shows how far it got.

The section adds what the fresh-clone report (``smoke_report.py``) does not show: the
pull's seconds and bytes, the first run from its start to ``/health/ready`` 200 and
``:3000`` 200, which image each app container runs and whether it was pulled, and the
uvicorn command with its worker count. Nothing here is a bar: PRD-262 US-004 sets them.
"""
from __future__ import annotations

import importlib.util
import os
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


smoke_boot = _sibling("smoke_boot")
layers = _sibling("image_layers")

NS_PER_SECOND = smoke_boot.NS_PER_SECOND
FRESH_LABEL, FRONTEND_LABEL = "fresh", "frontend"  # smoke-fresh-clone.sh's labels
TITLE = "### Images: pull, first run, and what each container runs (PRD-262 US-003)"


def _answered_ns(record: Optional[dict], probe: str) -> Optional[int]:
    """When ``probe`` first answered 200 in ``record``, in nanoseconds since the epoch."""
    seconds = (((record or {}).get("probes") or {}).get(probe) or {}).get("seconds")
    if seconds is None:
        return None
    return smoke_boot.parse_timestamp(record["started_at"]) + round(seconds * NS_PER_SECOND)


def first_run(boots: Sequence[dict], started_ns: Optional[int], since: str) -> dict:
    """Seconds from the run's start to the backend's first ``/health/ready`` 200 and ``:3000`` 200.

    ``since`` names the start: ``pull`` (the empty-cache pull began) or ``up`` (nothing
    was pulled: ``docker compose up`` began).
    """
    fresh = next((record for record in boots if record.get("label") == FRESH_LABEL), None)
    frontend = next((record for record in boots if record.get("label") == FRONTEND_LABEL), None)

    def elapsed(at_ns: Optional[int]) -> Optional[float]:
        if at_ns is None or started_ns is None:
            return None
        return round((at_ns - started_ns) / NS_PER_SECOND, 3)

    return {
        "since": since,
        "ready_s": elapsed(_answered_ns(fresh, smoke_boot.READY)),
        "ui_s": elapsed(_answered_ns(frontend, smoke_boot.UI)),
    }


def build(run: dict, pull: Optional[dict], processes: Optional[dict], images: Optional[dict],
          boots: Sequence[dict]) -> dict:
    """Everything the section shows, as one JSON-ready dict.

    ``run`` carries the mode, the tag and when ``compose up`` started (``up_started_ns``).
    """
    since, started = ("pull", pull["started_ns"]) if pull else ("up", run.get("up_started_ns"))
    return {
        "mode": run["mode"],
        "tag": run.get("tag") or "",
        "commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "pull": pull,
        "first_run": first_run(boots, started, since),
        "backend_processes": processes,
        "images": images,
    }


# --------------------------------------------------------------------------- Markdown


def _seconds(value: Optional[float]) -> str:
    return "—" if value is None else f"**{value:.1f} s**"


def _pull_lines(data: dict) -> List[str]:
    pull, images = data["pull"], data["images"] or {}
    if data["mode"] == "prebuilt":
        return ["- Pull: none. The images were built in this job, so the pull numbers "
                "(bytes, seconds, nothing built) come from the main runs."]
    if pull is None:
        return ["- Pull: not reached."]
    layer_bytes = images.get("app_layer_bytes")
    app = f" The app images' compressed layers on GHCR: {layers.mb(layer_bytes)}, each counted once." \
        if layer_bytes is not None else ""
    return [
        f"- Pull into an empty Docker cache ({pull['images_before']} image(s) were removed first): "
        f"{_seconds(pull['seconds'])}, **{layers.mb(pull['received_bytes'])}** received on "
        f"`{pull['interface']}` (every image, TCP/IP and TLS overhead included).{app}"
    ]


def _first_run_line(data: dict) -> str:
    run = data["first_run"]
    start = "the start of the pull" if run["since"] == "pull" else "`docker compose up`"
    return (f"- First run, from {start}: `/health/ready` 200 after {_seconds(run['ready_s'])}, "
            f":3000 200 after {_seconds(run['ui_s'])}.")


def _image_rows(images: Optional[dict]) -> List[str]:
    if not images:
        return ["", "The app images were not checked: the run stopped before that step."]
    pulled = images["mode"] == "pull"
    lines = ["", "| Service | Image | Revision | Pulled: RepoDigests holds the tag's index | Result |",
             "|---|---|---|---|---|"]
    for row in images["services"]:
        digest = (f"`{row['index_digest'][:19]}…`" if row["index_digest"] else "—") if pulled \
            else "not checked: built in this job"
        result = "ok" if not row["failures"] else "**FAIL**: " + "; ".join(row["failures"])
        lines.append(f"| {row['service']} | `{row['image']}` | `{(row['revision'] or 'none')[:12]}` "
                     f"| {digest} | {result} |")
    stray = images["stray"]
    lines += ["", "- No `local-*` tag and no compose-built image: "
              + ("ok." if not stray else "**FAIL**: " + "; ".join(stray))]
    return lines


def _process_line(processes: Optional[dict]) -> str:
    if not processes:
        return "- Backend processes: not checked (the run stopped before the fresh boot answered)."
    verdict = "ok" if not processes["failures"] else "**FAIL**: " + "; ".join(processes["failures"])
    return (f"- Backend (production stage): `{processes['command'] or 'no uvicorn process'}`; "
            f"`--reload`: {'yes' if processes['reload'] else 'no'}; workers: {processes['workers']} "
            f"(expected {processes['expected_workers']}). {verdict}.")


def markdown(data: dict) -> str:
    """The section, appended after smoke_report.py's boot tables."""
    head = [TITLE, "", f"Tag: `{data['tag'] or 'not set'}` ({data['mode']})."
            + (f" Images must carry revision `{data['images']['expected_revision'][:12]}`."
               if data["images"] and data["images"]["expected_revision"] else ""), ""]
    body = _pull_lines(data) + [_first_run_line(data), _process_line(data["backend_processes"])]
    return "\n".join(head + body + _image_rows(data["images"])) + "\n\n"
