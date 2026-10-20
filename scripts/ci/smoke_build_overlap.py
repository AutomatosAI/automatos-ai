#!/usr/bin/env python3
"""make build starts the datastores before it builds (PRD-262 US-039).

``make build`` runs ``docker compose up -d postgres redis minio`` before
``docker compose build``, so postgres's first-run initdb and health gate pass while the
app images build instead of after them. The fresh-clone smoke runs ``make build`` and
streams Docker's container ``start`` and ``health_status`` events, and its image ``tag``
events, to one JSON-lines file (``docker events --format '{{json .}}'``). BuildKit tags
an image the moment its build has finished, so the tag event of
``ghcr.io/automatosai/automatos-api:<tag>`` is when the backend's image was built. Once
make build has exited, this checks that:

* postgres's first ``health_status: healthy`` comes before that tag event. Compose's
  log cannot show this order: it prints ``postgres Healthy`` only while a later ``up``
  waits on the dependency, which is after ``backend Built``.

It also reports, from make build's start (``--started-ns``, taken by the smoke on the
same host), when postgres turned healthy and when each app image was built, and the
image build: make build's start to the last of the three images built, which the
smoke's report shows as its image build time (report only, no bar: PRD-262 §7). That
span includes make env and the datastores' pull before the build. ``--seconds-out``
gets that number alone, whole seconds, for the shell.

Every other time is a daemon event timestamp (``timeNano``). Exit 0 = postgres was
healthy before the backend image was built; 1 = it was not, or an event is missing.

    python3 scripts/ci/smoke_build_overlap.py --events "$OUT_DIR/events.jsonl" \\
        --tag local-0123abc --started-ns "$started_ns" --out "$OUT_DIR/build-overlap.json"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


report = _sibling("smoke_report")  # the event reader and Started → Healthy spans
images = _sibling("smoke_images")  # the app services and their published names

NS_PER_SECOND = report.NS_PER_SECOND
IMAGE_TYPE, TAG_ACTION = "image", "tag"
POSTGRES, BACKEND = "postgres", "backend"
TITLE = "make build: the datastores start while the images build (PRD-262 US-039)"


def image_refs(tag: str) -> Dict[str, str]:
    """Each app service's image at ``tag``, as docker-compose.yml names it."""
    return {service: f"{images.PUBLISHED_PREFIX}{name}:{tag}"
            for service, name in zip(images.APP_SERVICES, images.PUBLISHED_NAMES)}


def event_ns(event: dict) -> int:
    """An event's daemon timestamp in nanoseconds, read as smoke_report.py reads it."""
    return int(event.get("timeNano") or int(event.get("time") or 0) * NS_PER_SECOND)


def _is_tag_event(event: dict) -> bool:
    action = event.get("Action") or event.get("status") or ""
    return event.get("Type") == IMAGE_TYPE and action == TAG_ACTION


def tagged_ns(events: Sequence[dict], refs: Dict[str, str]) -> Dict[str, Optional[int]]:
    """When each service's image was first tagged, or None: Docker named no such image."""
    found: Dict[str, Optional[int]] = {service: None for service in refs}
    by_ref = {ref: service for service, ref in refs.items()}
    for event in sorted(events, key=event_ns):
        name = ((event.get("Actor") or {}).get("Attributes") or {}).get("name")
        service = by_ref.get(name)
        if service and _is_tag_event(event) and found[service] is None:
            found = {**found, service: event_ns(event)}
    return found


def first_healthy_ns(events: Sequence[dict], service: str) -> Optional[int]:
    """The first ``health_status: healthy`` of ``service``, or None."""
    healthy = [span["healthy_ns"] for span in report.service_spans(events)
               if span["service"] == service and span["healthy_ns"] is not None]
    return min(healthy) if healthy else None


def _since(at_ns: Optional[int], started_ns: Optional[int]) -> Optional[float]:
    if at_ns is None or started_ns is None:
        return None
    return round((at_ns - started_ns) / NS_PER_SECOND, 3)


def overlap_failures(healthy_ns: Optional[int], built_ns: Optional[int], backend_ref: str) -> List[str]:
    """Why the gate failed: postgres never healthy, no backend build, or healthy after it."""
    if built_ns is None:
        return [f"Docker reported no image tag event for {backend_ref}: the backend image was not built, "
                "or the event stream (--filter type=image --filter event=tag) missed it"]
    if healthy_ns is None:
        return ["postgres never reported healthy"]
    if healthy_ns >= built_ns:
        return [f"postgres turned healthy {(healthy_ns - built_ns) / NS_PER_SECOND:.1f} s after the backend "
                "image was built: make build must start postgres, redis and minio before it builds"]
    return []


def check(events: Sequence[dict], tag: str, started_ns: Optional[int]) -> dict:
    """The whole record: when postgres turned healthy, when each image was built, the gate."""
    refs = image_refs(tag)
    built = tagged_ns(events, refs)
    healthy_ns = first_healthy_ns(events, POSTGRES)
    times = [at for at in built.values() if at is not None]
    lead = None
    if healthy_ns is not None and built[BACKEND] is not None:
        lead = round((built[BACKEND] - healthy_ns) / NS_PER_SECOND, 3)
    return {
        "tag": tag,
        "started_ns": started_ns,
        "postgres_healthy_s": _since(healthy_ns, started_ns),
        "images": {service: {"ref": refs[service], "built_s": _since(built[service], started_ns)}
                   for service in refs},
        "lead_s": lead,
        "build_s": _since(max(times), started_ns) if len(times) == len(refs) else None,
        "failures": overlap_failures(healthy_ns, built[BACKEND], refs[BACKEND]),
    }


# --------------------------------------------------------------------------- Markdown


def _seconds(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1f} s"


def render_markdown(record: dict) -> str:
    """The step summary's section: the times from make build's start, then the gate."""
    lines = [f"### {TITLE}", "", "| From the start of `make build` to | |", "|---|---|",
             f"| postgres healthy | {_seconds(record['postgres_healthy_s'])} |"]
    lines += [f"| {service} image built (`{image['ref']}`) | {_seconds(image['built_s'])} |"
              for service, image in record["images"].items()]
    if record["failures"]:
        verdict = "**FAIL**: " + "; ".join(record["failures"])
    else:
        verdict = f"postgres turned healthy {record['lead_s']:.1f} s before the backend image was built: pass"
    lines += ["", f"{verdict}.",
              f"Image build, to the last of the three: {_seconds(record['build_s'])} (report only, no bar).", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- command


def _optional_int(text: str) -> Optional[int]:
    return int(text) if text.strip() else None


def build_parser() -> argparse.ArgumentParser:
    """Where the events are, the tag make build built, and where the records go."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", type=Path, required=True, help="docker events --format '{{json .}}' output")
    parser.add_argument("--tag", required=True, help="the tag make build built the app images under")
    parser.add_argument("--started-ns", type=_optional_int, default=None, help="when make build started")
    parser.add_argument("--out", type=Path, required=True, help="where the JSON record goes")
    parser.add_argument("--seconds-out", type=Path, help="the image build's whole seconds, or nothing")
    parser.add_argument("--summary", default="/dev/stdout", help="the Markdown is appended here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    record = check(report.read_json_lines(args.events), args.tag, args.started_ns)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.seconds_out is not None:
        seconds = record["build_s"]
        args.seconds_out.write_text("" if seconds is None else f"{round(seconds)}\n", encoding="utf-8")
    with open(args.summary, "a", encoding="utf-8") as summary:
        summary.write(render_markdown(record))
    for failure in record["failures"]:
        print(f"::error title=make build overlap::{failure}")
    if not record["failures"]:
        print(f"OK: postgres turned healthy {record['lead_s']:.1f} s before the backend image was built")
    return 1 if record["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
