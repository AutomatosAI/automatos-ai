#!/usr/bin/env python3
"""What the published-image smoke measures and asserts besides the boot (PRD-262 US-003).

``smoke-images.yml`` boots the images CI publishes, the way a self-hoster runs them,
through ``scripts/ci/smoke-fresh-clone.sh`` in one of two modes:

* ``SMOKE_IMAGES=pull`` (after images.yml publishes on main, or by hand with a tag):
  ``make up`` on the clean checkout pulls the app images from GHCR into an empty Docker
  cache (PRD-262 US-036). After a push it picks the tag itself; a weekly or manual run
  pins it;
* ``SMOKE_IMAGES=prebuilt`` (the pull-request variant): the job built the three amd64
  images and tagged them with the GHCR names, so compose neither pulls nor builds them.

The fresh-clone lane's source build (``SMOKE_IMAGES`` unset) has its images checked in
mode ``build``: each is tagged ``local-*`` and labelled with the checkout's commit.

The script times the boots with ``smoke_boot.py``, as the fresh-clone smoke does. This
tool does the rest, one command per step:

* ``target``: the tag a run boots and the commit its images must carry: ``sha-<7>``, or
  ``weekly-<yyyymmdd>`` after images.yml's weekly OS rebuild (PRD-262 US-011), and the
  tag to pin for ``make up`` (``pin``; none after a push, so make up picks it);
* ``base``: for the pull-request ``make up`` job, ``sha-<7>`` of the merge base, or of
  the nearest ancestor whose three images CI published;
* ``pull``: runs the pull (``make up``), timing it, counting the bytes the runner
  received meanwhile and, with ``--log``, keeping its output and counting the lines
  where a registry refused a pull for its rate limit (Docker Hub's 429);
* ``workers``: the backend's processes. This is the first CI boot of the production
  stage, so it must run without ``--reload`` and with the expected worker count;
* ``images``: each app container runs its ``ghcr.io/automatosai/automatos-*`` image,
  which carries the commit's ``org.opencontainers.image.revision``. A pulled image's
  ``RepoDigests`` holds the index digest its tag resolves to on GHCR (a local build
  has none), and the app images exist only under that tag. No tag is ``local-*``, and
  no compose-built ``<project>-*`` image exists. A local build (mode ``build``) is
  tagged ``local-*``, never with an official tag;
* ``report``: the step-summary section, and the JSON PRD-262 US-004 takes medians from.

The standard library only, like the other smoke tools.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


smoke_boot = _sibling("smoke_boot")
layers = _sibling("image_layers")
registry = _sibling("image_registry")
image_matrix = _sibling("image_matrix")  # images.yml's picker, which also names the weekly tag

MODES = ("pull", "prebuilt", "build")
APP_SERVICES = ("backend", "frontend", "workspace-worker")
PUBLISHED_NAMES = ("api", "frontend", "workspace-worker")  # automatos-<name> on GHCR
NOT_FOUND = 404
DANGLING = "<none>"
# How a registry refuses a pull over its rate limit: Docker Hub answers 429
# `toomanyrequests: You have reached your pull rate limit` (PRD-262 §9, Docker Hub on
# the first run).
RATE_LIMITED = re.compile(r"toomanyrequests|429 Too Many Requests|pull rate limit", re.I)
RATE_LIMIT_LINES_KEPT = 5
PUBLISHED_PREFIX = "ghcr.io/automatosai/automatos-"
REVISION_LABEL = "org.opencontainers.image.revision"
LOCAL_TAG_PREFIX = "local-"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA_TAG = re.compile(r"^sha-([0-9a-f]{7})$")
IMAGE_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
SHORT_SHA = 7  # docker/metadata-action's type=sha
TAG_PREFIXES = {"workflow_run": "sha", "pull_request": "pr"}
SCHEDULE_TRIGGER = "schedule"  # images.yml's weekly OS rebuild, which publishes weekly-<yyyymmdd>
ROUTE_TABLE, DEVICE_TABLE = Path("/proc/net/route"), Path("/proc/net/dev")
DEFAULT_ROUTE = "00000000"
# -ww: ps never cuts a command line short, so `--reload` at its end is seen.
TOP_ARGS = ("-eo", "pid,ppid,args", "-ww")
WORKER_MARK = "spawn_main"  # uvicorn starts each --workers process with multiprocessing's spawn
RELOAD_FLAG = "--reload"
ARCH_NAMES = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
PULL_FILE, PROCESSES_FILE, IMAGES_FILE = "pull.json", "processes.json", "images.json"
NS_PER_SECOND = smoke_boot.NS_PER_SECOND
READ_ERRORS = registry.READ_ERRORS


# --------------------------------------------------------------------------- target


def target(event: str, sha: str = "", tag: str = "", trigger: str = "", created: str = "") -> Dict[str, str]:
    """The tag this run boots, and the commit (or a ``sha-<7>`` tag's prefix) it must carry.

    * ``workflow_run`` (images.yml published on main): ``sha-<7>`` of that commit. When
      the weekly schedule started that run (``trigger`` ``schedule``), it published
      ``weekly-<yyyymmdd>`` of its ``created`` time instead, and that is booted;
    * ``pull_request``: ``pr-<7>`` of the commit the job checked out and built;
    * ``workflow_dispatch``: the tag given. A ``sha-<7>`` tag names its commit's
      prefix; any other tag (``edge``) must carry a revision, but not a known one.
    """
    if event in TAG_PREFIXES:
        if not FULL_SHA.match(sha):
            raise ValueError(f"not a full commit sha: {sha!r}")
        if event == "workflow_run" and trigger == SCHEDULE_TRIGGER:
            return {"tag": image_matrix.weekly_tag(created), "revision": sha}
        return {"tag": f"{TAG_PREFIXES[event]}-{sha[:SHORT_SHA]}", "revision": sha}
    if event != "workflow_dispatch":
        raise ValueError(f"smoke-images does not run on {event!r}")
    if not IMAGE_TAG.match(tag):
        raise ValueError(f"not an image tag: {tag!r}")
    named = SHA_TAG.match(tag)
    return {"tag": tag, "revision": named.group(1) if named else ""}


def pin_for(event: str, trigger: str, tag: str) -> str:
    """The tag smoke-images exports as AUTOMATOS_IMAGE_TAG; empty lets ``make up`` pick.

    After a push to main (or a manual images run there), ``make up`` on that clean
    checkout must pick the commit's ``sha-<7>`` itself (PRD-262 US-036). The weekly
    rebuild's tag, a manual run's tag and a pull request's prebuilt tag are pinned:
    make up would pick the commit's own build over the weekly one.
    """
    if event == "workflow_run" and trigger != SCHEDULE_TRIGGER:
        return ""
    return tag


def _published(ref: str) -> bool:
    """Whether ``ref``'s tag is on its registry; a read error other than 404 raises."""
    try:
        registry.resolve(ref)
    except urllib.error.HTTPError as error:
        if error.code == NOT_FOUND:
            return False
        raise
    return True


def published_base(candidates: Sequence[str]) -> Optional[Dict[str, str]]:
    """The first commit of ``candidates`` whose three app images CI published as ``sha-<7>``."""
    for sha in candidates:
        if not FULL_SHA.match(sha):
            raise ValueError(f"not a full commit sha: {sha!r}")
        tag = f"sha-{sha[:SHORT_SHA]}"
        if all(_published(f"{PUBLISHED_PREFIX}{name}:{tag}") for name in PUBLISHED_NAMES):
            return {"tag": tag, "revision": sha}
    return None


# --------------------------------------------------------------------------- pull


def default_interface(route_table: str) -> str:
    """The interface that carries the default route, from ``/proc/net/route``."""
    for line in route_table.splitlines()[1:]:
        fields = line.split()
        if len(fields) > 1 and fields[1] == DEFAULT_ROUTE:
            return fields[0]
    raise LookupError("/proc/net/route has no default route")


def received_bytes(device_table: str, interface: str) -> int:
    """The bytes ``interface`` has received so far, from ``/proc/net/dev``."""
    for line in device_table.splitlines()[2:]:
        name, separator, counters = line.partition(":")
        if separator and name.strip() == interface:
            return int(counters.split()[0])
    raise LookupError(f"/proc/net/dev has no interface {interface!r}")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def tee_runner(log: Path) -> Callable[[List[str]], int]:
    """A runner that shows the command's output and keeps a copy in ``log``."""
    def run(command: List[str]) -> int:
        with open(log, "w", encoding="utf-8") as kept, subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace") as process:
            for line in process.stdout:
                sys.stdout.write(line)
                kept.write(line)
        return process.returncode
    return run


def rate_limited(text: str) -> List[str]:
    """The output lines where a registry refused a pull over its rate limit."""
    return [line.strip() for line in text.splitlines() if RATE_LIMITED.search(line)]


def run_pull(
    command: Sequence[str],
    images_before: int,
    reader: Callable[[Path], str] = _read,
    clock: Callable[[], int] = time.time_ns,
    runner: Callable[[List[str]], int] = subprocess.call,
) -> Tuple[int, dict]:
    """Run ``command`` (``docker compose … pull``): its exit code, and what it cost.

    The bytes are those the runner's default-route interface received while it ran:
    every registry's layers with their TCP/IP and TLS overhead, plus whatever else the
    runner fetched in those seconds (a few kB).
    """
    interface = default_interface(reader(ROUTE_TABLE))
    before = received_bytes(reader(DEVICE_TABLE), interface)
    started = clock()
    code = runner(list(command))
    ended = clock()
    return code, {
        "started_ns": started,
        "seconds": round((ended - started) / NS_PER_SECOND, 3),
        "received_bytes": received_bytes(reader(DEVICE_TABLE), interface) - before,
        "interface": interface,
        "images_before": images_before,
        "exit_code": code,
    }


# --------------------------------------------------------------------------- docker


def docker_text(*args: str) -> str:
    """``docker <args>``'s standard output; a failing command raises."""
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=True).stdout


def docker_json(*args: str) -> list:
    """``docker inspect``-style JSON output (an array)."""
    return json.loads(docker_text(*args))


def docker_lines(*args: str) -> List[dict]:
    """``--format '{{json .}}'`` output: one JSON object per line."""
    return [json.loads(line) for line in docker_text(*args).splitlines() if line.strip()]


# --------------------------------------------------------------------------- workers


def parse_top(text: str) -> List[dict]:
    """``docker top <c> -eo pid,ppid,args`` rows: the pid, its parent's pid and the command line."""
    rows = []
    for line in text.splitlines()[1:]:
        fields = line.split(None, 2)
        if len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit():
            rows.append({"pid": int(fields[0]), "ppid": int(fields[1]), "args": fields[2]})
    return rows


def _runs_uvicorn(args: str) -> bool:
    """``uvicorn …`` or ``python …/uvicorn …``; never the ``sh -c "… && uvicorn …"`` around it."""
    return any(os.path.basename(token) == "uvicorn" for token in args.split()[:2])


def backend_processes(rows: Sequence[dict], expected_workers: int) -> dict:
    """The uvicorn command, whether it reloads, and how many workers serve.

    uvicorn starts each ``--workers`` process with multiprocessing's spawn, so the
    workers are the children of the uvicorn process that run ``spawn_main``. With no
    such child, the uvicorn process serves by itself: one worker.
    """
    servers = [row for row in rows if _runs_uvicorn(row["args"])]
    reloading = any(RELOAD_FLAG in row["args"].split() for row in rows)
    workers = None
    failures = []
    if len(servers) == 1:
        children = [row for row in rows if row["ppid"] == servers[0]["pid"] and WORKER_MARK in row["args"]]
        workers = len(children) or 1
    else:
        failures.append(f"expected one uvicorn process, found {len(servers)}")
    if reloading:
        failures.append("the backend runs uvicorn with --reload, as the development stage does")
    if workers is not None and workers != expected_workers:
        failures.append(f"{workers} uvicorn worker(s) serve, expected {expected_workers}")
    return {
        "command": servers[0]["args"] if len(servers) == 1 else None,
        "reload": reloading,
        "workers": workers,
        "expected_workers": expected_workers,
        "processes": list(rows),
        "failures": failures,
    }


# --------------------------------------------------------------------------- images


def revision_problem(label: str, expected: str) -> Optional[str]:
    """Why ``label`` (the image's revision) is wrong for ``expected``, or None."""
    if not FULL_SHA.match(label or ""):
        return f"no {REVISION_LABEL} label naming a commit"
    if expected and not label.startswith(expected):
        return f"{REVISION_LABEL} is {label[:12]}, expected {expected[:12]}"
    return None


def _digest_problem(ref: str, repo_digests: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """The index digest ``ref`` resolves to on its registry, and why the pulled image lacks it."""
    parts = registry.split_ref(ref)
    try:
        index = registry.resolve(ref)
    except READ_ERRORS as error:
        return None, f"{ref} could not be resolved on its registry: {error}"
    pinned = f"{parts.host}/{parts.repo}@{index}"
    if pinned in repo_digests:
        return index, None
    held = ", ".join(repo_digests) if repo_digests else "nothing (a locally built image)"
    return index, f"RepoDigests holds {held}, not {pinned}, the index {ref} resolves to"


def check_service(service: str, spec: dict, mode: str, revision: str) -> dict:
    """One app container: the image it runs, its revision label and, when pulled, its digest."""
    container = docker_json("inspect", spec["container_name"])[0]
    ref = container["Config"]["Image"]
    image = docker_json("image", "inspect", container["Image"])[0]
    label = ((image.get("Config") or {}).get("Labels") or {}).get(REVISION_LABEL, "")
    repo_digests = image.get("RepoDigests") or []
    failures = []
    if ref != spec["image"]:
        failures.append(f"runs {ref}, but compose names {spec['image']}")
    if not ref.startswith(PUBLISHED_PREFIX):
        failures.append(f"{ref} is not a {PUBLISHED_PREFIX}* image")
    problem = revision_problem(label, revision)
    if problem:
        failures.append(problem)
    index = None
    if mode == "pull":
        index, problem = _digest_problem(ref, repo_digests)
        if problem:
            failures.append(problem)
    if mode == "build" and not ref.rsplit(":", 1)[-1].startswith(LOCAL_TAG_PREFIX):
        failures.append(f"{ref} was built here under an official tag: a local build is tagged {LOCAL_TAG_PREFIX}*")
    return {
        "service": service, "container": spec["container_name"], "image": ref,
        "revision": label, "repo_digests": repo_digests, "index_digest": index, "failures": failures,
    }


def stray_images(listing: Iterable[dict], project: str, local_allowed: bool = False) -> List[str]:
    """``docker image ls`` rows that show a local build: a ``local-*`` tag or compose's own name.

    ``local_allowed``: a source build (mode ``build``) is meant to carry ``local-*``.
    """
    failures = []
    for row in listing:
        repository, tag = str(row.get("Repository", "")), str(row.get("Tag", ""))
        if tag.startswith(LOCAL_TAG_PREFIX) and not local_allowed:
            failures.append(f"{repository}:{tag} carries a {LOCAL_TAG_PREFIX}* tag")
        if repository.startswith(f"{project}-"):
            failures.append(f"{repository}:{tag} was built by docker compose (project {project})")
    return failures


def other_tags(listing: Iterable[dict], refs: Iterable[str]) -> List[str]:
    """App images also present under another tag than the one the stack runs.

    A ``local-*`` tag is left to ``stray_images``, which already reports it.
    """
    expected = dict(ref.rsplit(":", 1) for ref in refs)
    failures = []
    for row in listing:
        repository, tag = str(row.get("Repository", "")), str(row.get("Tag", ""))
        if repository in expected and tag not in (expected[repository], DANGLING) \
                and not tag.startswith(LOCAL_TAG_PREFIX):
            failures.append(f"{repository}:{tag} is here too: the app images must exist only under "
                            f"{expected[repository]}")
    return failures


def app_layer_bytes(refs: Iterable[str], arch: str) -> Optional[int]:
    """The app images' compressed layers for ``arch`` on their registry, each counted once."""
    try:
        return layers.unique_bytes(registry.image(ref, arch)[0]["layers"] for ref in refs)
    except READ_ERRORS as error:
        print(f"::warning title=smoke-images::the app images' layer sizes could not be read: {error}")
        return None


def check_images(config: dict, mode: str, revision: str, arch: str) -> dict:
    """Every app container's image, and the image list, against what ``mode`` allows.

    ``config`` is ``docker compose config --format json``: it names each service's image
    and container, and the project whose name a compose build would carry.
    """
    services = [check_service(name, config["services"][name], mode, revision) for name in APP_SERVICES]
    listing = docker_lines("image", "ls", "--format", "{{json .}}")
    stray = stray_images(listing, config["name"], local_allowed=mode == "build")
    extra = other_tags(listing, [row["image"] for row in services]) if mode == "pull" else []
    return {
        "mode": mode,
        "arch": arch,
        "expected_revision": revision,
        "services": services,
        "image_list": [{key: row.get(key) for key in ("Repository", "Tag", "ID", "Size")} for row in listing],
        "app_layer_bytes": app_layer_bytes([row["image"] for row in services], arch) if mode == "pull" else None,
        "stray": stray,
        "other_tags": extra,
        "failures": [f"{row['service']}: {failure}" for row in services for failure in row["failures"]]
        + stray + extra,
    }


# --------------------------------------------------------------------------- commands


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _say(failures: Sequence[str], what: str) -> int:
    for failure in failures:
        print(f"::error title=smoke-images::{failure}")
    if not failures:
        print(f"OK: {what}")
    return 1 if failures else 0


def _outputs(values: Dict[str, str], path: Optional[str]) -> None:
    """Print ``key=value`` lines and append them to ``path`` ($GITHUB_OUTPUT)."""
    lines = "".join(f"{key}={value}\n" for key, value in values.items())
    print(lines, end="")
    if path:
        with open(path, "a", encoding="utf-8") as outputs:
            outputs.write(lines)


def cmd_target(args: argparse.Namespace) -> int:
    """The tag, the revision and the tag to pin for make up, as job outputs."""
    chosen = target(args.event, args.sha, args.tag, args.trigger, args.created)
    _outputs({**chosen, "pin": pin_for(args.event, args.trigger, chosen["tag"])}, args.outputs)
    return 0


def cmd_base(args: argparse.Namespace) -> int:
    """sha-<7> of the merge base, or of the nearest candidate CI published, as job outputs."""
    listed = [line.strip() for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = listed if listed[:1] == [args.merge_base] else [args.merge_base, *listed]
    chosen = published_base(candidates)
    if chosen is None:
        print(f"::error title=smoke-images::CI published no images for the merge base {args.merge_base[:12]} "
              f"or the {len(candidates) - 1} commits before it")
        return 1
    if chosen["revision"] != args.merge_base:
        print(f"::warning title=smoke-images::CI published no images for the merge base {args.merge_base[:12]} "
              f"(a [skip ci] merge, or a failed images run): using its ancestor {chosen['revision'][:12]}")
    _outputs(chosen, args.outputs)
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    """Run the pull command after ``--`` and record its seconds, bytes and rate limits."""
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("smoke_images.py pull: give the pull command after --")
    if args.log is None:
        code, record = run_pull(command, args.images_before)
    else:
        code, record = run_pull(command, args.images_before, runner=tee_runner(args.log))
        limited = rate_limited(args.log.read_text(encoding="utf-8", errors="replace"))
        record = {**record, "rate_limited": len(limited), "rate_limit_lines": limited[:RATE_LIMIT_LINES_KEPT]}
        for line in limited[:RATE_LIMIT_LINES_KEPT]:
            print(f"::warning title=smoke-images::a registry refused a pull over its rate limit: {line}")
    _write(args.out, record)
    print(f"Pulled in {record['seconds']:.1f}s; {layers.mb(record['received_bytes'])} received on "
          f"{record['interface']}")
    return code


def cmd_workers(args: argparse.Namespace) -> int:
    """Check the backend's processes: no --reload, ``--expect`` workers."""
    found = backend_processes(parse_top(docker_text("top", args.container, *TOP_ARGS)), args.expect)
    _write(args.out, found)
    return _say(found["failures"], f"the backend runs `{found['command']}` with {found['workers']} worker(s)")


def cmd_images(args: argparse.Namespace) -> int:
    """Check the app containers' images against ``--mode``."""
    config = json.loads(args.config.read_text(encoding="utf-8"))
    found = check_images(config, args.mode, args.expect_revision, args.arch)
    _write(args.out, found)
    return _say(found["failures"], "every app container runs its published image, and nothing was built")


def _read_json(path: Path) -> Optional[dict]:
    """A step's JSON record, or None when the run stopped before that step."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_boots(path: Optional[Path]) -> List[dict]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_report(args: argparse.Namespace) -> int:
    """Append the images section to the step summary and write its JSON, pass or fail."""
    report = _sibling("smoke_images_report")
    data = report.build(
        {"mode": args.mode, "tag": args.tag, "up_started_ns": args.up_started_ns},
        pull=_read_json(args.dir / PULL_FILE),
        processes=_read_json(args.dir / PROCESSES_FILE),
        images=_read_json(args.dir / IMAGES_FILE),
        boots=_read_boots(args.results),
    )
    if args.json is not None:
        _write(args.json, data)
    with open(args.summary, "a", encoding="utf-8") as summary:
        summary.write(report.markdown(data))
    return 0


def _optional_int(text: str) -> Optional[int]:
    return int(text) if text.strip() else None


def build_parser() -> argparse.ArgumentParser:
    """One subcommand per smoke step."""
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="name", required=True)
    pick = commands.add_parser("target", help="the tag to boot and the revision its images carry")
    pick.add_argument("--event", required=True, help="workflow_run, pull_request or workflow_dispatch")
    pick.add_argument("--sha", default="", help="the commit the images were built from")
    pick.add_argument("--tag", default="", help="workflow_dispatch: the tag to boot")
    pick.add_argument("--trigger", default="", help="workflow_run: the event that started the images run")
    pick.add_argument("--created", default="", help="workflow_run: when the images run was created (ISO 8601)")
    pick.add_argument("--outputs", help="append tag=, revision= and pin= here ($GITHUB_OUTPUT)")
    base = commands.add_parser("base", help="sha-<7> of the merge base, or its nearest published ancestor")
    base.add_argument("--merge-base", required=True, help="the pull request's merge-base commit")
    base.add_argument("--candidates", type=Path, required=True,
                      help="a file of commits, newest first, from the merge base back")
    base.add_argument("--outputs", help="append tag= and revision= here ($GITHUB_OUTPUT)")
    pull = commands.add_parser("pull", help="time a pull and count the bytes received")
    pull.add_argument("--out", type=Path, required=True)
    pull.add_argument("--images-before", type=int, default=0, help="images in the cache before it was emptied")
    pull.add_argument("--log", type=Path, help="keep the command's output here and count rate-limited pulls")
    pull.add_argument("command", nargs=argparse.REMAINDER, help="-- docker compose … pull")
    workers = commands.add_parser("workers", help="the backend runs without --reload, with --expect workers")
    workers.add_argument("--container", required=True)
    workers.add_argument("--expect", type=int, required=True)
    workers.add_argument("--out", type=Path, required=True)
    images = commands.add_parser("images", help="each app container runs its published image")
    images.add_argument("--mode", choices=MODES, required=True)
    images.add_argument("--config", type=Path, required=True, help="docker compose config --format json")
    images.add_argument("--expect-revision", default="", help="the commit (or its prefix) the images carry")
    images.add_argument("--arch", default=ARCH_NAMES.get(platform.machine(), platform.machine()))
    images.add_argument("--out", type=Path, required=True)
    report = commands.add_parser("report", help="the step-summary section and its JSON")
    report.add_argument("--mode", choices=MODES, required=True)
    report.add_argument("--tag", default="")
    report.add_argument("--dir", type=Path, required=True, help="where pull/processes/images.json are")
    report.add_argument("--results", type=Path, help="the boot records smoke_boot.py appended")
    report.add_argument("--up-started-ns", type=_optional_int, default=None, help="when `compose up` started")
    report.add_argument("--json", type=Path)
    report.add_argument("--summary", default="/dev/stdout")
    return root


COMMANDS = {"target": cmd_target, "base": cmd_base, "pull": cmd_pull, "workers": cmd_workers,
            "images": cmd_images, "report": cmd_report}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.name](args)


if __name__ == "__main__":
    sys.exit(main())
