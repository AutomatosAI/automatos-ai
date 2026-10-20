#!/usr/bin/env python3
"""MinIO runs as its own user, and a volume root wrote still serves (PRD-262 US-041).

docker-compose.yml runs MinIO as its image's user, uid 65532, never root. A one-shot
service, minio-volume-owner, runs before it and re-owns the minio_data volume to
65532:65532 when root owns it: a volume the official minio/minio image wrote, or this
stack before US-041. The smokes run these checks on a running stack, one command per
step:

* ``server``: MinIO's container is healthy and runs as 65532 (its ``Config.User``, and
  the uid of every process ``docker top`` lists), and minio-volume-owner exited 0.
  ``--reowned yes`` needs the one-shot's log to say it re-owned the volume (an upgrade
  from a root-owned volume), ``no`` needs it silent, and ``any`` leaves it: a fresh
  volume takes the owner of an image's ``/data`` when Docker creates the containers
  that mount it (the one-shot's redis image, uid 999, then MinIO's), which need not be
  65532, so a first start may re-own it too;
* ``volume``: every path in a named volume is owned by ``--owner`` (``uid:gid``). A
  throwaway container of ``--image`` (the one-shot's image, the stack's redis, whose
  busybox has find and stat) lists them, with the volume mounted read-only and no
  network;
* ``put`` and ``get``: an object written to a bucket, or read back and compared byte for
  byte, inside a running MinIO container, through the mc client MinIO's image ships
  (``docker exec``, as that container's own user). The root credentials are the ones
  the container was started with (its ``Config.Env``), and they reach mc through the
  docker CLI's environment (``-e MC_HOST_<alias>``), never on a command line.

Each command writes its JSON record to ``--out`` and one line to ``--summary``. The
standard library only, like the other smoke tools. Exit 0 = the check holds, 1 = not.

    python3 scripts/ci/smoke_minio.py server --container automatos_minio \\
        --owner-container automatos_minio_volume_owner --reowned any --out "$OUT_DIR/minio.json"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

MINIO_UID = "65532"  # the Chainguard image's user (its config's User)
MINIO_OWNER = f"{MINIO_UID}:{MINIO_UID}"
# What minio-volume-owner logs when it re-owns the volume (docker-compose.yml).
REOWN_MARK = f"re-owned it to {MINIO_OWNER}"
REOWNED_CHOICES = ("yes", "no", "any")
TOP_ARGS = ("-eo", "pid,uid,args", "-ww")
HEALTHY, EXITED = "healthy", "exited"
VOLUME_MOUNT = "/data"
# MinIO's own defaults, which docker-compose.yml repeats (${MINIO_ROOT_USER:-minioadmin}).
DEFAULT_ROOT_USER = DEFAULT_ROOT_PASSWORD = "minioadmin"
MC_ALIAS = "smoke"
MC_HOST_ENV = f"MC_HOST_{MC_ALIAS}"
MINIO_ADDRESS = "127.0.0.1:9000"  # inside MinIO's own container
OFFENDERS_SHOWN = 5
OFFENDERS_KEPT = 20
EXCERPT_CHARS = 300
TITLE = "MinIO runs as its own user (PRD-262 US-041)"

Runner = Callable[..., subprocess.CompletedProcess]


def docker(args: Sequence[str], *, data: Optional[bytes] = None, env: Optional[Dict[str, str]] = None,
           runner: Runner = subprocess.run) -> subprocess.CompletedProcess:
    """``docker <args>``, bytes in and out. A non-zero exit is returned, not raised."""
    return runner(["docker", *args], input=data, capture_output=True, env=env, check=False)


def _text(output: Optional[bytes]) -> str:
    return (output or b"").decode("utf-8", "replace")


def _excerpt(proc: subprocess.CompletedProcess) -> str:
    return " ".join((_text(proc.stderr) or _text(proc.stdout)).split())[:EXCERPT_CHARS]


def inspect(container: str, runner: Runner = subprocess.run) -> dict:
    """``docker inspect`` of one container; an empty dict when there is no such container."""
    proc = docker(["inspect", container], runner=runner)
    found = json.loads(_text(proc.stdout) or "[]") if proc.returncode == 0 else []
    return found[0] if found else {}


# --------------------------------------------------------------------------- server


def parse_uids(text: str) -> List[dict]:
    """``docker top <c> -eo pid,uid,args`` rows: the pid, its uid and its command line. Pure."""
    rows = []
    for line in text.splitlines()[1:]:
        fields = line.split(None, 2)
        if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
            rows.append({"pid": int(fields[0]), "uid": fields[1], "args": fields[2] if len(fields) == 3 else ""})
    return rows


def _user_uid(user: str) -> str:
    """The uid part of a container's ``Config.User`` (``65532``, ``65532:65532``). Pure."""
    return user.split(":", 1)[0]


def minio_problems(inspected: dict, rows: Sequence[dict]) -> List[str]:
    """Why MinIO's container is not healthy, or runs as anyone but uid 65532. Pure."""
    if not inspected:
        return ["there is no MinIO container"]
    state = inspected.get("State") or {}
    health = (state.get("Health") or {}).get("Status")
    user = (inspected.get("Config") or {}).get("User") or ""
    failures = [] if health == HEALTHY else [f"MinIO is {health or state.get('Status') or 'not running'}, not healthy"]
    if _user_uid(user) != MINIO_UID:
        failures.append(f"MinIO's container runs as user {user or 'root (none set)'}, not {MINIO_UID}: "
                        "docker-compose.yml must leave its user to the image")
    if not rows:
        failures.append("docker top lists no MinIO process")
    others = sorted({row["uid"] for row in rows if row["uid"] != MINIO_UID})
    if others:
        failures.append(f"MinIO's processes run as uid {', '.join(others)}, not only {MINIO_UID}")
    return failures


def owner_problems(inspected: dict, log: str, reowned: str) -> List[str]:
    """Why minio-volume-owner did not finish the way this run expects. Pure."""
    if not inspected:
        return ["there is no minio-volume-owner container: MinIO started without the one-shot that re-owns its volume"]
    state = inspected.get("State") or {}
    failures = []
    if state.get("Status") != EXITED or state.get("ExitCode") != 0:
        failures.append(f"minio-volume-owner is {state.get('Status')} with exit code {state.get('ExitCode')}, "
                        "not exited 0")
    said = REOWN_MARK in log
    if reowned == "yes" and not said:
        failures.append("minio-volume-owner's log does not say it re-owned the volume, which root owned")
    if reowned == "no" and said:
        failures.append("minio-volume-owner re-owned a volume that was already MinIO's")
    return failures


def check_server(minio: dict, rows: Sequence[dict], owner: dict, log: str, reowned: str) -> dict:
    """MinIO's container and the one-shot that ran before it, against US-041. Pure."""
    return {
        "minio_user": (minio.get("Config") or {}).get("User"),
        "minio_health": ((minio.get("State") or {}).get("Health") or {}).get("Status"),
        "processes": [{"pid": row["pid"], "uid": row["uid"]} for row in rows],
        "owner_exit_code": (owner.get("State") or {}).get("ExitCode"),
        "owner_log": log.strip().splitlines()[-5:],
        "reowned": REOWN_MARK in log,
        "expected_reowned": reowned,
        "failures": minio_problems(minio, rows) + owner_problems(owner, log, reowned),
    }


def read_server(container: str, owner_container: str, runner: Runner = subprocess.run) -> Tuple[dict, list, dict, str]:
    """What ``check_server`` reads: both containers, MinIO's processes and the one-shot's log."""
    top = docker(["top", container, *TOP_ARGS], runner=runner)
    logs = docker(["logs", owner_container], runner=runner)
    rows = parse_uids(_text(top.stdout)) if top.returncode == 0 else []
    return inspect(container, runner), rows, inspect(owner_container, runner), _text(logs.stdout) + _text(logs.stderr)


# --------------------------------------------------------------------------- volume


def list_owners(volume: str, image: str, runner: Runner = subprocess.run) -> subprocess.CompletedProcess:
    """``uid:gid path`` for every path in ``volume``, from a throwaway container of ``image``."""
    return docker(["run", "--rm", "--network", "none", "--user", "0:0", "-v", f"{volume}:{VOLUME_MOUNT}:ro",
                   "--entrypoint", "find", image, VOLUME_MOUNT, "-exec", "stat", "-c", "%u:%g %n", "{}", ";"],
                  runner=runner)


def check_volume(listing: str, owner: str) -> dict:
    """Every path ``list_owners`` printed, against the owner it must have. Pure."""
    entries = [line.split(" ", 1) for line in listing.splitlines() if " " in line]
    offenders = [f"{path} ({found})" for found, path in entries if found != owner]
    failures = []
    if not entries:
        failures.append("the volume listing is empty, so its owners are unknown")
    if offenders:
        failures.append(f"{len(offenders)} of {len(entries)} paths in the volume are not owned by {owner}: "
                        f"{', '.join(offenders[:OFFENDERS_SHOWN])}")
    return {"owner": owner, "paths": len(entries), "offenders": offenders[:OFFENDERS_KEPT], "failures": failures}


# --------------------------------------------------------------------------- objects


def root_credentials(env: Sequence[str]) -> Tuple[str, str]:
    """MINIO_ROOT_USER and MINIO_ROOT_PASSWORD from a container's ``Config.Env``, or MinIO's defaults. Pure."""
    values = dict(entry.split("=", 1) for entry in env if "=" in entry)
    return (values.get("MINIO_ROOT_USER") or DEFAULT_ROOT_USER,
            values.get("MINIO_ROOT_PASSWORD") or DEFAULT_ROOT_PASSWORD)


def mc_host(user: str, password: str) -> str:
    """The ``MC_HOST_<alias>`` value for MinIO, seen from inside its own container. Pure."""
    return f"http://{urllib.parse.quote(user, safe='')}:{urllib.parse.quote(password, safe='')}@{MINIO_ADDRESS}"


def mc(container: str, host: str, args: Sequence[str], *, data: Optional[bytes] = None,
       runner: Runner = subprocess.run) -> subprocess.CompletedProcess:
    """``mc --quiet <args>`` inside ``container``; the alias's credentials come from this environment."""
    interactive = ["-i"] if data is not None else []
    return docker(["exec", *interactive, "-e", MC_HOST_ENV, container, "mc", "--quiet", *args],
                  data=data, env={**os.environ, MC_HOST_ENV: host}, runner=runner)


def _mc_failure(step: str, proc: subprocess.CompletedProcess) -> List[str]:
    return [f"`mc {step}` exited {proc.returncode}: {_excerpt(proc)}"] if proc.returncode else []


def put_object(container: str, host: str, bucket: str, key: str, text: str,
               runner: Runner = subprocess.run) -> dict:
    """Create ``bucket`` if it is missing and write ``text`` to ``key`` in it."""
    target = f"{MC_ALIAS}/{bucket}"
    failures = _mc_failure("mb", mc(container, host, ["mb", "--ignore-existing", target], runner=runner))
    if not failures:
        piped = mc(container, host, ["pipe", f"{target}/{key}"], data=text.encode("utf-8"), runner=runner)
        failures = _mc_failure("pipe", piped)
    return {"container": container, "object": f"{bucket}/{key}", "bytes": len(text.encode("utf-8")),
            "failures": failures}


def get_object(container: str, host: str, bucket: str, key: str, text: str,
               runner: Runner = subprocess.run) -> dict:
    """Read ``key`` back from ``bucket``: its bytes must be ``text``'s."""
    expected = text.encode("utf-8")
    read = mc(container, host, ["cat", f"{MC_ALIAS}/{bucket}/{key}"], runner=runner)
    failures = _mc_failure("cat", read)
    if not failures and read.stdout != expected:
        failures.append(f"{bucket}/{key} came back as {len(read.stdout or b'')} bytes that are not the "
                        f"{len(expected)} written")
    return {"container": container, "object": f"{bucket}/{key}", "bytes": len(read.stdout or b""),
            "failures": failures}


# --------------------------------------------------------------------------- commands


def _finish(args: argparse.Namespace, record: dict, line: str) -> int:
    """Write the record and the summary line; print the failures as annotations."""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failures = record["failures"]
    with open(args.summary, "a", encoding="utf-8") as summary:
        summary.write(f"**{TITLE}**: {line if not failures else '; '.join(failures)}\n\n")
    for failure in failures:
        print(f"::error title=smoke minio::{failure}")
    if not failures:
        print(f"OK: {line}")
    return 1 if failures else 0


def cmd_server(args: argparse.Namespace) -> int:
    """MinIO's container and minio-volume-owner's run."""
    record = check_server(*read_server(args.container, args.owner_container), args.reowned)
    done = "re-owned the volume" if record["reowned"] else "left the volume as it was"
    return _finish(args, record, f"MinIO is healthy and runs as uid {MINIO_UID}; minio-volume-owner exited 0 "
                                 f"and {done}")


def cmd_volume(args: argparse.Namespace) -> int:
    """Every path in the volume, against its expected owner.

    A running MinIO may remove a temporary file while find walks the volume, which makes
    find exit 1 having listed the rest, so the paths listed are judged and the exit code
    and its message are only recorded. A listing that failed outright is empty, and fails.
    """
    proc = list_owners(args.volume, args.image)
    record = {**check_volume(_text(proc.stdout), args.owner), "volume": args.volume,
              "listing_exit_code": proc.returncode, "listing_errors": _excerpt(proc) if proc.returncode else ""}
    return _finish(args, record, f"all {record['paths']} paths in {args.volume} are owned by {args.owner}")


def _object_command(args: argparse.Namespace, step: Callable[..., dict]) -> int:
    inspected = inspect(args.container)
    host = mc_host(*root_credentials((inspected.get("Config") or {}).get("Env") or []))
    record = step(args.container, host, args.bucket, args.key, args.text)
    verb = "wrote" if step is put_object else "read back"
    return _finish(args, record, f"{verb} {record['object']} ({record['bytes']} bytes) in {args.container}")


def build_parser() -> argparse.ArgumentParser:
    """One subcommand per smoke step."""
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="name", required=True)
    server = commands.add_parser("server", help="MinIO healthy as uid 65532; minio-volume-owner exited 0")
    server.add_argument("--container", required=True)
    server.add_argument("--owner-container", required=True)
    server.add_argument("--reowned", choices=REOWNED_CHOICES, required=True,
                        help="whether minio-volume-owner's log must say it re-owned the volume")
    volume = commands.add_parser("volume", help="every path in a volume has one owner")
    volume.add_argument("--volume", required=True)
    volume.add_argument("--image", required=True, help="an image with find and stat (the one-shot's)")
    volume.add_argument("--owner", required=True, help="uid:gid")
    subs = [server, volume]
    for name, help_text in (("put", "write an object through mc"), ("get", "read an object back through mc")):
        sub = commands.add_parser(name, help=help_text)
        for option in ("--container", "--bucket", "--key", "--text"):
            sub.add_argument(option, required=True)
        subs.append(sub)
    for sub in subs:
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--summary", default="/dev/stdout")
    return root


COMMANDS = {
    "server": cmd_server,
    "volume": cmd_volume,
    "put": lambda args: _object_command(args, put_object),
    "get": lambda args: _object_command(args, get_object),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.name](args)


if __name__ == "__main__":
    sys.exit(main())
