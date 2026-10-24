"""PRD-262 US-045: the worker boots safely and reports healthy quickly.

- entrypoint.sh re-owns the workspace volume with ``chown -h`` (a symlink itself,
  never the file it points to: an agent or a cloned repo can link to /etc/passwd) and
  only what the worker does not own yet. A chown that fails stays non-fatal under
  ``set -e``, as ``2>/dev/null || true`` made it, but is now logged;
- the Dockerfile copies the entrypoint with ``--chmod=755`` (no chmod RUN and its
  empty layer), probes every 2 s while the worker starts (``--start-interval``), and
  carries ``# syntax=docker/dockerfile:1`` on line 1, without which a build on Engine
  24's built-in frontend rejects ``--start-interval``;
- test.yml's prod-image-parity runs the real block as root in throwaway containers:
  a symlink to a root-owned file, and a read-only volume.

The entrypoint itself runs here under bash with stand-in stat, find, chown and setpriv,
so its control flow is proved in the required suite. setpriv replaced gosu (PRD-262,
row 47 of the 19 Oct size breakdown): it is in the base image's util-linux, and the
exec sets HOME to the worker's, which setpriv, unlike gosu, leaves as root's. The
smoke's 20 s bar and /metrics 200 are in test_prd262_smoke_worker_health.py.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "services" / "workspace-worker"
ENTRYPOINT = WORKER / "entrypoint.sh"
DOCKERFILE = (WORKER / "Dockerfile").read_text()
SYNTAX_LINE = "# syntax=docker/dockerfile:1"
LOG_LINE = "[entrypoint] some files could not be re-owned"
DROP = "setpriv --reuid=worker --regid=worker --init-groups --no-new-privs env HOME=/home/worker python -m main"
OWNERSHIP_FIND = (
    "find \"$WORKSPACE_VOLUME_PATH\" -mindepth 1 -path '*/projects' -prune -o ! -user worker "
    "-exec chown -h worker:worker {} + || echo \"" + LOG_LINE + "\" >&2"
)


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in re.sub(r"\\[ \t]*\n", " ", code).splitlines() if line.strip()]


INSTRUCTIONS = _instructions(DOCKERFILE)


def _only(predicate) -> str:
    found = [line for line in INSTRUCTIONS if predicate(line)]
    assert len(found) == 1, f"expected one matching instruction, found {found}"
    return found[0]


# --------------------------------------------------------------------------- entrypoint.sh


def test_the_ownership_walk_skips_owned_files_never_follows_a_link_and_logs_a_failure():
    lines = [line.strip() for line in ENTRYPOINT.read_text().splitlines()]
    assert OWNERSHIP_FIND in lines
    walks = [line for line in lines if line.startswith("find ")]
    assert walks == [OWNERSHIP_FIND], "one walk, and it no longer hides chown's errors"
    assert lines[1] == "set -e"
    assert lines[-1] == f'exec {DROP} "$@"'
    assert not [line for line in lines if "gosu" in line and not line.startswith("#")]


_STANDINS = {
    "stat": '#!/bin/sh\necho "$FAKE_UID"\n',
    "find": '#!/bin/sh\necho "find $*" >> "$CALL_LOG"\nexit "$FIND_EXIT"\n',
    "chown": '#!/bin/sh\necho "chown $*" >> "$CALL_LOG"\n',
    "setpriv": '#!/bin/sh\necho "setpriv $*"\n',
}


def _boot(tmp_path: Path, uid: str, find_exit: int) -> tuple[subprocess.CompletedProcess, list[str], Path]:
    """Run entrypoint.sh with the stand-ins first on PATH; (result, stand-in calls, volume)."""
    bin_dir, volume, log = tmp_path / "bin", tmp_path / "volume", tmp_path / "calls.log"
    bin_dir.mkdir()
    volume.mkdir()
    log.touch()
    for name, body in _STANDINS.items():
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IXUSR)
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(log),
           "FAKE_UID": uid, "FIND_EXIT": str(find_exit), "WORKSPACE_VOLUME_PATH": str(volume)}
    proc = subprocess.run(["bash", str(ENTRYPOINT), "--flag"], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=30)
    return proc, log.read_text(encoding="utf-8").splitlines(), volume


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_failed_chown_is_logged_and_the_worker_still_starts(tmp_path):
    proc, calls, volume = _boot(tmp_path, uid="0", find_exit=1)
    assert proc.returncode == 0, proc.stderr
    assert LOG_LINE in proc.stderr.splitlines()
    assert proc.stdout.splitlines()[-1] == f"{DROP} --flag"
    assert calls[0] == (f"find {volume} -mindepth 1 -path */projects -prune -o ! -user worker "
                        "-exec chown -h worker:worker {} +")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_clean_walk_logs_nothing(tmp_path):
    proc, calls, volume = _boot(tmp_path, uid="0", find_exit=0)
    assert proc.returncode == 0, proc.stderr
    assert LOG_LINE not in proc.stderr
    assert calls[-1] == f"chown worker:worker {volume}"
    assert proc.stdout.splitlines()[-1] == f"{DROP} --flag"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_volume_the_worker_already_owns_is_not_walked(tmp_path):
    proc, calls, _ = _boot(tmp_path, uid="1000", find_exit=1)
    assert proc.returncode == 0, proc.stderr
    assert calls == []
    assert proc.stdout.splitlines() == [f"{DROP} --flag"]


# --------------------------------------------------------------------------- the Dockerfile


def test_the_entrypoint_is_copied_executable_with_no_chmod_layer():
    copy = _only(lambda line: line.startswith("COPY") and "entrypoint.sh" in line)
    assert copy == "COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint.sh"
    assert not [line for line in INSTRUCTIONS if line.startswith("RUN") and "chmod +x" in line]
    assert INSTRUCTIONS.index(copy) < INSTRUCTIONS.index("COPY --chown=1000:1000 . .")
    assert INSTRUCTIONS[-2:] == ['ENTRYPOINT ["entrypoint.sh"]', "CMD []"]


def test_the_healthcheck_probes_every_2_s_while_the_worker_starts():
    healthcheck = _only(lambda line: line.startswith("HEALTHCHECK"))
    flags = dict(re.findall(r"--([a-z-]+)=(\S+)", healthcheck.split(" CMD ", 1)[0]))
    assert flags["start-interval"] == "2s"
    # Docker probes at the start interval only inside the start period.
    assert re.fullmatch(r"[1-9]\d*s", flags["start-period"]), flags
    assert "http://localhost:8081/health" in healthcheck.split(" CMD ", 1)[1]


def test_the_syntax_line_comes_first():
    """Engine 24's built-in frontend (BuildKit 0.11) rejects --start-interval; 0.12 parses it."""
    assert DOCKERFILE.splitlines()[0] == SYNTAX_LINE


# --------------------------------------------------------------------------- the parity job


def test_the_parity_job_runs_the_ownership_block_in_throwaway_containers():
    steps = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text())["jobs"]["prod-image-parity"]["steps"]
    step = next(step for step in steps if "entrypoint re-owns the volume" in step.get("name", ""))
    script = step["run"]
    for needle in (
        "docker volume create parity-wk-readonly",
        "-v parity-wk-readonly:/srv/readonly:ro --entrypoint bash parity-wk -s",
        'ln -s /tmp/outside "$vol/ws/link"',
        'check "the root-owned file the symlink points to" "$(owner /tmp/outside)" root',
        'check "the symlink" "$(owner "$vol/ws/link")" worker',
        'check "projects/" "$(owner "$vol/ws/projects")" root',
        'check "the ctime of a file the worker already owned"',
        "out=$(boot /srv/readonly); code=$?",
        "some files could not be re-owned$",
        '"started as uid 1000"',
        'check "the worker\'s HOME" "$(grep -o \'HOME=[^ ]*\' <<< "$out")" "HOME=/home/worker"',
    ):
        assert needle in script, needle
    assert script.rstrip().endswith('exit "$bad"\nSH')
    # It runs after the step that builds the image it needs.
    builds = "docker build -t parity-wk services/workspace-worker/"
    built_at = next(i for i, other in enumerate(steps) if builds in other.get("run", ""))
    assert built_at < steps.index(step)
