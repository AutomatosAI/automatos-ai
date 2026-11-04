"""WEB_CONCURRENCY=0: the local edition's uvicorn worker count, sized to the machine.

``envs/api.defaults`` sets ``WEB_CONCURRENCY=0``, "size it here". One worker is one
event loop, so a call that blocks it stalls every request, and a large server ran one
worker as a laptop did. Before a server command, ``orchestrator/docker-entrypoint.sh``
resolves 0 (or ``auto``, for an operator who types it) to one worker per CPU and per
4 GB of memory the container may use, at most 4, at least 1, exports the number and
logs ``uvicorn workers: N`` once, which the fresh-clone smoke reads.

The default is a number, not a word, because uvicorn does ``int(WEB_CONCURRENCY)``
(uvicorn 0.50.1, config.py:351): a backend image older than the resolver, run from a
newer checkout's envs/api.defaults, crashed on ``auto`` (the merge base's ``make up``
lane), and reads 0 as one process (main.py run(): workers <= 1 serves in-process). It reads the
container's own limits: the cgroup's cpu.max and memory.max (v1's cfs quota and
memory.limit_in_bytes on an older host), else nproc and /proc/meminfo, so Docker
Desktop's VM is what counts, not the Mac.

What does not change: a number passes through (the image's 4 on Railway, an operator's
own in envs/api.local), a one-off command never sees the resolution, and the
development stage's ``--reload`` runs one process.

The entrypoint runs for real under bash, with AUTOMATOS_SKIP_MIGRATIONS=true so no boot
step needs a database, a fake cgroup directory and meminfo file
(AUTOMATOS_CGROUP_DIR, AUTOMATOS_MEMINFO_FILE), and stub ``nproc`` and ``uvicorn``; the
uvicorn stub records the WEB_CONCURRENCY it was started with.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
_REPO = _ORCH.parent
_ENTRYPOINT = _ORCH / "docker-entrypoint.sh"
_DOCKERFILE = _ORCH / "Dockerfile"
_API_DEFAULTS = _REPO / "envs" / "api.defaults"

GIB = 1024 ** 3
_SERVER = ["uvicorn", "main:app", "--port", "8000"]
_LOGGED = re.compile(r"uvicorn workers: (\d+)")
_STUBS = {
    "nproc": 'echo "${FAKE_NPROC:-1}"',
    "uvicorn": 'echo "WEB_CONCURRENCY=${WEB_CONCURRENCY-<unset>} uvicorn $*" >> "$CALL_LOG"',
    "fake-app": 'echo "WEB_CONCURRENCY=${WEB_CONCURRENCY-<unset>} fake-app $*" >> "$CALL_LOG"',
}

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _machine(tmp_path: Path, files: dict[str, str], mem_total_kb: int | None) -> dict[str, str]:
    """A fake cgroup directory holding ``files`` and a meminfo with MemTotal; their env vars."""
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    for name, content in files.items():
        path = cgroup / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n", encoding="utf-8")
    meminfo = tmp_path / "meminfo"
    if mem_total_kb is not None:
        meminfo.write_text(f"MemTotal:       {mem_total_kb} kB\nMemFree:         1024 kB\n", encoding="utf-8")
    return {"AUTOMATOS_CGROUP_DIR": str(cgroup), "AUTOMATOS_MEMINFO_FILE": str(meminfo)}


def _run(tmp_path: Path, args: list[str], env: dict[str, str]) -> tuple[int, list[str], str]:
    """Run the entrypoint with stub binaries; return (exit code, calls, output)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in _STUBS.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.log"
    log.touch()
    full_env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "CALL_LOG": str(log),
        "AUTOMATOS_SKIP_MIGRATIONS": "true",
        **env,
    }
    proc = subprocess.run(
        ["bash", str(_ENTRYPOINT), *args],
        cwd=_ORCH,
        env=full_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    calls = [line.rstrip() for line in log.read_text(encoding="utf-8").splitlines()]
    return proc.returncode, calls, proc.stdout + proc.stderr


def _started_with(calls: list[str]) -> str:
    (call,) = calls
    return call.split(" ", 1)[0].removeprefix("WEB_CONCURRENCY=")


# --------------------------------------------------------------------------- the sizing


@pytest.mark.parametrize("value", ["0", "auto"])
@pytest.mark.parametrize("nproc, files, mem_total_kb, expected", [
    # A 12-core, 48 GB server: four workers, the most sizing runs.
    (12, {"cpu.max": "max 100000", "memory.max": "max"}, 48 * GIB // 1024, 4),
    # Docker Desktop's VM at 8 GB and 4 CPUs: no cgroup limit, its MemTotal counts.
    (4, {"cpu.max": "max 100000", "memory.max": "max"}, 8_039_988, 2),
    # A CPU quota below nproc (`cpus: 2`): the quota, rounded down.
    (8, {"cpu.max": "250000 100000", "memory.max": "max"}, 64 * GIB // 1024, 2),
    # A memory limit below MemTotal (`mem_limit: 8g`): the limit.
    (12, {"cpu.max": "max 100000", "memory.max": str(8 * GIB)}, 64 * GIB // 1024, 2),
    # A small machine: under 4 GB still runs one, never none.
    (2, {"cpu.max": "max 100000", "memory.max": "max"}, 3_900_000, 1),
    # Half a CPU: still one.
    (8, {"cpu.max": "50000 100000", "memory.max": "max"}, 64 * GIB // 1024, 1),
    # cgroup v1 (an older host): the cfs quota, and the unlimited v1 memory value.
    (16, {"cpu/cpu.cfs_quota_us": "300000", "cpu/cpu.cfs_period_us": "100000",
          "memory/memory.limit_in_bytes": "9223372036854771712"}, 32 * GIB // 1024, 3),
    # v1 with no quota (-1): nproc.
    (2, {"cpu/cpu.cfs_quota_us": "-1", "cpu/cpu.cfs_period_us": "100000"}, 32 * GIB // 1024, 2),
    # No cgroup files at all: nproc and MemTotal.
    (3, {}, 64 * GIB // 1024, 3),
    # Nothing readable about memory: one worker, as before sizing.
    (8, {}, None, 1),
], ids=["server-12c-48g", "docker-desktop-8g", "cpu-quota", "memory-limit", "small-machine",
        "half-a-cpu", "cgroup-v1", "cgroup-v1-no-quota", "no-cgroup", "memory-unknown"])
def test_zero_and_auto_size_the_workers_to_the_container(tmp_path, value, nproc, files, mem_total_kb, expected):
    env = {"WEB_CONCURRENCY": value, "FAKE_NPROC": str(nproc), **_machine(tmp_path, files, mem_total_kb)}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert _started_with(calls) == str(expected), "uvicorn is started with the sized number"
    assert [int(n) for n in _LOGGED.findall(out)] == [expected], "logged once, with the number it runs"
    assert f"WEB_CONCURRENCY={value}, sized here" in out


def test_the_log_line_says_what_it_read(tmp_path):
    env = {"WEB_CONCURRENCY": "0", "FAKE_NPROC": "4",
           **_machine(tmp_path, {"cpu.max": "max 100000", "memory.max": str(8 * GIB)}, 64 * GIB // 1024)}
    _, _, out = _run(tmp_path, _SERVER, env)
    assert ("uvicorn workers: 2 (WEB_CONCURRENCY=0, sized here: 4 CPUs, 8 GB of memory; "
            "one per CPU and per 4 GB, at most 4)") in out


# --------------------------------------------------------------------------- what passes through


@pytest.mark.parametrize("value", ["1", "4", "7"])
def test_a_number_passes_through_whatever_the_machine(tmp_path, value):
    """Railway runs the image's 4; an operator pins one in envs/api.local. Neither is resized."""
    env = {"WEB_CONCURRENCY": value, "FAKE_NPROC": "1", **_machine(tmp_path, {}, 2 * GIB // 1024)}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert _started_with(calls) == value
    assert f"uvicorn workers: {value} (WEB_CONCURRENCY)" in out


def test_no_value_stays_unset_and_says_nothing(tmp_path):
    code, calls, out = _run(tmp_path, _SERVER, {"FAKE_NPROC": "8", **_machine(tmp_path, {}, 64 * GIB // 1024)})
    assert code == 0, out
    assert _started_with(calls) == "<unset>"
    assert "uvicorn workers" not in out


def test_the_railway_image_cmd_keeps_its_four_workers(tmp_path):
    """The production CMD as Docker runs it, with the image's own ENV WEB_CONCURRENCY=4."""
    production = _DOCKERFILE.read_text(encoding="utf-8").split("FROM base as production", 1)[1]
    assert re.search(r"^ENV WEB_CONCURRENCY=4$", production, re.M)
    cmd = next(line for line in production.splitlines() if line.startswith("CMD ")).removeprefix("CMD ")
    env = {"WEB_CONCURRENCY": "4", "FAKE_NPROC": "1", **_machine(tmp_path, {}, 2 * GIB // 1024)}
    code, calls, out = _run(tmp_path, ["/bin/sh", "-c", cmd], env)
    assert code == 0, out
    assert _started_with(calls) == "4"


@pytest.mark.parametrize("value", ["0", "auto"])
def test_the_development_stages_reload_runs_one_process(tmp_path, value):
    """make dev: uvicorn --reload ignores the count, so 0 says one instead of a number it would not run."""
    development = _DOCKERFILE.read_text(encoding="utf-8").split("FROM base as development", 1)[1].split("\nFROM ", 1)[0]
    cmd = next(line for line in development.splitlines() if line.startswith("CMD ")).removeprefix("CMD ")
    assert "--reload" in cmd
    env = {"WEB_CONCURRENCY": value, "FAKE_NPROC": "12", **_machine(tmp_path, {}, 64 * GIB // 1024)}
    code, calls, out = _run(tmp_path, ["/bin/sh", "-c", cmd], env)
    assert code == 0, out
    assert _started_with(calls) == "1"
    assert f"uvicorn workers: 1 (WEB_CONCURRENCY={value}; --reload runs one process)" in out


def test_a_one_off_command_is_left_alone(tmp_path):
    """`docker compose run backend <cmd>`: no boot step, no resolution, no log line."""
    env = {"WEB_CONCURRENCY": "0", "FAKE_NPROC": "12", **_machine(tmp_path, {}, 64 * GIB // 1024)}
    code, calls, out = _run(tmp_path, ["fake-app", "--once"], env)
    assert code == 0, out
    assert calls == ["WEB_CONCURRENCY=0 fake-app --once"]
    assert "uvicorn workers" not in out


# --------------------------------------------------------------------------- the defaults


def test_the_local_edition_defaults_to_zero_a_number_an_older_image_can_read():
    """An older image's uvicorn int()s the value and serves 0 in one process, as before the
    resolver; the word `auto` stopped it with a ValueError."""
    values = [line.split("=", 1)[1] for line in _API_DEFAULTS.read_text(encoding="utf-8").splitlines()
              if line.startswith("WEB_CONCURRENCY=")]
    assert values == ["0"]
    assert int(values[0]) <= 1, "uvicorn's single-process branch, not its multiprocess supervisor"
