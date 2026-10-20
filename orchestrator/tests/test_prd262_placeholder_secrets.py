"""PRD-262 US-037: no install runs on the CHANGE_ME placeholder secrets.

``orchestrator/docker-secrets-guard.sh`` exits 1 when POSTGRES_PASSWORD, REDIS_PASSWORD
or API_KEY starts with CHANGE_ME, the values ``.env.example`` ships, and prints which
ones and how to rotate them: ``ALTER USER`` for Postgres, then the new value in
``.env``; a new value in ``.env`` alone for Redis and API_KEY, which are read at every
start. An unset or empty value passes: Helm and Railway set DATABASE_URL and REDIS_URL
instead of the two passwords, and a one-off ``docker run --rm <image> python -c …``
runs with no env.

Two callers print the same message: the backend's entrypoint, before it runs any
command but ``migrate`` (here), and ``make env`` on the host once an install has started
(test_prd262_make_env.py).

The guard and the entrypoint run for real under sh and bash, with the entrypoint's
database tools and server replaced by stubs that record their calls. The server is
started with AUTOMATOS_SKIP_MIGRATIONS=true, so no boot step runs before it whichever
entrypoint flag logic (PRD-262 US-029) is in place.
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
_GUARD = _ORCH / "docker-secrets-guard.sh"
_ENTRYPOINT = _ORCH / "docker-entrypoint.sh"
_DOCKERFILE = _ORCH / "Dockerfile"
_ENV_EXAMPLE = _REPO / ".env.example"
_KEYS = ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "API_KEY")
_GUARD_COPY = "COPY --chmod=0755 docker-secrets-guard.sh /usr/local/bin/docker-secrets-guard.sh"
# Before every command but `migrate` (the migration Job), from the path next to the entrypoint.
_GUARD_CALL = '[ "${1:-}" = "migrate" ] || "$(dirname "$0")/docker-secrets-guard.sh"'
_REAL = "0" * 48  # the shape make env writes: openssl rand -hex 24
_SYSTEM_PATH = f"/usr/bin{os.pathsep}/bin"

# The entrypoint's tools: each records its call. No boot step should reach them here.
_STUBS = {
    "pg_isready": 'echo "pg_isready $*" >> "$CALL_LOG"',
    "psql": 'echo "psql $*" >> "$CALL_LOG"',
    "alembic": 'echo "alembic $*" >> "$CALL_LOG"',
    "python": 'echo "python $*" >> "$CALL_LOG"',
    "uvicorn": 'echo "uvicorn $*" >> "$CALL_LOG"',
}
_SERVER = ["uvicorn", "main:app", "--port", "8000"]
_SERVED = "uvicorn main:app --port 8000"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("sh") is None,
                                reason="needs sh and bash")


def _example_values() -> dict:
    """The three secrets as .env.example ships them."""
    values = {}
    for line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key in _KEYS:
            values[key] = value
    return values


def _guard(env: dict) -> tuple[int, str]:
    proc = subprocess.run(["sh", str(_GUARD)], env={"PATH": _SYSTEM_PATH, **env}, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=30)
    return proc.returncode, proc.stdout + proc.stderr


def _entrypoint(tmp_path: Path, args: list[str], env: dict) -> tuple[int, list[str], str]:
    """The real entrypoint with stubbed tools: (exit code, calls, output)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in _STUBS.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.log"
    log.touch()
    proc = subprocess.run(["bash", str(_ENTRYPOINT), *args],
                          env={"PATH": f"{bin_dir}{os.pathsep}{_SYSTEM_PATH}", "CALL_LOG": str(log), **env},
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    return proc.returncode, log.read_text(encoding="utf-8").splitlines(), proc.stdout + proc.stderr


# --------------------------------------------------------------------------- the guard


def test_env_example_ships_the_three_secrets_as_change_me_placeholders():
    values = _example_values()
    assert sorted(values) == sorted(_KEYS)
    assert all(value.startswith("CHANGE_ME") for value in values.values()), values


@pytest.mark.parametrize("key", _KEYS)
def test_each_env_example_placeholder_is_refused_with_how_to_rotate_it(key):
    code, out = _guard({key: _example_values()[key]})
    assert code != 0, out
    assert f"Placeholder secrets: {key}. Each still starts with CHANGE_ME" in out
    assert "ALTER USER postgres PASSWORD" in out, "Postgres keeps its first password: rotate it there first"
    assert "REDIS_PASSWORD and API_KEY: a new value in .env alone" in out
    assert _example_values()[key] not in out, "the message never repeats a secret's value"


def test_the_three_unset_pass():
    assert _guard({}) == (0, "")


def test_empty_and_real_values_pass():
    assert _guard({key: "" for key in _KEYS}) == (0, "")
    assert _guard({key: _REAL for key in _KEYS}) == (0, "")


def test_only_a_value_that_starts_with_the_placeholder_is_refused():
    assert _guard({"API_KEY": f"x-CHANGE_ME-{_REAL}"}) == (0, "")
    code, out = _guard({**_example_values(), "REDIS_PASSWORD": _REAL})
    assert code != 0
    assert "Placeholder secrets: POSTGRES_PASSWORD API_KEY. Each still starts with CHANGE_ME" in out


def test_the_rotation_names_the_installs_own_postgres_user_and_database():
    code, out = _guard({"POSTGRES_PASSWORD": "CHANGE_ME", "POSTGRES_USER": "automatos", "POSTGRES_DB": "mine"})
    assert code != 0
    assert "psql -U automatos -d mine -c \"ALTER USER automatos PASSWORD '$new'\"" in out


def test_the_guard_is_executable_in_git():
    """The image copies it with --chmod, and make env and the entrypoint run it by path."""
    staged = subprocess.run(["git", "ls-files", "-s", "--", "orchestrator/docker-secrets-guard.sh"], cwd=_REPO,
                            capture_output=True, text=True, timeout=30, check=True).stdout.split()
    assert staged and staged[0] == "100755", staged


# --------------------------------------------------------------------------- the entrypoint


@pytest.mark.parametrize("key", _KEYS)
def test_the_entrypoint_starts_no_server_on_a_placeholder(tmp_path, key):
    env = {**{name: _REAL for name in _KEYS}, key: _example_values()[key], "AUTOMATOS_SKIP_MIGRATIONS": "true"}
    code, calls, out = _entrypoint(tmp_path, _SERVER, env)
    assert code != 0, out
    assert calls == [], "nothing ran: no boot step and no server"
    assert f"Placeholder secrets: {key}. Each still starts with CHANGE_ME" in out


def test_the_entrypoint_refuses_before_any_boot_step(tmp_path):
    env = {"API_KEY": "CHANGE_ME_secure_api_key", "AUTOMATOS_MIGRATE_ON_BOOT": "true",
           "POSTGRES_HOST": "db", "POSTGRES_USER": "postgres", "POSTGRES_PASSWORD": _REAL,
           "POSTGRES_DB": "orchestrator_db"}
    code, calls, out = _entrypoint(tmp_path, _SERVER, env)
    assert code != 0, out
    assert calls == [], "the guard runs before the database wait, the migrations and the seeds"


def test_the_entrypoint_starts_the_server_on_real_secrets(tmp_path):
    env = {**{name: _REAL for name in _KEYS}, "AUTOMATOS_SKIP_MIGRATIONS": "true"}
    code, calls, out = _entrypoint(tmp_path, _SERVER, env)
    assert code == 0, out
    assert calls == [_SERVED]
    assert "Placeholder" not in out


def test_a_one_off_with_no_env_runs_as_given(tmp_path):
    """test.yml's `docker run --rm parity-be python -c …` sets none of the three."""
    code, calls, out = _entrypoint(tmp_path, ["python", "-c", "import main"], {})
    assert code == 0, out
    assert calls == ["python -c import main"]


def test_the_entrypoint_calls_the_guard_next_to_itself_once():
    text = _ENTRYPOINT.read_text(encoding="utf-8")
    calls = [line.strip() for line in text.splitlines() if "docker-secrets-guard.sh" in line
             and not line.lstrip().startswith("#")]
    assert calls == [_GUARD_CALL]


# --------------------------------------------------------------------------- the image


def test_both_image_stages_carry_the_guard_next_to_the_entrypoint():
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert text.count(_GUARD_COPY) == 2, "the development and production stages both copy it"
    for stage in re.split(r"^FROM ", text, flags=re.M)[1:]:
        if _GUARD_COPY in stage:
            entrypoint = stage.index("/usr/local/bin/docker-entrypoint.sh")
            assert entrypoint < stage.index(_GUARD_COPY)


def test_the_dockerfile_declares_its_syntax_for_copy_chmod():
    """COPY --chmod needs the Dockerfile 1.x frontend; the directive is line 1 (Railway builds this file)."""
    assert _DOCKERFILE.read_text(encoding="utf-8").splitlines()[0] == "# syntax=docker/dockerfile:1"
