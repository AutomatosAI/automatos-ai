"""The backend entrypoint's run modes, exercised for real with fake binaries.

``orchestrator/docker-entrypoint.sh`` is the image ENTRYPOINT (copied in by the
Dockerfile) and has three ways to run:

* ``migrate`` — the wait-migrate-seed lifecycle, then exit 0 without starting
  the app (the Kubernetes migration Job).
* ``AUTOMATOS_MIGRATE_ON_BOOT=true`` — the lifecycle, then the CMD (compose,
  via ``envs/api.defaults``).
* neither — the CMD directly, touching nothing (the image default, so Railway,
  which builds this image and relies on the CMD's own ``alembic upgrade heads``,
  boots unchanged).

Each test runs the script under bash with ``pg_isready``, ``psql``, ``alembic``
and ``python`` replaced by stubs that record their arguments, so the assertions
are about what the script actually calls, in what order.
"""
from __future__ import annotations

import os
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

_POSTGRES_ENV = {
    "POSTGRES_HOST": "db",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": "postgres",
    "POSTGRES_PASSWORD": "pw",
    "POSTGRES_DB": "orchestrator_db",
}
_DATABASE_URL = "postgresql://u:p@db.internal:5432/app"

# psql logs its arguments and any SQL it reads on stdin (one line), and answers
# the two queries the script parses; FAKE_DB_STATE is the fresh-path state the
# script reads (empty | interrupted | existing), unset looks migrated.
_STUBS = {
    "pg_isready": 'echo "pg_isready $*" >> "$CALL_LOG"',
    "psql": (
        'echo "psql $* $(cat | tr \'\\n\' \' \')" >> "$CALL_LOG"\n'
        'case "$*" in\n'
        '  *automatos_fresh_init_incomplete*) echo "${FAKE_DB_STATE-existing}" ;;\n'
        '  *COUNT*) echo 3 ;;\n'
        "esac"
    ),
    "alembic": 'echo "alembic $*" >> "$CALL_LOG"\nexit "${FAKE_ALEMBIC_RC:-0}"',
    "python": (
        'echo "python $*" >> "$CALL_LOG"\n'
        'case "$*" in\n'
        '  *load_seed_data*) exit "${FAKE_SEED_RC:-0}" ;;\n'
        '  *init_fresh_db*) exit "${FAKE_INIT_RC:-0}" ;;\n'
        "esac"
    ),
    "fake-app": 'echo "fake-app $*" >> "$CALL_LOG"',
}

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _write_stubs(bin_dir: Path) -> None:
    bin_dir.mkdir()
    for name, body in _STUBS.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(tmp_path: Path, args: list[str], env: dict[str, str]) -> tuple[int, list[str], str]:
    """Run the entrypoint with stubbed binaries; return (exit code, calls, output)."""
    bin_dir = tmp_path / "bin"
    _write_stubs(bin_dir)
    log = tmp_path / "calls.log"
    log.touch()
    full_env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "CALL_LOG": str(log),
        **env,
    }
    proc = subprocess.run(
        ["bash", str(_ENTRYPOINT), *args],
        env=full_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    calls = [line.rstrip() for line in log.read_text(encoding="utf-8").splitlines()]
    return proc.returncode, calls, proc.stdout + proc.stderr


def _first(calls: list[str], prefix: str) -> int:
    return next(i for i, call in enumerate(calls) if call.startswith(prefix))


def test_default_runs_the_command_and_touches_nothing(tmp_path):
    """No flag, no `migrate`: exactly today's Railway boot — the CMD, nothing else."""
    code, calls, out = _run(tmp_path, ["fake-app", "--port", "8000"], dict(_POSTGRES_ENV))
    assert code == 0, out
    assert calls == ["fake-app --port 8000"]


def test_migrate_runs_the_lifecycle_in_order_and_exits(tmp_path):
    code, calls, out = _run(tmp_path, ["migrate"], dict(_POSTGRES_ENV))
    assert code == 0, out
    order = [
        _first(calls, "pg_isready"),
        _first(calls, "psql"),
        _first(calls, "alembic upgrade heads"),
        _first(calls, "python -m core.database.load_seed_data"),
    ]
    assert order == sorted(order), calls
    assert not any(c.startswith("fake-app") for c in calls)


def test_migrate_on_boot_runs_the_lifecycle_then_the_command(tmp_path):
    env = {**_POSTGRES_ENV, "AUTOMATOS_MIGRATE_ON_BOOT": "true"}
    code, calls, out = _run(tmp_path, ["fake-app"], env)
    assert code == 0, out
    assert calls[-1] == "fake-app"
    assert _first(calls, "alembic upgrade heads") < len(calls) - 1


def test_empty_database_is_initialized_before_migrating(tmp_path):
    env = {**_POSTGRES_ENV, "FAKE_DB_STATE": "empty"}
    code, calls, out = _run(tmp_path, ["migrate"], env)
    assert code == 0, out
    assert _first(calls, "python -m scripts.init_fresh_db") < _first(calls, "alembic upgrade heads")


def test_an_existing_database_never_touches_the_fresh_path(tmp_path):
    code, calls, out = _run(tmp_path, ["migrate"], dict(_POSTGRES_ENV))
    assert code == 0, out
    assert not any(c.startswith("python -m scripts.init_fresh_db") for c in calls), calls


def test_an_interrupted_fresh_build_is_resumed_before_migrating(tmp_path):
    """alembic_version exists from a build's first statement, so it alone must
    not route a half-built database to `alembic upgrade heads`: the build's
    marker sends it back to init_fresh_db, which finishes the build."""
    env = {**_POSTGRES_ENV, "FAKE_DB_STATE": "interrupted"}
    code, calls, out = _run(tmp_path, ["migrate"], env)
    assert code == 0, out
    assert "interrupted" in out
    assert _first(calls, "python -m scripts.init_fresh_db") < _first(calls, "alembic upgrade heads")


def test_a_failed_resume_fails_closed(tmp_path):
    env = {**_POSTGRES_ENV, "AUTOMATOS_MIGRATE_ON_BOOT": "true", "FAKE_DB_STATE": "interrupted", "FAKE_INIT_RC": "1"}
    code, calls, out = _run(tmp_path, ["fake-app"], env)
    assert code != 0, out
    assert not any(c.startswith("alembic upgrade heads") or c.startswith("fake-app") for c in calls), calls


def test_failed_migration_fails_closed(tmp_path):
    env = {**_POSTGRES_ENV, "AUTOMATOS_MIGRATE_ON_BOOT": "true", "FAKE_ALEMBIC_RC": "1"}
    code, calls, out = _run(tmp_path, ["fake-app"], env)
    assert code != 0, out
    assert not any(c.startswith("fake-app") for c in calls)


def test_failed_seed_fails_migrate(tmp_path):
    """A migration Job must not report success on an unseeded database."""
    code, calls, out = _run(tmp_path, ["migrate"], {**_POSTGRES_ENV, "FAKE_SEED_RC": "1"})
    assert code != 0, out
    assert "Migrate complete" not in out


def test_failed_seed_only_warns_on_boot(tmp_path):
    """Compose keeps today's lenient boot: a seed failure warns, the app starts."""
    env = {**_POSTGRES_ENV, "AUTOMATOS_MIGRATE_ON_BOOT": "true", "FAKE_SEED_RC": "1"}
    code, calls, out = _run(tmp_path, ["fake-app"], env)
    assert code == 0, out
    assert calls[-1] == "fake-app"
    assert "will continue anyway" in out


def test_database_url_is_used_when_postgres_host_is_unset(tmp_path):
    """Railway and Kubernetes secrets provide DATABASE_URL, not POSTGRES_HOST."""
    code, calls, out = _run(tmp_path, ["migrate"], {"DATABASE_URL": _DATABASE_URL})
    assert code == 0, out
    assert f"pg_isready -d {_DATABASE_URL}" in calls
    psql_calls = [c for c in calls if c.startswith("psql")]
    assert psql_calls and all(c.startswith(f"psql {_DATABASE_URL}") for c in psql_calls)


@pytest.mark.parametrize("driver", ["psycopg2", "asyncpg"])
def test_sqlalchemy_driver_suffix_is_dropped_for_libpq(tmp_path, driver):
    """pg_isready and psql reject postgresql+<driver>://, which the app accepts."""
    url = _DATABASE_URL.replace("postgresql://", f"postgresql+{driver}://")
    code, calls, out = _run(tmp_path, ["migrate"], {"DATABASE_URL": url})
    assert code == 0, out
    assert f"pg_isready -d {_DATABASE_URL}" in calls
    psql_calls = [c for c in calls if c.startswith("psql")]
    assert psql_calls and all(c.startswith(f"psql {_DATABASE_URL}") for c in psql_calls)


def test_local_edition_seeds_workspace_and_operator(tmp_path):
    env = {**_POSTGRES_ENV, "AUTH_EDITION": "local", "DEFAULT_WORKSPACE_ID": "00000000-0000-0000-0000-0000000000c1"}
    code, calls, out = _run(tmp_path, ["migrate"], env)
    assert code == 0, out
    assert any("INSERT INTO workspaces" in c for c in calls)
    assert any("INSERT INTO users" in c for c in calls)


def test_local_seed_values_are_bound_not_pasted_into_sql(tmp_path):
    """Workspace id and operator email reach SQL as psql variables, never inline."""
    hostile = "x'); DROP TABLE users; --"
    env = {**_POSTGRES_ENV, "AUTH_EDITION": "local", "DEFAULT_WORKSPACE_ID": hostile, "LOCAL_OPERATOR_EMAIL": hostile}
    code, calls, out = _run(tmp_path, ["migrate"], env)
    assert code == 0, out
    inserts = [c for c in calls if "INSERT INTO workspaces" in c or "INSERT INTO users" in c]
    assert len(inserts) == 2
    for call in inserts:
        sql = call.split("INSERT INTO", 1)[1]
        assert hostile not in sql, "a value was pasted into the SQL"
    assert any(f"workspace_id={hostile}" in c for c in inserts)
    assert any(f"operator_email={hostile}" in c for c in inserts)


def test_migrate_without_database_settings_fails_clearly(tmp_path):
    code, calls, out = _run(tmp_path, ["migrate"], {})
    assert code != 0
    assert "No database configured" in out
    assert calls == []


def test_dockerfile_copies_the_entrypoint_into_both_stages():
    """The image carries the real script; the old `exec "$@"` stubs are gone."""
    text = _DOCKERFILE.read_text(encoding="utf-8")
    copy = "COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh"
    assert text.count(copy) == 2, "development and production stages must both copy the entrypoint"
    assert "> /usr/local/bin/docker-entrypoint.sh" not in text, "a generated stub entrypoint is back"


def test_production_cmd_still_migrates_for_railway():
    """Railway runs the production CMD with the flag unset, so the CMD must migrate."""
    text = _DOCKERFILE.read_text(encoding="utf-8")
    production = text.split("FROM base as production", 1)[1]
    cmd = next(line for line in production.splitlines() if line.startswith("CMD "))
    assert "alembic upgrade heads" in cmd


def test_compose_defaults_turn_the_lifecycle_on():
    assert "AUTOMATOS_MIGRATE_ON_BOOT=true" in _API_DEFAULTS.read_text(encoding="utf-8").splitlines()
