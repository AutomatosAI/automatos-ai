"""The backend entrypoint's run modes, exercised for real with fake binaries.

``orchestrator/docker-entrypoint.sh`` is the image ENTRYPOINT (copied in by the
Dockerfile) and the only owner of migrations (PRD-262 US-029). It runs:

* ``migrate`` — the whole wait-migrate-seed lifecycle, then exit 0 without
  starting the app (the Kubernetes migration Job). It ignores
  ``AUTOMATOS_SKIP_MIGRATIONS``.
* a server command (uvicorn: the image CMD) — the boot steps first, and each
  flag has one meaning:

  * ``AUTOMATOS_SKIP_MIGRATIONS=true`` (the Helm api pods): none of them;
  * ``AUTOMATOS_MIGRATE_ON_BOOT=true`` (compose, via ``envs/api.defaults``): the
    migration step, then the seed loader and the local workspace and operator;
  * neither (Railway): the migration step only, never the seed loader, whose
    package seed would overwrite the packages curated in production.

* any other command — a one-off, run as given with no boot step: ``docker
  compose run backend <cmd>`` and test.yml's ``docker run --rm parity-be python
  -c …`` neither migrate nor seed.

The image CMDs run no alembic of their own: they exec uvicorn.

PRD-262 US-030: the migration step runs no alembic either when ``alembic_version``
already holds the code's head(s), which ``scripts/alembic_heads.py`` reads with no
import; it logs ``schema at head — alembic skipped``. A missing table still goes to
init_fresh_db first, and a database ahead of the code still goes to alembic, which
fails loudly.

PRD-262 US-031: before a server, the seed step starts the boot loader (``--boot``) only
when ``seed_fingerprint.loader`` changed: it pipes the stored value into ``python -m
core.seeds.fingerprint loader``, which exits 0 (and logs the skip line) when it
matches. The local workspace and operator are ensured either way. ``migrate`` runs the
full loader every time, without asking.

Each test runs the script under bash, from ``orchestrator/`` as the image runs it from
``/app``, with ``pg_isready``, ``psql``, ``alembic``, ``python`` and ``uvicorn``
replaced by stubs that record their arguments, so the assertions are about what the
script actually calls, in what order. The head check's ``python`` stub answers
``FAKE_AT_HEAD_RC`` (not at head by default), or, with ``REAL_PYTHON`` set, runs the
real reader against the real migration scripts.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
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
# the queries the script parses; FAKE_DB_STATE is the fresh-path state the
# script reads (empty | interrupted | existing), unset looks migrated.
# FAKE_DB_VERSION is alembic_version's rows (none by default) and FAKE_VERSION_RC
# that query's exit status. FAKE_STORED_FINGERPRINT is seed_fingerprint.loader's
# stored value (none by default) and FAKE_FINGERPRINT_READ_RC that query's status.
_STUBS = {
    "pg_isready": 'echo "pg_isready $*" >> "$CALL_LOG"',
    "psql": (
        'echo "psql $* $(cat | tr \'\\n\' \' \')" >> "$CALL_LOG"\n'
        'case "$*" in\n'
        '  *seed_fingerprint.loader*)\n'
        '    [ -z "${FAKE_STORED_FINGERPRINT:-}" ] || printf \'%s\\n\' "$FAKE_STORED_FINGERPRINT"\n'
        '    exit "${FAKE_FINGERPRINT_READ_RC:-0}" ;;\n'
        '  *automatos_fresh_init_incomplete*) echo "${FAKE_DB_STATE-existing}" ;;\n'
        '  *COUNT*) echo 3 ;;\n'
        '  *"SELECT version_num FROM alembic_version"*)\n'
        '    [ -z "${FAKE_DB_VERSION:-}" ] || printf \'%s\\n\' "$FAKE_DB_VERSION"\n'
        '    exit "${FAKE_VERSION_RC:-0}" ;;\n'
        "esac"
    ),
    "alembic": 'echo "alembic $*" >> "$CALL_LOG"\nexit "${FAKE_ALEMBIC_RC:-0}"',
    "python": (
        'echo "python $*" >> "$CALL_LOG"\n'
        'case "$*" in\n'
        '  *core.seeds.fingerprint*)\n'
        '    [ -z "${REAL_FINGERPRINT_PYTHON:-}" ] || exec "$REAL_FINGERPRINT_PYTHON" "$@"\n'
        '    echo "fingerprint stdin: $(cat | tr \'\\n\' \' \')" >> "$CALL_LOG"\n'
        '    exit "${FAKE_SEEDS_UNCHANGED_RC:-1}" ;;\n'
        '  *load_seed_data*) exit "${FAKE_SEED_RC:-0}" ;;\n'
        '  *init_fresh_db*) exit "${FAKE_INIT_RC:-0}" ;;\n'
        '  *alembic_heads*)\n'
        '    [ -z "${REAL_PYTHON:-}" ] || exec "$REAL_PYTHON" "$@"\n'
        '    echo "python stdin: $(cat | tr \'\\n\' \' \')" >> "$CALL_LOG"\n'
        '    exit "${FAKE_AT_HEAD_RC:-1}" ;;\n'
        "esac"
    ),
    "uvicorn": 'echo "uvicorn $*" >> "$CALL_LOG"',
    "fake-app": 'echo "fake-app $*" >> "$CALL_LOG"',
}
# The server the image CMD starts, and what its stub records.
_SERVER = ["uvicorn", "main:app", "--port", "8000"]
_SERVED = "uvicorn main:app --port 8000"
_SEED = "python -m core.database.load_seed_data"
_BOOT_SEED = f"{_SEED} --boot"
_SEED_CHECK = "python -m core.seeds.fingerprint loader"
_BANNER = "🚀 Starting Backend Application"
_HEAD_CHECK = "python scripts/alembic_heads.py"
_SKIPPED = "schema at head — alembic skipped"
_MIGRATING = "Running database migrations"
_LOCAL_ENV = {"AUTH_EDITION": "local", "DEFAULT_WORKSPACE_ID": "00000000-0000-0000-0000-0000000000c1"}

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
        cwd=_ORCH,  # the image's WORKDIR, /app: alembic.ini and scripts/ are relative to it
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


def _containing(calls: list[str], needle: str) -> int:
    return next(i for i, call in enumerate(calls) if needle in call)


def _alembic(calls: list[str]) -> list[str]:
    return [call for call in calls if call.startswith("alembic")]


def _cmd(stage: str) -> str:
    """A Dockerfile stage's CMD. Shell form: Docker runs it as `/bin/sh -c '<CMD>'`."""
    body = _DOCKERFILE.read_text(encoding="utf-8").split(f"FROM base as {stage}", 1)[1].split("\nFROM ", 1)[0]
    return next(line for line in body.splitlines() if line.startswith("CMD ")).removeprefix("CMD ")


# --------------------------------------------------------------------------- one-off commands


@pytest.mark.parametrize("flags", [{}, {"AUTOMATOS_MIGRATE_ON_BOOT": "true"}])
def test_a_one_off_command_runs_alone(tmp_path, flags):
    """`docker compose run backend <cmd>`: no migration and no seed, whatever the flags."""
    code, calls, out = _run(tmp_path, ["fake-app", "--port", "8000"], {**_POSTGRES_ENV, **flags})
    assert code == 0, out
    assert calls == ["fake-app --port 8000"]
    assert _BANNER not in out


def test_the_image_import_check_neither_migrates_nor_seeds(tmp_path):
    """test.yml's prod-image-parity runs `docker run --rm parity-be python -c …` with no database."""
    code, calls, out = _run(tmp_path, ["python", "-c", "import graspologic"], {"AUTOMATOS_MIGRATE_ON_BOOT": "true"})
    assert code == 0, out
    assert calls == ["python -c import graspologic"]


# --------------------------------------------------------------------------- server commands


def test_a_server_without_either_flag_migrates_and_never_seeds(tmp_path):
    """Railway: neither flag. The migration step runs; the seed loader and the local SQL never do."""
    code, calls, out = _run(tmp_path, _SERVER, {**_POSTGRES_ENV, **_LOCAL_ENV})
    assert code == 0, out
    assert _alembic(calls) == ["alembic upgrade heads"]
    assert calls[-1] == _SERVED
    assert not any(call.startswith(_SEED) for call in calls), calls
    assert not any("INSERT INTO" in call for call in calls), calls
    assert _BANNER in out


def test_migrate_on_boot_runs_the_lifecycle_then_the_server(tmp_path):
    env = {**_POSTGRES_ENV, **_LOCAL_ENV, "AUTOMATOS_MIGRATE_ON_BOOT": "true"}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert _alembic(calls) == ["alembic upgrade heads"], "one migration run per boot"
    order = [
        _first(calls, "alembic upgrade heads"),
        _first(calls, _SEED),
        _containing(calls, "INSERT INTO workspaces"),
        len(calls) - 1,
    ]
    assert order == sorted(order), calls
    assert calls[-1] == _SERVED
    assert _BANNER in out


def test_skip_migrations_runs_no_boot_step(tmp_path):
    """The Helm api pods: the migration Job owns the schema, so not even the database wait runs."""
    env = {"AUTOMATOS_SKIP_MIGRATIONS": "true", "AUTOMATOS_MIGRATE_ON_BOOT": "true"}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert calls == [_SERVED]
    assert _BANNER not in out


@pytest.mark.parametrize("command", [
    ["{bin}/uvicorn", "main:app"],
    ["python", "-m", "uvicorn", "main:app"],
    ["sh", "-c", "uvicorn main:app"],
    ["/bin/sh", "-c", "exec uvicorn main:app --port ${PORT:-8000}"],
])
def test_every_way_of_starting_uvicorn_is_a_server_command(tmp_path, command):
    args = [word.format(bin=tmp_path / "bin") if word.startswith("{bin}") else word for word in command]
    code, calls, out = _run(tmp_path, args, dict(_POSTGRES_ENV))
    assert code == 0, out
    assert _alembic(calls) == ["alembic upgrade heads"], calls
    assert _BANNER in out


@pytest.mark.parametrize("stage", ["development", "production"])
def test_the_image_cmds_migrate_once_through_the_entrypoint(tmp_path, stage):
    """Each stage's own CMD, as Docker runs it: one alembic run (the entrypoint's), then uvicorn."""
    code, calls, out = _run(tmp_path, ["/bin/sh", "-c", _cmd(stage)], dict(_POSTGRES_ENV))
    assert code == 0, out
    assert _alembic(calls) == ["alembic upgrade heads"], calls
    assert calls[-1].startswith("uvicorn main:app --host 0.0.0.0 --port 8000"), calls


def test_a_server_without_database_settings_fails_clearly(tmp_path):
    code, calls, out = _run(tmp_path, _SERVER, {})
    assert code != 0
    assert "No database configured" in out
    assert calls == []


def test_migrate_ignores_skip_migrations(tmp_path):
    """An explicit `migrate` is the migration Job: the api pods' flag does not turn it off."""
    code, calls, out = _run(tmp_path, ["migrate"], {**_POSTGRES_ENV, "AUTOMATOS_SKIP_MIGRATIONS": "true"})
    assert code == 0, out
    assert _alembic(calls) == ["alembic upgrade heads"]
    assert any(call.startswith(_SEED) for call in calls), calls


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
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code != 0, out
    assert not any(c.startswith("alembic upgrade heads") or c.startswith("uvicorn") for c in calls), calls


@pytest.mark.parametrize("flags", [{}, {"AUTOMATOS_MIGRATE_ON_BOOT": "true"}])
def test_failed_migration_fails_closed(tmp_path, flags):
    """Railway (no flag) and compose alike: uvicorn never starts on a half-built schema."""
    code, calls, out = _run(tmp_path, _SERVER, {**_POSTGRES_ENV, **flags, "FAKE_ALEMBIC_RC": "1"})
    assert code != 0, out
    assert not any(c.startswith("uvicorn") for c in calls)


# --------------------------------------------------------------------------- the head check (PRD-262 US-030)


def _heads_reader():
    spec = importlib.util.spec_from_file_location("alembic_heads_for_entrypoint", _ORCH / "scripts" / "alembic_heads.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code_head_and_parent() -> tuple[str, str]:
    reader = _heads_reader()
    graph = reader.read_graph()
    (head,) = reader.heads(graph)  # one head: test_prd209_alembic_single_head
    return head, graph[head][0]


_REAL_READER = {"REAL_PYTHON": sys.executable}


@pytest.mark.parametrize("flags", [{}, {"AUTOMATOS_MIGRATE_ON_BOOT": "true"}])
def test_a_boot_at_head_runs_no_alembic(tmp_path, flags):
    """The head check passes: no alembic, the skip line, and the rest of the boot as before."""
    env = {**_POSTGRES_ENV, **_LOCAL_ENV, **flags, "FAKE_AT_HEAD_RC": "0", "FAKE_DB_VERSION": "the_head"}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert _alembic(calls) == [], calls
    assert _SKIPPED in out and _MIGRATING not in out
    assert "python stdin: the_head" in calls, "the check reads alembic_version's rows on stdin"
    assert calls[-1] == _SERVED
    assert any(call.startswith(_SEED) for call in calls) is bool(flags), calls
    assert _BANNER in out


def test_the_check_reads_alembic_version_then_alembic_runs_when_not_at_head(tmp_path):
    code, calls, out = _run(tmp_path, _SERVER, {**_POSTGRES_ENV, "FAKE_DB_VERSION": "the_parent"})
    assert code == 0, out
    order = [_containing(calls, "SELECT version_num FROM alembic_version"), _first(calls, _HEAD_CHECK),
             _first(calls, "alembic upgrade heads"), len(calls) - 1]
    assert order == sorted(order), calls
    assert _alembic(calls) == ["alembic upgrade heads"]
    assert _MIGRATING in out and _SKIPPED not in out


def test_an_unreadable_alembic_version_goes_to_alembic(tmp_path):
    env = {**_POSTGRES_ENV, "FAKE_DB_VERSION": "the_head", "FAKE_VERSION_RC": "2", "FAKE_AT_HEAD_RC": "0"}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert not any(call.startswith(_HEAD_CHECK) for call in calls), calls
    assert _alembic(calls) == ["alembic upgrade heads"]


@pytest.mark.parametrize("flags", [{}, {"AUTOMATOS_MIGRATE_ON_BOOT": "true"}])
def test_the_real_reader_skips_alembic_at_the_codes_head(tmp_path, flags):
    head, _parent = _code_head_and_parent()
    env = {**_POSTGRES_ENV, **flags, **_REAL_READER, "FAKE_DB_VERSION": head}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert _alembic(calls) == [], calls
    assert _SKIPPED in out and f"the code's head(s): {head}" in out


def test_the_real_reader_migrates_a_database_behind_the_code(tmp_path):
    _head, parent = _code_head_and_parent()
    code, calls, out = _run(tmp_path, _SERVER, {**_POSTGRES_ENV, **_REAL_READER, "FAKE_DB_VERSION": parent})
    assert code == 0, out
    assert _alembic(calls) == ["alembic upgrade heads"], calls
    assert out.count(_MIGRATING) == 1 and _SKIPPED not in out


@pytest.mark.parametrize("stamped", ["", "prd999_from_newer_code"])
def test_an_empty_table_or_a_database_ahead_goes_to_alembic_which_fails_loudly(tmp_path, stamped):
    """A rollback leaves a version this code does not know: alembic refuses it, and uvicorn never starts."""
    env = {**_POSTGRES_ENV, **_REAL_READER, "FAKE_DB_VERSION": stamped, "FAKE_ALEMBIC_RC": "1"}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code != 0, out
    assert _alembic(calls) == ["alembic upgrade heads"], calls
    assert not any(call.startswith("uvicorn") for call in calls)
    assert "Migration failed" in out


def test_a_fresh_database_is_initialized_then_skips_alembic(tmp_path):
    """A missing alembic_version still goes to init_fresh_db, which stamps the head: no alembic run after it."""
    head, _parent = _code_head_and_parent()
    env = {**_POSTGRES_ENV, **_REAL_READER, "FAKE_DB_STATE": "empty", "FAKE_DB_VERSION": head,
           "AUTOMATOS_MIGRATE_ON_BOOT": "true"}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert _first(calls, "python -m scripts.init_fresh_db") < _first(calls, _HEAD_CHECK)
    assert _alembic(calls) == [] and _SKIPPED in out


def test_migrate_at_head_skips_alembic_and_still_seeds(tmp_path):
    head, _parent = _code_head_and_parent()
    code, calls, out = _run(tmp_path, ["migrate"], {**_POSTGRES_ENV, **_REAL_READER, "FAKE_DB_VERSION": head})
    assert code == 0, out
    assert _alembic(calls) == [] and _SKIPPED in out
    assert any(call.startswith(_SEED) for call in calls), calls
    assert "Migrate complete" in out


def test_failed_seed_fails_migrate(tmp_path):
    """A migration Job must not report success on an unseeded database."""
    code, calls, out = _run(tmp_path, ["migrate"], {**_POSTGRES_ENV, "FAKE_SEED_RC": "1"})
    assert code != 0, out
    assert "Migrate complete" not in out


def test_failed_seed_only_warns_on_boot(tmp_path):
    """Compose keeps today's lenient boot: a seed failure warns, the app starts."""
    env = {**_POSTGRES_ENV, "AUTOMATOS_MIGRATE_ON_BOOT": "true", "FAKE_SEED_RC": "1"}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert calls[-1] == _SERVED
    assert "will continue anyway" in out


# --------------------------------------------------------------------------- the seed fingerprint (PRD-262 US-031)


def test_a_boot_whose_seed_inputs_are_unchanged_starts_no_seed_loader(tmp_path):
    """The stored fingerprint matches: no loader process, and the local rows are still ensured."""
    env = {**_POSTGRES_ENV, **_LOCAL_ENV, "AUTOMATOS_MIGRATE_ON_BOOT": "true",
           "FAKE_STORED_FINGERPRINT": "the_stored_fingerprint", "FAKE_SEEDS_UNCHANGED_RC": "0"}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert not any(call.startswith(_SEED) for call in calls), calls
    assert "Loading seed data" not in out
    assert "fingerprint stdin: the_stored_fingerprint" in calls, "the check reads the stored value on stdin"
    order = [_containing(calls, "seed_fingerprint.loader"), _first(calls, _SEED_CHECK),
             _containing(calls, "INSERT INTO workspaces"), len(calls) - 1]
    assert order == sorted(order), calls
    assert calls[-1] == _SERVED


def test_a_boot_whose_seed_inputs_changed_runs_the_boot_loader(tmp_path):
    env = {**_POSTGRES_ENV, **_LOCAL_ENV, "AUTOMATOS_MIGRATE_ON_BOOT": "true"}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert [call for call in calls if call.startswith(_SEED)] == [_BOOT_SEED]
    assert _first(calls, _SEED_CHECK) < _first(calls, _BOOT_SEED) < _containing(calls, "INSERT INTO workspaces")
    assert "Loading seed data" in out


def test_an_unreadable_stored_fingerprint_runs_the_boot_loader(tmp_path):
    env = {**_POSTGRES_ENV, "AUTOMATOS_MIGRATE_ON_BOOT": "true", "FAKE_FINGERPRINT_READ_RC": "2",
           "FAKE_SEEDS_UNCHANGED_RC": "0"}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert not any(call.startswith(_SEED_CHECK) for call in calls), calls
    assert _BOOT_SEED in calls


def test_migrate_runs_the_full_loader_without_asking(tmp_path):
    env = {**_POSTGRES_ENV, "FAKE_SEEDS_UNCHANGED_RC": "0"}
    code, calls, out = _run(tmp_path, ["migrate"], env)
    assert code == 0, out
    assert not any("seed_fingerprint" in call or call.startswith(_SEED_CHECK) for call in calls), calls
    assert [call for call in calls if call.startswith(_SEED)] == [_SEED]


def test_the_entrypoint_reads_the_fingerprint_the_loader_stores():
    """The shell's query is core/seeds/fingerprint.py's, for the loader's key."""
    from core.seeds import fingerprint

    text = _ENTRYPOINT.read_text(encoding="utf-8")
    sql = re.search(r'^SEED_FINGERPRINT_SQL="(.*)"$', text, re.M).group(1)
    expected = (fingerprint.READ_SQL.replace(":category", f"'{fingerprint.SETTING_CATEGORY}'")
                .replace(":key", f"'{fingerprint.setting_key(fingerprint.LOADER)}'"))
    assert sql == expected
    assert re.search(rf"^SEED_FINGERPRINT_CHECK={re.escape(fingerprint.__name__)}$", text, re.M)


def _real_loader_fingerprint(env: dict) -> str:
    """The loader's fingerprint as the check computes it in this environment (config reads it,
    after orchestrator/.env, as the check does)."""
    code = ("from dotenv import load_dotenv; load_dotenv('.env'); "
            "from core.seeds.fingerprint import seed_fingerprint; print(seed_fingerprint('loader'))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=_ORCH, env=env, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("stored,skipped", [("current", True), ("0" * 64, False)])
def test_the_real_check_skips_the_loader_only_for_the_current_fingerprint(tmp_path, stored, skipped):
    """End to end: the shell pipes the stored value into the real check, which imports config."""
    base = {**_POSTGRES_ENV, **_LOCAL_ENV, "AUTOMATOS_MIGRATE_ON_BOOT": "true"}
    current = _real_loader_fingerprint({"PATH": f"/usr/bin{os.pathsep}/bin", **base})
    env = {**base, "REAL_FINGERPRINT_PYTHON": sys.executable,
           "FAKE_STORED_FINGERPRINT": current if stored == "current" else stored}
    code, calls, out = _run(tmp_path, _SERVER, env)
    assert code == 0, out
    assert ("seed_fingerprint.loader: seeds unchanged" in out) is skipped, out
    assert (_BOOT_SEED in calls) is not skipped, calls


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


@pytest.mark.parametrize("stage", ["development", "production"])
def test_the_cmds_exec_uvicorn_and_run_no_alembic(stage):
    """The entrypoint is the only owner of migrations; `exec` lets signals reach uvicorn (F416)."""
    cmd = _cmd(stage)
    assert cmd.startswith("exec uvicorn main:app "), cmd
    assert "alembic" not in cmd and "--workers" not in cmd, cmd


def test_the_production_cmd_takes_its_workers_from_web_concurrency():
    """Railway sets no WEB_CONCURRENCY: the image's 4 keeps its worker count; its shutdown is unchanged."""
    production = _DOCKERFILE.read_text(encoding="utf-8").split("FROM base as production", 1)[1]
    assert re.search(r"^ENV WEB_CONCURRENCY=4$", production, re.M), "the production image keeps Railway's 4 workers"
    assert "--timeout-graceful-shutdown" not in _cmd("production")
    assert "--port ${PORT:-8000}" in _cmd("production"), "Railway's PORT, expanded by the shell"


def test_the_header_documents_each_flag():
    header = _ENTRYPOINT.read_text(encoding="utf-8").split("\nset -e\n", 1)[0]
    for text in ("AUTOMATOS_SKIP_MIGRATIONS=true", "AUTOMATOS_MIGRATE_ON_BOOT=true", "Neither set (Railway)",
                 "docker-entrypoint.sh migrate", "one-off"):
        assert text in header, f"the entrypoint header does not document {text!r}"


def test_compose_defaults_turn_the_lifecycle_on():
    assert "AUTOMATOS_MIGRATE_ON_BOOT=true" in _API_DEFAULTS.read_text(encoding="utf-8").splitlines()


def test_compose_defaults_run_one_worker_that_stops_within_five_seconds():
    """The production stage under compose: one uvicorn process, and SSE streams cut after 5 s on stop."""
    lines = _API_DEFAULTS.read_text(encoding="utf-8").splitlines()
    assert "WEB_CONCURRENCY=1" in lines
    assert "UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN=5" in lines
