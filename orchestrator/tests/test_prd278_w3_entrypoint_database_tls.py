"""PRD-278 W3 (US-302): the entrypoint's libpq tools connect the way the app does.

``docker-entrypoint.sh`` runs ``pg_isready`` and ``psql`` against the database before it
migrates. libpq reads ``PGSSLMODE`` and ``PGSSLROOTCERT``, so the entrypoint sets them
from ``DATABASE_SSLMODE`` and ``DATABASE_SSLROOTCERT`` on those two commands only: the
Python steps (Alembic, the seed loader, the server) read the two settings through
config.py, and a process-wide PGSSLMODE would also reach the app's connections to a
user's own PostgreSQL. With neither setting, neither variable is set.

The script runs under bash with every command it calls replaced by a stub that records
the two variables it was given.
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
_ENTRYPOINT = _ORCH / "docker-entrypoint.sh"
CA_IN_POD = "/etc/automatos/database-ca/ca.crt"
_RECORD = 'echo "$(basename "$0") PGSSLMODE=${PGSSLMODE-unset} PGSSLROOTCERT=${PGSSLROOTCERT-unset}" >> "$CALL_LOG"'
_STUBS = {
    "pg_isready": _RECORD,
    "psql": _RECORD + '\ncat > /dev/null\ncase "$*" in *automatos_fresh_init_incomplete*) echo existing ;; *COUNT*) echo 3 ;; esac',
    "alembic": _RECORD,
    # Not at head (so alembic runs); every other Python step succeeds.
    "python": _RECORD + '\ncat > /dev/null\ncase "$*" in *alembic_heads*) exit 1 ;; esac',
}

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _address() -> str:
    return "postgresql" + "://" + "app" + "@" + "db.example.invalid" + ":5432/" + "app"


def _run(tmp_path: Path, env: dict) -> list:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in _STUBS.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.log"
    log.touch()
    proc = subprocess.run(
        ["bash", str(_ENTRYPOINT), "migrate"],
        cwd=_ORCH,
        env={"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(log),
             "DATABASE_URL": _address(), **env},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
    )
    calls = log.read_text(encoding="utf-8").splitlines()
    assert calls, proc.stdout + proc.stderr
    return calls


def _by_tool(calls: list) -> dict:
    found: dict = {}
    for line in calls:
        tool, rest = line.split(" ", 1)
        found.setdefault(tool, set()).add(rest)
    return found


def test_neither_setting_sets_neither_libpq_variable(tmp_path):
    tools = _by_tool(_run(tmp_path, {}))
    assert tools["pg_isready"] == {"PGSSLMODE=unset PGSSLROOTCERT=unset"}
    assert tools["psql"] == {"PGSSLMODE=unset PGSSLROOTCERT=unset"}


def test_verify_full_and_a_ca_reach_pg_isready_and_psql(tmp_path):
    tools = _by_tool(_run(tmp_path, {"DATABASE_SSLMODE": "verify-full", "DATABASE_SSLROOTCERT": CA_IN_POD}))
    expected = {f"PGSSLMODE=verify-full PGSSLROOTCERT={CA_IN_POD}"}
    assert tools["pg_isready"] == expected
    assert tools["psql"] == expected


def test_the_python_steps_are_not_given_the_libpq_variables(tmp_path):
    """Alembic and the seed loader read DATABASE_SSLMODE through config.py; PGSSLMODE in
    their environment would also reach connections to a user's own database."""
    tools = _by_tool(_run(tmp_path, {"DATABASE_SSLMODE": "verify-full", "DATABASE_SSLROOTCERT": CA_IN_POD}))
    seen = tools.get("python", set()) | tools.get("alembic", set())
    assert seen and seen == {"PGSSLMODE=unset PGSSLROOTCERT=unset"}


def test_every_libpq_tool_in_the_entrypoint_runs_through_the_tls_wrapper():
    """A new pg_isready or psql line that skips with_database_tls fails here."""
    lines = [line.strip() for line in _ENTRYPOINT.read_text(encoding="utf-8").splitlines()]
    invocations = [line for line in lines if not line.startswith("#")
                   and re.search(r"(^|[\s;(])(pg_isready|psql)\s", line)]
    assert invocations, "the entrypoint calls pg_isready and psql"
    for line in invocations:
        assert re.search(r"with_database_tls (pg_isready|psql)\b", line), line
