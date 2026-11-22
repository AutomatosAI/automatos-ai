"""PRD-278 W3 (US-302): the entrypoint's libpq tools connect the way the app does.

``docker-entrypoint.sh`` runs ``pg_isready`` and ``psql`` against the database before it
migrates. libpq reads ``PGSSLMODE`` and ``PGSSLROOTCERT``, so the entrypoint sets them
from ``DATABASE_SSLMODE`` and ``DATABASE_SSLROOTCERT`` on those two commands only: the
Python steps (Alembic, the seed loader, the server) read the two settings through
config.py, and a process-wide PGSSLMODE would also reach the app's connections to a
user's own PostgreSQL. With neither setting, neither variable is set and the address is
passed as given. With the mode set, the address's own ``sslmode`` and ``sslrootcert``, and
``requiressl`` and ``ssl``, which libpq turns into ``sslmode``, are dropped from what the tools
get, since libpq prefers the connection string to the environment; a query key written with
a percent-escape, which libpq decodes before matching, stops the script instead. The settings are read the way config.py's reader reads them (trimmed, the mode
lowercased); a value the app would refuse, verify-ca or verify-full without a CA file among
them, stops the script before it waits for the database. A mode that checks a certificate
(verify-ca, verify-full) also sets ``PGGSSENCMODE=disable`` and drops the address's own
``gssencmode``: libpq prefers GSSAPI encryption to TLS whatever sslmode says, and would then
check no certificate.

The script runs under bash with every command it calls replaced by a stub that records
the three variables it was given, and the addresses pg_isready and psql were given.
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
_RECORD = ('echo "$(basename "$0") PGSSLMODE=${PGSSLMODE-unset} PGSSLROOTCERT=${PGSSLROOTCERT-unset} '
           'PGGSSENCMODE=${PGGSSENCMODE-unset}" >> "$CALL_LOG"')
UNSET = "PGSSLMODE=unset PGSSLROOTCERT=unset PGGSSENCMODE=unset"
VERIFIED = f"PGSSLMODE=verify-full PGSSLROOTCERT={CA_IN_POD} PGGSSENCMODE=disable"
# The address: pg_isready's after -d, psql's first argument.
_RECORD_ADDRESS = ('for arg in "$@"; do case "$arg" in postgres*://*) echo "$(basename "$0") $arg" >> "$ADDRESS_LOG" ;; '
                   'esac; done')
_STUBS = {
    "pg_isready": _RECORD + "\n" + _RECORD_ADDRESS,
    "psql": _RECORD + "\n" + _RECORD_ADDRESS + '\ncat > /dev/null\ncase "$*" in *automatos_fresh_init_incomplete*) echo existing ;; *COUNT*) echo 3 ;; esac',
    "alembic": _RECORD,
    # Not at head (so alembic runs); every other Python step succeeds.
    "python": _RECORD + '\ncat > /dev/null\ncase "$*" in *alembic_heads*) exit 1 ;; esac',
}

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _address(query: str = "") -> str:
    return "postgresql" + "://" + "app" + "@" + "db.example.invalid" + ":5432/" + "app" + query


def _execute(tmp_path: Path, env: dict):
    """The entrypoint's `migrate`, under stubs: (the process, the calls, the addresses)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in _STUBS.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    log, addresses = tmp_path / "calls.log", tmp_path / "addresses.log"
    log.touch()
    addresses.touch()
    proc = subprocess.run(
        ["bash", str(_ENTRYPOINT), "migrate"],
        cwd=_ORCH,
        env={"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(log),
             "ADDRESS_LOG": str(addresses), "DATABASE_URL": _address(), **env},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
    )
    return (proc, log.read_text(encoding="utf-8").splitlines(),
            addresses.read_text(encoding="utf-8").splitlines())


def _run(tmp_path: Path, env: dict) -> list:
    proc, calls, _ = _execute(tmp_path, env)
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
    assert tools["pg_isready"] == {UNSET}
    assert tools["psql"] == {UNSET}


def test_verify_full_and_a_ca_reach_pg_isready_and_psql(tmp_path):
    tools = _by_tool(_run(tmp_path, {"DATABASE_SSLMODE": "verify-full", "DATABASE_SSLROOTCERT": CA_IN_POD}))
    expected = {VERIFIED}
    assert tools["pg_isready"] == expected
    assert tools["psql"] == expected


def test_the_mode_is_read_the_way_the_app_reads_it(tmp_path):
    tools = _by_tool(_run(tmp_path, {"DATABASE_SSLMODE": " VERIFY-FULL ", "DATABASE_SSLROOTCERT": f" {CA_IN_POD}"}))
    assert tools["pg_isready"] == {VERIFIED}


def test_the_settings_win_over_an_sslmode_and_sslrootcert_in_the_address(tmp_path):
    query = "?application_name=boot&sslmode=disable&sslrootcert=" + "/tmp/other.crt"
    proc, calls, addresses = _execute(tmp_path, {"DATABASE_URL": _address(query), "DATABASE_SSLMODE": "verify-full",
                                                 "DATABASE_SSLROOTCERT": CA_IN_POD})
    assert calls, proc.stdout + proc.stderr
    assert addresses and set(addresses) == {f"pg_isready {_address('?application_name=boot')}",
                                            f"psql {_address('?application_name=boot')}"}
    assert _by_tool(calls)["pg_isready"] == {VERIFIED}


@pytest.mark.parametrize("query", ["?sslmode=disable", "?sslrootcert=/tmp/other.crt&sslmode=disable"])
def test_the_address_loses_only_its_tls_keys(tmp_path, query):
    proc, calls, addresses = _execute(tmp_path, {"DATABASE_URL": _address(query), "DATABASE_SSLMODE": "require"})
    assert calls, proc.stdout + proc.stderr
    assert {line.split(" ", 1)[1] for line in addresses} == {_address()}


@pytest.mark.parametrize("query", ["?ss%6Cmode=disable", "?application_name=boot&%73slrootcert=" + "/tmp/other.crt",
                                   "?%72equiressl=0"])
def test_a_percent_encoded_query_name_stops_the_entrypoint_while_a_mode_is_set(tmp_path, query):
    """libpq decodes a key before it matches it, so the strip below could not drop this one."""
    env = {"DATABASE_URL": _address(query), "DATABASE_SSLMODE": "verify-full", "DATABASE_SSLROOTCERT": CA_IN_POD}
    proc, calls, _ = _execute(tmp_path, env)
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    assert calls == [], "nothing is run: no pg_isready, no psql, no Python step"
    assert len([line for line in output.splitlines() if "query parameter name in DATABASE_URL" in line]) == 1
    assert "db.example.invalid" not in output and query not in output


def test_a_percent_encoded_value_is_left_alone(tmp_path):
    query = "?application_name=boot%20lane"
    proc, calls, addresses = _execute(tmp_path, {"DATABASE_URL": _address(query), "DATABASE_SSLMODE": "require"})
    assert calls, proc.stdout + proc.stderr
    assert {line.split(" ", 1)[1] for line in addresses} == {_address(query)}


def test_the_keys_libpq_turns_into_sslmode_are_dropped_too(tmp_path):
    """requiressl and ssl become sslmode in libpq, over PGSSLMODE: with a mode set they go."""
    query = "?requiressl=0&application_name=boot&ssl=true"
    proc, calls, addresses = _execute(tmp_path, {"DATABASE_URL": _address(query), "DATABASE_SSLMODE": "verify-full",
                                                 "DATABASE_SSLROOTCERT": CA_IN_POD})
    assert calls, proc.stdout + proc.stderr
    assert {line.split(" ", 1)[1] for line in addresses} == {_address("?application_name=boot")}


@pytest.mark.parametrize("mode", ["disable", "allow", "prefer", "require"])
def test_a_mode_that_checks_no_certificate_leaves_gss_encryption_alone(tmp_path, mode):
    tools = _by_tool(_run(tmp_path, {"DATABASE_SSLMODE": mode}))
    expected = {f"PGSSLMODE={mode} PGSSLROOTCERT=unset PGGSSENCMODE=unset"}
    assert tools["pg_isready"] == expected and tools["psql"] == expected


def test_verify_ca_turns_gss_encryption_off(tmp_path):
    tools = _by_tool(_run(tmp_path, {"DATABASE_SSLMODE": "verify-ca", "DATABASE_SSLROOTCERT": CA_IN_POD}))
    expected = {f"PGSSLMODE=verify-ca PGSSLROOTCERT={CA_IN_POD} PGGSSENCMODE=disable"}
    assert tools["pg_isready"] == expected and tools["psql"] == expected


def test_the_addresses_gssencmode_is_dropped_beside_a_mode_that_checks_a_certificate(tmp_path):
    """libpq prefers the connection string to PGGSSENCMODE, so the address's own key goes."""
    query = "?gssencmode=prefer&application_name=boot"
    proc, calls, addresses = _execute(tmp_path, {"DATABASE_URL": _address(query), "DATABASE_SSLMODE": "verify-full",
                                                 "DATABASE_SSLROOTCERT": CA_IN_POD})
    assert calls, proc.stdout + proc.stderr
    assert {line.split(" ", 1)[1] for line in addresses} == {_address("?application_name=boot")}


@pytest.mark.parametrize("env", [{}, {"DATABASE_SSLMODE": "prefer"}, {"DATABASE_SSLMODE": "require"}])
def test_without_a_certificate_check_the_addresses_gssencmode_is_kept(tmp_path, env):
    query = "?gssencmode=require"
    proc, calls, addresses = _execute(tmp_path, {"DATABASE_URL": _address(query), **env})
    assert calls, proc.stdout + proc.stderr
    assert {line.split(" ", 1)[1] for line in addresses} == {_address(query)}


def test_with_neither_setting_an_encoded_or_translated_key_reaches_the_tools_as_given(tmp_path):
    query = "?ss%6Cmode=disable&requiressl=0&ssl=true"
    proc, calls, addresses = _execute(tmp_path, {"DATABASE_URL": _address(query)})
    assert calls, proc.stdout + proc.stderr
    assert {line.split(" ", 1)[1] for line in addresses} == {_address(query)}


def test_neither_setting_passes_the_address_as_given(tmp_path):
    proc, calls, addresses = _execute(tmp_path, {"DATABASE_URL": _address("?sslmode=disable")})
    assert calls, proc.stdout + proc.stderr
    assert {line.split(" ", 1)[1] for line in addresses} == {_address("?sslmode=disable")}


@pytest.mark.parametrize(("env", "message"), [
    ({"DATABASE_SSLMODE": "verify"}, "DATABASE_SSLMODE must be one of"),
    ({"DATABASE_SSLMODE": "require", "DATABASE_SSLROOTCERT": CA_IN_POD}, "DATABASE_SSLROOTCERT is used only with"),
    ({"DATABASE_SSLROOTCERT": CA_IN_POD}, "DATABASE_SSLROOTCERT is used only with"),
    ({"DATABASE_SSLMODE": "verify-full"}, "DATABASE_SSLMODE verify-ca and verify-full need DATABASE_SSLROOTCERT"),
    ({"DATABASE_SSLMODE": "verify-ca"}, "DATABASE_SSLMODE verify-ca and verify-full need DATABASE_SSLROOTCERT"),
    ({"DATABASE_SSLMODE": " VERIFY-FULL ", "DATABASE_SSLROOTCERT": "   "},
     "DATABASE_SSLMODE verify-ca and verify-full need DATABASE_SSLROOTCERT"),
])
def test_a_value_the_app_would_refuse_stops_the_entrypoint_before_it_waits(tmp_path, env, message):
    proc, calls, _ = _execute(tmp_path, env)
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    assert calls == [], "nothing is run: no pg_isready, no psql, no Python step"
    refusal = [line for line in output.splitlines() if message in line]
    assert len(refusal) == 1, output
    assert CA_IN_POD not in output and "db.example.invalid" not in output


def test_the_python_steps_are_not_given_the_libpq_variables(tmp_path):
    """Alembic and the seed loader read DATABASE_SSLMODE through config.py; PGSSLMODE in
    their environment would also reach connections to a user's own database."""
    tools = _by_tool(_run(tmp_path, {"DATABASE_SSLMODE": "verify-full", "DATABASE_SSLROOTCERT": CA_IN_POD}))
    seen = tools.get("python", set()) | tools.get("alembic", set())
    assert seen and seen == {UNSET}


def test_every_libpq_tool_in_the_entrypoint_runs_through_the_tls_wrapper():
    """A new pg_isready or psql line that skips with_database_tls fails here."""
    lines = [line.strip() for line in _ENTRYPOINT.read_text(encoding="utf-8").splitlines()]
    invocations = [line for line in lines if not line.startswith("#")
                   and re.search(r"(^|[\s;(])(pg_isready|psql)\s", line)]
    assert invocations, "the entrypoint calls pg_isready and psql"
    for line in invocations:
        assert re.search(r"with_database_tls (pg_isready|psql)\b", line), line
