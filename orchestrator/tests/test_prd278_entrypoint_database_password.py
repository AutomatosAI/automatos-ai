"""PRD-278: the entrypoint's pg_isready and psql get the database password in PGPASSWORD,
and no command line carries it.

``docker-entrypoint.sh`` runs ``pg_isready`` and ``psql`` before it migrates. In the
DATABASE_URL form it reads the password once, the way libpq reads a URI: after the first
colon of the user part, or in a ``password`` query key, which wins. The tools get the
address without it, and the password, percent-decoded once, in PGPASSWORD for their one
command. The POSTGRES_* form does what it did: psql gets POSTGRES_PASSWORD in PGPASSWORD,
and pg_isready none. Nothing is exported for the whole script, so the command it finally
execs has no PGPASSWORD. An address with no password, or an empty one, reaches the tools
as given. A password libpq could not decode stops the script before anything runs.

The script runs under bash with every command it calls replaced by a stub. The stubs record
their arguments and PGPASSWORD, separated by NUL bytes so that any character survives, and
the server's stub records its whole environment.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import NamedTuple, Optional
from urllib.parse import quote

import pytest

_ORCH = Path(__file__).resolve().parents[1]
_ENTRYPOINT = _ORCH / "docker-entrypoint.sh"
CA_IN_POD = "/etc/automatos/database-ca/ca.crt"
_HOST = "db.example.invalid"
# A made-up password: a raw colon (libpq's password starts after the user part's first one)
# and characters a URI must percent-encode, a % and a backslash among them.
_NEEDLE = "sesame"
_DECODED = "open:" + _NEEDLE + "@/?&=%\\ x"
_ENCODED = "open:" + quote(_NEEDLE + "@/?&=%\\ x", safe="")
_LIBPQ_TOOLS = ("pg_isready", "psql")

_RECORD = ('out="$(mktemp "$CALL_DIR/call.XXXXXX")"\n'
           'printf \'%s\\0\' "$(basename "$0")" "${PGPASSWORD+set}" "${PGPASSWORD-}" "$@" > "$out"')
_STUBS = {
    "pg_isready": _RECORD,
    "psql": _RECORD + '\ncat > /dev/null\ncase "$*" in *automatos_fresh_init_incomplete*) echo existing ;; *COUNT*) echo 3 ;; esac',
    "alembic": _RECORD,
    # Not at head (so alembic runs); every other Python step succeeds.
    "python": _RECORD + '\ncat > /dev/null\ncase "$*" in *alembic_heads*) exit 1 ;; esac',
    "uvicorn": 'env -0 > "$CALL_DIR/server.env"',
}

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


class _Call(NamedTuple):
    tool: str
    password: Optional[str]  # PGPASSWORD, None when unset
    args: tuple


def _address(userinfo: str = "app", query: str = "", scheme: str = "postgresql") -> str:
    """An application database address, built from parts (no literal credentialed URL)."""
    return scheme + "://" + userinfo + "@" + _HOST + ":5432/" + "app" + query


def _nul_fields(path: Path) -> list:
    return path.read_bytes().decode("utf-8", "surrogateescape").split("\0")[:-1]


def _read_call(path: Path) -> _Call:
    tool, is_set, value, *args = _nul_fields(path)
    return _Call(tool, value if is_set else None, tuple(args))


def _execute(tmp_path: Path, args: list, env: dict):
    """The entrypoint under stubs: (the process, every recorded call, the server's environment)."""
    bin_dir, calls_dir = tmp_path / "bin", tmp_path / "calls"
    bin_dir.mkdir()
    calls_dir.mkdir()
    for name, body in _STUBS.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    proc = subprocess.run(
        ["bash", str(_ENTRYPOINT), *args],
        cwd=_ORCH,
        env={"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_DIR": str(calls_dir), **env},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
    )
    calls = [_read_call(path) for path in sorted(calls_dir.glob("call.*"))]
    server = calls_dir / "server.env"
    server_env = dict(item.split("=", 1) for item in _nul_fields(server)) if server.exists() else None
    return proc, calls, server_env


def _migrate(tmp_path: Path, env: dict) -> list:
    proc, calls, _ = _execute(tmp_path, ["migrate"], env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return calls


def _tools(calls: list) -> list:
    found = [call for call in calls if call.tool in _LIBPQ_TOOLS]
    assert {call.tool for call in found} == set(_LIBPQ_TOOLS), calls
    return found


def _addresses(calls: list) -> set:
    return {arg for call in calls for arg in call.args if "://" in arg}


def _assert_no_argument_holds(calls: list, *texts: str) -> None:
    for call in calls:
        for text in texts:
            assert not any(text in arg for arg in call.args), f"an argument of {call.tool} holds the password"


# ── the DATABASE_URL form ───────────────────────────────────────────────────


def test_the_tools_get_the_address_without_its_password_and_the_password_decoded_in_pgpassword(tmp_path):
    calls = _migrate(tmp_path, {"DATABASE_URL": _address("app:" + _ENCODED)})
    tools = _tools(calls)
    assert {call.password for call in tools} == {_DECODED}
    assert _addresses(tools) == {_address()}
    _assert_no_argument_holds(calls, _NEEDLE)


@pytest.mark.parametrize(("database_url", "address"), [
    (_address("app", "?application_name=boot&password=" + quote(_DECODED, safe="")),
     _address("app", "?application_name=boot")),
    (_address("app:" + "other", "?password=" + quote(_DECODED, safe="")), _address("app")),
    (_address("app:" + _ENCODED, scheme="postgresql+psycopg2"), _address("app")),
    (_address("app:" + _ENCODED, "?connect_timeout=5"), _address("app", "?connect_timeout=5")),
], ids=["query-key", "query-key-wins", "driver-suffix", "query-kept"])
def test_every_place_libpq_reads_a_password_from_leaves_the_address(tmp_path, database_url, address):
    calls = _migrate(tmp_path, {"DATABASE_URL": database_url})
    tools = _tools(calls)
    assert {call.password for call in tools} == {_DECODED}
    assert _addresses(tools) == {address}
    _assert_no_argument_holds(calls, _NEEDLE)


@pytest.mark.parametrize("decoded", ["ends in a newline\n", "back\\slash and %41 kept", "café ☕",
                                     "plus+stays"])
def test_the_password_is_percent_decoded_once(tmp_path, decoded):
    encoded = quote(decoded, safe="+")
    calls = _migrate(tmp_path, {"DATABASE_URL": _address("app:" + encoded)})
    tools = _tools(calls)
    assert {call.password for call in tools} == {decoded}
    _assert_no_argument_holds(calls, encoded, decoded)


def test_the_tls_settings_still_replace_the_addresses_own_keys(tmp_path):
    database_url = _address("app:" + _ENCODED, "?sslmode=disable&application_name=boot&requiressl=0")
    calls = _migrate(tmp_path, {"DATABASE_URL": database_url, "DATABASE_SSLMODE": "verify-full",
                                "DATABASE_SSLROOTCERT": CA_IN_POD})
    tools = _tools(calls)
    assert _addresses(tools) == {_address("app", "?application_name=boot")}
    assert {call.password for call in tools} == {_DECODED}


@pytest.mark.parametrize("database_url", [_address("app"), _address("app:"), _address("app", "?password="),
                                          _address("app", "?sslmode=disable")],
                         ids=["no-password", "empty-password", "empty-query-key", "no-password-with-query"])
def test_an_address_without_a_password_reaches_the_tools_as_given(tmp_path, database_url):
    tools = _tools(_migrate(tmp_path, {"DATABASE_URL": database_url}))
    assert _addresses(tools) == {database_url}
    assert {call.password for call in tools} == {None}


def test_a_pgpassword_the_script_was_given_reaches_the_tools_unless_the_address_has_one(tmp_path):
    outer = {"PGPASSWORD": "from-the-environment"}
    plain, own = tmp_path / "plain", tmp_path / "own"
    plain.mkdir()
    own.mkdir()
    tools = _tools(_migrate(plain, {"DATABASE_URL": _address(), **outer}))
    assert {call.password for call in tools} == {"from-the-environment"}
    tools = _tools(_migrate(own, {"DATABASE_URL": _address("app:" + _ENCODED), **outer}))
    assert {call.password for call in tools} == {_DECODED}


@pytest.mark.parametrize(("userinfo", "query"), [
    ("app:bad%zz" + _NEEDLE, ""), ("app:" + _NEEDLE + "%4", ""), ("app:" + _NEEDLE + "%00", ""),
    ("app", "?password=" + _NEEDLE + "%G1"),
], ids=["not-hex", "one-digit", "nul", "query-key"])
def test_a_password_libpq_cannot_decode_stops_the_entrypoint_before_anything_runs(tmp_path, userinfo, query):
    proc, calls, _ = _execute(tmp_path, ["migrate"], {"DATABASE_URL": _address(userinfo, query)})
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    assert calls == [], "nothing is run: no pg_isready, no psql, no Python step"
    refusal = [line for line in output.splitlines() if "DATABASE_URL holds a password libpq cannot decode" in line]
    assert len(refusal) == 1, output
    assert _NEEDLE not in output and _HOST not in output


def test_the_python_steps_are_not_given_pgpassword(tmp_path):
    calls = _migrate(tmp_path, {"DATABASE_URL": _address("app:" + _ENCODED)})
    steps = [call for call in calls if call.tool in ("python", "alembic")]
    assert steps and {call.password for call in steps} == {None}


@pytest.mark.parametrize("env", [
    {"DATABASE_URL": _address("app:" + _ENCODED)},
    {"POSTGRES_HOST": "db", "POSTGRES_PORT": "5432", "POSTGRES_USER": "postgres",
     "POSTGRES_PASSWORD": "open-" + _NEEDLE, "POSTGRES_DB": "orchestrator_db"},
], ids=["database-url", "postgres-settings"])
def test_the_server_the_entrypoint_execs_is_not_given_pgpassword(tmp_path, env):
    proc, calls, server_env = _execute(tmp_path, ["uvicorn", "main:app"], env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _tools(calls) and server_env is not None, "the server was not started"
    assert "PGPASSWORD" not in server_env
    assert not any(_DECODED in value for value in server_env.values())


# ── the POSTGRES_* form, as before ──────────────────────────────────────────


def test_the_postgres_settings_give_psql_the_password_in_pgpassword_and_pg_isready_none(tmp_path):
    env = {"POSTGRES_HOST": "db", "POSTGRES_PORT": "5432", "POSTGRES_USER": "postgres",
           "POSTGRES_PASSWORD": "open-" + _NEEDLE, "POSTGRES_DB": "orchestrator_db"}
    calls = _migrate(tmp_path, env)
    tools = _tools(calls)
    assert {call.password for call in tools if call.tool == "psql"} == {"open-" + _NEEDLE}
    assert {call.password for call in tools if call.tool == "pg_isready"} == {None}
    _assert_no_argument_holds(calls, _NEEDLE)
