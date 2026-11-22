"""PRD-278 W3 (US-301, US-302): one function says how every connection to the app's own
database is secured, and each driver gets it in the form it takes.

``core.database.tls.database_tls()`` reads ``DATABASE_SSLMODE`` and ``DATABASE_SSLROOTCERT``:

* with neither set, no connection path gains a keyword, and the main engine keeps its
  production default (``sslmode=require`` added when the address names none);
* with ``verify-full`` and a CA file, libpq gets ``sslmode`` and ``sslrootcert`` (psycopg2,
  and SQLAlchemy through ``connect_args``), with ``gssencmode=disable`` as for every mode that
  checks a certificate, so libpq cannot use GSSAPI encryption in place of that check; asyncpg gets an ``ssl.SSLContext`` that
  requires the certificate, trusts that CA and checks the host name;
* a mode libpq doesn't know, a CA file without a mode that checks it, verify-ca or
  verify-full without a CA file, or a CA file that is missing or holds no certificate, stops
  start-up with a message that names the settings and never the file's path;
* a CA file replaced at the same path is loaded again for the next asyncpg connection.

The CA files are made at run time (``tests/helpers_prd278_w3_ca.py``).

No database: the drivers' ``connect`` and SQLAlchemy's ``create_engine`` are recorded.
"""
from __future__ import annotations

import ast
import asyncio
import os
import ssl
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from core.database import tls
from tests.helpers_prd278_w3_ca import write_ca_bundle

ORCH = Path(__file__).resolve().parents[1]
# A path that is never read: the settings are refused before the file is looked at.
CA_IN_POD = "/etc/automatos/database-ca/ca.crt"


def _address(query: str = "") -> str:
    """An application database address, built from parts (no literal credentialed URL)."""
    return "postgresql" + "://" + "app" + "@" + "db.example.invalid" + ":5432/" + "app" + query


@pytest.fixture
def settings(monkeypatch):
    """Set the two settings for one test; both start unset."""
    monkeypatch.setattr(config, "DATABASE_SSLMODE", None)
    monkeypatch.setattr(config, "DATABASE_SSLROOTCERT", None)

    def apply(sslmode=None, sslrootcert=None):
        monkeypatch.setattr(config, "DATABASE_SSLMODE", sslmode)
        monkeypatch.setattr(config, "DATABASE_SSLROOTCERT", sslrootcert)

    return apply


@pytest.fixture
def ca_file(tmp_path) -> str:
    """A CA certificate made for this test, as a PEM file's path."""
    return str(write_ca_bundle(tmp_path / "ca.crt"))


# ── neither setting: every path as before ────────────────────────────────────


def test_neither_setting_adds_nothing_for_any_driver(settings):
    found = tls.database_tls()
    assert found == tls.DatabaseTLS()
    assert not found.configured and not found.checks_certificate and not found.verifies_server
    assert found.libpq_params() == {}
    assert found.asyncpg_kwargs() == {}
    assert found.with_libpq_params({"host": "db", "port": 5432}) == {"host": "db", "port": 5432}


def test_blank_settings_count_as_unset(settings):
    settings(sslmode="  ", sslrootcert="")
    assert tls.database_tls() == tls.DatabaseTLS()


@pytest.mark.parametrize(("environment", "query", "expected_query"), [
    ("production", "", "?sslmode=require"),
    ("PRODUCTION", "", "?sslmode=require"),
    ("production", "?application_name=api", "?application_name=api&sslmode=require"),
    ("production", "?sslmode=disable", "?sslmode=disable"),
    ("production", "?sslmode=verify-full", "?sslmode=verify-full"),
    ("development", "", ""),
    (None, "", ""),
])
def test_the_main_engine_keeps_its_production_default_with_neither_setting(environment, query, expected_query):
    """Pinned: unset, production adds sslmode=require only when the address names none."""
    assert tls.main_engine_url(_address(query), environment, tls.DatabaseTLS()) == _address(expected_query)


def test_a_set_mode_replaces_the_production_default_and_leaves_the_address_alone():
    configured = tls.DatabaseTLS(sslmode="verify-full")
    assert tls.main_engine_url(_address(), "production", configured) == _address()


def test_the_main_engine_is_built_from_the_address_and_the_builder():
    """database.py's engine: main_engine_url's address, built by create_app_engine."""
    tree = ast.parse((ORCH / "core" / "database" / "database.py").read_text(encoding="utf-8"))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {"main_engine_url", "create_app_engine", "database_tls"} <= called
    assert "create_engine" not in called


class _EngineRecorder:
    """sqlalchemy.create_engine, recorded."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return MagicMock(name="engine")


def test_create_app_engine_passes_the_callers_arguments_unchanged_with_neither_setting(settings, monkeypatch):
    recorder = _EngineRecorder()
    monkeypatch.setattr(tls, "create_engine", recorder)
    tls.create_app_engine(_address(), pool_pre_ping=True, connect_args={"keepalives": 1})
    tls.create_app_engine(_address())
    assert recorder.calls == [
        (_address(), {"connect_args": {"keepalives": 1}, "pool_pre_ping": True}),
        (_address(), {"connect_args": {}}),
    ]


# ── verify-full with a CA file ───────────────────────────────────────────────


def test_verify_full_with_a_ca_gives_libpq_its_keywords(settings, ca_file):
    settings(sslmode="verify-full", sslrootcert=ca_file)
    found = tls.database_tls()
    assert found.libpq_params() == {"sslmode": "verify-full", "sslrootcert": ca_file, "gssencmode": "disable"}
    assert found.configured and found.checks_certificate and found.verifies_server


def test_the_settings_win_over_the_same_keys_and_the_callers_dict_is_not_changed(settings, ca_file):
    settings(sslmode="verify-full", sslrootcert=ca_file)
    given = {"host": "db", "sslmode": "disable"}
    merged = tls.database_tls().with_libpq_params(given)
    assert merged == {"host": "db", "sslmode": "verify-full", "sslrootcert": ca_file, "gssencmode": "disable"}
    assert given == {"host": "db", "sslmode": "disable"}


def test_create_app_engine_puts_the_settings_in_connect_args_over_the_callers(settings, monkeypatch, ca_file):
    settings(sslmode="verify-full", sslrootcert=ca_file)
    recorder = _EngineRecorder()
    monkeypatch.setattr(tls, "create_engine", recorder)
    tls.create_app_engine(_address(), connect_args={"keepalives": 1, "sslmode": "prefer"})
    ((url, kwargs),) = recorder.calls
    assert url == _address()
    assert kwargs["connect_args"] == {"keepalives": 1, "sslmode": "verify-full", "sslrootcert": ca_file,
                                      "gssencmode": "disable"}


def test_psycopg2_receives_the_settings_over_the_addresses_own_sslmode(settings, monkeypatch, ca_file):
    """End to end through SQLAlchemy: the address's query becomes connect keywords, and
    connect_args, where the settings are, replace them. psycopg2.connect is recorded and
    refuses, so nothing is dialled."""
    import psycopg2
    from sqlalchemy.pool import NullPool

    seen = []

    def refuse(*args, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("recorded; no connection is made")

    monkeypatch.setattr(psycopg2, "connect", refuse)
    settings(sslmode="verify-full", sslrootcert=ca_file)
    engine = tls.create_app_engine(_address("?sslmode=disable"), poolclass=NullPool)
    with pytest.raises(Exception):
        engine.connect()
    assert seen, "psycopg2.connect was not reached"
    assert seen[0]["sslmode"] == "verify-full" and seen[0]["sslrootcert"] == ca_file
    assert seen[0]["host"] == "db.example.invalid"


# ── the address's own keys that libpq reads as sslmode ──────────────────────
# libpq turns requiressl (1: require, else prefer) and, in a URI, ssl=true (require) into sslmode,
# applying keys in the order written. psycopg2's final connection string is recorded at
# psycopg2._connect and read back with libpq's own parser (parse_dsn), which applies them the same
# way, so each assertion is on the mode libpq would use.
ALIASED_QUERIES = ["?sslmode=disable&requiressl=0", "?sslmode=disable&ssl=true",
                   "?requiressl=1&sslmode=disable&requiressl=0", "?sslmode=disable&requiress%6C=0",
                   "?application_name=api&sslmode=disable&requiressl=0"]


@pytest.fixture
def libpq_strings(monkeypatch):
    """psycopg2's final connection strings, recorded where psycopg2.connect hands them to libpq."""
    import psycopg2

    seen = []

    def record(dsn, *args, **kwargs):
        seen.append(dsn)
        raise psycopg2.OperationalError("recorded; no connection is made")

    monkeypatch.setattr(psycopg2, "_connect", record)
    return seen


def _libpq_reads(dsn: str) -> dict:
    from psycopg2.extensions import parse_dsn

    return parse_dsn(dsn)


@pytest.mark.parametrize("query", ALIASED_QUERIES)
def test_libpq_ends_on_the_settings_mode_through_an_engine(settings, libpq_strings, ca_file, query):
    from sqlalchemy.pool import NullPool

    settings(sslmode="verify-full", sslrootcert=ca_file)
    engine = tls.create_app_engine(_address(query), poolclass=NullPool)
    with pytest.raises(Exception):
        engine.connect()
    assert libpq_strings, "psycopg2 built no connection string"
    used = _libpq_reads(libpq_strings[0])
    assert (used["sslmode"], used["sslrootcert"]) == ("verify-full", ca_file)
    assert "requiressl" not in libpq_strings[0] and "ssl=" not in libpq_strings[0]


@pytest.mark.parametrize("query", ["?sslmode=disable&requiressl=0", "?sslmode=disable&ssl=true"])
def test_libpq_ends_on_the_settings_mode_through_the_seed_loaders_dsn(settings, monkeypatch, libpq_strings,
                                                                      ca_file, query):
    """psycopg2.connect(url, **keywords) parses the address with libpq first, then the keywords replace it."""
    import psycopg2
    from core.database import load_seed_data

    monkeypatch.setattr("core.database.database.get_database_url", lambda: _address(query))
    settings(sslmode="verify-full", sslrootcert=ca_file)
    with pytest.raises(psycopg2.OperationalError):
        load_seed_data._connect()
    used = _libpq_reads(libpq_strings[0])
    assert (used["sslmode"], used["sslrootcert"]) == ("verify-full", ca_file)


def test_a_keyword_dict_loses_libpqs_other_sslmode_keys_and_the_callers_dict_is_kept(settings, ca_file):
    settings(sslmode="verify-full", sslrootcert=ca_file)
    given = {"host": "db", "sslmode": "disable", "requiressl": "0", "ssl": "true"}
    merged = tls.database_tls().with_libpq_params(given)
    assert merged == {"host": "db", "sslmode": "verify-full", "sslrootcert": ca_file, "gssencmode": "disable"}
    assert given == {"host": "db", "sslmode": "disable", "requiressl": "0", "ssl": "true"}


def test_an_address_without_those_keys_is_returned_as_given():
    address = _address("?sslmode=disable&application_name=api")
    assert tls.without_sslmode_aliases(address) is address


def test_only_those_keys_leave_the_address():
    kept = tls.without_sslmode_aliases(_address("?requiressl=0&application_name=api&ssl=true&sslmode=disable"))
    assert dict(kept.query) == {"application_name": "api", "sslmode": "disable"}
    assert (kept.host, kept.port, kept.database, kept.username) == ("db.example.invalid", 5432, "app", "app")


def test_with_neither_setting_the_address_and_the_dict_reach_the_builder_as_given(settings, monkeypatch):
    recorder = _EngineRecorder()
    monkeypatch.setattr(tls, "create_engine", recorder)
    address = _address("?sslmode=disable&requiressl=1&ssl=true")
    tls.create_app_engine(address, connect_args={"requiressl": "1"})
    assert recorder.calls == [(address, {"connect_args": {"requiressl": "1"}})]
    given = {"host": "db", "requiressl": "1", "ssl": "true"}
    assert tls.database_tls().with_libpq_params(given) == given


def test_with_neither_setting_libpq_receives_the_addresses_own_keys(settings, libpq_strings):
    from sqlalchemy.pool import NullPool

    engine = tls.create_app_engine(_address("?sslmode=disable&requiressl=1"), poolclass=NullPool)
    with pytest.raises(Exception):
        engine.connect()
    assert "sslmode=disable" in libpq_strings[0] and "requiressl=1" in libpq_strings[0]


def test_verify_full_with_a_ca_gives_asyncpg_a_context_that_checks_the_certificate_and_host(settings, ca_file):
    settings(sslmode="verify-full", sslrootcert=ca_file)
    context = tls.database_tls().asyncpg_kwargs()["ssl"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.get_ca_certs(), "the CA file was loaded"


def test_verify_ca_with_a_ca_checks_the_certificate_but_not_the_host(settings, ca_file):
    settings(sslmode="verify-ca", sslrootcert=ca_file)
    found = tls.database_tls()
    context = found.asyncpg_kwargs()["ssl"]
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is False
    assert found.checks_certificate and not found.verifies_server


def test_the_asyncpg_context_trusts_only_the_named_ca(settings, monkeypatch, ca_file):
    """Like libpq's sslrootcert: the context is made from that file, not the system's CAs."""
    made = []
    real = ssl.create_default_context

    def spy(purpose=ssl.Purpose.SERVER_AUTH, **kwargs):
        made.append((purpose, kwargs))
        return real(purpose, **kwargs)

    tls._verifying_context.cache_clear()
    monkeypatch.setattr(tls.ssl, "create_default_context", spy)
    settings(sslmode="verify-full", sslrootcert=ca_file)
    tls.database_tls().asyncpg_kwargs()
    tls._verifying_context.cache_clear()
    assert made == [(ssl.Purpose.SERVER_AUTH, {"cafile": ca_file})], "built once, from that file only"


@pytest.mark.parametrize("mode", ["disable", "allow", "prefer", "require"])
def test_a_mode_that_checks_no_certificate_reaches_asyncpg_as_text(settings, mode):
    """asyncpg reads the mode text the way libpq does. A checking mode needs a CA file (below)."""
    settings(sslmode=mode)
    assert tls.database_tls().asyncpg_kwargs() == {"ssl": mode}
    assert tls.database_tls().libpq_params() == {"sslmode": mode}


def test_only_verify_full_verifies_the_server():
    """The predicate the enterprise edition's start-up requires (PRD-278 D-6), and no wider."""
    verified = {mode for mode in tls.SSLMODES if tls.DatabaseTLS(sslmode=mode).verifies_server}
    checked = {mode for mode in tls.SSLMODES if tls.DatabaseTLS(sslmode=mode).checks_certificate}
    assert verified == {"verify-full"}
    assert checked == {"verify-ca", "verify-full"}
    assert not tls.DatabaseTLS().verifies_server


# ── GSSAPI encryption: libpq prefers it to TLS whatever sslmode says ─────────
# With Kerberos credentials at hand, libpq tries a GSSAPI-encrypted connection first and, if it
# is accepted, uses it and checks no certificate. A mode that checks a certificate turns that off.
GSS_QUERIES = ["?gssencmode=prefer", "?sslmode=disable&gssencmode=require", "?gss%65ncmode=prefer"]


def test_exactly_the_modes_that_check_a_certificate_turn_gss_encryption_off():
    """require checks no certificate: GSSAPI encryption, a channel Kerberos authenticates, may stay."""
    off = {mode for mode in tls.SSLMODES if "gssencmode" in tls.DatabaseTLS(sslmode=mode).libpq_params()}
    assert off == tls.CERTIFICATE_CHECKING_MODES == {"verify-ca", "verify-full"}
    assert tls.DatabaseTLS(sslmode="verify-ca").libpq_params()["gssencmode"] == "disable"
    assert "gssencmode" not in tls.DatabaseTLS().libpq_params()
    assert dict(tls.NO_GSS_ENCRYPTION) == {"gssencmode": "disable"}


def test_a_keyword_dicts_own_gssencmode_is_replaced(settings, ca_file):
    settings(sslmode="verify-full", sslrootcert=ca_file)
    given = {"host": "db", "gssencmode": "require"}
    assert tls.database_tls().with_libpq_params(given)["gssencmode"] == "disable"
    assert given == {"host": "db", "gssencmode": "require"}


@pytest.mark.parametrize("query", GSS_QUERIES)
def test_libpq_gets_no_gss_encryption_through_an_engine_whatever_the_address_says(settings, libpq_strings,
                                                                                   ca_file, query):
    from sqlalchemy.pool import NullPool

    settings(sslmode="verify-full", sslrootcert=ca_file)
    engine = tls.create_app_engine(_address(query), poolclass=NullPool)
    with pytest.raises(Exception):
        engine.connect()
    used = _libpq_reads(libpq_strings[0])
    assert (used["sslmode"], used["gssencmode"]) == ("verify-full", "disable")
    assert "gssencmode=prefer" not in libpq_strings[0] and "gssencmode=require" not in libpq_strings[0]


@pytest.mark.parametrize("query", GSS_QUERIES)
def test_libpq_gets_no_gss_encryption_through_the_seed_loaders_dsn(settings, monkeypatch, libpq_strings,
                                                                   ca_file, query):
    import psycopg2
    from core.database import load_seed_data

    monkeypatch.setattr("core.database.database.get_database_url", lambda: _address(query))
    settings(sslmode="verify-full", sslrootcert=ca_file)
    with pytest.raises(psycopg2.OperationalError):
        load_seed_data._connect()
    assert _libpq_reads(libpq_strings[0])["gssencmode"] == "disable"


@pytest.mark.parametrize("mode", [None, "prefer", "require"])
def test_without_a_certificate_check_the_addresses_gssencmode_reaches_libpq_as_given(settings, libpq_strings, mode):
    from sqlalchemy.pool import NullPool

    settings(sslmode=mode)
    engine = tls.create_app_engine(_address("?gssencmode=require"), poolclass=NullPool)
    with pytest.raises(Exception):
        engine.connect()
    assert _libpq_reads(libpq_strings[0])["gssencmode"] == "require"
    assert "gssencmode" not in tls.database_tls().libpq_params()


# ── values that can't work stop start-up, saying which setting ──────────────


def test_an_unknown_mode_is_refused_by_name(settings):
    settings(sslmode="verify")
    with pytest.raises(tls.DatabaseTLSConfigError, match="DATABASE_SSLMODE must be one of"):
        tls.database_tls()


@pytest.mark.parametrize("mode", [None, "disable", "prefer", "require"])
def test_a_ca_without_a_checking_mode_is_refused_without_quoting_its_path(settings, mode):
    settings(sslmode=mode, sslrootcert=CA_IN_POD)
    with pytest.raises(tls.DatabaseTLSConfigError, match="DATABASE_SSLROOTCERT is used only with") as refused:
        tls.database_tls()
    assert CA_IN_POD not in str(refused.value)


@pytest.mark.parametrize("rootcert", [None, "", "   "])
@pytest.mark.parametrize("mode", ["verify-ca", "verify-full", " VERIFY-FULL "])
def test_a_checking_mode_without_a_ca_is_refused_naming_both_settings(settings, mode, rootcert):
    """Refused at start, not after a wait: libpq would look for a root.crt the image does not hold."""
    settings(sslmode=mode, sslrootcert=rootcert)
    with pytest.raises(tls.DatabaseTLSConfigError) as refused:
        tls.database_tls()
    message = str(refused.value)
    assert message.startswith("DATABASE_SSLMODE verify-ca and verify-full need DATABASE_SSLROOTCERT")
    assert "a file" in message and CA_IN_POD not in message


# The engine module imported in a fresh interpreter, with psycopg2.connect replaced by a refusal
# that prints a marker: the import is the API's start, and nothing may connect before it refuses.
_IMPORT_THE_ENGINE_MODULE = (
    "import psycopg2\n"
    "def refuse(*args, **kwargs):\n"
    "    print('CONNECT_ATTEMPTED', flush=True)\n"
    "    raise SystemExit(3)\n"
    "psycopg2.connect = refuse\n"
    "import core.database.database\n"
    "print('ENGINE_MODULE_IMPORTED', flush=True)\n"
)


def _import_the_engine_module(**settings):
    """A fresh interpreter importing core/database/database.py as an enterprise API: (code, output)."""
    env = {key: value for key, value in os.environ.items() if key not in ("DATABASE_SSLMODE", "DATABASE_SSLROOTCERT")}
    env.update({"AUTH_EDITION": "enterprise", "DATABASE_URL": _address(), **settings})
    done = subprocess.run([sys.executable, "-c", _IMPORT_THE_ENGINE_MODULE], cwd=ORCH, env=env,
                          capture_output=True, text=True, timeout=120)
    return done.returncode, done.stdout + done.stderr


def test_an_enterprise_api_with_verify_full_and_no_ca_stops_at_the_engine_import_before_any_connection():
    """Refused by core.database.tls when the engine module is imported, not by the edition check."""
    code, output = _import_the_engine_module(DATABASE_SSLMODE="verify-full")
    assert code != 0, output
    assert "DatabaseTLSConfigError" in output and "need DATABASE_SSLROOTCERT" in output
    assert "CONNECT_ATTEMPTED" not in output and "ENGINE_MODULE_IMPORTED" not in output
    assert "db.example.invalid" not in output


def test_the_same_api_with_a_ca_file_imports_the_engine_module_and_connects_to_nothing(ca_file):
    """The control: only the missing CA file stopped the import above."""
    code, output = _import_the_engine_module(DATABASE_SSLMODE="verify-full", DATABASE_SSLROOTCERT=ca_file)
    assert code == 0, output[-2000:]
    assert "ENGINE_MODULE_IMPORTED" in output and "CONNECT_ATTEMPTED" not in output


def _no_certificate(tmp_path) -> str:
    path = tmp_path / "not-a-ca.pem"
    path.write_text("this file holds no certificate\n", encoding="utf-8")
    return str(path)


def _empty(tmp_path) -> str:
    path = tmp_path / "empty.pem"
    path.write_bytes(b"")
    return str(path)


@pytest.mark.parametrize("make_path", [
    lambda tmp_path: str(tmp_path / "missing.crt"),
    lambda tmp_path: str(tmp_path),
    _no_certificate,
    _empty,
], ids=["missing", "a-directory", "no-certificate", "empty"])
@pytest.mark.parametrize("mode", ["verify-ca", "verify-full"])
def test_an_unusable_ca_file_stops_start_up_without_quoting_its_path(settings, tmp_path, make_path, mode):
    path = make_path(tmp_path)
    settings(sslmode=mode, sslrootcert=path)
    with pytest.raises(tls.DatabaseTLSConfigError, match="DATABASE_SSLROOTCERT must name a readable file") as refused:
        tls.database_tls()
    assert path not in str(refused.value)
    # The OS error, which names the path, is not chained into the traceback either.
    assert refused.value.__cause__ is None
    assert refused.value.__context__ is None or refused.value.__suppress_context__


def test_a_ca_replaced_at_the_same_path_gives_the_next_asyncpg_connection_a_new_context(settings, tmp_path):
    """A ConfigMap update replaces the file under a running process: the next connection
    trusts the new file, as libpq would."""
    path = write_ca_bundle(tmp_path / "ca.crt", ["first-ca"])
    settings(sslmode="verify-full", sslrootcert=str(path))
    first = tls.database_tls().asyncpg_kwargs()["ssl"]
    assert tls.database_tls().asyncpg_kwargs()["ssl"] is first, "an unchanged file reuses its context"
    before = os.stat(path)
    write_ca_bundle(path, ["first-ca", "second-ca"])
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    second = tls.database_tls().asyncpg_kwargs()["ssl"]
    assert second is not first
    assert len(first.get_ca_certs()) == 1 and len(second.get_ca_certs()) == 2


def test_the_two_settings_are_config_settings():
    import json

    names = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"]
    assert {"DATABASE_SSLMODE", "DATABASE_SSLROOTCERT"} <= set(names)
    assert hasattr(config, "DATABASE_SSLMODE") and hasattr(config, "DATABASE_SSLROOTCERT")


def test_the_unused_config_database_url_helper_is_gone():
    """Config.get_database_url() is gone; the main engine and core/database/tls.py decide TLS."""
    assert not hasattr(type(config), "get_database_url")


# ── each connection path is given the settings ───────────────────────────────


def _connect_spy(rows=()):
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=list(rows))
    conn.close = AsyncMock()
    return AsyncMock(return_value=conn)


@pytest.mark.parametrize(("mode", "rootcert", "expected"), [
    (None, None, {}),
    ("require", None, {"ssl": "require"}),
])
def test_the_bm25_leg_connects_with_the_settings(settings, mode, rootcert, expected):
    from modules.rag import bm25_leg

    settings(sslmode=mode, sslrootcert=rootcert)
    connect = _connect_spy()
    with patch("asyncpg.connect", new=connect):
        asyncio.run(bm25_leg._fetch_rows("query", "00000000-0000-0000-0000-0000000000c1", 5))
    ((args, kwargs),) = [(c.args, c.kwargs) for c in connect.await_args_list]
    assert args == (config.DATABASE_URL,)
    assert kwargs == expected


def test_the_chunk_hydration_connects_with_a_verifying_context(settings, ca_file):
    from modules.rag import chunk_hydration

    settings(sslmode="verify-full", sslrootcert=ca_file)
    connect = _connect_spy()
    with patch("asyncpg.connect", new=connect):
        asyncio.run(chunk_hydration.fetch_chunk_texts({(10, 1)}, None))
    context = connect.await_args.kwargs["ssl"]
    assert isinstance(context, ssl.SSLContext) and context.check_hostname is True


def test_the_chunk_hydration_adds_nothing_with_neither_setting(settings):
    from modules.rag import chunk_hydration

    connect = _connect_spy()
    with patch("asyncpg.connect", new=connect):
        asyncio.run(chunk_hydration.fetch_chunk_texts({(10, 1)}, None))
    assert connect.await_args.kwargs == {}


def test_the_scheduler_lock_keeps_its_keepalive_and_gains_the_settings(settings, monkeypatch, ca_file):
    from core.jobs import leadership

    recorder = _EngineRecorder()
    monkeypatch.setattr(tls, "create_engine", recorder)
    leadership.SchedulerLock(_address())._connect()
    settings(sslmode="verify-full", sslrootcert=ca_file)
    leadership.SchedulerLock(_address())._connect()
    (_, unset), (_, verified) = recorder.calls
    assert unset["connect_args"] == leadership.KEEPALIVE
    assert verified["connect_args"] == {**leadership.KEEPALIVE, "sslmode": "verify-full", "sslrootcert": ca_file,
                                        "gssencmode": "disable"}


def test_the_seed_loader_connects_with_the_settings(settings, monkeypatch, ca_file):
    from core.database import load_seed_data

    connect = MagicMock(name="psycopg2.connect")
    monkeypatch.setattr(load_seed_data.psycopg2, "connect", connect)
    monkeypatch.setattr("core.database.database.get_database_url", lambda: _address())
    settings(sslmode="verify-full", sslrootcert=ca_file)
    load_seed_data._connect()
    connect.assert_called_once_with(_address(), sslmode="verify-full", sslrootcert=ca_file, gssencmode="disable")


class _StubEmbeddings:
    def get_provider_info(self):
        return {"provider": "stub"}


def test_the_ingestion_managers_connections_carry_the_settings(settings, monkeypatch, ca_file):
    """Every psycopg2.connect(**self.db_config) in the manager, whoever built the dict."""
    pytest.importorskip("docx")
    from modules.rag.ingestion import manager as mgr_mod

    monkeypatch.setattr("core.llm.create_embedding_manager", lambda *a, **kw: _StubEmbeddings())
    given = {"host": "db", "port": 5432, "database": "app", "user": "app"}
    assert mgr_mod.DocumentManager(db_config=given, workspace_id="ws").db_config == given
    settings(sslmode="verify-full", sslrootcert=ca_file)
    secured = mgr_mod.DocumentManager(db_config=given, workspace_id="ws").db_config
    assert secured == {**given, "sslmode": "verify-full", "sslrootcert": ca_file, "gssencmode": "disable"}
