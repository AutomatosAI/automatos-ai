"""PRD-278 W3 (US-301, US-302): one function says how every connection to the app's own
database is secured, and each driver gets it in the form it takes.

``core.database.tls.database_tls()`` reads ``DATABASE_SSLMODE`` and ``DATABASE_SSLROOTCERT``:

* with neither set, no connection path gains a keyword, and the main engine keeps its
  production default (``sslmode=require`` added when the address names none);
* with ``verify-full`` and a CA file, libpq gets ``sslmode`` and ``sslrootcert`` (psycopg2,
  and SQLAlchemy through ``connect_args``), and asyncpg gets an ``ssl.SSLContext`` that
  requires the certificate, trusts that CA and checks the host name;
* a mode libpq doesn't know, or a CA file without a mode that checks it, stops start-up
  with a message that names the setting and never the file's path.

No database: the drivers' ``connect`` and SQLAlchemy's ``create_engine`` are recorded.
"""
from __future__ import annotations

import ast
import asyncio
import ssl
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import certifi
import pytest

from config import config
from core.database import tls

ORCH = Path(__file__).resolve().parents[1]
# A real PEM bundle on disk, so the asyncpg context is built from an actual CA file.
CA_FILE = certifi.where()
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


def test_verify_full_with_a_ca_gives_libpq_both_keywords(settings):
    settings(sslmode="verify-full", sslrootcert=CA_IN_POD)
    found = tls.database_tls()
    assert found.libpq_params() == {"sslmode": "verify-full", "sslrootcert": CA_IN_POD}
    assert found.configured and found.checks_certificate and found.verifies_server


def test_the_settings_win_over_the_same_keys_and_the_callers_dict_is_not_changed(settings):
    settings(sslmode="verify-full", sslrootcert=CA_IN_POD)
    given = {"host": "db", "sslmode": "disable"}
    merged = tls.database_tls().with_libpq_params(given)
    assert merged == {"host": "db", "sslmode": "verify-full", "sslrootcert": CA_IN_POD}
    assert given == {"host": "db", "sslmode": "disable"}


def test_create_app_engine_puts_the_settings_in_connect_args_over_the_callers(settings, monkeypatch):
    settings(sslmode="verify-full", sslrootcert=CA_IN_POD)
    recorder = _EngineRecorder()
    monkeypatch.setattr(tls, "create_engine", recorder)
    tls.create_app_engine(_address(), connect_args={"keepalives": 1, "sslmode": "prefer"})
    ((url, kwargs),) = recorder.calls
    assert url == _address()
    assert kwargs["connect_args"] == {"keepalives": 1, "sslmode": "verify-full", "sslrootcert": CA_IN_POD}


def test_psycopg2_receives_the_settings_over_the_addresses_own_sslmode(settings, monkeypatch):
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
    settings(sslmode="verify-full", sslrootcert=CA_IN_POD)
    engine = tls.create_app_engine(_address("?sslmode=disable"), poolclass=NullPool)
    with pytest.raises(Exception):
        engine.connect()
    assert seen, "psycopg2.connect was not reached"
    assert seen[0]["sslmode"] == "verify-full" and seen[0]["sslrootcert"] == CA_IN_POD
    assert seen[0]["host"] == "db.example.invalid"


def test_verify_full_with_a_ca_gives_asyncpg_a_context_that_checks_the_certificate_and_host(settings):
    settings(sslmode="verify-full", sslrootcert=CA_FILE)
    context = tls.database_tls().asyncpg_kwargs()["ssl"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.get_ca_certs(), "the CA file was loaded"


def test_verify_ca_with_a_ca_checks_the_certificate_but_not_the_host(settings):
    settings(sslmode="verify-ca", sslrootcert=CA_FILE)
    found = tls.database_tls()
    context = found.asyncpg_kwargs()["ssl"]
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is False
    assert found.checks_certificate and not found.verifies_server


def test_the_asyncpg_context_trusts_only_the_named_ca(settings, monkeypatch):
    """Like libpq's sslrootcert: the context is made from that file, not the system's CAs."""
    made = []
    real = ssl.create_default_context

    def spy(purpose=ssl.Purpose.SERVER_AUTH, **kwargs):
        made.append((purpose, kwargs))
        return real(purpose, **kwargs)

    tls._verifying_context.cache_clear()
    monkeypatch.setattr(tls.ssl, "create_default_context", spy)
    settings(sslmode="verify-full", sslrootcert=CA_FILE)
    tls.database_tls().asyncpg_kwargs()
    tls._verifying_context.cache_clear()
    assert made == [(ssl.Purpose.SERVER_AUTH, {"cafile": CA_FILE})]


@pytest.mark.parametrize("mode", ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"])
def test_a_mode_without_a_ca_reaches_asyncpg_as_text(settings, mode):
    """asyncpg reads the mode text the way libpq does (its own root.crt for verify-*)."""
    settings(sslmode=mode)
    assert tls.database_tls().asyncpg_kwargs() == {"ssl": mode}
    assert tls.database_tls().libpq_params() == {"sslmode": mode}


def test_only_verify_full_verifies_the_server():
    """The predicate a later edition check can require (PRD-278 D-6)."""
    verified = {mode for mode in tls.SSLMODES if tls.DatabaseTLS(sslmode=mode).verifies_server}
    checked = {mode for mode in tls.SSLMODES if tls.DatabaseTLS(sslmode=mode).checks_certificate}
    assert verified == {"verify-full"}
    assert checked == {"verify-ca", "verify-full"}
    assert not tls.DatabaseTLS().verifies_server


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


def test_the_two_settings_are_config_settings():
    import json

    names = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"]
    assert {"DATABASE_SSLMODE", "DATABASE_SSLROOTCERT"} <= set(names)
    assert hasattr(config, "DATABASE_SSLMODE") and hasattr(config, "DATABASE_SSLROOTCERT")


def test_the_unused_config_database_url_helper_is_gone():
    """Config.get_database_url() skipped TLS by a substring match on the address; the
    main engine and core/database/tls.py decide it now."""
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


def test_the_chunk_hydration_connects_with_a_verifying_context(settings):
    from modules.rag import chunk_hydration

    settings(sslmode="verify-full", sslrootcert=CA_FILE)
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


def test_the_scheduler_lock_keeps_its_keepalive_and_gains_the_settings(settings, monkeypatch):
    from core.jobs import leadership

    recorder = _EngineRecorder()
    monkeypatch.setattr(tls, "create_engine", recorder)
    leadership.SchedulerLock(_address())._connect()
    settings(sslmode="verify-full", sslrootcert=CA_IN_POD)
    leadership.SchedulerLock(_address())._connect()
    (_, unset), (_, verified) = recorder.calls
    assert unset["connect_args"] == leadership.KEEPALIVE
    assert verified["connect_args"] == {**leadership.KEEPALIVE, "sslmode": "verify-full", "sslrootcert": CA_IN_POD}


def test_the_seed_loader_connects_with_the_settings(settings, monkeypatch):
    from core.database import load_seed_data

    connect = MagicMock(name="psycopg2.connect")
    monkeypatch.setattr(load_seed_data.psycopg2, "connect", connect)
    monkeypatch.setattr("core.database.database.get_database_url", lambda: _address())
    settings(sslmode="verify-full", sslrootcert=CA_IN_POD)
    load_seed_data._connect()
    connect.assert_called_once_with(_address(), sslmode="verify-full", sslrootcert=CA_IN_POD)


class _StubEmbeddings:
    def get_provider_info(self):
        return {"provider": "stub"}


def test_the_ingestion_managers_connections_carry_the_settings(settings, monkeypatch):
    """Every psycopg2.connect(**self.db_config) in the manager, whoever built the dict."""
    pytest.importorskip("docx")
    from modules.rag.ingestion import manager as mgr_mod

    monkeypatch.setattr("core.llm.create_embedding_manager", lambda *a, **kw: _StubEmbeddings())
    given = {"host": "db", "port": 5432, "database": "app", "user": "app"}
    assert mgr_mod.DocumentManager(db_config=given, workspace_id="ws").db_config == given
    settings(sslmode="verify-full", sslrootcert=CA_IN_POD)
    secured = mgr_mod.DocumentManager(db_config=given, workspace_id="ws").db_config
    assert secured == {**given, "sslmode": "verify-full", "sslrootcert": CA_IN_POD}
