"""PRD-278 W3 (US-302): every connection the orchestrator opens to its own database is
secured by ``core/database/tls.py``, and a new one opened another way fails here.

An AST walk of every ``.py`` file under ``orchestrator/`` (its tests and conftest files
aside) finds each call that opens a PostgreSQL connection or builds an engine:
SQLAlchemy's ``create_engine``, ``engine_from_config`` and their async forms,
``psycopg2.connect``, ``psycopg.connect``, ``asyncpg.connect`` and ``asyncpg.create_pool``,
under whatever local name the file imports them as. A call to
``core.database.tls.create_app_engine`` (under any name) is not one: it applies the
settings itself. Every call found must be in exactly one of:

* ``APP_DATABASE``: a connection to the application's own database, which passes the
  settings. Each call there is checked for the form named: ``**database_tls().<x>()``, a
  ``connect_args`` built from ``database_tls()``, or ``**db_config`` where the file's only
  ``db_config`` is built by ``with_libpq_params``.
* ``USER_DATA_SOURCES``: a connection to a database a user saved (NL2SQL, database
  knowledge, the credential testers). It takes that user's own settings, never these.
* ``NOT_COVERED``: operator scripts run by hand, and retired scripts nothing runs. Named
  one by one, so a new file is never let through by a pattern.

What it does not catch: a connection opened through ``getattr`` or another indirection,
one built by a third-party library, and ``pg_isready``/``psql`` (the entrypoint's test
checks those). Which database a call reaches is decided by the lists, not proven.
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pytest

ORCH = Path(__file__).resolve().parents[1]
# Tests (and their conftest files) connect to the test database as they please; a virtual
# environment or installed packages are not the orchestrator's code.
SKIPPED_DIRS = frozenset({"tests", "node_modules", "site-packages", "__pycache__"})

DRIVER_CALLS = frozenset({
    "sqlalchemy.create_engine",
    "sqlalchemy.engine.create_engine",
    "sqlalchemy.engine_from_config",
    "sqlalchemy.engine.engine_from_config",
    "sqlalchemy.ext.asyncio.create_async_engine",
    "sqlalchemy.ext.asyncio.async_engine_from_config",
    "psycopg2.connect",
    "psycopg.connect",
    "asyncpg.connect",
    "asyncpg.create_pool",
})
# The call that builds every other engine on the app's database, from the settings.
ENGINE_BUILDER_FILE = "core/database/tls.py"
TLS_METHODS = frozenset({"libpq_params", "asyncpg_kwargs", "with_libpq_params"})

APP_DATABASE: Dict[str, str] = {
    "alembic/env.py": "engine_from_config for an alembic.ini address, connect_args=database_tls().libpq_params()",
    "api/documents.py": "psycopg2.connect(**db_config); db_config = database_tls().with_libpq_params({...})",
    "core/database/load_seed_data.py": "psycopg2.connect(url, **database_tls().libpq_params())",
    "modules/rag/bm25_leg.py": "asyncpg.connect(url, **database_tls().asyncpg_kwargs())",
    "modules/rag/chunk_hydration.py": "asyncpg.connect(url, **database_tls().asyncpg_kwargs())",
    "modules/rag/ingestion/manager.py":
        "psycopg2.connect(**self.db_config); __init__ sets self.db_config = database_tls().with_libpq_params(...)",
}

USER_DATA_SOURCES: Dict[str, str] = {
    "api/database_knowledge.py": "runs a question's SQL on the database a user saved as a knowledge source",
    "core/credentials/service.py": "tests a PostgreSQL credential a user saved, with its own host and settings",
    "core/credentials/tester.py": "tests a PostgreSQL credential a user saved, with its own ssl_mode",
    "modules/nl2sql/schema/grounding.py": "NL2SQL: reads the schema of a user's own data source",
    "modules/nl2sql/schema/introspection.py": "NL2SQL: introspects a user's own data source",
    "modules/nl2sql/service.py": "NL2SQL: queries a user's own data source",
}

_OPERATOR_SCRIPT = ("an operator's one-off script, run by hand with `python -m scripts.<name>`; "
                    "not run by the API, the worker, the entrypoint or the migration Job")
_RETIRED = "a retired script that nothing runs; it reads a file this repository no longer has"
NOT_COVERED: Dict[str, str] = {
    **{f"scripts/{name}.py": _OPERATOR_SCRIPT for name in (
        "backfill_prd129_deliverables", "benchmark_blog_pipeline", "clerk_cutover_relink",
        "consolidate_workspace", "debug_cloud_connections", "diagnose_sync_failure",
        "migrate_blog_to_workspace", "migrate_to_s3_vectors", "reingest_documents",
        "repair_catalog_kb_types", "seed_agent_personas_v2", "seed_blog_playbook",
        "seed_llm_marketplace", "seed_marketplace_agents_v2", "seed_nightly_test_playbook",
        "seed_recipes_marketplace", "seed_starter_agents",
    )},
    "core/database/init_database.py": _RETIRED,
    "core/database/update_tool_metadata.py": _RETIRED,
}


def _parse(rel: str) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # an old file's escape sequence is not this test's concern
        return ast.parse((ORCH / rel).read_text(encoding="utf-8"), filename=rel)


def _orchestrator_sources() -> Iterator[Tuple[str, ast.Module]]:
    for path in sorted(ORCH.rglob("*.py")):
        parts = path.relative_to(ORCH).parts
        if path.name == "conftest.py" or SKIPPED_DIRS & set(parts) or any(p.startswith(".") for p in parts):
            continue
        rel = path.relative_to(ORCH).as_posix()
        yield rel, _parse(rel)


def _imported_names(tree: ast.Module) -> Dict[str, str]:
    """Each local name a file binds by import, to the dotted name it stands for."""
    names: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                names[local] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def _dotted(func: ast.AST, names: Dict[str, str]) -> Optional[str]:
    """The call's target as a dotted name, when it starts at an imported name."""
    parts: List[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name) or func.id not in names:
        return None
    return ".".join([names[func.id], *reversed(parts)])


def _driver_calls(tree: ast.Module) -> List[ast.Call]:
    names = _imported_names(tree)
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _dotted(node.func, names) in DRIVER_CALLS]


def _calls_tls(node: ast.AST) -> bool:
    """True when ``node`` holds a call to one of DatabaseTLS's driver forms."""
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in TLS_METHODS
               for n in ast.walk(node))


def _db_config_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name) and node.id == "db_config":
        return "db_config"
    if isinstance(node, ast.Attribute) and node.attr == "db_config":
        return "self.db_config"
    return None


def _db_config_is_secured(tree: ast.Module) -> bool:
    """Every assignment to ``db_config`` / ``self.db_config`` is built by with_libpq_params."""
    assigned = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(_db_config_name(target) for target in node.targets)]
    return bool(assigned) and all(isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                                  and node.value.func.attr == "with_libpq_params" for node in assigned)


def _passes_the_settings(call: ast.Call, tree: ast.Module) -> bool:
    for keyword in call.keywords:
        if keyword.arg is None and _calls_tls(keyword.value):
            return True  # **database_tls().libpq_params() / .asyncpg_kwargs()
        if keyword.arg == "connect_args" and _calls_tls(keyword.value):
            return True
        if keyword.arg is None and _db_config_name(keyword.value) and _db_config_is_secured(tree):
            return True
    return False


def _found() -> Dict[str, List[ast.Call]]:
    found: Dict[str, List[ast.Call]] = {}
    for rel, tree in _orchestrator_sources():
        calls = _driver_calls(tree)
        if calls:
            found[rel] = calls
    return found


FOUND = _found()


def test_every_connection_is_listed_with_its_reason():
    listed = {ENGINE_BUILDER_FILE, *APP_DATABASE, *USER_DATA_SOURCES, *NOT_COVERED}
    unlisted = {rel: [call.lineno for call in calls] for rel, calls in FOUND.items() if rel not in listed}
    assert not unlisted, (
        "These files open a database connection outside core/database/tls.py. A connection to "
        "the app's own database uses create_app_engine or database_tls(); a user's own data "
        f"source goes in USER_DATA_SOURCES with its reason: {unlisted}"
    )


def test_the_lists_do_not_overlap_and_name_files_that_still_connect():
    groups = [set(APP_DATABASE), set(USER_DATA_SOURCES), set(NOT_COVERED)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    stale = sorted(set().union(*groups) - set(FOUND))
    assert not stale, f"listed but no longer connecting (take them off the list): {stale}"


def test_the_engine_builder_is_the_only_create_engine_on_the_apps_database():
    calls = FOUND.get(ENGINE_BUILDER_FILE, [])
    assert len(calls) == 1, "core/database/tls.py builds engines in one place: create_app_engine"
    (call,) = calls
    assert any(k.arg == "connect_args" and _calls_tls(k.value) for k in call.keywords)


@pytest.mark.parametrize("rel", sorted(APP_DATABASE))
def test_each_app_database_connection_passes_the_settings(rel):
    tree = _parse(rel)
    unsecured = [call.lineno for call in _driver_calls(tree) if not _passes_the_settings(call, tree)]
    assert not unsecured, f"{rel}: a connection without database_tls() ({APP_DATABASE[rel]}), lines {unsecured}"


@pytest.mark.parametrize("rel", [
    "core/database/database.py",
    "core/jobs/leadership.py",
    "modules/tools/formatting/result_formatter.py",
    "api/recipe_executor.py",
    "scripts/init_fresh_db.py",
    "scripts/generate_schema_baseline.py",
    "alembic/env.py",
])
def test_the_engines_on_the_apps_database_come_from_the_builder(rel):
    """The main engine, the scheduler's lock, the formatter's engine, a run's own engine,
    the start-up scripts and Alembic: each builds through create_app_engine."""
    tree = _parse(rel)
    names = _imported_names(tree)
    built = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and _dotted(node.func, names) == "core.database.tls.create_app_engine"]
    assert built, f"{rel} builds no engine through core.database.tls.create_app_engine"


def test_the_guard_sees_a_connection_under_any_imported_name():
    """The walk resolves aliases, so renaming the import does not hide a connection."""
    source = (
        "import psycopg2 as pg\n"
        "from sqlalchemy import create_engine as make\n"
        "import asyncpg\n"
        "from core.database.tls import create_app_engine as create_engine\n"
        "pg.connect('x')\nmake('x')\nasyncpg.connect('x')\ncreate_engine('x')\n"
    )
    found = sorted(call.lineno for call in _driver_calls(ast.parse(source)))
    assert found == [5, 6, 7]
