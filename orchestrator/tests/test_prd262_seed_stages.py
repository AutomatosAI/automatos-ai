"""PRD-262 US-031: the two boot seed stages run when their inputs change, not on every boot.

* **The loader stage** (``python -m core.database.load_seed_data --boot``, which the
  entrypoint starts only when ``seed_fingerprint.loader`` changed) stores its fingerprint
  only after every section ran clean; a failed section leaves the stored value as it was.
  The boot loader leaves out what the lifespan leader seeds (credential types, system
  settings, packages, the local first-run content); ``migrate`` keeps the full loader.
  Its ``__main__`` flushes and ends with ``os._exit``.
* **The leader stage** (``core/seeds/leader_stage.py`` plus ``main._run_starters``) is
  skipped when ``seed_fingerprint.leader`` matches, and stores it only after its
  sections and the starters pass ran clean. The local first-run seed and the reaper run
  on every boot, under the same lock.
* **Credential types** are written only when the seed changes them, so a second load
  (a second boot that runs the stage) leaves every ``updated_at`` as it was; LinkedIn's
  schema override lives in ``core/seeds/`` and is applied by that upsert.

The real-Postgres tests work in a throwaway schema and skip without a database.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

import core.database.load_seed_data as loader
from core.seeds import fingerprint as fp
from core.seeds import leader_stage
from core.seeds.credential_type_overrides import LINKEDIN_COMMUNITY_MANAGEMENT, with_overrides

_ORCH = Path(__file__).resolve().parents[1]
_MAIN = _ORCH / "main.py"
FINGERPRINT = "f" * 64


# --------------------------------------------------------------------------- the loader stage


@pytest.fixture
def loader_run(monkeypatch):
    """main() with its fingerprint, store and pool stubbed; records the order of events."""
    events = []
    monkeypatch.setattr(loader, "seed_fingerprint", lambda stage: events.append(("fingerprint", stage)) or FINGERPRINT)
    monkeypatch.setattr(loader, "_store_loader_fingerprint", lambda value: events.append(("stored", value)))
    monkeypatch.setattr(loader, "_close_database_pool", lambda: None)

    def run(argv, load):
        def fake_load(**kwargs):
            events.append(("load", kwargs))
            return load()

        monkeypatch.setattr(loader, "load_seed_data", fake_load)
        return loader.main(argv), events

    return run


def test_the_boot_loader_stores_its_fingerprint_after_a_clean_run(loader_run):
    code, events = loader_run(["--boot"], lambda: True)
    assert code == 0
    assert events == [
        ("fingerprint", fp.LOADER),  # taken before any seed runs
        ("load", {"load_credentials": False, "load_platform_defaults": True, "boot": True}),
        ("stored", FINGERPRINT),
    ]


def test_a_section_that_logged_an_error_leaves_the_stored_fingerprint_as_it_was(loader_run):
    """The loader's sections catch their own failure and log it: the run still exits 0."""
    def a_section_fails():
        logging.getLogger("core.database.load_seed_data").exception("Error loading personas")
        return True

    code, events = loader_run(["--boot"], a_section_fails)
    assert code == 0
    assert not [event for event in events if event[0] == "stored"]


def test_a_failed_load_stores_nothing_and_fails(loader_run):
    code, events = loader_run(["--boot"], lambda: False)
    assert code == 1
    assert not [event for event in events if event[0] == "stored"]


def test_migrate_runs_the_full_loader_and_stores_its_fingerprint(loader_run):
    code, events = loader_run([], lambda: True)
    assert code == 0
    assert events[1:] == [("load", {"load_credentials": True, "load_platform_defaults": True, "boot": False}),
                          ("stored", FINGERPRINT)]


def test_a_credentials_only_run_stores_nothing(loader_run):
    code, events = loader_run(["--credentials-only"], lambda: True)
    assert code == 0
    assert not [event for event in events if event[0] == "stored"]


def test_the_boot_loader_leaves_out_what_the_lifespan_leader_seeds():
    left_out = set(loader._ALL_SECTIONS) - set(loader.BOOT_SECTIONS)
    assert left_out == {loader._seed_system_settings, loader._seed_packages, loader._seed_local_edition_first_run}
    assert [s for s in loader._ALL_SECTIONS if s in loader.BOOT_SECTIONS] == list(loader.BOOT_SECTIONS), "order kept"


def test_the_boot_mode_runs_the_boot_sections_and_writes_no_credential_type(monkeypatch):
    from unittest.mock import MagicMock

    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (418,)
    monkeypatch.setattr(loader, "_connect", lambda: connection)
    ran = []
    monkeypatch.setattr(loader, "_load_platform_defaults", lambda sections=loader._ALL_SECTIONS: ran.append(sections))

    assert loader.load_seed_data(load_credentials=False, load_platform_defaults=True, boot=True)
    assert ran == [loader.BOOT_SECTIONS]
    assert not [c for c in cursor.execute.call_args_list if "INSERT INTO credential_types" in c.args[0]]


def test_a_skills_seed_that_only_printed_its_failure_is_logged_as_one(monkeypatch, caplog):
    """seed_skills/seed_patterns print a failure and return; an empty table after them is an error."""
    import core.seeds.seed_skills as seed_skills

    monkeypatch.setattr(seed_skills, "seed_skills", lambda: None)
    monkeypatch.setattr(seed_skills, "seed_patterns", lambda: None)

    def empty():
        raise RuntimeError("still empty after their seeders ran: skills")

    monkeypatch.setattr(loader, "_require_seeded_tables", empty)
    with caplog.at_level(logging.ERROR, logger=loader.__name__):
        loader._seed_skills_and_patterns()
    assert "Error loading skills/patterns" in caplog.text


@pytest.mark.parametrize("counts,empty", [((32, 5), None), ((0, 5), "skills"), ((32, 0), "patterns")])
def test_an_empty_skills_or_patterns_table_after_its_seeder_is_an_error(monkeypatch, counts, empty):
    answers = dict(zip(("skills", "patterns"), counts))

    class _Db:
        def execute(self, statement):
            table = str(statement).rsplit(" ", 1)[-1]
            return type("Result", (), {"scalar": lambda _self: answers[table]})()

    @contextmanager
    def session():
        yield _Db()

    monkeypatch.setattr("core.database.database.get_db_session", session)
    if empty is None:
        loader._require_seeded_tables()
    else:
        with pytest.raises(RuntimeError, match=f"still empty after their seeders ran: {empty}"):
            loader._require_seeded_tables()


def test_the_loader_flushes_and_ends_without_the_interpreter_teardown():
    tree = ast.parse((_ORCH / "core" / "database" / "load_seed_data.py").read_text(encoding="utf-8"))
    block = next(node for node in tree.body if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test))
    statements = [ast.unparse(statement) for statement in block.body]
    assert statements[-1] == "os._exit(status)"
    for flush in ("logging.shutdown()", "sys.stdout.flush()", "sys.stderr.flush()"):
        assert statements.index(flush) < len(statements) - 1, flush


# --------------------------------------------------------------------------- the leader stage


@pytest.fixture
def leader(monkeypatch):
    """The leader stage with its fingerprint, stored value, store and sections stubbed."""
    state = {"stored": None, "writes": [], "ran": []}
    monkeypatch.setattr(leader_stage, "seed_fingerprint", lambda stage: FINGERPRINT)
    monkeypatch.setattr(leader_stage, "_stored", lambda: state["stored"])
    monkeypatch.setattr(leader_stage, "store_fingerprint", lambda db, stage, value: state["writes"].append((stage, value)))

    @contextmanager
    def session():
        yield object()

    monkeypatch.setattr("core.database.database.get_db_session", session)

    def sections(*names_and_failures):
        def section(name, fails):
            def run():
                state["ran"].append(name)
                if fails:
                    raise RuntimeError(f"{name} failed")
            return run

        monkeypatch.setattr(leader_stage, "SECTIONS", tuple((name, section(name, fails))
                                                            for name, fails in names_and_failures))

    state["sections"] = sections
    return state


def test_an_unchanged_fingerprint_skips_the_leaders_seeds(leader, caplog):
    leader["stored"] = FINGERPRINT
    leader["sections"](("System prompts", False))
    with caplog.at_level(logging.INFO, logger=leader_stage.__name__):
        seeds = leader_stage.run_leader_seeds()
    assert (seeds.ran, seeds.clean) == (False, True)
    assert leader["ran"] == []
    assert fp.skipped_line(fp.LEADER, FINGERPRINT) in caplog.text
    assert leader_stage.finish_leader_stage(seeds, starters_clean=True) is False
    assert leader["writes"] == []


def test_a_changed_fingerprint_runs_every_section_and_stores_after_a_clean_stage(leader):
    leader["stored"] = "0" * 64
    leader["sections"](("a", False), ("b", False))
    seeds = leader_stage.run_leader_seeds()
    assert (seeds.ran, seeds.clean, leader["ran"]) == (True, True, ["a", "b"])
    assert leader_stage.finish_leader_stage(seeds, starters_clean=True) is True
    assert leader["writes"] == [(fp.LEADER, FINGERPRINT)]


def test_a_failed_section_leaves_the_leaders_stored_fingerprint_as_it_was(leader):
    leader["sections"](("a", True), ("b", False))
    seeds = leader_stage.run_leader_seeds()
    assert (seeds.ran, seeds.clean, leader["ran"]) == (True, False, ["a", "b"])
    assert leader_stage.finish_leader_stage(seeds, starters_clean=True) is False
    assert leader["writes"] == []


def test_a_failed_starters_pass_leaves_the_leaders_stored_fingerprint_as_it_was(leader):
    leader["sections"](("a", False))
    seeds = leader_stage.run_leader_seeds()
    assert leader_stage.finish_leader_stage(seeds, starters_clean=False) is False
    assert leader["writes"] == []


def test_the_leaders_sections_are_the_old_lifespan_block_in_order():
    assert [name for name, _section in leader_stage.SECTIONS] == [
        "System prompts", "System settings", "Credential types", "Auto persona doctrine backfill",
        "Built-in skills", "Socials package", "Marketplace packages",
    ]


def test_the_starters_run_only_after_the_leader_ran_its_seeds(monkeypatch):
    import main

    calls = []
    monkeypatch.setattr(main, "_seed_document_starters_everywhere", lambda: calls.append("documents") or True)
    monkeypatch.setattr(main, "_seed_social_starters_where_on", lambda: calls.append("socials") or False)
    monkeypatch.setattr(leader_stage, "finish_leader_stage",
                        lambda seeds, clean: calls.append(("finish", seeds.fingerprint, clean)))
    cases = [
        (None, []),  # not the boot leader
        (leader_stage.LeaderSeeds(FINGERPRINT, ran=False, clean=True), []),  # unchanged: skipped
        (leader_stage.LeaderSeeds(FINGERPRINT, ran=True, clean=True),
         ["documents", "socials", ("finish", FINGERPRINT, False)]),
    ]
    for seeds, expected in cases:
        calls.clear()

        asyncio.run(main._run_starters(seeds))
        assert calls == expected, seeds


def _boot_phase_1_core() -> ast.AsyncFunctionDef:
    tree = ast.parse(_MAIN.read_text(encoding="utf-8"))
    return next(node for node in tree.body
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "_boot_phase_1_core")


def test_the_leader_runs_the_gated_seeds_then_the_first_run_seed_and_the_reaper_under_one_lock():
    function = _boot_phase_1_core()
    lock = next(node for node in ast.walk(function) if isinstance(node, ast.With)
                and "boot_leader_lock" in ast.unparse(node.items[0].context_expr))
    body = ast.unparse(lock)
    order = [body.index(call) for call in ("if not is_leader", "run_leader_seeds()",
                                           "seed_local_first_run_on_boot()", "reap_orphaned_runs(db)", "return seeds")]
    assert order == sorted(order)


def test_main_py_no_longer_carries_the_seed_block_or_the_linkedin_update():
    source = _MAIN.read_text(encoding="utf-8")
    for gone in ("seed_system_prompts", "seed_builtin_skills", "seed_packages(", LINKEDIN_COMMUNITY_MANAGEMENT,
                 "UPDATE credential_types"):
        assert gone not in source, gone


# --------------------------------------------------------------------------- the LinkedIn override


def test_the_linkedin_override_is_applied_to_a_new_dict():
    entry = {"name": LINKEDIN_COMMUNITY_MANAGEMENT, "category": "social", "description": "from the seed file",
             "schema_definition": [{"name": "scope"}], "icon": "linkedin"}
    original = json.loads(json.dumps(entry))
    written = with_overrides(entry)
    assert [field["name"] for field in written["schema_definition"]] == [
        "client_id", "client_secret", "access_token", "refresh_token", "organization_urn"]
    assert written["description"] == "LinkedIn Community Management API — posts images to organization pages"
    assert written["icon"] == "linkedin"
    assert entry == original, "the seed file's entry is not changed"
    other = {"name": "openAiApi", "schema_definition": []}
    assert with_overrides(other) == other and with_overrides(other) is not other


# --------------------------------------------------------------------------- real Postgres


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text

    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the seed-stage database tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def schema(engine):
    """A schema of its own, so the shared test database's tables are never touched."""
    from sqlalchemy import text

    name = f"prd262_us031_{uuid.uuid4().hex[:12]}"
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{name}"'))
    yield name
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA "{name}" CASCADE'))


def _create(engine, schema: str, table) -> None:
    """The model's own table, built in ``schema``."""
    from sqlalchemy import MetaData

    metadata = MetaData()
    table.to_metadata(metadata, schema=schema)
    metadata.create_all(engine)


@contextmanager
def _psycopg2_in(schema: str):
    import psycopg2

    from core.database.database import get_database_url

    conn = psycopg2.connect(get_database_url())
    try:
        with conn.cursor() as cursor:
            cursor.execute(f'SET search_path TO "{schema}"')
        conn.commit()
        yield conn
    finally:
        conn.close()


def _stamps(engine, schema: str) -> dict:
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(text(f'SELECT name, created_at, updated_at FROM "{schema}".credential_types')).all()
    return {row.name: (row.created_at, row.updated_at) for row in rows}


@pytest.mark.parametrize("json_type", ["json", "jsonb"])
def test_a_second_credential_type_load_rewrites_no_row(engine, schema, json_type, caplog):
    """Across a warm restart that runs the stage, credential_types.updated_at is unchanged."""
    from sqlalchemy import text

    from core.models.credentials import CredentialType

    _create(engine, schema, CredentialType.__table__)
    if json_type == "jsonb":  # whichever the database was built with, json or jsonb
        with engine.begin() as conn:
            for column in ("schema_definition", "test_endpoint"):
                conn.execute(text(f'ALTER TABLE "{schema}".credential_types ALTER COLUMN {column} '
                                  f"TYPE jsonb USING {column}::jsonb"))
    seed = json.loads(loader._CREDENTIAL_TYPES_FILE.read_text(encoding="utf-8"))
    with _psycopg2_in(schema) as conn:
        loader._load_credential_types(conn)
        first = _stamps(engine, schema)
        with caplog.at_level(logging.INFO, logger=loader.__name__):
            loader._load_credential_types(conn)
    assert len(first) == len(seed)
    assert _stamps(engine, schema) == first
    assert f"Credential types: 0 written, {len(seed)} unchanged, 0 failed" in caplog.text


def test_a_changed_seed_entry_rewrites_only_its_row_and_linkedin_carries_its_override(engine, schema, tmp_path,
                                                                                       monkeypatch):
    from sqlalchemy import text

    from core.models.credentials import CredentialType

    _create(engine, schema, CredentialType.__table__)
    seed = json.loads(loader._CREDENTIAL_TYPES_FILE.read_text(encoding="utf-8"))
    linkedin = next(entry for entry in seed if entry["name"] == LINKEDIN_COMMUNITY_MANAGEMENT)
    other = next(entry for entry in seed if entry["name"] != LINKEDIN_COMMUNITY_MANAGEMENT)
    fixture = tmp_path / "credential_types_seed.json"
    fixture.write_text(json.dumps([linkedin, other]), encoding="utf-8")
    monkeypatch.setattr(loader, "_CREDENTIAL_TYPES_FILE", fixture)
    with _psycopg2_in(schema) as conn:
        loader._load_credential_types(conn)
        first = _stamps(engine, schema)
        fixture.write_text(json.dumps([linkedin, {**other, "display_name": "Renamed"}]), encoding="utf-8")
        loader._load_credential_types(conn)
    second = _stamps(engine, schema)
    assert second[LINKEDIN_COMMUNITY_MANAGEMENT] == first[LINKEDIN_COMMUNITY_MANAGEMENT]
    assert second[other["name"]][1] > first[other["name"]][1]
    with engine.connect() as conn:
        schema_definition = conn.execute(
            text(f'SELECT schema_definition FROM "{schema}".credential_types WHERE name = :name'),
            {"name": LINKEDIN_COMMUNITY_MANAGEMENT},
        ).scalar_one()
    assert schema_definition == with_overrides(linkedin)["schema_definition"]


def test_a_stored_fingerprint_reads_back_and_a_new_one_replaces_it(engine, schema):
    from sqlalchemy import text

    from core.models.system_settings import SystemSetting

    _create(engine, schema, SystemSetting.__table__)
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        assert fp.stored_fingerprint(conn, fp.LOADER) is None
        fp.store_fingerprint(conn, fp.LOADER, "a" * 64)
        fp.store_fingerprint(conn, fp.LOADER, "b" * 64)
        assert fp.stored_fingerprint(conn, fp.LOADER) == "b" * 64
        assert fp.stored_fingerprint(conn, fp.LEADER) is None
        rows = conn.execute(text("SELECT category, key, value_type, is_sensitive, is_required, created_by "
                                 "FROM system_settings")).all()
    assert [tuple(row) for row in rows] == [("boot", "seed_fingerprint.loader", "string", False, False, "system")]
