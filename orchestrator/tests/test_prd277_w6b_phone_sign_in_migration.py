"""PRD-277 Wave 6 part 2 (US-603): prd277_w6b_phone_sign_in_20261121, the part's one migration.

* The revision chains onto ``prd277_w4b_step_up_20261120`` and is the one head; both head pins name it.
* Its upgrade widens ``ck_mobile_devices_kind`` to allow ``oidc`` and lets an ``oidc`` row hold a code and token
  hash (``ck_mobile_devices_clerk_holds_no_token``), each dropped only if it exists, added back ``NOT VALID`` and
  then validated; it adds four nullable columns to ``mobile_devices`` and four to ``oidc_pending_sign_ins``,
  each only when missing. It writes no row, touches no other table and sets no ``lock_timeout`` (no revision
  does). The model declares the same names and sizes.
* On Postgres, inside a transaction rolled back at the end: over the tables ``create_all`` built, the upgrade
  run twice changes nothing and leaves both checks validated; the downgrade drops the eight columns, keeps an
  ``oidc`` row already there and refuses a new one; the upgrade from there gives back exactly what
  ``create_all`` built.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

import core.models  # noqa: F401  (registers every table on Base)
from core.models.mobile_devices import MobileDevice
from core.models.oidc_pending_sign_ins import OidcPendingSignIn

ORCH = Path(__file__).resolve().parents[1]
VERSIONS = ORCH / "alembic" / "versions"
REVISION, PARENT = "prd277_w6b_phone_sign_in_20261121", "prd277_w4b_step_up_20261120"
MIGRATION = VERSIONS / f"{REVISION}.py"
TABLES = ("mobile_devices", "oidc_pending_sign_ins")
DEVICE_COLUMNS = {"sign_in_challenge": 43, "sign_in_install_id": 128, "sign_in_app_link": 16,
                  "sign_in_confirm_hash": 64}
PENDING_COLUMNS = {"phone_app_link": 16, "phone_code_challenge": 43, "phone_install_id": 128, "phone_platform": 16}
CHECKS = ("ck_mobile_devices_kind", "ck_mobile_devices_clerk_holds_no_token")
ROW_WRITES = re.compile(r"\b(DELETE FROM|TRUNCATE|UPDATE \w+ SET|INSERT INTO)\b")


def _assigned() -> dict:
    found = {}
    for node in ast.parse(MIGRATION.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return found


def _statements(step: str) -> list:
    spec = importlib.util.spec_from_file_location(f"{MIGRATION.stem}_{uuid.uuid4().hex[:8]}", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    statements: list = []
    module.op = SimpleNamespace(execute=statements.append)
    getattr(module, step)()
    return statements


def _parents(down) -> list:
    return list(down) if isinstance(down, (list, tuple)) else [down] if down else []


def _revisions() -> dict:
    revisions = {}
    for path in VERSIONS.glob("*.py"):
        values = {}
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ("revision", "down_revision"):
                    values[node.targets[0].id] = ast.literal_eval(node.value)
        if isinstance(values.get("revision"), str):
            revisions[values["revision"]] = values.get("down_revision")
    return revisions


# --------------------------------------------------------------------------- the file


def test_the_revision_chains_onto_wave_4_part_2_and_is_the_one_head():
    assert (_assigned()["revision"], _assigned()["down_revision"]) == (REVISION, PARENT)
    revisions = _revisions()
    assert [rev for rev, down in revisions.items() if PARENT in _parents(down)] == [REVISION]
    heads = set(revisions) - {parent for down in revisions.values() for parent in _parents(down)}
    assert heads == {REVISION}
    single_head = (ORCH / "tests" / "test_prd209_alembic_single_head.py").read_text(encoding="utf-8")
    assert f'EXPECTED_HEAD = "{REVISION}"' in single_head
    chain = (ORCH / "tests" / "test_prd236_w1_routes.py").read_text(encoding="utf-8")
    assert f'(versions / "{REVISION}.py")' in chain and f'down_revision = "{PARENT}"' in chain


def test_the_upgrade_widens_two_checks_adds_eight_columns_and_writes_no_row():
    upgrade = _statements("upgrade")
    assert not any(ROW_WRITES.search(statement) for statement in upgrade)
    assert all(re.match(rf"ALTER TABLE ({'|'.join(TABLES)}) ", statement) for statement in upgrade)
    assert "CHECK (kind IN ('paired', 'clerk', 'oidc')) NOT VALID;" in " ".join(upgrade)
    for name in CHECKS:
        dropped = upgrade.index(f"ALTER TABLE mobile_devices DROP CONSTRAINT IF EXISTS {name};")
        added_back = upgrade[dropped + 1]
        assert added_back.startswith(f"ALTER TABLE mobile_devices ADD CONSTRAINT {name} CHECK (")
        assert added_back.endswith(") NOT VALID;")
        assert upgrade[dropped + 2] == f"ALTER TABLE mobile_devices VALIDATE CONSTRAINT {name};"
    added = {match[2]: (match[1], int(match[3])) for match in
             (re.match(r"ALTER TABLE (\w+) ADD COLUMN IF NOT EXISTS (\w+) VARCHAR\((\d+)\);$", s) for s in upgrade)
             if match}
    assert added == {**{name: ("mobile_devices", size) for name, size in DEVICE_COLUMNS.items()},
                     **{name: ("oidc_pending_sign_ins", size) for name, size in PENDING_COLUMNS.items()}}
    assert len(upgrade) == 3 * len(CHECKS) + len(added)
    assert not any("NOT NULL" in statement or "DEFAULT" in statement for statement in upgrade)
    assert not any("lock_timeout" in statement for statement in upgrade)


def test_the_downgrade_drops_the_eight_columns_and_puts_the_narrow_checks_back_not_valid():
    downgrade = _statements("downgrade")
    dropped = {re.match(r"ALTER TABLE \w+ DROP COLUMN IF EXISTS (\w+);$", s)[1] for s in downgrade
               if "DROP COLUMN" in s}
    assert dropped == set(DEVICE_COLUMNS) | set(PENDING_COLUMNS)
    restored = [s for s in downgrade if " ADD CONSTRAINT " in s]
    assert len(restored) == 2 and all(s.endswith(" NOT VALID;") for s in restored)
    assert "'oidc'" not in " ".join(restored)
    assert not any(ROW_WRITES.search(statement) for statement in downgrade)


def test_the_models_declare_the_same_columns():
    for model, columns in ((MobileDevice, DEVICE_COLUMNS), (OidcPendingSignIn, PENDING_COLUMNS)):
        for name, size in columns.items():
            column = model.__table__.c[name]
            assert column.nullable and column.type.length == size and not column.foreign_keys, name


# --------------------------------------------------------------------------- Postgres (CI)


def _run(db, step: str) -> None:
    for statement in _statements(step):
        db.execute(text(statement))


def _schema(db) -> tuple:
    """Both tables' columns, by name, and the two checks' definitions."""
    columns = db.execute(text(
        "SELECT table_name, column_name, data_type, character_maximum_length, is_nullable, column_default "
        "FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = ANY(:tables) "
        "ORDER BY table_name, column_name"), {"tables": list(TABLES)}).all()
    checks = db.execute(text(
        "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = ANY(:names) ORDER BY conname"
    ), {"names": list(CHECKS)}).all()
    return tuple(tuple(row) for row in columns), tuple(tuple(row) for row in checks)


def _seed_user(db) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
                      {"u": f"prd277-w6b-{tag}", "e": f"prd277-w6b-{tag}@example.test"}).scalar()


def _oidc_row(db, workspace_id, user_id, *, with_token: bool = True) -> None:
    db.execute(text("INSERT INTO mobile_devices (id, workspace_id, user_id, kind, status, token_hash) "
                    "VALUES (:id, CAST(:ws AS uuid), :u, 'oidc', 'active', :t)"),
               {"id": uuid.uuid4(), "ws": workspace_id, "u": user_id,
                "t": uuid.uuid4().hex * 2 if with_token else None})


@pytest.fixture
def tables(db_session, seed_workspace):
    """The two tables as the models built them, a person and a workspace; rolled back with ``db_session``."""
    db_session.execute(text("SET LOCAL lock_timeout = '5s'"))
    return SimpleNamespace(db=db_session, workspace_id=seed_workspace(), user_id=_seed_user(db_session))


def test_over_create_all_the_upgrade_changes_nothing_twice(tables):
    by_model = _schema(tables.db)
    assert len(by_model[1]) == len(CHECKS)
    _run(tables.db, "upgrade")
    _run(tables.db, "upgrade")
    assert _schema(tables.db) == by_model
    validated = tables.db.execute(text("SELECT bool_and(convalidated) FROM pg_constraint WHERE conname = ANY(:names)"),
                                  {"names": list(CHECKS)}).scalar_one()
    assert validated is True
    _oidc_row(tables.db, tables.workspace_id, tables.user_id)


def test_the_downgrade_keeps_an_oidc_row_refuses_a_new_one_and_the_upgrade_gives_back_the_model(tables):
    from sqlalchemy.exc import IntegrityError

    db = tables.db
    by_model = _schema(db)
    _oidc_row(db, tables.workspace_id, tables.user_id)
    _run(db, "downgrade")

    columns = {row[1] for row in _schema(db)[0]}
    assert not (set(DEVICE_COLUMNS) | set(PENDING_COLUMNS)) & columns
    assert db.execute(text("SELECT count(*) FROM mobile_devices WHERE user_id = :u AND kind = 'oidc'"),
                      {"u": tables.user_id}).scalar() == 1
    with pytest.raises(IntegrityError, match="ck_mobile_devices_kind"):
        with db.begin_nested():
            _oidc_row(db, tables.workspace_id, tables.user_id, with_token=False)

    _run(db, "upgrade")  # the kept row holds under the wider checks
    assert _schema(db) == by_model
