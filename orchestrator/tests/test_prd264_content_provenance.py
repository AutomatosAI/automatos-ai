"""PRD-264 EUAI-10 — the provenance record and the public detection route.

Anyone holding a file can ask whether Automatos AI made it, with no account and no
upload: the /verify page hashes the file in the browser and asks
``GET /api/public/provenance/{sha256}``. Proven here:

* ``content_provenance`` (model and Alembic revision ``prd264_content_provenance``)
  holds one row per marked file: the hash, kind, manifest id, generator, date and the
  workspace (for tenancy and erasure only). The revision survives a create_all database.
* ``record_marked_file`` (the writer EUAI-09 and EUAI-06b call) validates its input and
  keeps the first row when a workspace records the same file twice.
* The route takes 64 lower-case hex characters only (422 otherwise), answers yes with
  the kind and the date for a known hash and no for an unknown one, never returns
  anything about the workspace, and is rate-limited per IP (429 with Retry-After).
* It is mounted (router manifest) and listed in the route manifest, and is a plain def.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
MIGRATION = ORCH / "alembic" / "versions" / "prd264_content_provenance.py"
ROUTE = "/api/public/provenance/{sha256}"
COLUMNS = ("id", "workspace_id", "sha256", "kind", "manifest_id", "generator", "created_at")
INDEX = "ix_content_provenance_sha256"
UNIQUE = "uq_content_provenance_ws_sha256"
KNOWN = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # sha256("abc")
UNKNOWN = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # sha256("")
GENERATOR = "media-render/prd264-test"
MANIFEST_ID = "urn:c2pa:prd264-test-manifest"
FIRST_MADE = datetime(2026, 10, 19, 23, 30, tzinfo=timezone.utc)
LATER = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- fake Redis


class _FakePipeline:
    def __init__(self, redis):
        self._redis = redis
        self._ops = []

    def __getattr__(self, name):
        def _queue(*args, **kwargs):
            self._ops.append((name, args, kwargs))
            return self

        return _queue

    def execute(self):
        return [getattr(self._redis, name)(*args, **kwargs) for name, args, kwargs in self._ops]


class _FakeRedis:
    """The sorted-set commands RateLimitStore uses, on a dict."""

    def __init__(self):
        self.z = {}

    def zremrangebyscore(self, key, low, high):
        marks = self.z.get(key, {})
        doomed = [m for m, score in marks.items() if float(low) <= score <= float(high)]
        for member in doomed:
            marks.pop(member)
        return len(doomed)

    def zcard(self, key):
        return len(self.z.get(key, {}))

    def zrange(self, key, start, end, withscores=False):
        items = sorted(self.z.get(key, {}).items(), key=lambda kv: kv[1])[start:end + 1]
        return items if withscores else [m for m, _ in items]

    def zadd(self, key, mapping):
        self.z.setdefault(key, {}).update(mapping)
        return len(mapping)

    def expire(self, key, ttl):
        return True

    def pipeline(self):
        return _FakePipeline(self)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def fake_redis(monkeypatch):
    from api import provenance
    from api.widgets.rate_limit import RateLimitStore

    redis = _FakeRedis()
    monkeypatch.setattr(provenance, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    return redis


@pytest.fixture
def client(db_session, fake_redis):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import provenance
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(provenance.router)
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app)


def _mark(db, workspace_id, sha256=KNOWN, kind="video", created_at=FIRST_MADE):
    from core.models.content_provenance import ContentProvenance

    row = ContentProvenance(workspace_id=UUID(str(workspace_id)), sha256=sha256, kind=kind,
                            generator=GENERATOR, manifest_id=MANIFEST_ID, created_at=created_at)
    db.add(row)
    db.flush()
    return row


# --------------------------------------------------------------------------- table


def test_the_model_and_the_migration_declare_one_shape():
    from sqlalchemy import CheckConstraint, UniqueConstraint

    from core.models import content_provenance as model

    table = model.ContentProvenance.__table__
    assert tuple(column.name for column in table.columns) == COLUMNS
    assert {index.name: [c.name for c in index.columns] for index in table.indexes} == {INDEX: ["sha256"]}
    (unique,) = [c for c in table.constraints if isinstance(c, UniqueConstraint)]
    assert (unique.name, [c.name for c in unique.columns]) == (UNIQUE, ["workspace_id", "sha256"])
    (fk,) = table.c.workspace_id.foreign_keys
    assert (fk.target_fullname, fk.ondelete) == ("workspaces.id", "CASCADE")
    checks = {c.name: str(c.sqltext) for c in table.constraints if isinstance(c, CheckConstraint)}
    assert set(re.findall(r"'([a-z]+)'", checks["ck_content_provenance_kind"])) == set(model.PROVENANCE_KINDS)
    assert checks["ck_content_provenance_sha256"] == "sha256 ~ '^[0-9a-f]{64}$'"

    src = MIGRATION.read_text()
    assert 'revision = "prd264_content_provenance"' in src
    assert 'down_revision = "auto_haiku_default_20261019"' in src
    (ddl,) = re.findall(r"CREATE TABLE IF NOT EXISTS content_provenance \((.*?)\n\s*\)\n", src, re.S)
    lines = [line.strip() for line in ddl.strip().splitlines()]
    assert tuple(line.split()[0] for line in lines if not line.startswith("CONSTRAINT")) == COLUMNS
    assert "REFERENCES workspaces(id) ON DELETE CASCADE" in ddl
    assert f"CONSTRAINT {UNIQUE} UNIQUE (workspace_id, sha256)" in ddl
    assert "CHECK (kind IN ('video', 'image', 'audio', 'document'))" in ddl
    assert "CHECK (sha256 ~ '^[0-9a-f]{64}$')" in ddl
    assert f"CREATE INDEX IF NOT EXISTS {INDEX} " in src and "ON content_provenance (sha256)" in src
    assert f"DROP INDEX IF EXISTS {INDEX};" in src and "DROP TABLE IF EXISTS content_provenance;" in src


def test_the_table_holds_no_prompt_user_or_file_name():
    from core.models.content_provenance import ContentProvenance

    names = set(ContentProvenance.__table__.columns.keys())
    for forbidden in ("prompt", "user_id", "file_name", "filename", "path", "email"):
        assert forbidden not in names


def _upgrade_statements():
    spec = importlib.util.spec_from_file_location("prd264_content_provenance_under_test", MIGRATION)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    statements = []
    migration.op = SimpleNamespace(execute=statements.append)
    migration.upgrade()
    return statements


def _index_def(db):
    return db.execute(text("SELECT indexdef FROM pg_indexes WHERE tablename = 'content_provenance' "
                           "AND indexname = :name"), {"name": INDEX}).scalar()


def test_the_upgrade_keeps_a_create_all_table_and_adds_only_its_missing_index(db_session):
    """create_all built the table; the upgrade runs over it and puts back a missing index.
    Everything runs inside db_session's transaction and is rolled back."""
    assert "(sha256)" in _index_def(db_session)
    db_session.execute(text(f"DROP INDEX {INDEX}"))
    assert _index_def(db_session) is None

    for statement in _upgrade_statements():
        db_session.execute(text(statement))

    assert "(sha256)" in _index_def(db_session)
    columns = db_session.execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'content_provenance' "
        "ORDER BY ordinal_position")).scalars().all()
    assert tuple(columns) == COLUMNS


@pytest.mark.parametrize("bad, check", [
    ({"kind": "text"}, "ck_content_provenance_kind"),
    ({"sha256": KNOWN.upper()}, "ck_content_provenance_sha256"),
])
def test_the_table_refuses_a_kind_or_a_hash_the_code_never_writes(db_session, seed_workspace, bad, check):
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError, match=check):
        _mark(db_session, seed_workspace(), **bad)


# --------------------------------------------------------------------------- writer and answer


def test_the_hash_validator_takes_64_lower_case_hex_characters_only():
    from services.content_provenance import is_sha256_hex

    assert is_sha256_hex(KNOWN) is True
    for bad in (KNOWN.upper(), KNOWN[1:], KNOWN + "0", KNOWN[1:] + "g", KNOWN + "\n", " " + KNOWN[1:],
                "", None, 12, b"a" * 64, "../" + KNOWN[3:]):
        assert is_sha256_hex(bad) is False, bad


def test_the_writer_records_a_file_once_per_workspace(db_session, seed_workspace):
    from core.models.content_provenance import ContentProvenance
    from services.content_provenance import record_marked_file

    ws = seed_workspace()
    assert record_marked_file(db_session, ws, KNOWN, "image", GENERATOR, manifest_id=MANIFEST_ID) is True
    assert record_marked_file(db_session, UUID(ws), KNOWN, "video", "another-generator") is False

    (row,) = db_session.query(ContentProvenance).filter(ContentProvenance.workspace_id == UUID(ws)).all()
    assert (row.sha256, row.kind, row.generator, row.manifest_id) == (KNOWN, "image", GENERATOR, MANIFEST_ID)
    assert row.created_at is not None


@pytest.mark.parametrize("bad", [
    {"sha256": KNOWN.upper()},
    {"sha256": "abc"},
    {"kind": "text"},
    {"generator": "  "},
    {"generator": "g" * 121},
    {"manifest_id": ""},
    {"manifest_id": "m" * 256},
    {"workspace_id": "not-a-workspace"},
])
def test_the_writer_refuses_bad_input(db_session, seed_workspace, bad):
    from services.content_provenance import record_marked_file

    args = {"workspace_id": seed_workspace(), "sha256": KNOWN, "kind": "audio", "generator": GENERATOR,
            "manifest_id": None, **bad}
    with pytest.raises(ValueError):
        record_marked_file(db_session, **args)


def test_the_answer_is_the_first_mark_and_holds_nothing_about_the_workspace(db_session, seed_workspace):
    from services.content_provenance import provenance_answer

    first_ws, later_ws = seed_workspace(), seed_workspace()
    _mark(db_session, later_ws, kind="image", created_at=LATER)
    _mark(db_session, first_ws, kind="video", created_at=FIRST_MADE)

    answer = provenance_answer(db_session, KNOWN)
    assert answer == {"made_by_automatos_ai": True, "kind": "video", "date": "2026-10-19"}
    assert provenance_answer(db_session, UNKNOWN) == {"made_by_automatos_ai": False}


# --------------------------------------------------------------------------- the route


def test_a_known_hash_says_yes_with_kind_and_date_and_nothing_else(client, db_session, seed_workspace):
    ws = seed_workspace()
    _mark(db_session, ws)

    response = client.get(f"/api/public/provenance/{KNOWN}")

    assert response.status_code == 200
    assert response.json() == {"made_by_automatos_ai": True, "kind": "video", "date": "2026-10-19"}
    for private in (ws, GENERATOR, MANIFEST_ID, "workspace"):
        assert private not in response.text


def test_an_unknown_hash_says_no(client):
    response = client.get(f"/api/public/provenance/{UNKNOWN}")

    assert response.status_code == 200
    assert response.json() == {"made_by_automatos_ai": False}


@pytest.mark.parametrize("bad", [KNOWN.upper(), KNOWN[1:], KNOWN + "0", KNOWN[1:] + "z", KNOWN + "%0A",
                                 "1%20OR%201=1", "%2E%2E%2Fworkspaces"])
def test_anything_but_a_lower_case_sha256_is_refused(client, bad):
    response = client.get(f"/api/public/provenance/{bad}")

    assert response.status_code in (404, 422)  # an encoded "/" never reaches the handler
    assert "made_by_automatos_ai" not in response.text


def test_a_malformed_hash_is_refused_with_a_clear_422(client):
    from api.provenance import BAD_HASH_DETAIL

    response = client.get(f"/api/public/provenance/{KNOWN.upper()}")

    assert response.status_code == 422
    assert response.json() == {"detail": BAD_HASH_DETAIL}


def test_each_ip_gets_its_checks_a_minute_then_429(client, monkeypatch):
    from config import config

    monkeypatch.setattr(config, "PROVENANCE_RATE_PER_MINUTE", 2)
    first_ip = {"X-Forwarded-For": "203.0.113.7"}

    assert client.get(f"/api/public/provenance/{UNKNOWN}", headers=first_ip).status_code == 200
    # A malformed request counts too: the limit runs before the hash is read.
    assert client.get("/api/public/provenance/nope", headers=first_ip).status_code == 422
    limited = client.get(f"/api/public/provenance/{UNKNOWN}", headers=first_ip)

    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1
    assert "made_by_automatos_ai" not in limited.text
    other_ip = client.get(f"/api/public/provenance/{UNKNOWN}", headers={"X-Forwarded-For": "198.51.100.4"})
    assert other_ip.status_code == 200


def test_the_window_lets_an_ip_check_again_once_it_has_passed(client, fake_redis, monkeypatch):
    from config import config

    monkeypatch.setattr(config, "PROVENANCE_RATE_PER_MINUTE", 1)
    ip = {"X-Forwarded-For": "192.0.2.9"}
    assert client.get(f"/api/public/provenance/{UNKNOWN}", headers=ip).status_code == 200
    assert client.get(f"/api/public/provenance/{UNKNOWN}", headers=ip).status_code == 429

    stale = time.time() - 120
    for marks in fake_redis.z.values():
        for member in list(marks):
            marks[member] = stale

    assert client.get(f"/api/public/provenance/{UNKNOWN}", headers=ip).status_code == 200


def _bucket_keys(fake_redis):
    return sorted(key for key in fake_redis.z if ":provenance:ip:" in key)


def test_a_client_ip_that_is_not_an_address_never_becomes_a_redis_key(client, fake_redis):
    from api.provenance import UNPARSABLE_IP_BUCKET

    for junk in ("not-an-ip", "x" * 4000, "203.0.113.7:8080", "<script>"):
        response = client.get(f"/api/public/provenance/{UNKNOWN}", headers={"X-Forwarded-For": junk})
        assert response.status_code == 200

    keys = _bucket_keys(fake_redis)
    assert len(keys) == 1
    assert keys[0].endswith(UNPARSABLE_IP_BUCKET)


def test_an_ipv6_caller_is_counted_per_64_so_rotating_addresses_inside_it_gains_nothing(
    client, fake_redis, monkeypatch,
):
    from config import config

    monkeypatch.setattr(config, "PROVENANCE_RATE_PER_MINUTE", 1)
    first = {"X-Forwarded-For": "2001:db8:1:2::1"}
    same_64 = {"X-Forwarded-For": "2001:db8:1:2:ffff:ffff:ffff:fffe"}
    other_64 = {"X-Forwarded-For": "2001:db8:1:3::1"}

    assert client.get(f"/api/public/provenance/{UNKNOWN}", headers=first).status_code == 200
    assert client.get(f"/api/public/provenance/{UNKNOWN}", headers=same_64).status_code == 429
    assert client.get(f"/api/public/provenance/{UNKNOWN}", headers=other_64).status_code == 200
    assert [key.rsplit(":ip:", 1)[1] for key in _bucket_keys(fake_redis)] == [
        "2001:db8:1:2::/64", "2001:db8:1:3::/64",
    ]


def test_an_ipv4_mapped_ipv6_address_shares_its_ipv4_bucket(client, monkeypatch):
    from config import config

    monkeypatch.setattr(config, "PROVENANCE_RATE_PER_MINUTE", 1)

    plain = client.get(f"/api/public/provenance/{UNKNOWN}", headers={"X-Forwarded-For": "203.0.113.9"})
    mapped = client.get(f"/api/public/provenance/{UNKNOWN}", headers={"X-Forwarded-For": "::ffff:203.0.113.9"})

    assert plain.status_code == 200
    assert mapped.status_code == 429


def test_the_route_needs_no_account_and_ignores_one_it_is_sent(client):
    headers = {"Authorization": "Bearer not-a-real-token", "X-Workspace-ID": str(uuid4())}

    response = client.get(f"/api/public/provenance/{UNKNOWN}", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"made_by_automatos_ai": False}


def test_the_route_is_mounted_listed_and_runs_off_the_event_loop():
    from api import provenance
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    assert RouterSpec("api.provenance") in MANIFEST_ROUTERS
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert {"method": "GET", "path": ROUTE} in manifest["routes"]
    assert [route.path for route in provenance.router.routes] == [ROUTE]
    assert not inspect.iscoroutinefunction(provenance.check_provenance)
    assert not inspect.iscoroutinefunction(provenance.limit_checks_per_ip)


def test_the_rate_setting_is_named_in_the_config_surface():
    from config import config

    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))
    assert "PROVENANCE_RATE_PER_MINUTE" in surface["settings"]
    assert config.PROVENANCE_RATE_PER_MINUTE >= 1


# --------------------------------------------------------------------------- the hash stays private (FR-17)


def test_every_answer_tells_browsers_and_proxies_not_to_keep_it(client, db_session, seed_workspace, monkeypatch):
    from config import config

    _mark(db_session, seed_workspace())
    monkeypatch.setattr(config, "PROVENANCE_RATE_PER_MINUTE", 3)
    ip = {"X-Forwarded-For": "203.0.113.21"}

    answers = [client.get(f"/api/public/provenance/{digest}", headers=ip) for digest in (KNOWN, UNKNOWN, "nope")]
    limited = client.get(f"/api/public/provenance/{KNOWN}", headers=ip)

    assert [a.status_code for a in answers] == [200, 200, 422]
    assert limited.status_code == 429
    for response in (*answers, limited):
        assert response.headers["Cache-Control"] == "no-store"


def test_a_redis_outage_lets_the_check_through_and_names_this_limiter(client, monkeypatch, caplog):
    import logging

    from api import provenance
    from api.widgets.rate_limit import RateLimitStore

    monkeypatch.setattr(provenance, "_rate_store", RateLimitStore(redis_factory=lambda: None))

    with caplog.at_level(logging.ERROR, logger="api.widgets.rate_limit"):
        response = client.get(f"/api/public/provenance/{UNKNOWN}", headers={"X-Forwarded-For": "203.0.113.22"})

    assert response.status_code == 200   # fail-open is unchanged
    (line,) = [r.getMessage() for r in caplog.records if "FAIL-OPEN" in r.getMessage()]
    assert line.startswith(f"{provenance.RATE_LIMITER_LABEL} FAIL-OPEN")
    assert UNKNOWN not in line


def test_another_limiter_s_fail_open_line_is_unchanged(caplog):
    import logging

    from api.widgets.rate_limit import DEFAULT_LIMITER_LABEL, RateLimitStore

    with caplog.at_level(logging.ERROR, logger="api.widgets.rate_limit"):
        allowed = RateLimitStore(redis_factory=lambda: None).check("ip:192.0.2.1", 5)[0]

    assert allowed is True
    assert DEFAULT_LIMITER_LABEL == "widget rate limiter"
    assert any(r.getMessage().startswith("widget rate limiter FAIL-OPEN") for r in caplog.records)


def test_a_database_error_answers_503_and_logs_no_hash(client, monkeypatch, caplog):
    import logging

    from sqlalchemy.exc import OperationalError

    from api import provenance

    def _down(_db, sha256):
        raise OperationalError("SELECT ... WHERE sha256 = %(sha256_1)s", {"sha256_1": sha256},
                               Exception("server closed the connection"))

    monkeypatch.setattr(provenance, "provenance_answer", _down)

    with caplog.at_level(logging.DEBUG):
        response = client.get(f"/api/public/provenance/{KNOWN}")

    assert response.status_code == 503
    assert response.json() == {"detail": provenance.LOOKUP_FAILED_DETAIL}
    assert response.headers["Cache-Control"] == "no-store"
    assert any("OperationalError" in r.getMessage() for r in caplog.records)
    # The test's own HTTP client logs its request URL; only the server side counts here.
    server_side = [r for r in caplog.records if not r.name.startswith(("httpx", "httpcore"))]
    assert not [r for r in server_side if KNOWN in r.getMessage() or (r.exc_text and KNOWN in r.exc_text)]
