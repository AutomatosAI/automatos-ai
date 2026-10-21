"""PRD-261 C-05g — the hosted revocation feed, ``GET /api/themes/revocations`` (schema 1), and its parser.

The real router (``api/themes_revocations_feed.py``) runs on a mini FastAPI app over in-memory
SQLite holding ``theme_packages`` and ``theme_revocations``; no request-context override is
installed, so a 200 is an unauthenticated one. Pins (plan §2.2.7, §2.6 C-05g):

* the hosted edition answers 200 with no credentials, ``{schema: 1, generatedAt, revocations,
  revokedKeys}``; the local edition answers 404; the route is ``/api/themes/revocations``, never
  under ``/api/marketplace/themes``;
* only revocations covering a listed Marketplace package go out: a private upload's, a pending or
  rejected submission's and a pulled (``origin`` 'feed') row never appear;
* a row lifted within ``THEME_REVOCATIONS_LIFTED_DAYS`` appears with ``liftedAt``, an older one doesn't;
* ``Cache-Control: public, max-age=300``, a weak ETag that ignores ``generatedAt``, ``If-None-Match`` → 304;
* ``revokedKeys`` are the keys file's revoked keys; a keys file that doesn't load is a 503;
* a built feed a local edition would refuse for its size (over ``THEME_REVOCATIONS_FEED_MAX_KB``, or
  over ``FEED_MAX_ROWS`` rows) is a 503 with a plain message and an error log, never a 200 or a 304;
* ``parse_feed`` round-trips ``build_feed`` and refuses the whole feed for a wrong schema, oversize,
  more than 10,000 rows, a bad digest, id or range, a reason that isn't one line of plain text (over
  2000 characters, empty, a line break, a bidi override or another control character); an unknown
  level parses as ``disable``; a hosted reason with line breaks or a bidi override goes out on one
  line, so its row stays in the feed.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.themes_revocations_feed as feed_api  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
from config import config  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import ThemePackage, ThemeRevocation  # noqa: E402
from services.themes import revocation_feed, signature_verify  # noqa: E402
from services.themes.revocation_feed import (  # noqa: E402
    FeedError,
    build_feed,
    feed_bytes,
    feed_etag,
    iso_time,
    parse_feed,
)

FEED = "/api/themes/revocations"
TABLES = [m.__table__ for m in (ThemePackage, ThemeRevocation)]
VECTOR = json.loads((_ORCH / "tests" / "fixtures" / "themes" / "signing-vector.json").read_text(encoding="utf-8"))
REVOKED_KEY, ACTIVE_KEY = VECTOR["ed25519"]["cases"][:2]
DROP = object()  # a field the wire helpers leave out
UPPER_UUID = "ABCDEF00-0000-4000-8000-0000000000AB"


@pytest.fixture
def api(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    session = sessionmaker(bind=engine)()
    app = FastAPI()
    app.include_router(feed_api.router)
    app.dependency_overrides[get_db] = lambda: session
    try:
        yield SimpleNamespace(db=session, client=TestClient(app))
    finally:
        session.close()
        engine.dispose()


def _package(db, theme_id="acme/ink", version="1.0.0", owner="marketplace", status="approved"):
    workspace_id = uuid.uuid4() if owner == "workspace" else None
    row = ThemePackage(
        theme_id=theme_id, version=version, owner_type=owner, workspace_id=workspace_id,
        digest=uuid.uuid4().hex * 2, manifest={"id": theme_id, "version": version},
        file_manifest={}, storage_prefix=f"themes/{theme_id}/{version}/", status=status,
        reviewed=owner == "marketplace" and status in ("approved", "deactivated"),
    )
    db.add(row)
    db.commit()
    return row


def _revoke(db, minutes_ago=0, lifted_days_ago=None, **fields):
    now = datetime.now(timezone.utc)
    lifted = None if lifted_days_ago is None else now - timedelta(days=lifted_days_ago)
    values = {"level": "disable", "reason": "leaks data", **fields}
    row = ThemeRevocation(created_at=now - timedelta(minutes=minutes_ago), lifted_at=lifted, **values)
    db.add(row)
    db.commit()
    return row


def _ids(api):
    resp = api.client.get(FEED)
    assert resp.status_code == 200, resp.text
    return {row["id"] for row in resp.json()["revocations"]}


def _calls(dependant):
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def test_the_hosted_edition_serves_the_feed_with_no_credentials(api):
    package = _package(api.db)
    revocation = _revoke(api.db, digest=package.digest)
    resp = api.client.get(FEED)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["schema"], body["revokedKeys"]) == (1, [])
    assert datetime.fromisoformat(body["generatedAt"]).tzinfo is not None
    [row] = body["revocations"]
    assert set(row) == {"id", "digest", "level", "reason", "createdAt", "liftedAt"}
    assert (row["id"], row["digest"], row["level"], row["reason"], row["liftedAt"]) == (
        str(revocation.id), package.digest, "disable", "leaks data", None,
    )
    assert row["createdAt"] == iso_time(revocation.created_at)


def test_the_route_is_public_plain_and_not_under_the_marketplace():
    [route] = feed_api.router.routes
    assert isinstance(route, APIRoute)
    assert (route.methods, route.path) == ({"GET"}, FEED)
    assert not route.path.startswith("/api/marketplace/themes")
    assert not inspect.iscoroutinefunction(route.endpoint)
    assert _calls(route.dependant) == [get_db]  # no request context, no workspace permission


def test_the_local_edition_answers_404(api, monkeypatch):
    _revoke(api.db, digest=_package(api.db).digest)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    assert api.client.get(FEED).status_code == 404


def test_a_private_uploads_revocation_never_appears(api):
    listed = _package(api.db)
    gone = _package(api.db, theme_id="acme/old", status="deactivated")
    private = _package(api.db, version="2.0.0", owner="workspace")
    secret = _package(api.db, theme_id="ws/secret", owner="workspace")
    pending = _package(api.db, theme_id="acme/new", status="pending")
    _package(api.db, theme_id="acme/bad", status="rejected")
    shown = [
        _revoke(api.db, digest=listed.digest),
        _revoke(api.db, digest=gone.digest),
        _revoke(api.db, theme_id="acme/ink"),
        _revoke(api.db, theme_id="acme/ink", version_range="1.0.0"),
    ]
    for hidden in ({"digest": private.digest}, {"digest": secret.digest}, {"theme_id": "ws/secret"},
                   {"theme_id": "acme/ink", "version_range": "^2.0.0"}, {"digest": pending.digest},
                   {"theme_id": "acme/new"}, {"theme_id": "acme/bad"},
                   {"digest": listed.digest, "origin": "feed"}):
        _revoke(api.db, **hidden)
    assert _ids(api) == {str(row.id) for row in shown}


def test_lifted_rows_appear_within_the_window(api, monkeypatch):
    digest = _package(api.db).digest
    live = _revoke(api.db, digest=digest)
    recent = _revoke(api.db, digest=digest, lifted_days_ago=10)
    _revoke(api.db, digest=digest, lifted_days_ago=400)
    rows = {row["id"]: row for row in api.client.get(FEED).json()["revocations"]}
    assert set(rows) == {str(live.id), str(recent.id)}
    assert rows[str(live.id)]["liftedAt"] is None
    assert rows[str(recent.id)]["liftedAt"] == iso_time(recent.lifted_at)
    monkeypatch.setattr(config, revocation_feed.LIFTED_DAYS_SETTING, 5, raising=False)
    assert _ids(api) == {str(live.id)}


def test_cache_control_a_weak_etag_and_304(api):
    digest = _package(api.db).digest
    _revoke(api.db, digest=digest)
    first = api.client.get(FEED)
    etag = first.headers["etag"]
    assert first.headers["cache-control"] == "public, max-age=300" and etag.startswith('W/"')
    for sent in (etag, f'"other", {etag.removeprefix("W/")}', "*"):
        again = api.client.get(FEED, headers={"If-None-Match": sent})
        assert (again.status_code, again.content, again.headers["etag"]) == (304, b"", etag)
        assert again.headers["cache-control"] == "public, max-age=300"
    feed = first.json()
    assert feed_etag({**feed, "generatedAt": "2030-01-01T00:00:00+00:00"}) == etag
    _revoke(api.db, digest=digest, level="warn")
    changed = api.client.get(FEED, headers={"If-None-Match": etag})
    assert changed.status_code == 200 and changed.headers["etag"] != etag


def test_revoked_keys_come_from_the_keys_file(api, monkeypatch, tmp_path):
    keys = [{"keyId": case["keyId"], "spki": case["spki"], "status": status, "added": "2026-10-20"}
            for case, status in ((REVOKED_KEY, "revoked"), (ACTIVE_KEY, "active"))]
    path = tmp_path / "signing-keys.json"
    path.write_text(json.dumps({"schema": 1, "keys": keys}), encoding="utf-8")
    monkeypatch.setattr(signature_verify, "KEYS_PATH", path)
    assert api.client.get(FEED).json()["revokedKeys"] == [REVOKED_KEY["keyId"]]
    path.write_text("not json", encoding="utf-8")
    broken = api.client.get(FEED)
    assert broken.status_code == 503
    assert broken.json()["detail"] == feed_api.FEED_DOWN


def test_a_hosted_reason_goes_out_on_one_line_and_its_row_stays_in_the_feed(api):
    # review.deactivate stores a reason with its line breaks; the parser refuses one with them.
    package = _package(api.db)
    row = _revoke(api.db, digest=package.digest, reason="Reported twice:\n  sends chat off-site\u202e\r\nfixed in 1.1")
    blank = _revoke(api.db, digest=_package(api.db, theme_id="acme/other").digest, reason="\u202e\u202c")
    wire = {item["id"]: item["reason"] for item in api.client.get(FEED).json()["revocations"]}
    assert wire == {
        str(row.id): "Reported twice: sends chat off-site fixed in 1.1",
        str(blank.id): revocation_feed.FEED_REASON_FALLBACK,
    }
    parsed = {str(item.id): item.reason for item in parse_feed(feed_bytes(build_feed(api.db))).revocations}
    assert parsed == wire


def test_a_feed_too_large_for_a_local_edition_is_503_and_logged(api, monkeypatch, caplog):
    _revoke(api.db, digest=_package(api.db).digest, reason="r" * 1500)  # one row, about 1.8 KB of feed
    etag = feed_etag(build_feed(api.db))
    monkeypatch.setattr(config, revocation_feed.FEED_MAX_KB_SETTING, 1, raising=False)
    with caplog.at_level(logging.ERROR, logger=feed_api.logger.name):
        refused = api.client.get(FEED)
        unconfirmed = api.client.get(FEED, headers={"If-None-Match": etag})
    for resp in (refused, unconfirmed):
        assert (resp.status_code, resp.json()) == (503, {"detail": feed_api.FEED_TOO_LARGE})
    logged = [r.getMessage() for r in caplog.records if r.name == feed_api.logger.name and r.levelno == logging.ERROR]
    assert len(logged) == 2 and all("not served" in message for message in logged)
    monkeypatch.setattr(config, revocation_feed.FEED_MAX_KB_SETTING, 2, raising=False)
    assert api.client.get(FEED).status_code == 200


def test_a_feed_with_more_rows_than_a_local_edition_reads_is_503(api, monkeypatch):
    digest = _package(api.db).digest
    _revoke(api.db, digest=digest)
    _revoke(api.db, digest=digest, level="warn")
    assert api.client.get(FEED).status_code == 200
    monkeypatch.setattr(revocation_feed, "FEED_MAX_ROWS", 1)
    resp = api.client.get(FEED)
    assert (resp.status_code, resp.json()) == (503, {"detail": feed_api.FEED_TOO_LARGE})


def test_an_unreadable_hosted_range_goes_out_for_every_version_and_a_long_reason_is_cut(api):
    _package(api.db)
    row = _revoke(api.db, theme_id="acme/ink", version_range="~1.0", reason="r" * 2500)
    [wire] = api.client.get(FEED).json()["revocations"]
    assert (wire["id"], wire["themeId"], wire["versionRange"]) == (str(row.id), "acme/ink", None)
    assert wire["reason"] == "r" * revocation_feed.FEED_REASON_MAX


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def _wire(**fields):
    row = {"id": str(uuid.uuid4()), "digest": "ab" * 32, "level": "disable", "reason": "leaks data",
           "createdAt": "2026-10-20T10:00:00+00:00", "liftedAt": None, **fields}
    return {key: value for key, value in row.items() if value is not DROP}


def _feed(rows=None, **fields):
    feed = {"schema": 1, "generatedAt": "2026-10-21T09:00:00+00:00",
            "revocations": [_wire()] if rows is None else rows, "revokedKeys": [], **fields}
    return json.dumps({key: value for key, value in feed.items() if value is not DROP}).encode("utf-8")


def _summary(item):
    return (str(item.id), item.digest, item.theme_id, item.version_range, item.level, item.reason,
            iso_time(item.created_at), iso_time(item.lifted_at))


def test_parse_feed_round_trips_build_feed(api):
    package = _package(api.db)
    _revoke(api.db, minutes_ago=3, digest=package.digest, level="warn", reason="slow")
    _revoke(api.db, minutes_ago=2, theme_id="acme/ink", version_range="^1.0.0", lifted_days_ago=1)
    _revoke(api.db, minutes_ago=1, theme_id="acme/ink")
    feed = build_feed(api.db)
    parsed = parse_feed(feed_bytes(feed))
    wire = [(row["id"], row.get("digest"), row.get("themeId"), row.get("versionRange"), row["level"],
             row["reason"], row["createdAt"], row["liftedAt"]) for row in feed["revocations"]]
    assert len(wire) == 3 and [_summary(item) for item in parsed.revocations] == wire
    assert (iso_time(parsed.generated_at), list(parsed.revoked_keys)) == (feed["generatedAt"], feed["revokedKeys"])
    assert [_summary(item) for item in parse_feed(api.client.get(FEED).content).revocations] == wire


BAD_ROWS = {
    "digest in capitals": {"digest": "AB" * 32},
    "digest too short": {"digest": "ab" * 31},
    "theme id in capitals": {"digest": DROP, "themeId": "Acme/Ink"},
    "theme id without a slug": {"digest": DROP, "themeId": "acme"},
    "tilde range": {"digest": DROP, "themeId": "acme/ink", "versionRange": "~1.0.0"},
    "short range": {"digest": DROP, "themeId": "acme/ink", "versionRange": "1.0"},
    "range as a number": {"digest": DROP, "themeId": "acme/ink", "versionRange": 1},
    "digest and theme id": {"themeId": "acme/ink"},
    "no target": {"digest": DROP},
    "digest with a range": {"versionRange": "^1.0.0"},
    "reason over 2000": {"reason": "x" * 2001},
    "reason not text": {"reason": None},
    "reason with NUL": {"reason": "bad\x00"},
    "reason empty": {"reason": ""},
    "reason of spaces": {"reason": "   "},
    "reason with a line break": {"reason": "two\nlines"},
    "reason with a bidi override": {"reason": "safe \u202e txt.exe"},
    "reason with a bell": {"reason": "ring\x07"},
    "id not a uuid": {"id": "not-a-uuid"},
    "id in capitals": {"id": UPPER_UUID},
    "no id": {"id": DROP},
    "createdAt without an offset": {"createdAt": "2026-10-20T10:00:00"},
    "createdAt not a time": {"createdAt": "yesterday"},
    "liftedAt a date": {"liftedAt": "2026-10-20"},
}
BAD_FEEDS = {
    "schema 2": {"schema": 2},
    "schema as text": {"schema": "1"},
    "schema true": {"schema": True},
    "no schema": {"schema": DROP},
    "revocations not a list": {"revocations": {}},
    "a row that isn't an object": {"revocations": ["ab" * 32]},
    "no generatedAt": {"generatedAt": DROP},
    "generatedAt without an offset": {"generatedAt": "2026-10-21T09:00:00"},
    "no revokedKeys": {"revokedKeys": DROP},
    "revokedKeys not a list": {"revokedKeys": "ed25519:0123456789abcdef"},
    "a revoked key that isn't ed25519": {"revokedKeys": ["rsa:0123456789abcdef"]},
}


@pytest.mark.parametrize("change", list(BAD_ROWS.values()), ids=list(BAD_ROWS))
def test_one_bad_row_refuses_the_whole_feed(change):
    with pytest.raises(FeedError):
        parse_feed(_feed([_wire(), _wire(**change), _wire()]))


@pytest.mark.parametrize("change", list(BAD_FEEDS.values()), ids=list(BAD_FEEDS))
def test_a_bad_feed_is_refused(change):
    with pytest.raises(FeedError):
        parse_feed(_feed(**change))


@pytest.mark.parametrize("data", [b"{", b"\xff\xfe", b"[]", b"[" * 100_000], ids=["truncated", "not utf-8",
                                                                                 "a list", "nested too deep"])
def test_not_a_json_object_is_refused(data):
    with pytest.raises(FeedError):
        parse_feed(data)


def test_a_duplicate_id_refuses_the_feed():
    twice = _wire()
    with pytest.raises(FeedError):
        parse_feed(_feed([twice, _wire(), dict(twice, reason="again")]))


def test_oversize_refuses_the_feed(monkeypatch):
    data = _feed([_wire() for _ in range(10)])
    with pytest.raises(FeedError):
        parse_feed(data, max_bytes=len(data) - 1)
    assert len(parse_feed(data, max_bytes=len(data)).revocations) == 10
    monkeypatch.setattr(config, revocation_feed.FEED_MAX_KB_SETTING, 1, raising=False)
    assert len(data) > 1024
    with pytest.raises(FeedError):
        parse_feed(data)


def test_more_than_ten_thousand_rows_refuses_the_feed():
    rows = [_wire(id=str(uuid.UUID(int=number))) for number in range(revocation_feed.FEED_MAX_ROWS + 1)]
    roomy = 64 * 1024 * 1024
    with pytest.raises(FeedError):
        parse_feed(_feed(rows), max_bytes=roomy)
    assert len(parse_feed(_feed(rows[:-1]), max_bytes=roomy).revocations) == revocation_feed.FEED_MAX_ROWS


@pytest.mark.parametrize("level", ["quarantine", None, 3, ["warn"]])
def test_an_unknown_level_parses_as_disable(level):
    [item] = parse_feed(_feed([_wire(level=level)])).revocations
    assert item.level == "disable"
    assert [row.level for row in parse_feed(_feed([_wire(level="warn")])).revocations] == ["warn"]


def test_a_theme_target_and_the_revoked_keys_parse():
    key_id = REVOKED_KEY["keyId"]
    row = _wire(digest=DROP, themeId="acme/ink", versionRange="^1.2.0", liftedAt="2026-10-21T08:00:00Z")
    feed = parse_feed(_feed([row], revokedKeys=[key_id, key_id]))
    [item] = feed.revocations
    assert (item.digest, item.theme_id, item.version_range) == (None, "acme/ink", "^1.2.0")
    assert iso_time(item.lifted_at) == "2026-10-21T08:00:00+00:00" and feed.revoked_keys == (key_id,)
