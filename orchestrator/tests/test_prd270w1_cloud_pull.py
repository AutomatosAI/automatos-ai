"""PRD-270 US-270-03a (§5.1, §5.3, K13, K14) — Sync with Automatos Cloud: the local edition's pull of the catalogue.

On Postgres with committing sessions (``conftest.new_session``), in the local edition. Automatos Cloud is
an ``httpx.MockTransport`` that records each request, behind a client built with
``follow_redirects=True``: a 3xx answered once proves the fetch turns redirects off itself. Packages are
real ``.autotheme`` files signed with the RFC 8032 vector's seeds (``tests/helpers_theme_publish.py``);
the keys file is the test's own, with both vector keys active. Storage is a fake S3 that must stay empty.
Each test's themes are under its own publisher and swept on teardown, with the one status row. Proven:

* a good catalogue writes Cloud rows (approved, reviewed, ``source`` 'cloud', no stored file, one
  'approved' event with no actor) that ``listings.all_listings`` lists with a verified signature, and
  records the success, the catalogue's time and the counts;
* a tampered signature, a digest that isn't the files', a themeJson that doesn't hash, a file type a
  path isn't served as, a manifest of another id, a version for a newer host, a package on another host,
  a non-object, a type that isn't a name and a key the revocation feed revoked are each refused with
  their reason and counted, nothing written for them;
* items of another type are skipped (never refused, the Sync succeeds); an item with no type is refused;
* a theme id a local-origin listing holds is left whole and counted refused; a private upload with the
  same digest is untouched; a version held with other files is refused, in the database or earlier in
  the same catalogue; a second Sync is unchanged and a newer version is updated; a row a unique index
  refuses (written meanwhile) refuses only its item;
* a database error mid-write, a refused catalogue, a keys file that won't load, an unexpected error and
  every fetch failure (redirect, oversize declared or streamed, encoded, timeout, 404, unreachable) write
  nothing and record only the attempt and the error: the previous success stays;
* off (no request, no status write) when ``AUTOMATOS_CLOUD_URL`` is empty and on SaaS; refused and
  recorded when it names a path, credentials or plain http to another host; a second Sync while one runs
  is answered "already running" and records nothing;
* the request carries exactly Accept, ``Accept-Encoding: identity`` and the User-Agent, and no
  Authorization, Cookie or X-Workspace-ID; ``urls`` derives the catalogue and file URLs from the origin.
"""
from __future__ import annotations

import base64
import gzip
import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
import sqlalchemy as sa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlalchemy.exc import SQLAlchemyError

import core.models  # noqa: F401  (registers every table the foreign keys name)
from config import config
from core.models.marketplace_cloud import MARKETPLACE_CLOUD_SYNC_ROW_ID, MarketplaceCloudSync
from core.models.themes import THEME_REVOCATION_SYNC_ROW_ID, ThemePackage, ThemeReviewEvent, ThemeRevocationSync
from services.marketplace_cloud import feed, fetch, pull, urls
from services.themes import listings, serve, signature_verify, storage
from services.themes import digest as theme_digest
from services.themes.contract import THEME_HOST_VERSION
from services.themes.package import file_kind, read_theme_package
from tests import helpers_theme_publish as hp
from tests.fixtures.themes import build
from tests.test_prd270w1_catalog_feed import ITEM_KEYS

CLOUD = "https://cloud.example.test"
CATALOG_URL = CLOUD + "/api/marketplace/catalog"
GENERATED = "2026-10-30T09:00:00+00:00"
RELEASE = feed.KIT_RELEASES + "play-v1.1.0/automatos.play-1.1.0.autotheme"
SENT_HEADERS = {"host", "accept", "accept-encoding", "user-agent", "connection"}


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(sa.text("SELECT source FROM theme_packages LIMIT 1"))
            conn.execute(sa.text("SELECT refused_count FROM marketplace_cloud_sync LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Sync tests need Postgres with the PRD-270 schema: {exc}")
    yield eng
    eng.dispose()


class Cloud:
    """Automatos Cloud as a mock transport: one answer for every request, each request recorded."""

    def __init__(self) -> None:
        self.requests = []
        self.status, self.body, self.headers, self.raises = 200, b"", {}, None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises(request)
        return httpx.Response(self.status, content=self.body, headers=self.headers)

    def factory(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), follow_redirects=True)


def _keys(tmp_path):
    """Both vector keys active."""
    entries = [{"keyId": case["keyId"], "spki": case["spki"], "status": "active", "added": "2026-10-20"}
               for case in (hp.SIGNER, hp.OTHER_SIGNER)]
    path = tmp_path / "signing-keys.json"
    path.write_text(json.dumps({"schema": 1, "keys": entries}), encoding="utf-8")
    return path


SYNC_COLUMNS = tuple(column.key for column in MarketplaceCloudSync.__table__.columns)


def _sweep(session, publisher):
    """This test's theme rows and their events, and the one status row (``_status_kept`` keeps what was there)."""
    mine = {"p": f"{publisher}/%"}
    session.execute(sa.text(
        "DELETE FROM theme_review_events WHERE package_id IN (SELECT id FROM theme_packages WHERE theme_id LIKE :p)"),
        mine)
    session.execute(sa.text("DELETE FROM theme_packages WHERE theme_id LIKE :p"), mine)
    session.execute(sa.text("DELETE FROM marketplace_cloud_sync"))


def _status_kept(session):
    """The status row as it was before the test (a dev database has a real one), to put back after it."""
    row = session.get(MarketplaceCloudSync, MARKETPLACE_CLOUD_SYNC_ROW_ID)
    return None if row is None else {name: getattr(row, name) for name in SYNC_COLUMNS}


@pytest.fixture
def cloud(engine, new_session, monkeypatch, tmp_path):
    publisher = f"cloud{uuid.uuid4().hex[:10]}"
    s3, mock = hp.FakeS3(), Cloud()
    monkeypatch.setattr(storage, "get_s3_client", lambda: s3)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(serve, "get_s3_client", lambda: s3)
    monkeypatch.setattr(signature_verify, "KEYS_PATH", _keys(tmp_path))
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "on", raising=False)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", CLOUD)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_FEED_MAX_KB", 1024)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_TIMEOUT_SECONDS", 15)
    first = new_session()
    kept = _status_kept(first)
    _sweep(first, publisher)
    first.commit()
    state = SimpleNamespace(publisher=publisher, s3=s3, mock=mock, new=new_session, tmp=tmp_path)
    state.theme = lambda slug: f"{publisher}/{slug}"
    yield state
    sweep = new_session.sweep()
    _sweep(sweep, publisher)
    sweep.execute(sa.text("DELETE FROM workspaces WHERE name = :n"), {"n": publisher})
    if kept is not None:
        sweep.add(MarketplaceCloudSync(**kept))
    sweep.commit()


# --------------------------------------------------------------------------- the wire


def _pkg(theme_id, version="1.0.0", source=hp.SOURCE_BYTES):
    return hp.package(theme_id, version, source)


def _item(data, case=hp.SIGNER, **changes):
    """One catalogue item for these package bytes, as the hosted catalogue writes it (``catalog._item``)."""
    checked = read_theme_package(data)
    manifest = checked.manifest
    item = {
        "type": "theme", "id": manifest["id"], "listingId": manifest["id"].replace("/", "."),
        "version": manifest["version"], "digest": checked.digest, "name": manifest.get("name"),
        "description": manifest.get("description"), "publisher": {"name": "Automatos", "official": True},
        "engines": manifest.get("engines"), "uses": [], "pricing": "Free",
        "signature": json.loads(hp.envelope(data, case)), "files": checked.file_manifest,
        "themeJson": base64.b64encode(checked.files["theme.json"].data).decode("ascii"),
        "package": urls.package_path(checked.digest), "cover": None, "phoneCover": None, "presets": [],
        "otherPreviews": [], "otherPhonePreviews": [],
    }
    return {**item, **changes}


def _catalog(*items, **fields):
    return json.dumps({"schema": 1, "generatedAt": GENERATED, "items": list(items), **fields}).encode("utf-8")


def _sync(env, body=None):
    if body is not None:
        env.mock.body = body
    return pull.sync_catalog(env.new(), client_factory=env.mock.factory)


def _cloud_rows(env):
    session = env.new()
    rows = session.query(ThemePackage).filter(ThemePackage.theme_id.like(f"{env.publisher}/%")).all()
    session.close()
    return sorted(rows, key=lambda row: (row.theme_id, row.version, row.owner_type))


def _status_row(env):
    session = env.new()
    row = session.get(MarketplaceCloudSync, MARKETPLACE_CLOUD_SYNC_ROW_ID)
    session.close()
    return row


def _columns(row):
    return {column.name: getattr(row, column.key) for column in ThemePackage.__table__.columns}


def _add(env, *objects):
    session = env.new()
    session.add_all(objects)
    session.commit()
    session.close()


def _row(data, owner="marketplace", workspace_id=None, source=None):
    checked = read_theme_package(data)
    theme_id, version = checked.manifest["id"], checked.manifest["version"]
    return ThemePackage(
        theme_id=theme_id, version=version, owner_type=owner, workspace_id=workspace_id, digest=checked.digest,
        manifest=checked.manifest, file_manifest=checked.file_manifest,
        storage_prefix=storage.theme_prefix(theme_id, version, workspace_id), status="approved", reviewed=True,
        source=source,
    )


NEWER_HOST = {"automatos": "^9.0.0", "sdk": "^1.0.0"}


def _unread_item(theme_id, version="1.0.0", **manifest_fields):
    """A signed item for the hello files with theme.json changed, built without ``read_theme_package`` (which
    refuses a manifest this host can't run): its file list, digest and envelope made here."""
    members = {**build.hello_members(), hp.SOURCE_FILE: hp.SOURCE_BYTES,
               build.MANIFEST: build.changed_manifest({"id": theme_id, "version": version, **manifest_fields})}
    files = theme_digest.file_manifest({path: (data, file_kind(path).content_type) for path, data in members.items()})
    payload = signature_verify.payload_for(theme_id, version, {path: row["sha256"] for path, row in files.items()})
    seed = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(hp.SIGNER["seed"]))
    envelope = {
        "form": "automatos-theme-signature/v1", "algorithm": "ed25519", "keyId": hp.SIGNER["keyId"], "id": theme_id,
        "version": version, "digest": payload["digest"],
        "signature": base64.b64encode(seed.sign(signature_verify.canonical_bytes(payload))).decode("ascii"),
    }
    return {"type": "theme", "id": theme_id, "version": version, "digest": payload["digest"], "files": files,
            "themeJson": base64.b64encode(members[build.MANIFEST]).decode("ascii"),
            "package": urls.package_path(payload["digest"]), "signature": envelope}


# --------------------------------------------------------------------------- a good Sync


def test_a_good_catalogue_writes_cloud_rows_the_local_listings_show(cloud):
    themes_before = pull.sync_status(cloud.new())["themes"]
    arcade, ink = _pkg(cloud.theme("arcade")), _pkg(cloud.theme("ink"))
    outcome = _sync(cloud, _catalog(_item(arcade), _item(ink)))
    assert outcome.ok and outcome.counts.view() == {"new": 2, "updated": 0, "unchanged": 0, "refused": 0}
    rows = _cloud_rows(cloud)
    assert [(row.theme_id, row.version) for row in rows] == [(cloud.theme(name), "1.0.0") for name in ("arcade", "ink")]
    for row, data in zip(rows, (arcade, ink)):
        checked = read_theme_package(data)
        assert (row.owner_type, row.workspace_id, row.status, row.reviewed, row.source) == (
            "marketplace", None, "approved", True, "cloud")
        assert (row.scan_verdict, row.submitted_by, row.scan_report) == ("not_run", None, None)
        assert row.digest == checked.digest and row.manifest == checked.manifest
        assert row.file_manifest == checked.file_manifest
        assert row.signature == json.loads(hp.envelope(data))["signature"]
        assert row.storage_prefix == storage.theme_prefix(row.theme_id, row.version)
    session = cloud.new()
    events = session.query(ThemeReviewEvent).filter(ThemeReviewEvent.package_id.in_([row.id for row in rows])).all()
    assert sorted((e.event, e.from_status, e.to_status, e.actor_user_id, json.dumps(e.detail)) for e in events) == [
        ("approved", None, "approved", None, '{"source": "cloud"}')] * 2
    listed = listings.all_listings(session)
    for row in rows:
        assert [version.id for version in listed[row.theme_id]] == [row.id]
        assert listings.signature_verdict(listed[row.theme_id][0]).verified
    assert cloud.s3.objects == {}  # no stored file: the package is fetched on install
    status = pull.sync_status(session)
    session.close()
    assert status["enabled"] and status["cloudUrl"] == CLOUD and status["lastError"] is None
    assert status["generatedAt"] == GENERATED and status["lastSuccessAt"] == status["lastAttemptAt"]
    assert status["counts"] == {"new": 2, "updated": 0, "unchanged": 0, "refused": 0}
    assert status["themes"] == themes_before + 2 and status["stale"] is False


def test_the_wire_item_is_the_hosted_catalogues_shape_and_urls_derive_from_the_origin(cloud, monkeypatch):
    assert set(_item(_pkg(cloud.theme("arcade")))) == ITEM_KEYS
    digest = "ab" * 32
    assert urls.configured_origin() == CLOUD and urls.catalog_url(CLOUD) == CATALOG_URL
    assert urls.package_url(CLOUD, digest) == f"{CLOUD}/api/marketplace/catalog/files/{digest}/package.autotheme"
    assert urls.cloud_file_url(digest, "assets/previews/a.png") == (
        f"{CLOUD}/api/marketplace/catalog/files/{digest}/assets/previews/a.png")
    for bad in ("", "https://cloud.example.test/api", "https://u:p@cloud.example.test", "http://cloud.example.test"):
        monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", bad)
        assert urls.configured_origin() is None and urls.cloud_file_url(digest, "x.png") is None
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", CLOUD + "/")
    assert urls.configured_origin() == CLOUD


# --------------------------------------------------------------------------- refused items


def _flipped(envelope):
    raw = bytearray(base64.b64decode(envelope["signature"]))
    raw[0] ^= 0xFF
    return {**envelope, "signature": base64.b64encode(bytes(raw)).decode("ascii")}


def _other_digest(env, data):
    other = read_theme_package(_pkg(env.theme("other"))).digest
    item = _item(data)
    return {**item, "digest": other, "package": urls.package_path(other),
            "signature": {**item["signature"], "digest": other}}


def _typed_as_html(data):
    item = _item(data)
    return {**item, "files": {**item["files"], "theme.json": {**item["files"]["theme.json"], "type": "text/html"}}}


BAD_ITEMS = {
    "a tampered signature": (lambda env, data: _item(data, signature=_flipped(_item(data)["signature"])),
                             feed.NOT_SIGNED.format(code="SIGNATURE_BAD")),
    "a digest that isn't the files'": (_other_digest, feed.DIGEST_MISMATCH),
    "a themeJson that doesn't hash": (
        lambda env, data: _item(data, themeJson=base64.b64encode(b'{"id": "x"}').decode("ascii")),
        feed.THEME_JSON_MISMATCH),
    "a file typed as it isn't served": (lambda env, data: _typed_as_html(data), feed.BAD_FILES),
    "a manifest of another id": (lambda env, data: _item(data, id=env.theme("mint")), feed.OTHER_THEME),
    "a package on another host": (lambda env, data: _item(data, package="https://evil.example.test/p.autotheme"),
                                  feed.BAD_PACKAGE),
    "an item that isn't an object": (lambda env, data: ["not", "an", "item"], feed.NOT_AN_OBJECT),
    "a type that isn't a name": (lambda env, data: _item(data, type=None), feed.BAD_TYPE),
    "a version for a newer host": (lambda env, data: _unread_item(env.theme("ink"), engines=NEWER_HOST),
                                   feed.NEEDS_NEWER_HOST),
    "a release path that climbs": (lambda env, data: _item(data, package=RELEASE.rsplit("/", 2)[0] + "/../x/y"),
                                   feed.BAD_PACKAGE),
}


def _revoke(env, key_id):
    """The revocation feed's stored revoked keys name ``key_id``; the earlier value comes back on teardown."""
    session = env.new()
    row = session.get(ThemeRevocationSync, THEME_REVOCATION_SYNC_ROW_ID)
    before = None if row is None else list(row.revoked_key_ids or [])
    if row is None:
        session.add(ThemeRevocationSync(id=THEME_REVOCATION_SYNC_ROW_ID, revoked_key_ids=[key_id]))
    else:
        row.revoked_key_ids = [*before, key_id]
    session.commit()
    session.close()
    return before


def _unrevoke(env, before):
    session = env.new()
    row = session.get(ThemeRevocationSync, THEME_REVOCATION_SYNC_ROW_ID)
    if before is None:
        session.delete(row)
    else:
        row.revoked_key_ids = before
    session.commit()
    session.close()


def _assert_one_refused(env, outcome, catalog_body, reason):
    assert outcome.ok and outcome.counts.view() == {"new": 1, "updated": 0, "unchanged": 0, "refused": 1}
    assert [row.theme_id for row in _cloud_rows(env)] == [env.theme("arcade")]
    assert _status_row(env).refused_count == 1
    session = env.new()
    assert feed.parse_catalog(catalog_body, session).refused == (f"Item 2: {reason}",)
    session.close()


@pytest.mark.parametrize("case", sorted(BAD_ITEMS))
def test_each_bad_item_is_refused_counted_and_never_written(cloud, case):
    build, reason = BAD_ITEMS[case]
    body = _catalog(_item(_pkg(cloud.theme("arcade"))), build(cloud, _pkg(cloud.theme("ink"))))
    _assert_one_refused(cloud, _sync(cloud, body), body, reason)


def test_a_key_the_revocation_feed_revoked_is_refused(cloud):
    ink = _pkg(cloud.theme("ink"))
    body = _catalog(_item(_pkg(cloud.theme("arcade"))), _item(ink, hp.OTHER_SIGNER))
    before = _revoke(cloud, hp.OTHER_SIGNER["keyId"])
    try:
        _assert_one_refused(cloud, _sync(cloud, body), body, feed.NOT_SIGNED.format(code="SIGNATURE_BAD"))
    finally:
        _unrevoke(cloud, before)


def test_other_types_are_skipped_and_an_item_with_no_type_is_refused(cloud):
    arcade = _item(_pkg(cloud.theme("arcade")))
    others = [{"type": "widget", "id": "widget-x", "name": "X"}, {"type": "voice", "id": "voice-y"}]
    outcome = _sync(cloud, _catalog(arcade, *others))
    assert outcome.ok and outcome.counts.skipped == 2
    assert outcome.counts.view() == {"new": 1, "updated": 0, "unchanged": 0, "refused": 0}
    assert _status_row(cloud).refused_count == 0 and _status_row(cloud).last_error is None
    untyped = {key: value for key, value in _item(_pkg(cloud.theme("ink"))).items() if key != "type"}
    outcome = _sync(cloud, _catalog(arcade, untyped))
    assert outcome.ok and outcome.counts.view() == {"new": 0, "updated": 0, "unchanged": 1, "refused": 1}
    assert [row.theme_id for row in _cloud_rows(cloud)] == [cloud.theme("arcade")]


def test_a_design_kit_release_url_is_an_allowed_package(cloud):
    outcome = _sync(cloud, _catalog(_item(_pkg(cloud.theme("arcade")), package=RELEASE)))
    assert outcome.ok and outcome.counts.new == 1


# --------------------------------------------------------------------------- what is never touched


def test_a_theme_id_a_local_listing_holds_is_left_whole(cloud):
    local = _row(_pkg(cloud.theme("arcade"), "0.9.0"))
    _add(cloud, local)
    before = _columns(_cloud_rows(cloud)[0])
    body = _catalog(*(_item(_pkg(cloud.theme("arcade"), version)) for version in ("1.0.0", "1.1.0")))
    outcome = _sync(cloud, body)
    assert outcome.ok and outcome.counts.view() == {"new": 0, "updated": 0, "unchanged": 0, "refused": 2}
    rows = _cloud_rows(cloud)
    assert len(rows) == 1 and _columns(rows[0]) == before


def test_a_private_upload_with_the_same_digest_is_untouched(cloud):
    workspace = uuid.uuid4()
    session = cloud.new()
    session.execute(sa.text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n)"),
                    {"id": str(workspace), "n": cloud.publisher})
    session.commit()
    session.close()
    data = _pkg(cloud.theme("arcade"))
    _add(cloud, _row(data, owner="workspace", workspace_id=workspace))
    before = _columns(_cloud_rows(cloud)[0])
    outcome = _sync(cloud, _catalog(_item(data)))
    assert outcome.ok and outcome.counts.new == 1
    private, cloud_row = sorted(_cloud_rows(cloud), key=lambda row: row.owner_type, reverse=True)
    assert _columns(private) == before
    assert (cloud_row.owner_type, cloud_row.source, cloud_row.digest) == ("marketplace", "cloud", private.digest)


def test_a_second_sync_is_unchanged_and_a_newer_version_is_updated(cloud):
    first = _item(_pkg(cloud.theme("arcade")))
    assert _sync(cloud, _catalog(first)).counts.new == 1
    again = _sync(cloud, _catalog(first))
    assert again.ok and again.counts.view() == {"new": 0, "updated": 0, "unchanged": 1, "refused": 0}
    newer = _item(_pkg(cloud.theme("arcade"), "1.1.0"))
    moved = _sync(cloud, _catalog(newer, first))
    assert moved.ok and moved.counts.view() == {"new": 0, "updated": 1, "unchanged": 1, "refused": 0}
    session = cloud.new()
    versions = listings.listed_versions(session, cloud.theme("arcade"))
    events = session.query(ThemeReviewEvent).filter(ThemeReviewEvent.package_id.in_([v.id for v in versions])).count()
    session.close()
    assert [row.version for row in versions] == ["1.1.0", "1.0.0"] and events == 2
    assert _status_row(cloud).updated_count == 1 and _status_row(cloud).unchanged_count == 1


def test_a_version_held_with_other_files_is_refused(cloud):
    assert _sync(cloud, _catalog(_item(_pkg(cloud.theme("arcade"))))).counts.new == 1
    before = _columns(_cloud_rows(cloud)[0])
    other = _item(_pkg(cloud.theme("arcade"), source=b"export default { other: true }\n"))
    outcome = _sync(cloud, _catalog(other))
    assert outcome.ok and outcome.counts.view() == {"new": 0, "updated": 0, "unchanged": 0, "refused": 1}
    rows = _cloud_rows(cloud)
    assert len(rows) == 1 and _columns(rows[0]) == before


# --------------------------------------------------------------------------- failures write nothing


def test_two_versions_of_a_new_theme_are_both_new_and_a_second_copy_of_a_version_is_refused(cloud):
    one, two = _item(_pkg(cloud.theme("arcade"))), _item(_pkg(cloud.theme("arcade"), "1.1.0"))
    rival = _item(_pkg(cloud.theme("arcade"), source=b"export default { rival: true }\n"))
    outcome = _sync(cloud, _catalog(two, one, rival, one))
    assert outcome.ok and outcome.counts.view() == {"new": 2, "updated": 0, "unchanged": 1, "refused": 1}
    assert [(row.version, row.digest) for row in _cloud_rows(cloud)] == [("1.0.0", one["digest"]), ("1.1.0", two["digest"])]


def test_a_row_written_meanwhile_refuses_only_its_item(cloud, monkeypatch):
    held = _row(_pkg(cloud.theme("arcade")), source="cloud")
    _add(cloud, held)
    before = _columns(_cloud_rows(cloud)[0])
    empty = pull.Held(frozenset(), frozenset(), frozenset(), frozenset())
    monkeypatch.setattr(pull, "held_state", lambda db, items: empty)  # as if another Sync wrote it after the read
    rival = _item(_pkg(cloud.theme("arcade"), source=b"export default { rival: true }\n"))
    outcome = _sync(cloud, _catalog(rival, _item(_pkg(cloud.theme("ink")))))
    assert outcome.ok and outcome.counts.view() == {"new": 1, "updated": 0, "unchanged": 0, "refused": 1}
    arcade, ink = _cloud_rows(cloud)
    assert _columns(arcade) == before and (ink.theme_id, ink.source) == (cloud.theme("ink"), "cloud")
    session = cloud.new()
    assert session.query(ThemeReviewEvent).filter(ThemeReviewEvent.package_id == ink.id).count() == 1
    session.close()
    assert _status_row(cloud).refused_count == 1


def test_an_unexpected_error_rolls_back_and_is_recorded(cloud, monkeypatch):
    def broken(data, db):
        raise RuntimeError("a bug")

    monkeypatch.setattr(pull, "parse_catalog", broken)
    _assert_failed_and_nothing_written(cloud, _sync(cloud, _catalog()), pull.UNEXPECTED)


def test_a_database_error_mid_write_rolls_everything_back(cloud, monkeypatch):
    assert _sync(cloud, _catalog(_item(_pkg(cloud.theme("arcade"))))).ok
    good = _status_row(cloud)
    real, calls = pull.write_item, []

    def failing(db, item, at):
        calls.append(item.theme_id)
        if len(calls) == 2:
            raise SQLAlchemyError("the database went away")
        return real(db, item, at)

    monkeypatch.setattr(pull, "write_item", failing)
    outcome = _sync(cloud, _catalog(_item(_pkg(cloud.theme("ink"))), _item(_pkg(cloud.theme("mint")))))
    assert not outcome.ok and outcome.error == pull.NOT_SAVED and len(calls) == 2
    assert [row.theme_id for row in _cloud_rows(cloud)] == [cloud.theme("arcade")]
    after = _status_row(cloud)
    assert after.last_error == pull.NOT_SAVED and after.last_attempt_at > good.last_attempt_at
    assert (after.last_success_at, after.feed_generated_at) == (good.last_success_at, good.feed_generated_at)
    assert (after.new_count, after.refused_count) == (good.new_count, good.refused_count) == (1, 0)


OVERSIZE = b" " * 2048


def _stream(body):
    return httpx.Response(200, content=iter([body[:512], body[512:]]))


FETCH_FAILURES = {
    "a redirect": (dict(status=302, headers={"location": CLOUD + "/elsewhere"}),
                   fetch.REDIRECTED.format(status=302)),
    "a 404": (dict(status=404), fetch.BAD_STATUS.format(status=404)),
    "an encoded body": (dict(headers={"content-encoding": "gzip"}, body=gzip.compress(b'{"schema": 1}')),
                        fetch.ENCODED),
    "an oversize declared body": (dict(body=b" " * 2048), fetch.TOO_LARGE.format(kb=1)),
    "a timeout": (dict(raises=lambda request: httpx.ReadTimeout("slow", request=request)), fetch.TIMED_OUT),
    "an unreachable host": (dict(raises=lambda request: httpx.ConnectError("refused", request=request)),
                            fetch.UNREACHABLE.format(error="ConnectError")),
}


def _assert_failed_and_nothing_written(env, outcome, message, requests=1):
    assert not outcome.ok and not outcome.off and outcome.error == message
    assert len(env.mock.requests) == requests and _cloud_rows(env) == []
    row = _status_row(env)
    assert row.last_error == message and row.last_attempt_at is not None and row.last_success_at is None


@pytest.mark.parametrize("case", sorted(FETCH_FAILURES))
def test_a_failed_fetch_writes_nothing(cloud, monkeypatch, case):
    answer, message = FETCH_FAILURES[case]
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_FEED_MAX_KB", 1)
    for name, value in answer.items():
        setattr(cloud.mock, name, value)
    outcome = _sync(cloud, answer.get("body"))
    _assert_failed_and_nothing_written(cloud, outcome, message)


def test_an_oversize_streamed_body_is_cut_off(cloud, monkeypatch):
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_FEED_MAX_KB", 1)
    monkeypatch.setattr(cloud.mock, "handler", lambda request: cloud.mock.requests.append(request) or _stream(OVERSIZE))
    _assert_failed_and_nothing_written(cloud, _sync(cloud), fetch.TOO_LARGE.format(kb=1))


CATALOG_REFUSALS = {
    "not JSON": (b"<html>", feed.NOT_JSON),
    "not UTF-8": (b'{"schema": 1, "x": "\xff"}', feed.NOT_JSON),
    "schema 2": (json.dumps({"schema": 2, "generatedAt": GENERATED, "items": []}).encode(), feed.NOT_SCHEMA),
    "no generatedAt": (json.dumps({"schema": 1, "items": []}).encode(), feed.NOT_SCHEMA),
    "items not a list": (json.dumps({"schema": 1, "generatedAt": GENERATED, "items": {}}).encode(), feed.NOT_SCHEMA),
}


@pytest.mark.parametrize("case", sorted(CATALOG_REFUSALS))
def test_a_refused_catalogue_writes_nothing(cloud, case):
    body, message = CATALOG_REFUSALS[case]
    _assert_failed_and_nothing_written(cloud, _sync(cloud, body), message)


def test_too_many_items_or_too_many_bytes_refuse_the_whole_catalogue(cloud, monkeypatch):
    monkeypatch.setattr(feed, "CATALOG_MAX_ITEMS", 1)
    body = _catalog(_item(_pkg(cloud.theme("arcade"))), _item(_pkg(cloud.theme("ink"))))
    _assert_failed_and_nothing_written(cloud, _sync(cloud, body), feed.TOO_MANY.format(count=1))
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_FEED_MAX_KB", 1)
    with pytest.raises(feed.CatalogError, match="larger than 1 KB"):
        feed.parse_catalog(b" " * 2048, None)


def test_a_keys_file_that_wont_load_writes_nothing(cloud, monkeypatch):
    monkeypatch.setattr(signature_verify, "KEYS_PATH", cloud.tmp / "missing.json")
    outcome = _sync(cloud, _catalog(_item(_pkg(cloud.theme("arcade")))))
    _assert_failed_and_nothing_written(cloud, outcome, pull.KEYS_UNREADABLE)


# --------------------------------------------------------------------------- when it runs, and what it sends


@pytest.mark.parametrize(("url", "message"), [
    ("https://cloud.example.test/api", fetch.URL_NOT_ORIGIN),
    ("https://cloud.example.test/?x=1", fetch.URL_NOT_ORIGIN),
    ("https://user:secret@cloud.example.test", fetch.URL_CREDENTIALS),
    ("http://cloud.example.test", fetch.URL_NOT_HTTPS),
    ("ftp://cloud.example.test", fetch.URL_NOT_HTTPS),
])
def test_a_url_that_isnt_an_origin_is_refused_before_any_request(cloud, monkeypatch, url, message):
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", url)
    _assert_failed_and_nothing_written(cloud, _sync(cloud, _catalog()), message, requests=0)
    assert "secret" not in (_status_row(cloud).last_error or "")
    assert pull.sync_status(cloud.new())["cloudUrl"] is None


@pytest.mark.parametrize(("edition", "url"), [("local", ""), ("local", "   "), ("saas", CLOUD)])
def test_off_when_the_url_is_empty_or_on_saas(cloud, monkeypatch, edition, url):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", url)
    outcome = _sync(cloud, _catalog(_item(_pkg(cloud.theme("arcade")))))
    assert outcome.off and not outcome.ok and outcome.error == pull.SYNC_OFF
    assert cloud.mock.requests == [] and _cloud_rows(cloud) == [] and _status_row(cloud) is None
    assert pull.sync_status(cloud.new())["enabled"] is False


def test_a_loopback_http_origin_is_allowed(cloud, monkeypatch):
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", "http://127.0.0.1:8000/")
    assert _sync(cloud, _catalog()).ok
    assert str(cloud.mock.requests[0].url) == "http://127.0.0.1:8000/api/marketplace/catalog"


def test_the_request_is_anonymous_and_says_the_host_version(cloud):
    assert _sync(cloud, _catalog(_item(_pkg(cloud.theme("arcade"))))).ok
    (request,) = cloud.mock.requests
    assert request.method == "GET" and str(request.url) == CATALOG_URL
    sent = {name.lower() for name in request.headers}
    assert sent <= SENT_HEADERS
    assert not sent & {"authorization", "cookie", "x-workspace-id"}
    assert request.headers["user-agent"] == f"Automatos/{THEME_HOST_VERSION} (local)" == fetch.USER_AGENT
    assert request.headers["accept-encoding"] == "identity"
    assert request.headers["accept"] == "application/json"


def test_one_sync_at_a_time_a_second_is_answered_already_running(cloud):
    with pull._SYNC_LOCK:
        outcome = _sync(cloud, _catalog(_item(_pkg(cloud.theme("arcade")))))
    assert outcome.busy and not outcome.ok and outcome.error == pull.ALREADY_RUNNING
    assert cloud.mock.requests == [] and _cloud_rows(cloud) == [] and _status_row(cloud) is None
    assert _sync(cloud).ok and len(cloud.mock.requests) == 1


def test_the_status_says_stale_after_a_week_without_a_success():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    assert pull.is_stale(None, now)
    assert pull.is_stale(datetime(2026, 10, 22, tzinfo=timezone.utc), now)
    assert not pull.is_stale(datetime(2026, 10, 24, tzinfo=timezone.utc), now)
