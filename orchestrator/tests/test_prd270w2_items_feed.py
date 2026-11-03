"""PRD-270 W2 US-270-13a (§5.3, §5.6, K13) — locally: the catalogue's item entries read, changed bundles fetched.

Automatos Cloud is an ``httpx.MockTransport`` that answers by URL and records every request, behind a client
built with ``follow_redirects=True`` (W1's ``tests/test_prd270w1_cloud_pull.py``) and handed in through W1's
client-factory seam: a 3xx answered once proves the fetch turns redirects off itself. The bundles are the
corpus's canonical fixtures (``tests/fixtures/marketplace_bundles``), and each entry is what the hosted
catalogue lists for one (``catalog_items.entry_of``). Proven:

* W1's ``feed.parse_catalog`` reads an entry of each bundle type onto ``entries``, in order; an item of an
  unknown type is skipped: neither an entry nor refused, and W1's own refusals stay empty;
* a good entry fetches one bundle, from exactly ``<AUTOMATOS_CLOUD_URL>/api/marketplace/catalog/bundles/
  <revision>``, with W1's three headers and User-Agent only (no Authorization, Cookie or X-Workspace-ID);
  an entry whose ``bundle`` names another host fetches from the derived URL anyway;
* trust 'official' (or none), a bad externalId, version or revision, an over-cap display field, a missing
  one and a second entry of one type and externalId are refused, with a reason and a log line that echo
  nothing the entry held;
* a body whose sha256 isn't the revision, an oversize body (declared or streamed), a redirect, an encoded
  body, a 404, a timeout, an unreachable host, a body that isn't UTF-8 JSON, isn't canonical, isn't a valid
  bundle or is another item's each refuse that item only: the next entry is still fetched;
* a setting that isn't an origin (a path, credentials) refuses before any request;
* on Postgres, an entry a mapping holds at the same revision is unchanged and fetches nothing; at another
  revision, of another type or with no mapping, it is fetched;
* ``items_feed`` imports W1's ``fetch.py`` and ``urls.py``, never ``feed.py`` or ``pull.py``.
"""
from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import json
import logging
import re
import uuid
from pathlib import Path

import httpx
import pytest
import sqlalchemy as sa

import core.models  # noqa: F401  (registers every table the foreign keys name)
from config import config
from services.marketplace_bundles import items_feed
from services.marketplace_bundles.canonical import canonical_bytes
from services.marketplace_bundles.catalog_items import entry_of
from services.marketplace_bundles.exported import ExportedItem, ItemEntry
from services.marketplace_bundles.items_feed import CatalogEntry, Entries, EntryRefused, RefusedEntry
from services.marketplace_cloud import feed, fetch
from services.themes import signature_verify
from tests import helpers_theme_publish as hp

ORCH = Path(__file__).resolve().parents[1]
FIXTURES = ORCH / "tests" / "fixtures" / "marketplace_bundles"
TYPE_ORDER = ("agent", "playbook", "package", "plugin")
CLOUD = "https://cloud.example.test"
BUNDLES = CLOUD + "/api/marketplace/catalog/bundles/"
EVIL = "https://evil.example.test"
GENERATED = "2026-10-30T09:00:00+00:00"
SENT_HEADERS = {"host", "accept", "accept-encoding", "user-agent", "connection"}
MARK = "SECRET-MARK-7f3a"
HEX32 = re.compile(r"[0-9a-f]{32}")


def _bundle(kind):
    return json.loads((FIXTURES / f"{kind}.json").read_text(encoding="utf-8"))


def _revision(data):
    return hashlib.sha256(data).hexdigest()


def _entry(bundle, **changes):
    """The hosted catalogue's entry for ``bundle`` (``catalog_items.entry_of``), with ``changes``."""
    exported = ItemEntry(bundle["type"], bundle["externalId"], bundle["version"], frozenset())
    entry = entry_of(ExportedItem(exported, bundle, _revision(canonical_bytes(bundle))))
    return {**entry, **changes}


def _catalog(*items):
    return json.dumps({"schema": 1, "generatedAt": GENERATED, "items": list(items)}).encode("utf-8")


class Cloud:
    """Automatos Cloud as a mock transport: a responder per URL (404 for any other), each request recorded."""

    def __init__(self) -> None:
        self.requests = []
        self.answers = {}

    def answer(self, url, body=b"", status=200, headers=None):
        self.answers[url] = lambda request: httpx.Response(status, content=body, headers=headers or {})

    def answer_with(self, url, respond):
        """``respond(request)`` answers ``url``: a response, or it raises."""
        self.answers[url] = respond

    def serve(self, bundle):
        """``bundle``'s canonical bytes at its derived URL; its catalogue entry."""
        data = canonical_bytes(bundle)
        self.answer(BUNDLES + _revision(data), data)
        return _entry(bundle)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        respond = self.answers.get(str(request.url))
        return httpx.Response(404) if respond is None else respond(request)

    def factory(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), follow_redirects=True)

    def urls(self):
        return [str(request.url) for request in self.requests]


@pytest.fixture
def cloud(monkeypatch, tmp_path):
    entries = [{"keyId": case["keyId"], "spki": case["spki"], "status": "active", "added": "2026-10-20"}
               for case in (hp.SIGNER, hp.OTHER_SIGNER)]
    keys = tmp_path / "signing-keys.json"
    keys.write_text(json.dumps({"schema": 1, "keys": entries}), encoding="utf-8")
    monkeypatch.setattr(signature_verify, "KEYS_PATH", keys)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", CLOUD)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_FEED_MAX_KB", 1024)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_BUNDLE_MAX_KB", 256)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_TIMEOUT_SECONDS", 15)
    return Cloud()


def _fetch(cloud, *entries):
    return items_feed.fetch_bundles([items_feed.parse_item_entry(entry) for entry in entries], cloud.factory)


# --------------------------------------------------------------------------- the catalogue's entries


def test_parse_catalog_reads_each_bundle_type_onto_entries_and_skips_an_unknown_type(cloud):
    bundles = [_bundle(kind) for kind in TYPE_ORDER]
    unknown = [{"type": "widget", "id": "widget-x", "name": "X"}, {"type": "voice", "externalId": "0" * 32}]
    parsed = feed.parse_catalog(_catalog(unknown[0], *(_entry(b) for b in bundles), unknown[1]), None)
    assert parsed.items == () and parsed.refused == () and parsed.entries.refused == ()
    assert [entry.type for entry in parsed.entries.listed] == list(TYPE_ORDER)
    for entry, bundle in zip(parsed.entries.listed, bundles):
        revision = _revision(canonical_bytes(bundle))
        assert entry == CatalogEntry(bundle["type"], bundle["externalId"], bundle["version"], revision, bundle["name"])
    # the unknown types are counted nowhere: no entry, no refusal of either reader
    assert len(parsed.entries.listed) + len(parsed.entries.refused) == len(TYPE_ORDER)


def test_a_theme_only_catalogue_has_no_entries_and_an_item_not_of_a_bundle_type_is_no_entry(cloud):
    assert feed.parse_catalog(_catalog(), None).entries == items_feed.NO_ENTRIES
    for item in ({"type": "theme"}, {"type": "widget"}, {"type": ["agent"]}, {"name": "x"}, "agent", None):
        assert not items_feed.is_bundle_item(item)
        assert items_feed.NO_ENTRIES.read(item) == items_feed.NO_ENTRIES
    with pytest.raises(EntryRefused, match="item types"):
        items_feed.parse_item_entry({"type": "theme"})


def _over(text_max):
    return MARK + "x" * text_max


ENTRY_REFUSALS = {
    "trust official": (dict(trust="official"), items_feed.NOT_REVIEWED),
    "trust Reviewed": (dict(trust="Reviewed"), items_feed.NOT_REVIEWED),
    "trust missing": (dict(trust=None), items_feed.NOT_REVIEWED),
    "externalId upper-case": (dict(externalId="A" * 32), "entry.externalId is not in the form"),
    "externalId short": (dict(externalId="a" * 31), "entry.externalId is not in the form"),
    "externalId a path": (dict(externalId="../../" + MARK), "entry.externalId is not in the form"),
    "externalId missing": (dict(externalId=None), "entry.externalId is missing"),
    "revision short": (dict(revision="0" * 63), "entry.revision is not in the form"),
    "revision upper-case": (dict(revision="A" * 64), "entry.revision is not in the form"),
    "revision a path": (dict(revision="../" + "0" * 61), "entry.revision is not in the form"),
    "revision a number": (dict(revision=7), "entry.revision has the wrong type"),
    "version off its pattern": (dict(version="-1"), "entry.version is not in the form"),
    "name over its cap": (dict(name=_over(255)), "entry.name is longer than 255"),
    "name empty": (dict(name=""), "entry.name is empty"),
    "description over its cap": (dict(description=_over(4000)), "entry.description is longer than 4000"),
    "category over its cap": (dict(category=_over(100)), "entry.category is longer than 100"),
    "icon an image URL": (dict(icon="http://x/i.png"), "entry.icon is not in the form"),
    "too many tags": (dict(tags=[f"t{n}" for n in range(21)]), "entry.tags has more than 20"),
    "a tag over its cap": (dict(tags=[_over(64)]), "entry.tags[0] is longer than 64"),
    "publisher name over its cap": (dict(publisher={"name": _over(100)}), "entry.publisher.name is longer than 100"),
    "publisher with another key": (dict(publisher={"name": "P", "official": True}), "entry.publisher has a key"),
    "a type no bundle has": (dict(type="nonsense"), "item types"),
}


@pytest.mark.parametrize("case", sorted(ENTRY_REFUSALS))
def test_an_entry_that_fails_a_check_is_refused_echoing_nothing(cloud, caplog, case):
    changes, reason = ENTRY_REFUSALS[case]
    entry = {key: value for key, value in _entry(_bundle("agent"), **changes).items() if value is not None}
    with pytest.raises(EntryRefused) as refused:
        items_feed.parse_item_entry(entry)
    assert reason in str(refused.value) and MARK not in str(refused.value)
    caplog.set_level(logging.WARNING, logger=items_feed.__name__)
    parsed = feed.parse_catalog(_catalog(entry, _entry(_bundle("plugin"))), None)
    assert [entry.type for entry in parsed.entries.listed] == ["plugin"] and parsed.refused == ()
    if items_feed.is_bundle_item(entry):
        assert parsed.entries.refused == (RefusedEntry("agent", str(refused.value)),)
        shown = entry["externalId"] if HEX32.fullmatch(str(entry.get("externalId"))) else "-"
        assert f"agent {shown} refused: {refused.value}" in caplog.text
        assert MARK not in caplog.text and "../" not in caplog.text
    assert cloud.requests == []  # reading the catalogue fetches nothing


def test_a_second_entry_of_one_type_and_external_id_is_refused(cloud):
    agent = _bundle("agent")
    other_version = _entry({**agent, "version": "1.0.1"})
    parsed = feed.parse_catalog(_catalog(_entry(agent), other_version, _entry({**agent, "type": "plugin"})), None)
    assert [(entry.type, entry.version) for entry in parsed.entries.listed] == [("agent", "1.0.0"), ("plugin", "1.0.0")]
    assert parsed.entries.refused == (RefusedEntry("agent", items_feed.LISTED_TWICE),)


def test_the_entry_keys_are_k11_s_less_its_bundle_path():
    assert set(items_feed.ENTRY_KEYS) == set(_entry(_bundle("agent"))) - {"bundle"}
    assert items_feed.ENTRY_TRUST == "reviewed" and items_feed.BUNDLE_TYPES == TYPE_ORDER


# --------------------------------------------------------------------------- fetching a bundle


@pytest.mark.parametrize("kind", TYPE_ORDER)
def test_a_good_entry_fetches_one_bundle_from_exactly_the_derived_url(cloud, kind):
    bundle = _bundle(kind)
    entry = cloud.serve(bundle)
    fetched = _fetch(cloud, entry)
    assert fetched.refused == () and len(fetched.bundles) == 1
    assert fetched.bundles[0].bundle == bundle and fetched.bundles[0].entry.external_id == bundle["externalId"]
    assert cloud.urls() == [BUNDLES + entry["revision"]]
    request = cloud.requests[0]
    assert request.method == "GET" and set(request.headers.keys()) == SENT_HEADERS
    assert request.headers["user-agent"] == fetch.USER_AGENT
    assert request.headers["accept"] == fetch.ACCEPT_JSON and request.headers["accept-encoding"] == "identity"
    for header in ("authorization", "cookie", "x-workspace-id"):
        assert header not in request.headers


@pytest.mark.parametrize("named", [EVIL + "/steal", "//evil.example.test/x", "/api/elsewhere", "file:///etc/passwd"])
def test_an_entry_whose_bundle_names_another_place_fetches_from_the_derived_url_anyway(cloud, named):
    bundle = _bundle("agent")
    entry = {**cloud.serve(bundle), "bundle": named}
    fetched = _fetch(cloud, entry)
    assert [item.bundle for item in fetched.bundles] == [bundle]
    assert cloud.urls() == [BUNDLES + entry["revision"]]


def _other_bundle(bundle, **changes):
    return {**copy.deepcopy(bundle), **changes}


def _answered(cloud, entry, body, **answer):
    """``entry`` claims the revision of ``body``, which the cloud answers at the derived URL."""
    revised = {**entry, "revision": _revision(body)}
    cloud.answer(BUNDLES + revised["revision"], body, **answer)
    return revised


def _stream(body):
    return httpx.Response(200, content=iter([body[:512], body[512:]]))


def _raise(error):
    raise error


def _case_wrong_sha(cloud, entry):
    cloud.answer(BUNDLES + entry["revision"], canonical_bytes(_other_bundle(_bundle("playbook"), name="Changed")))
    return entry, items_feed.NOT_THE_REVISION


def _case_declared_oversize(cloud, entry):
    return _answered(cloud, entry, b" " * (256 * 1024 + 1)), items_feed.NOT_FETCHED.format(
        reason=fetch.TOO_LARGE.format(kb=256))


def _case_streamed_oversize(cloud, entry):
    body = b" " * (256 * 1024 + 1)
    revised = {**entry, "revision": _revision(body)}
    cloud.answer_with(BUNDLES + revised["revision"], lambda request: _stream(body))
    return revised, items_feed.NOT_FETCHED.format(reason=fetch.TOO_LARGE.format(kb=256))


def _case_redirect(cloud, entry):
    cloud.answer(BUNDLES + entry["revision"], status=302, headers={"location": EVIL + "/bundle"})
    return entry, items_feed.NOT_FETCHED.format(reason=fetch.REDIRECTED.format(status=302))


def _case_encoded(cloud, entry):
    data = canonical_bytes(_bundle("playbook"))
    cloud.answer(BUNDLES + entry["revision"], gzip.compress(data), headers={"content-encoding": "gzip"})
    return entry, items_feed.NOT_FETCHED.format(reason=fetch.ENCODED)


def _case_404(cloud, entry):
    return entry, items_feed.NOT_FETCHED.format(reason=fetch.BAD_STATUS.format(status=404))


def _case_timeout(cloud, entry):
    cloud.answer_with(BUNDLES + entry["revision"], lambda request: _raise(httpx.ReadTimeout("slow", request=request)))
    return entry, items_feed.NOT_FETCHED.format(reason=fetch.TIMED_OUT)


def _case_unreachable(cloud, entry):
    cloud.answer_with(BUNDLES + entry["revision"],
                      lambda request: _raise(httpx.ConnectError("refused", request=request)))
    return entry, items_feed.NOT_FETCHED.format(reason=fetch.UNREACHABLE.format(error="ConnectError"))


def _case_not_json(cloud, entry):
    return _answered(cloud, entry, b"<html>" + MARK.encode()), items_feed.NOT_JSON


def _case_not_utf8(cloud, entry):
    return _answered(cloud, entry, b'{"bundle": 1, "name": "\xff"}'), items_feed.NOT_JSON


def _case_not_canonical(cloud, entry):
    body = json.dumps(_bundle("playbook"), indent=2, sort_keys=True).encode("utf-8")
    return _answered(cloud, entry, body), items_feed.NOT_CANONICAL


def _case_duplicate_key(cloud, entry):
    data = canonical_bytes(_bundle("playbook")).decode("utf-8")
    body = ('{"name":"' + MARK + '",' + data[1:]).encode("utf-8")
    return _answered(cloud, entry, body), items_feed.NOT_CANONICAL


def _case_invalid_bundle(cloud, entry):
    body = canonical_bytes(_other_bundle(_bundle("playbook"), extra=MARK))
    return _answered(cloud, entry, body), items_feed.BAD_BUNDLE.format(
        message="bundle has a key the bundle format does not allow.")


def _case_another_item(cloud, entry):
    body = canonical_bytes(_other_bundle(_bundle("playbook"), externalId="f" * 32))
    return _answered(cloud, entry, body), items_feed.OTHER_ITEM


def _case_another_version(cloud, entry):
    body = canonical_bytes(_other_bundle(_bundle("playbook"), version="9.9.9"))
    return _answered(cloud, entry, body), items_feed.OTHER_ITEM


def _case_another_type(cloud, entry):
    body = canonical_bytes(_other_bundle(_bundle("agent"), externalId=entry["externalId"], name="Another"))
    return _answered(cloud, entry, body), items_feed.OTHER_ITEM


FETCH_REFUSALS = {
    "a body whose sha256 isn't the revision": _case_wrong_sha,
    "an oversize declared body": _case_declared_oversize,
    "an oversize streamed body": _case_streamed_oversize,
    "a redirect": _case_redirect,
    "an encoded body": _case_encoded,
    "a 404": _case_404,
    "a timeout": _case_timeout,
    "an unreachable host": _case_unreachable,
    "a body that isn't JSON": _case_not_json,
    "a body that isn't UTF-8": _case_not_utf8,
    "a body that isn't canonical": _case_not_canonical,
    "a body with a duplicate key": _case_duplicate_key,
    "an invalid bundle": _case_invalid_bundle,
    "another item's bundle": _case_another_item,
    "another version's bundle": _case_another_version,
    "another type's bundle": _case_another_type,
}


@pytest.mark.parametrize("case", sorted(FETCH_REFUSALS))
def test_a_failed_bundle_refuses_that_item_only(cloud, caplog, case):
    good = cloud.serve(_bundle("agent"))
    failing, reason = FETCH_REFUSALS[case](cloud, _entry(_bundle("playbook")))
    caplog.set_level(logging.WARNING, logger=items_feed.__name__)
    fetched = _fetch(cloud, failing, good)
    assert fetched.refused == (RefusedEntry("playbook", reason),)
    assert [item.bundle for item in fetched.bundles] == [_bundle("agent")]
    assert cloud.urls() == [BUNDLES + failing["revision"], BUNDLES + good["revision"]]  # no redirect followed
    assert MARK not in reason and MARK not in caplog.text
    assert f"playbook {failing['externalId']} refused" in caplog.text


@pytest.mark.parametrize(("url", "message"), [
    (CLOUD + "/api", fetch.URL_NOT_ORIGIN),
    ("https://user:" + MARK + "@cloud.example.test", fetch.URL_CREDENTIALS),
    ("http://cloud.example.test", fetch.URL_NOT_HTTPS),
    ("", fetch.CLOUD_OFF),
])
def test_a_setting_that_isnt_an_origin_refuses_every_entry_once_before_any_request(
        cloud, monkeypatch, caplog, url, message):
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", url)
    caplog.set_level(logging.WARNING, logger=items_feed.__name__)
    fetched = _fetch(cloud, cloud.serve(_bundle("agent")), cloud.serve(_bundle("plugin")))
    reason = items_feed.NOT_FETCHED.format(reason=message)
    assert fetched.bundles == () and fetched.refused == (RefusedEntry("agent", reason), RefusedEntry("plugin", reason))
    assert cloud.requests == [] and MARK not in reason and MARK not in caplog.text
    assert caplog.text.count(reason) == 1  # one line for the setting, not one per entry
    with pytest.raises(EntryRefused, match=re.escape(reason)):
        items_feed.fetch_bundle(items_feed.parse_item_entry(cloud.serve(_bundle("agent"))), cloud.factory)
    assert cloud.requests == []


def test_an_entry_built_with_a_bad_revision_is_refused_before_any_request(cloud):
    entry = CatalogEntry("agent", "0" * 32, "1.0.0", "../" + "0" * 61, "X")
    with pytest.raises(EntryRefused, match="entry.revision"):
        items_feed.fetch_bundle(entry, cloud.factory)
    assert cloud.requests == []


def test_the_bundle_url_is_the_origin_and_the_bundle_route():
    revision = "ab" * 32
    assert items_feed.bundle_url(CLOUD, revision) == f"{CLOUD}/api/marketplace/catalog/bundles/{revision}"


def test_items_feed_imports_w1_s_fetch_and_urls_but_never_feed_or_pull():
    tree = ast.parse((ORCH / "services" / "marketplace_bundles" / "items_feed.py").read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert {"services.marketplace_cloud.fetch", "services.marketplace_cloud.urls"} <= imported
    assert not imported & {"services.marketplace_cloud.feed", "services.marketplace_cloud.pull",
                           "services.marketplace_cloud", "httpx"}


# --------------------------------------------------------------------------- unchanged entries (Postgres)


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(sa.text("SELECT revision FROM marketplace_cloud_items LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the unchanged-entry tests need Postgres with the PRD-270 W2 schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def held(engine, new_session):
    """A marketplace agent this test made, and a mapping of a fresh agent bundle to it at its revision."""
    tag = f"prd270w2-feed-{uuid.uuid4().hex[:10]}"
    bundle = _other_bundle(_bundle("agent"), externalId=uuid.uuid4().hex)
    revision = _revision(canonical_bytes(bundle))
    session = new_session()
    agent_id = session.execute(sa.text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type) "
        "VALUES (:name, 'custom', NULL, 'active', CAST('{}' AS json), 'marketplace') RETURNING id"
    ), {"name": tag}).scalar()
    session.execute(sa.text(
        "INSERT INTO marketplace_cloud_items (item_type, external_id, version, revision, agent_id) "
        "VALUES ('agent', :external_id, :version, :revision, :agent_id)"
    ), {"external_id": bundle["externalId"], "version": bundle["version"], "revision": revision, "agent_id": agent_id})
    session.commit()
    yield bundle, session
    sweep = new_session.sweep()
    sweep.execute(sa.text("DELETE FROM agents WHERE name = :name"), {"name": tag})  # the mapping cascades
    sweep.commit()


def test_an_unchanged_entry_fetches_nothing_and_a_changed_one_is_fetched(cloud, held):
    bundle, session = held
    unchanged = items_feed.parse_item_entry(cloud.serve(bundle))
    assert items_feed.changed_entries(session, [unchanged]) == ()
    assert items_feed.fetch_bundles(items_feed.changed_entries(session, [unchanged]), cloud.factory).bundles == ()
    assert cloud.requests == []
    moved = items_feed.parse_item_entry(cloud.serve(_other_bundle(bundle, description="A newer description.")))
    as_plugin = unchanged._replace(type="plugin")
    unmapped = items_feed.parse_item_entry(cloud.serve(_other_bundle(bundle, externalId=uuid.uuid4().hex)))
    changed = items_feed.changed_entries(session, [unchanged, moved, as_plugin, unmapped])
    assert changed == (moved, as_plugin, unmapped)
    assert items_feed.changed_entries(session, []) == ()


def test_entries_is_immutable_reading_returns_new_entries():
    first = items_feed.NO_ENTRIES
    after = first.read(_entry(_bundle("agent")))
    assert first == Entries() and len(after.listed) == 1 and after is not first
