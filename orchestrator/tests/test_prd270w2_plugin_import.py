"""PRD-270 W2 US-270-15a (§5.6, K15) — locally: Automatos Cloud's plugins, checked like an upload and stored.

Automatos Cloud is an ``httpx.MockTransport`` answering by URL and recording every request (W1's client-factory
seam, as ``test_prd270w2_items_feed.py``), the object store a fake S3 client monkeypatched into
``core.services.marketplace_s3`` (the ``test_prd261_theme_files.py:632`` pattern), and ``run_async`` is
``asyncio.run``, recording each coroutine it runs. The LLM scan and ``PluginScanService.scan_plugin`` are
patched to fail any test that reaches them. Proven:

* the zip rules are the upload's own (the constants pinned against ``plugin_upload_service.py``'s lines), and
  ``package_files`` refuses 501 files, an unsafe path, a file or a total over its cap, a ratio over 100, a name
  twice and a zip it can't read, with a reason that echoes nothing the package held; the local static scan
  reads in chunks, keeps each finding's line in the file, and refuses a critical finding, a line over its cap
  and too many findings;
* on Postgres, a good plugin is stored under ``plugins/<slug>/<version>/``, approved and active, its scan row
  holding Automatos Cloud's report and this install's static result, its mapping and its 'cloud-sync' history
  written, its SKILL.md a global skill with package_slug the slug; only the derived URLs are requested, the
  static scan, the store and the materialiser each ran through ``run_async``, and no transaction was open
  during a fetch, a storage write or after the pass;
* a package over the cap, with 501 files, a '..' path, a ratio over 100, another manifest slug or version, or
  a critical static finding stores nothing and writes no row; so does a storage failure after a partial write,
  a database failure removes the files it stored, and files left under the prefix before refuse the import and
  are removed (the next Sync imports it); a writer that committed the same slug meanwhile keeps its files;
* a slug a row without a mapping holds is refused before its package is fetched, and that row and its files
  stay as they were; a changed revision at the same plugin version is refused; the same revision is unchanged
  and fetches nothing; the sixth plugin by name waits and is never fetched;
* a newer version moves the same row and keeps its enablements, its old files removed; a failed update keeps
  the row and its files at the old version; an update keeps a local deactivation and makes no skill; a mapped
  item whose bundle names another slug is refused;
* the module never names the LLM scan or the upload flow.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import io
import json
import logging
import random
import uuid
import warnings
import zipfile
from pathlib import Path

import httpx
import pytest
import sqlalchemy as sa

import core.models  # noqa: F401  (registers every table the foreign keys name)
from config import config
from core.services import marketplace_s3, plugin_security_scanner
from services.marketplace_bundles import items_feed, plugin_import
from services.marketplace_bundles.canonical import canonical_bytes
from services.marketplace_bundles.catalog_items import entry_of
from services.marketplace_bundles.catalog_paths import plugin_package_path
from services.marketplace_bundles.exported import ExportedItem, ItemEntry
from services.marketplace_bundles.plugin_import import PluginRefused
from services.marketplace_cloud.fetch import ACCEPT_PACKAGE, USER_AGENT

ORCH = Path(__file__).resolve().parents[1]
FIXTURES = ORCH / "tests" / "fixtures" / "marketplace_bundles"
MODULE = ORCH / "services" / "marketplace_bundles" / "plugin_import.py"
UPLOAD_SERVICE = ORCH / "core" / "services" / "plugin_upload_service.py"
S3_SERVICE = ORCH / "core" / "services" / "marketplace_s3.py"
CLOUD = "https://cloud.example.test"
BUNDLES = CLOUD + "/api/marketplace/catalog/bundles/"
MARK = "SECRET-MARK-7f3a"
BINARY = bytes(range(256))
APPEND_LINE = "    open(notes_path, 'a')"  # a medium static finding (file_write), never critical
SENT_HEADERS = {"host", "accept", "accept-encoding", "user-agent", "connection"}


# --------------------------------------------------------------------------- the plugin, its bundle and package


def _external_id(slug):
    return hashlib.sha256(slug.encode()).hexdigest()[:32]


def _bundle(slug, version="1.0.0", name="Test Research Kit", **changes):
    base = json.loads((FIXTURES / "plugin.json").read_text(encoding="utf-8"))
    plugin = {**base["plugin"], "slug": slug}
    return {**base, "externalId": _external_id(slug), "version": version, "name": name, "plugin": plugin, **changes}


def _skill_md(skill, body="Summarise the sources."):
    return f"---\nname: {skill}\ndescription: A test skill.\n---\n\n# Summarise\n\n{body}\n".encode()


def _files(slug, version, skill=None, manifest=None, extra=None, body="Summarise the sources."):
    """A plugin's files: its manifest (``manifest`` overriding its keys), a SKILL.md (its skill named after the
    slug unless ``skill`` names it), a command with a medium static finding and a binary file."""
    named = {"slug": slug, "name": "Test Research Kit", "version": version, **(manifest or {})}
    skill = skill or f"{slug}-skill"
    return {
        "manifest.json": json.dumps(named).encode(),
        f"skills/{skill}/SKILL.md": _skill_md(skill, body),
        "commands/save-notes.md": f"Save the notes:\n\n{APPEND_LINE}\n".encode(),
        "assets/logo.png": BINARY,
        **(extra or {}),
    }


def _zip(files, compression=zipfile.ZIP_DEFLATED, folders=()):
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # zipfile warns on a name written twice
        with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
            for folder in folders:
                archive.writestr(folder, b"")
            for path, data in (files.items() if isinstance(files, dict) else files):
                archive.writestr(path, data)
    return buffer.getvalue()


def _incompressible(size, seed=0):
    return random.Random(seed).randbytes(size)


def _package_url(slug, version):
    return CLOUD + plugin_package_path(slug, version)


class Cloud:
    """Automatos Cloud as a mock transport: a responder per URL (404 for any other), each request recorded."""

    def __init__(self):
        self.requests = []
        self.answers = {}
        self.on_request = None  # called on every request (the open-transaction check)

    def answer(self, url, body):
        self.answers[url] = body

    def serve(self, bundle, package):
        """``bundle`` at its derived URL and ``package`` at its plugin's; the parsed catalogue entry."""
        data = canonical_bytes(bundle)
        revision = hashlib.sha256(data).hexdigest()
        self.answer(BUNDLES + revision, data)
        self.answer(_package_url(bundle["plugin"]["slug"], bundle["version"]), package)
        item = ExportedItem(ItemEntry("plugin", bundle["externalId"], bundle["version"], frozenset()), bundle, revision)
        return items_feed.parse_item_entry(entry_of(item))

    def handler(self, request):
        self.requests.append(request)
        if self.on_request is not None:
            self.on_request()
        body = self.answers.get(str(request.url))
        return httpx.Response(404) if body is None else httpx.Response(200, content=body)

    def factory(self):
        return httpx.Client(transport=httpx.MockTransport(self.handler), follow_redirects=True)

    def urls(self):
        return [str(request.url) for request in self.requests]


class _Body:
    def __init__(self, data):
        self.data = data

    def read(self, amt=None):
        return self.data if amt is None else self.data[:amt]


class Bucket:
    """The four S3 calls MarketplaceS3Service makes, on a dict of (bucket, key) -> bytes."""

    def __init__(self):
        self.objects = {}
        self.puts = 0
        self.fail_put_at = None  # the put (1-based) that raises
        self.on_write = None  # called on every put and delete (the open-transaction check)

    def put_object(self, Bucket, Key, Body, **_):  # noqa: N803 — boto3's argument names
        if self.on_write is not None:
            self.on_write()
        self.puts += 1
        if self.fail_put_at is not None and self.puts >= self.fail_put_at:
            raise OSError("storage went away")
        self.objects[(Bucket, Key)] = bytes(Body)

    def list_objects_v2(self, Bucket, Prefix, **_):  # noqa: N803
        keys = sorted(key for bucket, key in self.objects if bucket == Bucket and key.startswith(Prefix))
        return {"Contents": [{"Key": key} for key in keys], "IsTruncated": False}

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        if self.on_write is not None:
            self.on_write()
        for item in Delete["Objects"]:
            self.objects.pop((Bucket, item["Key"]), None)
        return {"Deleted": Delete["Objects"]}

    def keys(self, prefix=""):
        return sorted(key for _, key in self.objects if key.startswith(prefix))


class Runner:
    """``run_async``: ``asyncio.run``, recording the name of every coroutine it runs."""

    def __init__(self):
        self.ran = []

    def __call__(self, coroutine):
        self.ran.append(coroutine.__name__)
        return asyncio.run(coroutine)

    def kinds(self):
        """The coroutines it ran, each name once, in the order first run."""
        return list(dict.fromkeys(self.ran))


def _no_model(*_args, **_kwargs):
    raise AssertionError("a Sync never calls a model")


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(plugin_security_scanner, "llm_security_scan", _no_model)
    monkeypatch.setattr(plugin_security_scanner.PluginScanService, "scan_plugin", _no_model)


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", CLOUD)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_BUNDLE_MAX_KB", 256)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_TIMEOUT_SECONDS", 15)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", 5)
    monkeypatch.setattr(config, "PLUGIN_MAX_UPLOAD_SIZE_MB", 10)
    return Cloud()


@pytest.fixture
def bucket(monkeypatch):
    fake = Bucket()
    monkeypatch.setattr(marketplace_s3, "get_s3_client", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(marketplace_s3, "ensure_bucket", lambda *_args, **_kwargs: True)
    return fake


# --------------------------------------------------------------------------- the zip rules (no database)


def test_the_zip_rules_are_the_upload_s_own():
    upload = {line.strip() for line in UPLOAD_SERVICE.read_text(encoding="utf-8").splitlines()}
    pinned = {
        "MAX_FILE_COUNT = 500": plugin_import.MAX_FILE_COUNT == 500,
        "MAX_FILE_UNCOMPRESSED_SIZE = 10 * 1024 * 1024  # 10 MB per file":
            plugin_import.MAX_FILE_UNCOMPRESSED_SIZE == 10 * 1024 * 1024,
        "MAX_TOTAL_UNCOMPRESSED_SIZE = 50 * 1024 * 1024  # 50 MB total":
            plugin_import.MAX_TOTAL_UNCOMPRESSED_SIZE == 50 * 1024 * 1024,
        "MAX_COMPRESSION_RATIO = 100  # reject suspiciously high ratios (zip bomb)":
            plugin_import.MAX_COMPRESSION_RATIO == 100,
    }
    for line, same in pinned.items():
        assert line in upload and same, line
    extract = S3_SERVICE.read_text(encoding="utf-8")
    assert 'if member_path.is_absolute() or ".." in member_path.parts:' in extract


def test_package_files_reads_a_good_package_and_leaves_folders_out():
    files = _files("test-kit", "1.0.0", "test-skill")
    assert plugin_import.package_files(_zip(files, folders=("skills/", "assets/"))) == files


def _many_files():
    return {**_files("test-kit", "1.0.0"), **{f"notes/n{index}.md": b"x" for index in range(497)}}


def _twice():
    return [("manifest.json", b"{}"), ("a.md", b"one"), ("a.md", b"two")]


def _bad_crc():
    data = bytearray(_zip({"a.md": b"plain text that is stored"}, compression=zipfile.ZIP_STORED))
    at = data.index(b"plain text")
    data[at] ^= 0x01
    return bytes(data)


@pytest.mark.parametrize(("package", "reason"), [
    (lambda: _zip(_many_files()), plugin_import.TOO_MANY_FILES.format(count=500)),
    (lambda: _zip({f"../{MARK}.md": b"x"}), plugin_import.UNSAFE_PATH),
    (lambda: _zip({f"/{MARK}.md": b"x"}), plugin_import.UNSAFE_PATH),
    (lambda: _zip({f"skills\\{MARK}.md": b"x"}), plugin_import.UNSAFE_PATH),
    (lambda: _zip({f"./{MARK}.md": b"x"}), plugin_import.UNSAFE_PATH),
    (lambda: _zip({".": b"x"}), plugin_import.UNSAFE_PATH),
    (lambda: _zip({f"{MARK}.bin": _incompressible(10 * 1024 * 1024 + 1)}, compression=zipfile.ZIP_STORED),
     plugin_import.FILE_TOO_LARGE.format(mb=10)),
    (lambda: _zip({f"{MARK}.txt": b"\0" * (1024 * 1024)}), plugin_import.RATIO_TOO_HIGH.format(ratio=100)),
    (lambda: _zip({"a.bin": _incompressible(6 * 1024 * 1024), "b.bin": _incompressible(5 * 1024 * 1024, 1)},
                  compression=zipfile.ZIP_STORED), plugin_import.FILES_TOO_LARGE.format(mb=10)),
    (lambda: _zip(_twice()), plugin_import.NAMED_TWICE),
    (lambda: f"not a zip {MARK}".encode(), plugin_import.NOT_A_ZIP),
    (_bad_crc, plugin_import.NOT_A_ZIP),
])
def test_package_files_refuses_what_the_upload_refuses_echoing_nothing(cloud, package, reason):
    with pytest.raises(PluginRefused) as refused:
        plugin_import.package_files(package())
    assert str(refused.value) == reason
    assert MARK not in str(refused.value)


def test_the_total_cap_is_the_smaller_of_the_upload_s_and_the_s3_cap(cloud, monkeypatch):
    monkeypatch.setattr(config, "PLUGIN_MAX_UPLOAD_SIZE_MB", 1)
    package = _zip({"a.bin": _incompressible(700 * 1024), "b.bin": _incompressible(400 * 1024, 1)},
                   compression=zipfile.ZIP_STORED)
    with pytest.raises(PluginRefused, match="larger than 1 MB in all"):
        plugin_import.package_files(package)


def test_the_local_scan_reads_in_chunks_and_keeps_each_finding_s_line():
    lines = ["Plain text."] * 450
    lines[419] = APPEND_LINE  # line 420: past two chunks of SCAN_CHUNK_LINES
    runner = Runner()
    result = plugin_import.local_scan({"commands/notes.md": "\n".join(lines), "README.md": "Plain."}, runner)
    assert (result.status, [(f.file, f.line, f.severity) for f in result.findings]) == (
        "flagged", [("commands/notes.md", 420, "medium")],
    )
    assert runner.ran == ["static_scan"] * 4  # README.md, then notes.md in three chunks
    assert plugin_import.local_scan({"a.md": "Plain."}, Runner()).status == "passed"


@pytest.mark.parametrize(("texts", "reason"), [
    ({"a.md": "x" * 8193}, plugin_import.LINE_TOO_LONG.format(count=8192)),
    ({"a.md": "    os.makedirs(here)\n" * 201}, plugin_import.TOO_MANY_FINDINGS.format(count=200)),
    # Plain text the scan reads, never run: `eval(` is one of its critical patterns.
    ({"a.md": "Plain.\n" * 300 + "    eval(text)\n"}, plugin_import.CRITICAL_FINDING),
])
def test_the_local_scan_refuses_a_long_line_too_many_findings_and_a_critical_one(texts, reason):
    with pytest.raises(PluginRefused) as refused:
        plugin_import.local_scan(texts, Runner())
    assert str(refused.value) == reason


def test_the_first_plugins_by_name_are_chosen_and_the_rest_wait():
    names = ["Test Plugin F", "test plugin b", "Test Plugin A", "Test Plugin E", "Test Plugin C", "Test Plugin D"]
    entries = [items_feed.CatalogEntry("plugin", f"{index:032x}", "1.0.0", "a" * 64, name)
               for index, name in enumerate(names)]
    chosen, waiting = plugin_import.chosen_entries(entries, 5)
    assert [entry.name for entry in chosen] == [
        "Test Plugin A", "test plugin b", "Test Plugin C", "Test Plugin D", "Test Plugin E",
    ]
    assert [entry.name for entry in waiting] == ["Test Plugin F"]
    assert plugin_import.chosen_entries(entries, 0) == ((), tuple(sorted(entries, key=lambda e: e.name.casefold())))


class _NoRows:
    """A session for the passes that read no row: ``changed_entries`` is patched, and rollback does nothing."""

    def rollback(self):
        return None


def test_other_types_are_left_to_the_item_pass_and_nothing_is_fetched(cloud, bucket):
    agent = items_feed.CatalogEntry("agent", "0" * 32, "1.0.0", "a" * 64, "Test Agent")
    imported = plugin_import.import_plugins(_NoRows(), [agent], cloud.factory, Runner())
    assert imported.counts.view() == {"new": 0, "updated": 0, "unchanged": 0, "refused": 0}
    assert imported.waiting == 0 and cloud.requests == [] and bucket.puts == 0


def test_a_setting_that_isnt_an_origin_refuses_every_chosen_plugin_before_any_request(cloud, bucket, monkeypatch):
    monkeypatch.setattr(plugin_import, "changed_entries", lambda _db, entries: tuple(entries))
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", "")
    entry = cloud.serve(_bundle("test-kit"), _zip(_files("test-kit", "1.0.0")))
    imported = plugin_import.import_plugins(_NoRows(), [entry], cloud.factory, Runner())
    assert imported.counts.refused == 1 and cloud.requests == [] and bucket.puts == 0


class _Rows:
    """A session stand-in: its one query answers ``found`` (or fails), and it counts its rollbacks."""

    def __init__(self, found=None, fails=False):
        self.found, self.fails, self.rollbacks = found, fails, 0

    def query(self, *_columns):
        if self.fails:
            raise sa.exc.OperationalError("SELECT", {}, Exception("the database went away"))
        return self

    def filter(self, *_criteria):
        return self

    def first(self):
        return self.found

    def rollback(self):
        self.rollbacks += 1


class _Store:
    """The storage seam's delete, recorded."""

    def __init__(self):
        self.deleted = []

    async def delete_plugin(self, slug, version):
        self.deleted.append((slug, version))
        return 0


def _seams(store=None):
    return plugin_import.Seams(CLOUD, None, Runner(), store or _Store())


def test_a_failed_read_of_the_rows_refuses_the_plugin_and_ends_the_read():
    entry = items_feed.CatalogEntry("plugin", "0" * 32, "1.0.0", "a" * 64, "Test Kit")
    rows = _Rows(fails=True)
    with pytest.raises(PluginRefused) as refused:
        plugin_import.held_target(rows, entry, "test-kit", "1.0.0")
    assert str(refused.value) == plugin_import.NOT_READ and rows.rollbacks == 1


def test_anything_unforeseen_in_the_checks_refuses_that_plugin_only(monkeypatch):
    def broken(_db, _fetched, _seams):
        raise RuntimeError("the scanner broke")

    monkeypatch.setattr(plugin_import, "checked_plugin", broken)
    entry = items_feed.CatalogEntry("plugin", "0" * 32, "1.0.0", "a" * 64, "Test Kit")
    rows = _Rows()
    verdict = plugin_import.import_plugin(rows, items_feed.FetchedBundle(entry, _bundle("test-kit")), _seams())
    assert verdict == "refused" and rows.rollbacks == 1


def test_files_a_row_names_or_may_name_are_kept_and_others_removed():
    store = _Store()
    plugin_import._discard(_Rows(found=("a-row",)), _seams(store), "test-kit", "1.0.0")
    plugin_import._discard(_Rows(fails=True), _seams(store), "test-kit", "1.0.0")  # unsure: kept
    assert store.deleted == []
    plugin_import._discard(_Rows(found=None), _seams(store), "test-kit", "1.0.0")
    assert store.deleted == [("test-kit", "1.0.0")]


def test_plugin_import_never_names_the_llm_scan_or_the_upload_flow():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    banned = {"llm_security_scan", "PluginScanService", "scan_plugin", "PluginUploadService", "upload_plugin"}
    assert not (names | imported) & banned
    upload_imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                      and node.module == "core.services.plugin_upload_service" for alias in node.names}
    assert upload_imports == {"_auto_categorise"}


# --------------------------------------------------------------------------- the import (Postgres)


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            for probe in (
                "SELECT plugin_id FROM marketplace_cloud_items LIMIT 1",
                "SELECT security_scan_id, materialized_skill_ids FROM marketplace_plugins LIMIT 1",
                "SELECT scanned_by FROM plugin_security_scans LIMIT 1",
                "SELECT details FROM plugin_sync_history LIMIT 1",
                "SELECT package_slug FROM skills LIMIT 1",
                "SELECT plugin_id FROM workspace_enabled_plugins LIMIT 1",
            ):
                conn.execute(sa.text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the plugin import tests need Postgres with the PRD-270 W2 schema: {exc}")
    yield eng
    eng.dispose()


class Made:
    """The slugs, skill names and workspaces a test made, swept after it."""

    def __init__(self):
        self.slugs, self.workspaces = [], []

    def slug(self):
        made = f"t270pi-{uuid.uuid4().hex[:12]}"
        self.slugs.append(made)
        return made


@pytest.fixture
def made(engine, new_session):
    tracked = Made()
    yield tracked
    sweep = new_session.sweep()
    for slug in tracked.slugs:
        for statement in (
            "DELETE FROM skills WHERE package_slug = :slug",
            "DELETE FROM plugin_sync_history WHERE plugin_slug = :slug",
            "DELETE FROM marketplace_plugins WHERE slug = :slug",  # its mapping and enablements cascade
            "DELETE FROM plugin_security_scans WHERE plugin_slug = :slug",
        ):
            sweep.execute(sa.text(statement), {"slug": slug})
    for workspace in tracked.workspaces:
        sweep.execute(sa.text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": workspace})
    sweep.commit()


def _import(new_session, cloud, *entries, runner=None):
    return plugin_import.import_plugins(new_session(), list(entries), cloud.factory, runner or Runner())


def _rows(new_session, slug):
    """Every row the import writes for ``slug``: the plugin, its scans, its history and its mapping."""
    session = new_session()
    found = {
        table: session.execute(sa.text(f"SELECT count(*) FROM {table} WHERE {column} = :slug"), {"slug": slug}).scalar()
        for table, column in (("marketplace_plugins", "slug"), ("plugin_security_scans", "plugin_slug"),
                              ("plugin_sync_history", "plugin_slug"))
    }
    found["marketplace_cloud_items"] = session.execute(sa.text(
        "SELECT count(*) FROM marketplace_cloud_items WHERE item_type = 'plugin' AND external_id = :id"
    ), {"id": _external_id(slug)}).scalar()
    session.rollback()
    return found


NO_ROWS = {"marketplace_plugins": 0, "plugin_security_scans": 0, "plugin_sync_history": 0, "marketplace_cloud_items": 0}


def _plugin(new_session, slug):
    session = new_session()
    row = session.execute(sa.text(
        "SELECT p.*, s.overall_verdict, s.scanned_by, s.static_scan_status, s.static_findings, s.llm_scan_status,"
        " s.llm_risk_score, s.llm_findings, s.llm_summary, s.llm_model_used, s.plugin_version AS scan_version"
        " FROM marketplace_plugins p JOIN plugin_security_scans s ON s.id = p.security_scan_id WHERE p.slug = :slug"
    ), {"slug": slug}).mappings().one()
    session.rollback()
    return row


def _scalar(new_session, statement, **params):
    session = new_session()
    value = session.execute(sa.text(statement), params).scalar()
    session.rollback()
    return value


def test_a_good_plugin_is_stored_approved_and_its_skill_becomes_a_global_skill(cloud, bucket, made, new_session):
    slug, skill = made.slug(), f"t270pi-skill-{uuid.uuid4().hex[:8]}"
    bundle = _bundle(slug)
    entry = cloud.serve(bundle, _zip(_files(slug, "1.0.0", skill)))
    runner, session, open_during_fetch, open_during_store = Runner(), new_session(), [], []
    cloud.on_request = lambda: open_during_fetch.append(session.in_transaction())
    bucket.on_write = lambda: open_during_store.append(session.in_transaction())
    imported = plugin_import.import_plugins(session, [entry], cloud.factory, runner)
    assert imported.counts.view() == {"new": 1, "updated": 0, "unchanged": 0, "refused": 0} and imported.waiting == 0
    assert cloud.urls() == [BUNDLES + entry.revision, _package_url(slug, "1.0.0")]
    # No transaction is held across a network read or a storage write, nor left open after the pass.
    assert open_during_fetch == [False, False]
    assert open_during_store and not any(open_during_store) and not session.in_transaction()
    package_request = cloud.requests[1]
    assert set(package_request.headers.keys()) <= SENT_HEADERS
    assert package_request.headers["accept"] == ACCEPT_PACKAGE and package_request.headers["user-agent"] == USER_AGENT
    assert runner.kinds() == ["static_scan", "extract_plugin", "list_plugin_files", "materialize_plugin"]
    prefix = f"plugins/{slug}/1.0.0/"
    assert bucket.keys(prefix) == sorted(prefix + path for path in _files(slug, "1.0.0", skill))
    row = _plugin(new_session, slug)
    assert (row["approval_status"], row["approved_by"], row["is_active"]) == ("approved", "automatos-cloud", True)
    assert row["approved_at"] is not None and row["security_status"] == "safe"
    assert (row["version"], row["s3_path"], row["name"]) == ("1.0.0", prefix, bundle["name"])
    assert (row["source_type"], row["source_url"]) == ("automatos-cloud", CLOUD)
    assert row["long_description"] == bundle["plugin"]["longDescription"] and row["license"] == "MIT"
    assert (row["skills_count"], row["commands_count"], row["token_estimate"]) == (2, 1, 12000)
    assert list(row["recommended_models"]) == bundle["plugin"]["recommendedModels"]
    report = bundle["plugin"]["scan"]
    assert (row["overall_verdict"], row["scanned_by"], row["scan_version"]) == ("safe", "automatos-cloud", "1.0.0")
    assert (row["llm_scan_status"], row["llm_risk_score"]) == ("passed", report["riskScore"])
    assert (row["llm_findings"], row["llm_summary"], row["llm_model_used"]) == (report["findings"], report["summary"], None)
    assert row["static_scan_status"] == "flagged"  # THIS install's result over the package it stored
    assert [(f["file"], f["line"], f["severity"]) for f in row["static_findings"]] == [
        ("commands/save-notes.md", 3, "medium"),
    ]
    assert _scalar(new_session, "SELECT revision FROM marketplace_cloud_items WHERE plugin_id = CAST(:id AS uuid)",
                   id=str(row["id"])) == entry.revision
    assert _scalar(new_session, "SELECT action FROM plugin_sync_history WHERE plugin_id = CAST(:id AS uuid)",
                   id=str(row["id"])) == "cloud-sync"
    skills = new_session().execute(sa.text(
        "SELECT name, package_slug, workspace_id, is_active, skill_source FROM skills WHERE package_slug = :slug"
    ), {"slug": slug}).all()
    assert [tuple(found) for found in skills] == [(skill, slug, None, True, f"plugin:{slug}")]
    assert len(row["materialized_skill_ids"]) == 1


def _over_the_cap(slug):
    return _zip({**_files(slug, "1.0.0"), "big.bin": _incompressible(1100 * 1024)}, compression=zipfile.ZIP_STORED)


def _critical(slug):
    # Plain text the static scan reads, never run: `eval(` is one of its critical patterns.
    return _zip(_files(slug, "1.0.0", extra={"commands/run.md": b"Then run:\n\n    eval(user_text)\n"}))


@pytest.mark.parametrize(("package", "reason"), [
    pytest.param(_over_the_cap, "Its package couldn't be fetched: Automatos Cloud's answer is larger than 1024 KB.",
                 id="over-the-cap"),
    pytest.param(lambda slug: _zip({**_files(slug, "1.0.0"), **{f"n/{i}.md": b"x" for i in range(497)}}),
                 plugin_import.TOO_MANY_FILES.format(count=500), id="501-files"),
    pytest.param(lambda slug: _zip(_files(slug, "1.0.0", extra={"../escape.md": b"x"})), plugin_import.UNSAFE_PATH,
                 id="dotdot-path"),
    pytest.param(lambda slug: _zip(_files(slug, "1.0.0", extra={"zeros.txt": b"\0" * (1024 * 1024)})),
                 plugin_import.RATIO_TOO_HIGH.format(ratio=100), id="ratio-over-100"),
    pytest.param(lambda slug: _zip(_files(slug, "1.0.0", manifest={"slug": "another-plugin"})),
                 "Its package isn't the bundle's plugin: its manifest.json names another slug or version",
                 id="another-manifest-slug"),
    pytest.param(lambda slug: _zip(_files(slug, "1.0.0", manifest={"version": "1.0.1"})),
                 "Its package isn't the bundle's plugin: its manifest.json names another slug or version",
                 id="another-manifest-version"),
    pytest.param(_critical, plugin_import.CRITICAL_FINDING, id="critical-static-finding"),
])
def test_a_package_that_fails_a_check_stores_nothing_and_writes_no_row(
    cloud, bucket, made, new_session, monkeypatch, caplog, package, reason,
):
    monkeypatch.setattr(config, "PLUGIN_MAX_UPLOAD_SIZE_MB", 1)
    slug = made.slug()
    with caplog.at_level(logging.WARNING, logger=plugin_import.__name__):
        imported = _import(new_session, cloud, cloud.serve(_bundle(slug), package(slug)))
    assert imported.counts.view() == {"new": 0, "updated": 0, "unchanged": 0, "refused": 1}
    assert f"plugin {_external_id(slug)} refused: {reason}" in caplog.text
    assert bucket.puts == 0 and bucket.keys() == []
    assert _rows(new_session, slug) == NO_ROWS


def test_a_storage_failure_after_a_partial_write_leaves_no_object_and_no_row(cloud, bucket, made, new_session):
    slug = made.slug()
    bucket.fail_put_at = 2
    imported = _import(new_session, cloud, cloud.serve(_bundle(slug), _zip(_files(slug, "1.0.0"))))
    assert imported.counts.refused == 1 and imported.counts.new == 0
    assert bucket.puts >= 2 and bucket.keys() == []
    assert _rows(new_session, slug) == NO_ROWS


def test_a_database_failure_removes_the_stored_files_and_writes_no_row(cloud, bucket, made, new_session, monkeypatch):
    slug = made.slug()
    real = plugin_import.scan_row
    monkeypatch.setattr(plugin_import, "scan_row", lambda checked: _without_version(real(checked)))
    imported = _import(new_session, cloud, cloud.serve(_bundle(slug), _zip(_files(slug, "1.0.0"))))
    assert imported.counts.refused == 1
    assert bucket.puts > 0 and bucket.keys() == []
    assert _rows(new_session, slug) == NO_ROWS


def _without_version(scan):
    scan.plugin_version = None  # NOT NULL: the flush fails
    return scan


def test_a_slug_held_by_a_row_without_a_mapping_is_refused_and_that_row_and_its_files_stay(
    cloud, bucket, made, new_session,
):
    slug = made.slug()
    session = new_session()
    session.execute(sa.text(
        "INSERT INTO marketplace_plugins (id, slug, name, version, s3_path, description, approval_status, is_active)"
        " VALUES (CAST(:id AS uuid), :slug, 'Operator Plugin', '0.9.0', :path, 'Ours.', 'pending', true)"
    ), {"id": str(uuid.uuid4()), "slug": slug, "path": f"plugins/{slug}/0.9.0/"})
    session.commit()
    bucket.objects[(config.MARKETPLACE_S3_BUCKET, f"plugins/{slug}/0.9.0/manifest.json")] = b"{}"
    before = dict(_scalar_row(new_session, slug))
    entry = cloud.serve(_bundle(slug, version="0.9.0"), _zip(_files(slug, "0.9.0")))
    imported = _import(new_session, cloud, entry)
    assert imported.counts.refused == 1
    assert cloud.urls() == [BUNDLES + entry.revision]  # refused before its package is fetched
    assert dict(_scalar_row(new_session, slug)) == before
    assert bucket.keys() == [f"plugins/{slug}/0.9.0/manifest.json"] and bucket.puts == 0
    assert _rows(new_session, slug) == {**NO_ROWS, "marketplace_plugins": 1}


def _scalar_row(new_session, slug):
    session = new_session()
    row = session.execute(sa.text("SELECT * FROM marketplace_plugins WHERE slug = :slug"), {"slug": slug}).mappings().one()
    session.rollback()
    return row


def test_the_sixth_plugin_by_name_waits_and_is_never_fetched(cloud, bucket, made, new_session):
    served = {}
    for letter in "FBDACE":
        slug = made.slug()
        served[letter] = (slug, cloud.serve(_bundle(slug, name=f"Test Plugin {letter}"), _zip(_files(slug, "1.0.0"))))
    imported = _import(new_session, cloud, *(entry for _, entry in served.values()))
    assert imported.counts.view() == {"new": 5, "updated": 0, "unchanged": 0, "refused": 0} and imported.waiting == 1
    waiting_slug, waiting_entry = served["F"]
    assert not any(waiting_slug in url or waiting_entry.revision in url for url in cloud.urls())
    assert _rows(new_session, waiting_slug) == NO_ROWS
    assert all(_rows(new_session, served[letter][0])["marketplace_plugins"] == 1 for letter in "ABCDE")


def test_the_same_revision_again_is_unchanged_and_fetches_nothing(cloud, bucket, made, new_session):
    slug = made.slug()
    entry = cloud.serve(_bundle(slug), _zip(_files(slug, "1.0.0")))
    assert _import(new_session, cloud, entry).counts.new == 1
    cloud.requests.clear()
    again = _import(new_session, cloud, entry)
    assert again.counts.view() == {"new": 0, "updated": 0, "unchanged": 1, "refused": 0}
    assert cloud.requests == []


def test_a_changed_revision_at_the_same_version_is_refused_and_the_row_kept(cloud, bucket, made, new_session):
    slug = made.slug()
    assert _import(new_session, cloud, cloud.serve(_bundle(slug), _zip(_files(slug, "1.0.0")))).counts.new == 1
    before = dict(_scalar_row(new_session, slug))
    changed = cloud.serve(_bundle(slug, description="A newer description."), _zip(_files(slug, "1.0.0")))
    cloud.requests.clear()
    imported = _import(new_session, cloud, changed)
    assert imported.counts.refused == 1
    assert cloud.urls() == [BUNDLES + changed.revision]
    assert dict(_scalar_row(new_session, slug)) == before
    assert _rows(new_session, slug)["plugin_security_scans"] == 1


def test_a_newer_version_moves_the_row_and_keeps_its_enablements(cloud, bucket, made, new_session):
    slug, skill = made.slug(), f"t270pi-skill-{uuid.uuid4().hex[:8]}"
    first = cloud.serve(_bundle(slug), _zip(_files(slug, "1.0.0", skill)))
    assert _import(new_session, cloud, first).counts.new == 1
    old = dict(_scalar_row(new_session, slug))
    workspace = str(uuid.uuid4())
    made.workspaces.append(workspace)
    session = new_session()
    session.execute(sa.text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd270w2 plugin import')"),
                    {"id": workspace})
    session.execute(sa.text(
        "INSERT INTO workspace_enabled_plugins (workspace_id, plugin_id) VALUES (CAST(:ws AS uuid), CAST(:plugin AS uuid))"
    ), {"ws": workspace, "plugin": str(old["id"])})
    session.commit()
    newer = cloud.serve(_bundle(slug, version="1.1.0"), _zip(_files(slug, "1.1.0", skill, body="Summarise, then cite.")))
    imported = _import(new_session, cloud, newer)
    assert imported.counts.view() == {"new": 0, "updated": 1, "unchanged": 0, "refused": 0}
    moved = dict(_scalar_row(new_session, slug))
    assert moved["id"] == old["id"] and moved["version"] == "1.1.0" and moved["s3_path"] == f"plugins/{slug}/1.1.0/"
    assert moved["security_scan_id"] != old["security_scan_id"] and moved["approval_status"] == "approved"
    assert bucket.keys(f"plugins/{slug}/1.0.0/") == [] and bucket.keys(f"plugins/{slug}/1.1.0/") != []
    assert _scalar(new_session, "SELECT count(*) FROM workspace_enabled_plugins WHERE plugin_id = CAST(:id AS uuid)",
                   id=str(old["id"])) == 1
    mapping = new_session().execute(sa.text(
        "SELECT version, revision FROM marketplace_cloud_items WHERE plugin_id = CAST(:id AS uuid)"
    ), {"id": str(old["id"])}).one()
    assert tuple(mapping) == ("1.1.0", newer.revision)
    prompts = new_session().execute(sa.text(
        "SELECT prompt_template FROM skills WHERE package_slug = :slug AND name = :name"), {"slug": slug, "name": skill}
    ).scalars().all()
    assert len(prompts) == 1 and "Summarise, then cite." in prompts[0]
    history = new_session().execute(sa.text(
        "SELECT details FROM plugin_sync_history WHERE plugin_slug = :slug ORDER BY started_at, completed_at"
    ), {"slug": slug}).scalars().all()
    assert [(d["outcome"], d["version"], d["previous_version"]) for d in history] == [
        ("new", "1.0.0", None), ("updated", "1.1.0", "1.0.0"),
    ]


def test_files_left_under_the_prefix_refuse_the_import_and_are_removed_for_the_next_sync(
    cloud, bucket, made, new_session,
):
    slug = made.slug()
    stale = f"plugins/{slug}/1.0.0/skills/stale-skill/SKILL.md"
    bucket.objects[(config.MARKETPLACE_S3_BUCKET, stale)] = _skill_md("stale-skill")
    entry = cloud.serve(_bundle(slug), _zip(_files(slug, "1.0.0")))
    assert _import(new_session, cloud, entry).counts.refused == 1
    assert bucket.keys() == [] and _rows(new_session, slug) == NO_ROWS
    assert _import(new_session, cloud, entry).counts.new == 1
    assert stale not in bucket.keys()
    names = new_session().execute(sa.text("SELECT name FROM skills WHERE package_slug = :slug"), {"slug": slug})
    assert names.scalars().all() == [f"{slug}-skill"]


def test_a_writer_that_committed_the_slug_meanwhile_keeps_its_files(cloud, bucket, made, new_session, monkeypatch):
    slug = made.slug()
    prefix = f"plugins/{slug}/1.0.0/"
    real = plugin_import.scan_row

    def after_another_writer(checked):
        other = new_session()
        other.execute(sa.text(
            "INSERT INTO marketplace_plugins (id, slug, name, version, s3_path, approval_status, is_active)"
            " VALUES (CAST(:id AS uuid), :slug, 'Uploaded Meanwhile', '1.0.0', :path, 'pending', true)"
        ), {"id": str(uuid.uuid4()), "slug": slug, "path": prefix})
        other.commit()
        return real(checked)

    monkeypatch.setattr(plugin_import, "scan_row", after_another_writer)
    imported = _import(new_session, cloud, cloud.serve(_bundle(slug), _zip(_files(slug, "1.0.0"))))
    assert imported.counts.refused == 1  # the unique slug refused this import's row
    assert bucket.keys(prefix) == sorted(prefix + path for path in _files(slug, "1.0.0"))
    assert _rows(new_session, slug) == {**NO_ROWS, "marketplace_plugins": 1}
    assert _scalar(new_session, "SELECT name FROM marketplace_plugins WHERE slug = :slug", slug=slug) == (
        "Uploaded Meanwhile"
    )


def test_a_failed_update_keeps_the_row_and_its_files_at_the_old_version(cloud, bucket, made, new_session):
    slug = made.slug()
    assert _import(new_session, cloud, cloud.serve(_bundle(slug), _zip(_files(slug, "1.0.0")))).counts.new == 1
    before, old_keys = dict(_scalar_row(new_session, slug)), bucket.keys(f"plugins/{slug}/1.0.0/")
    bucket.fail_put_at = bucket.puts + 2
    newer = cloud.serve(_bundle(slug, version="1.1.0"), _zip(_files(slug, "1.1.0")))
    assert _import(new_session, cloud, newer).counts.refused == 1
    assert dict(_scalar_row(new_session, slug)) == before
    assert bucket.keys(f"plugins/{slug}/1.0.0/") == old_keys and bucket.keys(f"plugins/{slug}/1.1.0/") == []
    assert _scalar(new_session, "SELECT version FROM marketplace_cloud_items WHERE external_id = :id",
                   id=_external_id(slug)) == "1.0.0"


def test_an_update_keeps_a_local_deactivation_and_makes_no_skill(cloud, bucket, made, new_session):
    slug = made.slug()
    assert _import(new_session, cloud, cloud.serve(_bundle(slug), _zip(_files(slug, "1.0.0")))).counts.new == 1
    session = new_session()
    session.execute(sa.text(
        "UPDATE marketplace_plugins SET is_active = false, approval_status = 'rejected' WHERE slug = :slug"
    ), {"slug": slug})
    session.commit()
    runner = Runner()
    newer = cloud.serve(_bundle(slug, version="1.1.0"), _zip(_files(slug, "1.1.0", body="A changed body.")))
    assert _import(new_session, cloud, newer, runner=runner).counts.updated == 1
    row = dict(_scalar_row(new_session, slug))
    assert (row["version"], row["is_active"], row["approval_status"]) == ("1.1.0", False, "rejected")
    assert "materialize_plugin" not in runner.kinds()
    prompt = _scalar(new_session, "SELECT prompt_template FROM skills WHERE package_slug = :slug", slug=slug)
    assert "A changed body." not in prompt


def test_a_mapped_item_whose_bundle_names_another_slug_is_refused(cloud, bucket, made, new_session):
    first, other = made.slug(), made.slug()
    assert _import(new_session, cloud, cloud.serve(_bundle(first), _zip(_files(first, "1.0.0")))).counts.new == 1
    before = dict(_scalar_row(new_session, first))
    renamed = cloud.serve(_bundle(other, version="2.0.0", externalId=_external_id(first)), _zip(_files(other, "2.0.0")))
    cloud.requests.clear()
    assert _import(new_session, cloud, renamed).counts.refused == 1
    assert cloud.urls() == [BUNDLES + renamed.revision]
    assert dict(_scalar_row(new_session, first)) == before and _rows(new_session, other)["marketplace_plugins"] == 0


def test_rows_that_change_while_a_plugin_is_checked_refuse_it_before_anything_is_stored(
    cloud, bucket, made, new_session, monkeypatch,
):
    slug = made.slug()
    real, reads = plugin_import.held_target, []

    def moving(db, entry, slug_, version):
        reads.append(slug_)
        target = real(db, entry, slug_, version)
        return target if len(reads) == 1 else target._replace(verdict="updated")

    monkeypatch.setattr(plugin_import, "held_target", moving)
    imported = _import(new_session, cloud, cloud.serve(_bundle(slug), _zip(_files(slug, "1.0.0"))))
    assert imported.counts.refused == 1 and len(reads) == 2
    assert bucket.puts == 0 and _rows(new_session, slug) == NO_ROWS


def test_an_update_whose_row_moved_meanwhile_is_refused_and_leaves_that_row(
    cloud, bucket, made, new_session, monkeypatch,
):
    slug = made.slug()
    assert _import(new_session, cloud, cloud.serve(_bundle(slug), _zip(_files(slug, "1.0.0")))).counts.new == 1
    real = plugin_import.scan_row

    def after_another_move(checked):
        other = new_session()
        other.execute(sa.text("UPDATE marketplace_plugins SET version = '1.0.5' WHERE slug = :slug"), {"slug": slug})
        other.commit()
        return real(checked)

    monkeypatch.setattr(plugin_import, "scan_row", after_another_move)
    newer = cloud.serve(_bundle(slug, version="1.1.0"), _zip(_files(slug, "1.1.0")))
    assert _import(new_session, cloud, newer).counts.refused == 1
    assert _scalar(new_session, "SELECT version FROM marketplace_plugins WHERE slug = :slug", slug=slug) == "1.0.5"
    assert bucket.keys(f"plugins/{slug}/1.1.0/") == [] and bucket.keys(f"plugins/{slug}/1.0.0/") != []
    assert _rows(new_session, slug)["plugin_security_scans"] == 1


def test_a_materialiser_failure_leaves_the_plugin_committed_and_no_transaction_open(
    cloud, bucket, made, new_session, monkeypatch,
):
    class Broken:
        def __init__(self, _db):
            pass

        async def materialize_plugin(self, _plugin):
            raise RuntimeError("the materialiser broke")

    monkeypatch.setattr(plugin_import, "SkillMaterializer", Broken)
    slug, session = made.slug(), new_session()
    entry = cloud.serve(_bundle(slug), _zip(_files(slug, "1.0.0")))
    assert plugin_import.import_plugins(session, [entry], cloud.factory, Runner()).counts.new == 1
    assert not session.in_transaction()
    assert _rows(new_session, slug) == {
        "marketplace_plugins": 1, "plugin_security_scans": 1, "plugin_sync_history": 1, "marketplace_cloud_items": 1,
    }
    assert _scalar(new_session, "SELECT count(*) FROM skills WHERE package_slug = :slug", slug=slug) == 0
