"""PRD-270 W2 US-270-15a (§5.6, K15) — the plugin import's checks, without a database.

The shared builders and fakes are ``tests/helpers_prd270w2_plugins.py``'s; the import on Postgres is
``tests/test_prd270w2_plugin_import.py``. Proven here:

* the zip rules are the upload's own (the constants pinned against ``plugin_upload_service.py``'s lines), and
  ``package_files`` refuses 501 files, an unsafe path ('..', absolute, a backslash, not normal, '.'), a file or
  a total over its cap, a ratio over 100, a name twice and a zip it can't read, with a reason that echoes
  nothing the package held;
* the local static scan reads in chunks, keeps each finding's line in the file, and refuses a critical finding,
  a line over its cap and too many findings;
* the first plugins by name are chosen and the rest wait; other types are the item pass's; a setting that isn't
  an origin refuses before any request;
* a failed read of the rows refuses the plugin and ends the read; anything unforeseen in the checks refuses that
  plugin only; files a row names (or may name) are kept, others removed;
* the module never names the LLM scan or the upload flow.
"""
from __future__ import annotations

import ast
import zipfile

import pytest
import sqlalchemy as sa

from config import config
from services.marketplace_bundles import items_feed, plugin_import
from services.marketplace_bundles.plugin_import import PluginRefused
from tests.helpers_prd270w2_plugins import (
    APPEND_LINE, CLOUD, MARK, MODULE, S3_SERVICE, UPLOAD_SERVICE, Runner, bundle_of, cloud_settings, fake_bucket,
    files_of, incompressible, refuse_models, zip_of,
)


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    refuse_models(monkeypatch)


@pytest.fixture
def cloud(monkeypatch):
    return cloud_settings(monkeypatch)


@pytest.fixture
def bucket(monkeypatch):
    return fake_bucket(monkeypatch)


# --------------------------------------------------------------------------- the zip rules


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
    files = files_of("test-kit", "1.0.0", "test-skill")
    assert plugin_import.package_files(zip_of(files, folders=("skills/", "assets/"))) == files


def _many_files():
    return {**files_of("test-kit", "1.0.0"), **{f"notes/n{index}.md": b"x" for index in range(497)}}


def _twice():
    return [("manifest.json", b"{}"), ("a.md", b"one"), ("a.md", b"two")]


def _bad_crc():
    data = bytearray(zip_of({"a.md": b"plain text that is stored"}, compression=zipfile.ZIP_STORED))
    at = data.index(b"plain text")
    data[at] ^= 0x01
    return bytes(data)


@pytest.mark.parametrize(("package", "reason"), [
    (lambda: zip_of(_many_files()), plugin_import.TOO_MANY_FILES.format(count=500)),
    (lambda: zip_of({f"../{MARK}.md": b"x"}), plugin_import.UNSAFE_PATH),
    (lambda: zip_of({f"/{MARK}.md": b"x"}), plugin_import.UNSAFE_PATH),
    (lambda: zip_of({f"skills\\{MARK}.md": b"x"}), plugin_import.UNSAFE_PATH),
    (lambda: zip_of({f"./{MARK}.md": b"x"}), plugin_import.UNSAFE_PATH),
    (lambda: zip_of({".": b"x"}), plugin_import.UNSAFE_PATH),
    (lambda: zip_of({f"{MARK}.bin": incompressible(10 * 1024 * 1024 + 1)}, compression=zipfile.ZIP_STORED),
     plugin_import.FILE_TOO_LARGE.format(mb=10)),
    (lambda: zip_of({f"{MARK}.txt": b"\0" * (1024 * 1024)}), plugin_import.RATIO_TOO_HIGH.format(ratio=100)),
    (lambda: zip_of({"a.bin": incompressible(6 * 1024 * 1024), "b.bin": incompressible(5 * 1024 * 1024, 1)},
                  compression=zipfile.ZIP_STORED), plugin_import.FILES_TOO_LARGE.format(mb=10)),
    (lambda: zip_of(_twice()), plugin_import.NAMED_TWICE),
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
    package = zip_of({"a.bin": incompressible(700 * 1024), "b.bin": incompressible(400 * 1024, 1)},
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
    entry = cloud.serve(bundle_of("test-kit"), zip_of(files_of("test-kit", "1.0.0")))
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
    one = plugin_import.Targeted(items_feed.FetchedBundle(entry, bundle_of("test-kit")), plugin_import.Target("new"))
    verdict = plugin_import.import_plugin(rows, one, _seams())
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
