"""PRD-270 W2 US-270-15a (§5.6, K15) — locally: Automatos Cloud's plugins, checked like an upload and stored.

Automatos Cloud is an ``httpx.MockTransport`` answering by URL and recording every request (W1's client-factory
seam, as ``test_prd270w2_items_feed.py``), the object store a fake S3 client monkeypatched into
``core.services.marketplace_s3`` (the ``test_prd261_theme_files.py:632`` pattern), and ``run_async`` is
``asyncio.run``, recording each coroutine it runs (``tests/helpers_prd270w2_plugins.py``). The LLM scan and
``PluginScanService.scan_plugin`` are patched to fail any test that reaches them. The zip rules, the local scan
and the pass's seams without a database are ``tests/test_prd270w2_plugin_import_checks.py``'s. Proven here, on
Postgres:

* a good plugin is stored under ``plugins/<slug>/<version>/``, approved and active, its scan row holding
  Automatos Cloud's report and this install's static result, its mapping and its 'cloud-sync' history written,
  its SKILL.md a global skill with package_slug the slug; only the derived URLs are requested, the static scan,
  the store and the materialiser each ran through ``run_async``, and no transaction was open during a fetch, a
  storage write or after the pass;
* a package over the cap, with 501 files, a '..' path, a ratio over 100, another manifest slug or version, or
  a critical static finding stores nothing and writes no row; so does a storage failure after a partial write,
  a database failure removes the files it stored, and files left under the prefix before refuse the import and
  are removed (the next Sync imports it); a writer that committed the same slug meanwhile keeps its files;
* a slug a row without a mapping holds is refused before its package is fetched, and that row and its files
  stay as they were; a changed revision at the same plugin version is refused; the same revision is unchanged
  and fetches nothing; the sixth plugin by name waits, its bundle read (its slug checked before the cap,
  P270W2-RVW-2) and its package never fetched; rows that change while a plugin is checked refuse it before
  anything is stored;
* a newer version moves the same row and keeps its enablements, its old files removed; a failed update keeps
  the row and its files at the old version; an update whose row moved meanwhile is refused; an update keeps a
  local deactivation and makes no skill; a mapped item whose bundle names another slug is refused; a
  materialiser failure leaves the plugin committed and no transaction open.
"""
from __future__ import annotations

import logging
import uuid
import zipfile

import pytest
import sqlalchemy as sa

import core.models  # noqa: F401  (registers every table the foreign keys name)
from config import config
from services.marketplace_bundles import plugin_import
from services.marketplace_cloud.fetch import ACCEPT_PACKAGE, USER_AGENT
from tests.helpers_prd270w2_plugins import (
    BUNDLES, CLOUD, SENT_HEADERS, Runner, bundle_of, cloud_settings, external_id, fake_bucket, files_of,
    incompressible, package_url, refuse_models, skill_md, zip_of,
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
    ), {"id": external_id(slug)}).scalar()
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
    bundle = bundle_of(slug)
    entry = cloud.serve(bundle, zip_of(files_of(slug, "1.0.0", skill)))
    runner, session, open_during_fetch, open_during_store = Runner(), new_session(), [], []
    cloud.on_request = lambda: open_during_fetch.append(session.in_transaction())
    bucket.on_write = lambda: open_during_store.append(session.in_transaction())
    imported = plugin_import.import_plugins(session, [entry], cloud.factory, runner)
    assert imported.counts.view() == {"new": 1, "updated": 0, "unchanged": 0, "refused": 0} and imported.waiting == 0
    assert cloud.urls() == [BUNDLES + entry.revision, package_url(slug, "1.0.0")]
    # No transaction is held across a network read or a storage write, nor left open after the pass.
    assert open_during_fetch == [False, False]
    assert open_during_store and not any(open_during_store) and not session.in_transaction()
    package_request = cloud.requests[1]
    assert set(package_request.headers.keys()) <= SENT_HEADERS
    assert package_request.headers["accept"] == ACCEPT_PACKAGE and package_request.headers["user-agent"] == USER_AGENT
    assert runner.kinds() == ["static_scan", "extract_plugin", "list_plugin_files", "materialize_plugin"]
    prefix = f"plugins/{slug}/1.0.0/"
    assert bucket.keys(prefix) == sorted(prefix + path for path in files_of(slug, "1.0.0", skill))
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
    return zip_of({**files_of(slug, "1.0.0"), "big.bin": incompressible(1100 * 1024)}, compression=zipfile.ZIP_STORED)


def _critical(slug):
    # Plain text the static scan reads, never run: `eval(` is one of its critical patterns.
    return zip_of(files_of(slug, "1.0.0", extra={"commands/run.md": b"Then run:\n\n    eval(user_text)\n"}))


@pytest.mark.parametrize(("package", "reason"), [
    pytest.param(_over_the_cap, "Its package couldn't be fetched: Automatos Cloud's answer is larger than 1024 KB.",
                 id="over-the-cap"),
    pytest.param(lambda slug: zip_of({**files_of(slug, "1.0.0"), **{f"n/{i}.md": b"x" for i in range(497)}}),
                 plugin_import.TOO_MANY_FILES.format(count=500), id="501-files"),
    pytest.param(lambda slug: zip_of(files_of(slug, "1.0.0", extra={"../escape.md": b"x"})), plugin_import.UNSAFE_PATH,
                 id="dotdot-path"),
    pytest.param(lambda slug: zip_of(files_of(slug, "1.0.0", extra={"zeros.txt": b"\0" * (1024 * 1024)})),
                 plugin_import.RATIO_TOO_HIGH.format(ratio=100), id="ratio-over-100"),
    pytest.param(lambda slug: zip_of(files_of(slug, "1.0.0", manifest={"slug": "another-plugin"})),
                 "Its package isn't the bundle's plugin: its manifest.json names another slug or version",
                 id="another-manifest-slug"),
    pytest.param(lambda slug: zip_of(files_of(slug, "1.0.0", manifest={"version": "1.0.1"})),
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
        imported = _import(new_session, cloud, cloud.serve(bundle_of(slug), package(slug)))
    assert imported.counts.view() == {"new": 0, "updated": 0, "unchanged": 0, "refused": 1}
    assert f"plugin {external_id(slug)} refused: {reason}" in caplog.text
    assert bucket.puts == 0 and bucket.keys() == []
    assert _rows(new_session, slug) == NO_ROWS


def test_a_storage_failure_after_a_partial_write_leaves_no_object_and_no_row(cloud, bucket, made, new_session):
    slug = made.slug()
    bucket.fail_put_at = 2
    imported = _import(new_session, cloud, cloud.serve(bundle_of(slug), zip_of(files_of(slug, "1.0.0"))))
    assert imported.counts.refused == 1 and imported.counts.new == 0
    assert bucket.puts >= 2 and bucket.keys() == []
    assert _rows(new_session, slug) == NO_ROWS


def test_a_database_failure_removes_the_stored_files_and_writes_no_row(cloud, bucket, made, new_session, monkeypatch):
    slug = made.slug()
    real = plugin_import.scan_row
    monkeypatch.setattr(plugin_import, "scan_row", lambda checked: _without_version(real(checked)))
    imported = _import(new_session, cloud, cloud.serve(bundle_of(slug), zip_of(files_of(slug, "1.0.0"))))
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
    entry = cloud.serve(bundle_of(slug, version="0.9.0"), zip_of(files_of(slug, "0.9.0")))
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


def test_the_sixth_plugin_by_name_waits_and_its_package_is_never_fetched(cloud, bucket, made, new_session):
    served = {}
    for letter in "FBDACE":
        slug = made.slug()
        served[letter] = (slug, cloud.serve(bundle_of(slug, name=f"Test Plugin {letter}"), zip_of(files_of(slug, "1.0.0"))))
    imported = _import(new_session, cloud, *(entry for _, entry in served.values()))
    assert imported.counts.view() == {"new": 5, "updated": 0, "unchanged": 0, "refused": 0} and imported.waiting == 1
    waiting_slug, waiting_entry = served["F"]
    named = [url for url in cloud.urls() if waiting_slug in url or waiting_entry.revision in url]
    assert named == [BUNDLES + waiting_entry.revision]  # its slug checked before the cap; no package fetched
    assert _rows(new_session, waiting_slug) == NO_ROWS
    assert all(_rows(new_session, served[letter][0])["marketplace_plugins"] == 1 for letter in "ABCDE")


def test_the_same_revision_again_is_unchanged_and_fetches_nothing(cloud, bucket, made, new_session):
    slug = made.slug()
    entry = cloud.serve(bundle_of(slug), zip_of(files_of(slug, "1.0.0")))
    assert _import(new_session, cloud, entry).counts.new == 1
    cloud.requests.clear()
    again = _import(new_session, cloud, entry)
    assert again.counts.view() == {"new": 0, "updated": 0, "unchanged": 1, "refused": 0}
    assert cloud.requests == []


def test_a_changed_revision_at_the_same_version_is_refused_and_the_row_kept(cloud, bucket, made, new_session):
    slug = made.slug()
    assert _import(new_session, cloud, cloud.serve(bundle_of(slug), zip_of(files_of(slug, "1.0.0")))).counts.new == 1
    before = dict(_scalar_row(new_session, slug))
    changed = cloud.serve(bundle_of(slug, description="A newer description."), zip_of(files_of(slug, "1.0.0")))
    cloud.requests.clear()
    imported = _import(new_session, cloud, changed)
    assert imported.counts.refused == 1
    assert cloud.urls() == [BUNDLES + changed.revision]
    assert dict(_scalar_row(new_session, slug)) == before
    assert _rows(new_session, slug)["plugin_security_scans"] == 1


def test_a_newer_version_moves_the_row_and_keeps_its_enablements(cloud, bucket, made, new_session):
    slug, skill = made.slug(), f"t270pi-skill-{uuid.uuid4().hex[:8]}"
    first = cloud.serve(bundle_of(slug), zip_of(files_of(slug, "1.0.0", skill)))
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
    newer = cloud.serve(bundle_of(slug, version="1.1.0"), zip_of(files_of(slug, "1.1.0", skill, body="Summarise, then cite.")))
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
    bucket.objects[(config.MARKETPLACE_S3_BUCKET, stale)] = skill_md("stale-skill")
    entry = cloud.serve(bundle_of(slug), zip_of(files_of(slug, "1.0.0")))
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
    imported = _import(new_session, cloud, cloud.serve(bundle_of(slug), zip_of(files_of(slug, "1.0.0"))))
    assert imported.counts.refused == 1  # the unique slug refused this import's row
    assert bucket.keys(prefix) == sorted(prefix + path for path in files_of(slug, "1.0.0"))
    assert _rows(new_session, slug) == {**NO_ROWS, "marketplace_plugins": 1}
    assert _scalar(new_session, "SELECT name FROM marketplace_plugins WHERE slug = :slug", slug=slug) == (
        "Uploaded Meanwhile"
    )


def test_a_failed_update_keeps_the_row_and_its_files_at_the_old_version(cloud, bucket, made, new_session):
    slug = made.slug()
    assert _import(new_session, cloud, cloud.serve(bundle_of(slug), zip_of(files_of(slug, "1.0.0")))).counts.new == 1
    before, old_keys = dict(_scalar_row(new_session, slug)), bucket.keys(f"plugins/{slug}/1.0.0/")
    bucket.fail_put_at = bucket.puts + 2
    newer = cloud.serve(bundle_of(slug, version="1.1.0"), zip_of(files_of(slug, "1.1.0")))
    assert _import(new_session, cloud, newer).counts.refused == 1
    assert dict(_scalar_row(new_session, slug)) == before
    assert bucket.keys(f"plugins/{slug}/1.0.0/") == old_keys and bucket.keys(f"plugins/{slug}/1.1.0/") == []
    assert _scalar(new_session, "SELECT version FROM marketplace_cloud_items WHERE external_id = :id",
                   id=external_id(slug)) == "1.0.0"


def test_an_update_keeps_a_local_deactivation_and_makes_no_skill(cloud, bucket, made, new_session):
    slug = made.slug()
    assert _import(new_session, cloud, cloud.serve(bundle_of(slug), zip_of(files_of(slug, "1.0.0")))).counts.new == 1
    session = new_session()
    session.execute(sa.text(
        "UPDATE marketplace_plugins SET is_active = false, approval_status = 'rejected' WHERE slug = :slug"
    ), {"slug": slug})
    session.commit()
    runner = Runner()
    newer = cloud.serve(bundle_of(slug, version="1.1.0"), zip_of(files_of(slug, "1.1.0", body="A changed body.")))
    assert _import(new_session, cloud, newer, runner=runner).counts.updated == 1
    row = dict(_scalar_row(new_session, slug))
    assert (row["version"], row["is_active"], row["approval_status"]) == ("1.1.0", False, "rejected")
    assert "materialize_plugin" not in runner.kinds()
    prompt = _scalar(new_session, "SELECT prompt_template FROM skills WHERE package_slug = :slug", slug=slug)
    assert "A changed body." not in prompt


def test_a_mapped_item_whose_bundle_names_another_slug_is_refused(cloud, bucket, made, new_session):
    first, other = made.slug(), made.slug()
    assert _import(new_session, cloud, cloud.serve(bundle_of(first), zip_of(files_of(first, "1.0.0")))).counts.new == 1
    before = dict(_scalar_row(new_session, first))
    renamed = cloud.serve(bundle_of(other, version="2.0.0", externalId=external_id(first)), zip_of(files_of(other, "2.0.0")))
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
    imported = _import(new_session, cloud, cloud.serve(bundle_of(slug), zip_of(files_of(slug, "1.0.0"))))
    assert imported.counts.refused == 1 and len(reads) == 2
    assert bucket.puts == 0 and _rows(new_session, slug) == NO_ROWS


def test_an_update_whose_row_moved_meanwhile_is_refused_and_leaves_that_row(
    cloud, bucket, made, new_session, monkeypatch,
):
    slug = made.slug()
    assert _import(new_session, cloud, cloud.serve(bundle_of(slug), zip_of(files_of(slug, "1.0.0")))).counts.new == 1
    real = plugin_import.scan_row

    def after_another_move(checked):
        other = new_session()
        other.execute(sa.text("UPDATE marketplace_plugins SET version = '1.0.5' WHERE slug = :slug"), {"slug": slug})
        other.commit()
        return real(checked)

    monkeypatch.setattr(plugin_import, "scan_row", after_another_move)
    newer = cloud.serve(bundle_of(slug, version="1.1.0"), zip_of(files_of(slug, "1.1.0")))
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
    entry = cloud.serve(bundle_of(slug), zip_of(files_of(slug, "1.0.0")))
    assert plugin_import.import_plugins(session, [entry], cloud.factory, Runner()).counts.new == 1
    assert not session.in_transaction()
    assert _rows(new_session, slug) == {
        "marketplace_plugins": 1, "plugin_security_scans": 1, "plugin_sync_history": 1, "marketplace_cloud_items": 1,
    }
    assert _scalar(new_session, "SELECT count(*) FROM skills WHERE package_slug = :slug", slug=slug) == 0
