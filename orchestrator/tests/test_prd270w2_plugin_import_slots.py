"""P270W2-RVW-2 (PRD-270 §5.6, K15) — a plugin refused for a lasting reason never holds a Sync's plugin places
ahead of plugins that can import.

The shared builders and fakes are ``tests/helpers_prd270w2_plugins.py``'s: Automatos Cloud an
``httpx.MockTransport`` recording every request (the fetch counter), the object store a fake S3 client,
``run_async`` as ``asyncio.run``, and a model call fails the test. Proven:

* with AUTOMATOS_CLOUD_PLUGINS_PER_SYNC = 1, a plugin refused SLUG_HELD and one refused SAME_VERSION that sort
  first by name take no place: the importable one is imported on the FIRST Sync, neither refused one's package
  is fetched, and nothing waits; with two importable ones behind a refused one, only the second waits;
* a plugin whose package this install's scan refused on Sync 1 is not fetched again on Sync 2 before a plugin
  not yet tried, and is tried again once a place is free;
* two entries naming one slug: the first is imported, and the second, whose slug check before the cap passed,
  is refused by the rows read again after its checks, before any of its files is stored;
* without a database: a plugin refused here at its revision comes after every one not yet tried (at another
  revision it is not yet tried); a remembered refusal is kept only while its entry is still new or changed; a
  slug check that refuses or breaks takes no place, ends its read and leaves the pass going on, and is never
  remembered as a package refusal.

A cap of 0 still requests nothing and every new or changed plugin waits: ``test_prd270w2_re_resolve.py`` and
``test_prd270w2_sync_routes.py`` pin it.
"""
from __future__ import annotations

import logging
import uuid

import pytest
import sqlalchemy as sa

import core.models  # noqa: F401  (registers every table the foreign keys name)
from config import config
from services.marketplace_bundles import items_feed, plugin_import
from services.marketplace_bundles.plugin_import import PluginRefused
from tests.helpers_prd270w2_plugins import (
    BUNDLES, Runner, bundle_of, cloud_settings, external_id, fake_bucket, files_of, package_url, refuse_models,
    zip_of,
)

COUNTS = ("new", "updated", "unchanged", "refused")


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    refuse_models(monkeypatch)


@pytest.fixture(autouse=True)
def nothing_remembered():
    plugin_import.forget_refused()
    yield
    plugin_import.forget_refused()


@pytest.fixture
def cloud(monkeypatch):
    return cloud_settings(monkeypatch)


@pytest.fixture
def bucket(monkeypatch):
    return fake_bucket(monkeypatch)


def _counts(**given):
    return {name: given.get(name, 0) for name in COUNTS}


def _entry(name, revision="a" * 64, index=0):
    return items_feed.CatalogEntry("plugin", f"{index:032x}", "1.0.0", revision, name)


# --------------------------------------------------------------------------- the order and the memory (no database)


def test_a_plugin_refused_here_at_its_revision_comes_after_every_plugin_not_yet_tried():
    first, second, third = (_entry(f"Test Plugin {letter}", index=i) for i, letter in enumerate("ABC"))
    refused = frozenset({plugin_import.identity(first)})
    chosen, waiting = plugin_import.chosen_entries([third, first, second], 2, refused)
    assert (chosen, waiting) == ((second, third), (first,))
    moved = first._replace(revision="b" * 64)  # its revision moved: not yet tried at this one
    assert plugin_import.chosen_entries([third, moved, second], 2, refused) == ((moved, second), (third,))


def test_a_remembered_refusal_is_kept_only_while_its_entry_is_still_new_or_changed():
    first, second = _entry("Test Plugin A", index=1), _entry("Test Plugin B", index=2)
    plugin_import.remember_refused([first, second], [first])
    assert plugin_import.refused_before() == {plugin_import.identity(first)}
    plugin_import.remember_refused([first, second], [second])
    assert plugin_import.refused_before() == {plugin_import.identity(first), plugin_import.identity(second)}
    moved = first._replace(revision="b" * 64)
    plugin_import.remember_refused([moved], [])  # the first's revision moved, the second was imported
    assert plugin_import.refused_before() == frozenset()


class _Rows:
    """A session for a pass whose row reads are patched: it counts its rollbacks."""

    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


def test_a_slug_check_that_refuses_or_breaks_takes_no_place_and_the_pass_goes_on(cloud, bucket, monkeypatch):
    def checked(_db, _entry, slug, _version):
        if slug == "test-kit-a":
            raise RuntimeError("the rows broke")
        raise PluginRefused(plugin_import.SLUG_HELD)

    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", 1)
    monkeypatch.setattr(plugin_import, "changed_entries", lambda _db, entries: tuple(entries))
    monkeypatch.setattr(plugin_import, "held_target", checked)
    entries = [cloud.serve(bundle_of(f"test-kit-{letter}", name=f"Test Plugin {letter}"),
                           zip_of(files_of(f"test-kit-{letter}", "1.0.0"))) for letter in "ab"]
    rows = _Rows()

    imported = plugin_import.import_plugins(rows, entries, cloud.factory, Runner())

    assert imported.counts.view() == _counts(refused=2) and imported.waiting == 0
    assert cloud.urls() == [BUNDLES + entry.revision for entry in entries] and bucket.puts == 0
    assert rows.rollbacks == 2  # the mappings' read, then the broken slug check's
    assert plugin_import.refused_before() == frozenset()


# --------------------------------------------------------------------------- the pass (Postgres)


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
            ):
                conn.execute(sa.text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the plugin place tests need Postgres with the PRD-270 W2 schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def made(engine, new_session):
    """A slug factory; every row of each slug made is swept after the test."""
    slugs = []

    def slug():
        slugs.append(f"t270ps-{uuid.uuid4().hex[:12]}")
        return slugs[-1]

    yield slug
    sweep = new_session.sweep()
    for one in slugs:
        for statement in (
            "DELETE FROM skills WHERE package_slug = :slug",
            "DELETE FROM plugin_sync_history WHERE plugin_slug = :slug",
            "DELETE FROM marketplace_plugins WHERE slug = :slug",  # its mapping cascades
            "DELETE FROM plugin_security_scans WHERE plugin_slug = :slug",
        ):
            sweep.execute(sa.text(statement), {"slug": one})
    sweep.commit()


def _import(new_session, cloud, *entries):
    return plugin_import.import_plugins(new_session(), list(entries), cloud.factory, Runner())


def _good(cloud, slug, name, **changes):
    return cloud.serve(bundle_of(slug, name=name, **changes), zip_of(files_of(slug, "1.0.0")))


def _critical(cloud, slug, name):
    # Plain text the static scan reads, never run: `eval(` is one of its critical patterns.
    files = files_of(slug, "1.0.0", extra={"commands/run.md": b"Then run:\n\n    eval(user_text)\n"})
    return cloud.serve(bundle_of(slug, name=name), zip_of(files))


def _hold_slug(new_session, slug):
    """An operator's plugin row with ``slug`` and no mapping: a Cloud plugin of that slug is refused SLUG_HELD."""
    session = new_session()
    session.execute(sa.text(
        "INSERT INTO marketplace_plugins (id, slug, name, version, s3_path, description, approval_status, is_active)"
        " VALUES (CAST(:id AS uuid), :slug, 'Operator Plugin', '1.0.0', :path, 'Ours.', 'pending', true)"
    ), {"id": str(uuid.uuid4()), "slug": slug, "path": f"plugins/{slug}/1.0.0/"})
    session.commit()


def _plugin_row(new_session, slug):
    session = new_session()
    row = session.execute(sa.text("SELECT * FROM marketplace_plugins WHERE slug = :slug"), {"slug": slug}).mappings()
    found = [dict(one) for one in row.all()]
    session.rollback()
    return found


def test_plugins_refused_by_their_slug_or_version_take_no_place_and_the_importable_one_imports_at_once(
    cloud, bucket, made, new_session, monkeypatch, caplog,
):
    held, same, good = made(), made(), made()
    _hold_slug(new_session, held)
    assert _import(new_session, cloud, _good(cloud, same, "Test Plugin B")).counts.new == 1
    held_before, same_before = _plugin_row(new_session, held), _plugin_row(new_session, same)
    entries = (
        _good(cloud, held, "Test Plugin A"),
        _good(cloud, same, "Test Plugin B", description="A newer description at the same version."),
        _good(cloud, good, "Test Plugin C"),
    )
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", 1)
    cloud.requests.clear()

    with caplog.at_level(logging.WARNING, logger=plugin_import.__name__):
        imported = _import(new_session, cloud, *entries)

    assert imported.counts.view() == _counts(new=1, refused=2) and imported.waiting == 0
    assert cloud.urls() == [*(BUNDLES + entry.revision for entry in entries), package_url(good, "1.0.0")]
    assert f"plugin {external_id(held)} refused: {plugin_import.SLUG_HELD}" in caplog.text
    assert f"plugin {external_id(same)} refused: {plugin_import.SAME_VERSION}" in caplog.text
    assert _plugin_row(new_session, held) == held_before and _plugin_row(new_session, same) == same_before
    assert [row["approval_status"] for row in _plugin_row(new_session, good)] == ["approved"]


def test_behind_a_refused_plugin_only_the_importable_plugins_past_the_cap_wait(
    cloud, bucket, made, new_session, monkeypatch,
):
    held, first, second = made(), made(), made()
    _hold_slug(new_session, held)
    entries = (_good(cloud, held, "Test Plugin A"), _good(cloud, first, "Test Plugin B"),
               _good(cloud, second, "Test Plugin C"))
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", 1)

    imported = _import(new_session, cloud, *entries)

    assert imported.counts.view() == _counts(new=1, refused=1) and imported.waiting == 1
    assert package_url(second, "1.0.0") not in cloud.urls() and _plugin_row(new_session, second) == []
    assert len(_plugin_row(new_session, first)) == 1


def test_a_plugin_whose_package_was_refused_is_not_fetched_again_before_one_not_yet_tried(
    cloud, bucket, made, new_session, monkeypatch,
):
    refused, fresh = made(), made()
    entries = (_critical(cloud, refused, "Test Plugin A"), _good(cloud, fresh, "Test Plugin Z"))
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", 1)

    one = _import(new_session, cloud, *entries)

    assert one.counts.view() == _counts(refused=1) and one.waiting == 1
    assert package_url(refused, "1.0.0") in cloud.urls() and package_url(fresh, "1.0.0") not in cloud.urls()
    cloud.requests.clear()

    two = _import(new_session, cloud, *entries)

    assert two.counts.view() == _counts(new=1) and two.waiting == 1
    assert cloud.urls() == [*(BUNDLES + entry.revision for entry in entries), package_url(fresh, "1.0.0")]
    assert len(_plugin_row(new_session, fresh)) == 1 and _plugin_row(new_session, refused) == []
    cloud.requests.clear()

    three = _import(new_session, cloud, *entries)  # a place is free: it is tried again, and refused again

    assert three.counts.view() == _counts(unchanged=1, refused=1) and three.waiting == 0
    assert cloud.urls() == [BUNDLES + entries[0].revision, package_url(refused, "1.0.0")]
    assert bucket.keys(f"plugins/{refused}/") == []


def test_two_entries_naming_one_slug_import_the_first_and_refuse_the_second_before_it_is_stored(
    cloud, bucket, made, new_session, caplog,
):
    slug = made()
    files = files_of(slug, "1.0.0")
    first = cloud.serve(bundle_of(slug, name="Test Plugin A"), zip_of(files))
    second_id = external_id(f"{slug}-second")
    second = cloud.serve(bundle_of(slug, name="Test Plugin B", externalId=second_id), zip_of(files))

    with caplog.at_level(logging.WARNING, logger=plugin_import.__name__):
        imported = _import(new_session, cloud, first, second)

    assert imported.counts.view() == _counts(new=1, refused=1) and imported.waiting == 0
    assert f"plugin {second_id} refused: {plugin_import.SLUG_HELD}" in caplog.text
    assert bucket.puts == len(files)  # only the first's files were stored
    assert [row["name"] for row in _plugin_row(new_session, slug)] == ["Test Plugin A"]
    session = new_session()
    mapped = session.execute(sa.text(
        "SELECT external_id FROM marketplace_cloud_items WHERE item_type = 'plugin' AND external_id IN (:a, :b)"
    ), {"a": first.external_id, "b": second_id}).scalars().all()
    session.rollback()
    assert mapped == [first.external_id]
