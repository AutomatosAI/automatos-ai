"""PRD-270 W2 US-270-13c (K8, K14) — the round trip: each fixture imported here, then exported again.

The four canonical bundles of ``tests/fixtures/marketplace_bundles/`` are imported as a Sync imports them: the
plugin through the plugin pass (``plugin_import.import_plugins``, Automatos Cloud an ``httpx.MockTransport``, the
object store a fake S3 client, ``tests/helpers_prd270w2_plugins.py``), then the agent, the Playbook and the
package through the item pass (``import_items``). The Cloud rows they make are then exported with the
US-270-12 builders (``exported_agents``, ``exported_plugins``, ``exported_playbooks``, ``exported_packages``),
and each bundle's canonical bytes are the fixture's, but for what is this install's own identity, which the
test maps and names:

* an agent's or a Playbook's row version carries the revision (``local_version``): set back to the bundle's;
* an agent's and a Playbook's externalId (and every ``{item}`` naming them) is the hash of this install's own
  row (its public_id, its ``cloud-<externalId>`` template id): mapped to Automatos Cloud's;
* an agent's and a Playbook's publisher is the name of their creator on Automatos Cloud, which a row here keeps
  in ``created_by`` and the export doesn't read: it writes Automatos;
* a plugin's category and icon are this install's own plugin category's, and its report's scannedAt the time
  of this install's scan row;
* a package has no category, icon or publisher column (``export_more``'s round trip maps the same three).

The plugin's static report is NOT mapped: its package is built so this install's static scan finds exactly
what Automatos Cloud's found. Each test runs in one transaction rolled back at its end.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

import core.models  # noqa: F401  (registers every table the foreign keys name)
from core.models.core import Agent, LLMModel, Skill, WorkflowTemplate
from core.models.marketplace_cloud import MarketplaceCloudItem
from services.marketplace_bundles import package_members, plugin_import
from services.marketplace_bundles.canonical import canonical_bytes, local_version, revision_of
from services.marketplace_bundles.export_agents import exported_agents
from services.marketplace_bundles.export_packages import exported_packages
from services.marketplace_bundles.export_playbooks import exported_playbooks
from services.marketplace_bundles.export_plugins import exported_plugins
from services.marketplace_bundles.import_items import import_items
from services.marketplace_bundles.items_feed import CatalogEntry, FetchedBundle
from services.marketplace_bundles.lineage import external_id
from tests.helpers_prd270w2_plugins import (
    APPEND_LINE, Runner, cloud_settings, fake_bucket, files_of, refuse_models, zip_of,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "marketplace_bundles"
SEEDED_AGENT_SLUG = "content-strategist"  # the Playbook's and the package's {slug}
SEEDED_TEMPLATE_ID = "test-seeded-playbook"  # the package's {templateId}
NOTES_FINDING_LINE = 12  # the plugin fixture's static finding: an append on line 12 of its command
ITEM_TYPES = ("agent", "playbook", "package")


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            for probe in (
                "SELECT agent_id, recipe_id, package_id, plugin_id FROM marketplace_cloud_items LIMIT 1",
                "SELECT security_scan_id, materialized_skill_ids FROM marketplace_plugins LIMIT 1",
                "SELECT scanned_by FROM plugin_security_scans LIMIT 1",
                "SELECT package_slug FROM skills LIMIT 1",
                "SELECT tool_id, enabled FROM agent_tool_assignments LIMIT 1",
                "SELECT plugin_id, priority FROM agent_assigned_plugins LIMIT 1",
            ):
                conn.execute(sa.text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the round trip needs Postgres with the PRD-270 W2 schema: {exc}")
    yield eng
    eng.dispose()


@contextmanager
def _rolled_back(engine):
    with engine.connect() as conn:
        trans = conn.begin()
        session = Session(bind=conn, join_transaction_mode="create_savepoint")
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            yield session
        finally:
            session.close()
            trans.rollback()


@pytest.fixture
def db(engine):
    with _rolled_back(engine) as session:
        yield session


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    refuse_models(monkeypatch)


# --------------------------------------------------------------------------- the fixtures


def _bytes(name: str) -> bytes:
    return (FIXTURES / f"{name}.json").read_bytes().rstrip(b"\n")


def _bundle(name: str) -> dict:
    return json.loads(_bytes(name))


def _fetched(bundle: dict) -> FetchedBundle:
    entry = CatalogEntry(bundle["type"], bundle["externalId"], bundle["version"], revision_of(bundle), bundle["name"])
    return FetchedBundle(entry, bundle)


def _plugin_package(bundle: dict) -> bytes:
    """The plugin's files: its manifest, the summarise-sources skill the agent and the package name, and a
    command whose line 12 appends to a file (the one finding of the fixture's static report)."""
    slug, version = bundle["plugin"]["slug"], bundle["version"]
    notes = "\n".join(["Save the notes.", *([""] * (NOTES_FINDING_LINE - 2)), APPEND_LINE]) + "\n"
    files = files_of(slug, version, skill="summarise-sources", extra={"commands/save-notes.md": notes.encode()})
    return zip_of(files)


def _hold_what_the_fixtures_name(db) -> None:
    """This install's rows the fixtures name beside the plugin: the cite-sources skill, the model, a seeded
    agent with the {slug} and a seeded Playbook with the {templateId}, each made when the database has none."""
    if not db.query(Skill.id).filter(Skill.name == "cite-sources", Skill.package_slug.is_(None),
                                     Skill.workspace_id.is_(None), Skill.is_active.is_(True)).first():
        db.add(Skill(name="cite-sources", description="Test skill", skill_type="technical", is_active=True))
    if not db.query(LLMModel.id).filter(LLMModel.model_id == "test-vendor/test-model-1").first():
        db.add(LLMModel(provider="test-vendor", serving_provider="openrouter", model_id="test-vendor/test-model-1",
                        display_name="Test Model 1", context_window=1000, max_output_tokens=100))
    if not db.query(Agent.id).filter(Agent.slug == SEEDED_AGENT_SLUG, Agent.owner_type == "marketplace",
                                     Agent.is_approved.is_(True), Agent.workspace_id.is_(None)).first():
        db.add(Agent(name="Content Strategist", agent_type="custom", owner_type="marketplace", owner_id="marketplace",
                     workspace_id=None, is_approved=True, slug=SEEDED_AGENT_SLUG, version="1.0.0"))
    if not db.query(WorkflowTemplate.id).filter(WorkflowTemplate.template_id == SEEDED_TEMPLATE_ID).first():
        db.add(WorkflowTemplate(template_id=SEEDED_TEMPLATE_ID, name="Test Seeded Playbook", description="Seeded.",
                                owner_type="marketplace", owner_id="marketplace", is_approved=True, is_public=True,
                                template_definition={}, steps=[], created_by="seed"))
    db.commit()


def _cloud_row(db, bundle: dict):
    mapping = (
        db.query(MarketplaceCloudItem)
        .filter(MarketplaceCloudItem.item_type == bundle["type"],
                MarketplaceCloudItem.external_id == bundle["externalId"])
        .one()
    )
    if bundle["type"] == "agent":
        return db.get(Agent, mapping.agent_id)
    return db.get(WorkflowTemplate, mapping.recipe_id)


def _exported(db) -> dict:
    """Every item the US-270-12 builders export from this database, by type and externalId."""
    agents, plugins = exported_agents(db), exported_plugins(db)
    playbooks = exported_playbooks(db, agents)
    packages = exported_packages(db, agents, playbooks, plugin_items=plugins)
    return {
        kind: {item.entry.external_id: item for item in items}
        for kind, items in (("agent", agents), ("playbook", playbooks), ("package", packages), ("plugin", plugins))
    }


def _mapped(bundle: dict, fixture: dict, keys, identity: dict) -> bytes:
    """``bundle``'s canonical bytes with ``fixture``'s values at ``keys`` and each of this install's externalIds
    replaced by Automatos Cloud's (an externalId is 32 hex characters: nothing else in a bundle matches one)."""
    text = canonical_bytes({**bundle, **{key: fixture[key] for key in keys}}).decode("utf-8")
    for local, cloud in identity.items():
        text = text.replace(local, cloud)
    return text.encode("utf-8")


# --------------------------------------------------------------------------- the round trip


def test_each_fixture_imported_here_exports_its_own_bytes_but_for_this_installs_identity(db, monkeypatch):
    cloud, _bucket = cloud_settings(monkeypatch), fake_bucket(monkeypatch)
    monkeypatch.setattr(package_members, "seeded_template_ids", lambda: frozenset({SEEDED_TEMPLATE_ID}))
    fixtures = {name: _bundle(name) for name in ("agent", "playbook", "package", "plugin")}

    entry = cloud.serve(fixtures["plugin"], _plugin_package(fixtures["plugin"]))
    plugins = plugin_import.import_plugins(db, [entry], cloud.factory, Runner())
    assert plugins.counts.view() == {"new": 1, "updated": 0, "unchanged": 0, "refused": 0}
    _hold_what_the_fixtures_name(db)
    items = import_items(db, [_fetched(fixtures[name]) for name in ITEM_TYPES])
    assert items.ok and all(counts["new"] == 1 for counts in items.view().values()), items.view()

    agent_row, playbook_row = _cloud_row(db, fixtures["agent"]), _cloud_row(db, fixtures["playbook"])
    for row, name in ((agent_row, "agent"), (playbook_row, "playbook")):
        assert row.version == local_version(fixtures[name]["version"], revision_of(fixtures[name]))
        row.version = fixtures[name]["version"]  # mapped: the revision on a row's version is this install's
    db.flush()
    identity = {
        external_id("agent", f"row:{agent_row.public_id}"): fixtures["agent"]["externalId"],
        external_id("playbook", f"row:{playbook_row.template_id}"): fixtures["playbook"]["externalId"],
    }
    local_id = {cloud_id: local for local, cloud_id in identity.items()}
    found = _exported(db)

    agent = found["agent"][local_id[fixtures["agent"]["externalId"]]].bundle
    assert agent["publisher"] == {"name": "Automatos"}
    assert _mapped(agent, fixtures["agent"], ("publisher",), identity) == _bytes("agent")

    playbook = found["playbook"][local_id[fixtures["playbook"]["externalId"]]].bundle
    assert playbook["playbook"]["agents"]["a1"] == {"item": local_id[fixtures["agent"]["externalId"]]}
    assert _mapped(playbook, fixtures["playbook"], ("publisher",), identity) == _bytes("playbook")

    package = found["package"][fixtures["package"]["externalId"]].bundle
    assert (package["category"], package["icon"], package["publisher"]) == ("", "", {"name": "Automatos"})
    assert _mapped(package, fixtures["package"], ("category", "icon", "publisher"), identity) == _bytes("package")

    plugin = found["plugin"][fixtures["plugin"]["externalId"]].bundle
    assert plugin["plugin"]["scan"]["staticFindings"] == fixtures["plugin"]["plugin"]["scan"]["staticFindings"]
    scan = {**plugin["plugin"]["scan"], "scannedAt": fixtures["plugin"]["plugin"]["scan"]["scannedAt"]}
    as_cloud = {**plugin, "plugin": {**plugin["plugin"], "scan": scan}}
    assert _mapped(as_cloud, fixtures["plugin"], ("category", "icon"), identity) == _bytes("plugin")
