"""PRD-270 W2 US-270-12b (K9, K10) — Automatos Cloud's export of admin-made packages and approved plugins.

``services/marketplace_bundles/export_packages.exported_packages`` and ``export_plugins.exported_plugins``
on Postgres: the rows are flushed on one session and never committed (the teardown rolls them back), and
every assertion reads only this test's items, by their externalIds, so rows the suite seeded never
matter. Proven here:

* a seeded package never exports; an admin package whose members all resolve exports with each member's
  reference as K8 says (a seeded agent by slug, an exported agent by item, a seeded Playbook by template
  id, an exported Playbook by item, a skill by name and plugin, a plugin by slug, a model by id, a tool
  by app), and a member with no name of its own takes its target's;
* a package with a member on a workspace agent, an unapproved marketplace agent, a plugin the export does
  not list (pending, or one whose report the format refuses), a workspace's or an inactive skill, or a
  skill an unlisted plugin made is left out, logged;
* a pending, rejected, inactive, 'review_required' or 'blocked' plugin, a 'safe' row whose scan says
  otherwise and one whose scan is of another version never export; a safe approved plugin exports with
  its report;
* no S3 bucket or path, approved_by, source URL, matched_text, pattern, evidence, scanner, setup key off
  the format or matching key off the format reaches any bundle's bytes;
* the round trip: rows built from the plugin fixture export exactly its bytes, and rows built from the
  package fixture export its bytes but for the three envelope fields a package row has no column for
  (category, icon, publisher: the export writes none, none and Automatos, which the test maps).

Without a database the pure checks still run: the seeded package slugs, the fixtures' externalIds, the
author and counts fitted, the scan report kept to the format, the setup and matching keys, and each
member type's reference.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import create_engine, text

from core.models.core import Agent, Skill, WorkflowTemplate as WorkflowRecipe
from core.models.marketplace_packages import MarketplacePackage
from core.models.marketplace_plugins import MarketplacePlugin, PluginCategory, PluginSecurityScan
from services.marketplace_bundles import export_packages, package_members
from services.marketplace_bundles.canonical import canonical_bytes
from services.marketplace_bundles.export_agents import exported_agents
from services.marketplace_bundles.export_packages import exported_packages
from services.marketplace_bundles.export_playbooks import exported_playbooks
from services.marketplace_bundles.export_plugins import author_of, bounded, exported_plugins, models_of
from services.marketplace_bundles.lineage import external_id
from services.marketplace_bundles.package_members import MemberRefs, Target, raw_ref
from services.marketplace_bundles.scan_report import scan_report
from services.marketplace_bundles.seeded import seeded_package_slugs
from services.marketplace_bundles.validate import validate_bundle

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "marketplace_bundles"
# The natural keys the fixtures' externalIds were hashed from, and the rows the package fixture names.
FIXTURE_PACKAGE_SLUG = "test-research-pack"
FIXTURE_PLUGIN_SLUG = "test-research-kit"
FIXTURE_AGENT_PUBLIC_ID = "00000000-0000-4000-8000-000000000001"
FIXTURE_TEMPLATE_ID = "test-weekly-digest"
FIXTURE_SEEDED_TEMPLATE_ID = "test-seeded-playbook"
FIXTURE_CATEGORY = "research"
# What a package row has no column for: the export writes these, the round trip maps them.
PACKAGE_ENVELOPE = {"category": "", "icon": "", "publisher": {"name": "Automatos"}}
MADE_UP = "sk-test-made-up"
WHEN = datetime(2026, 10, 30, 9, 15, 0)
TABLES = (
    "users", "workspaces", "agents", "workflow_recipes", "skills", "marketplace_packages", "marketplace_plugins",
    "plugin_security_scans", "plugin_categories",
)


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for table in TABLES:
                c.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the export tests need Postgres with the marketplace tables: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def cloud(engine, new_session):
    """One uncommitted session: every row is flushed, and the teardown rolls them all back."""
    db = new_session()
    try:
        yield NS(db=db, tag=uuid.uuid4().hex[:10])
    finally:
        new_session.sweep().close()


# --- rows --------------------------------------------------------------------------------------


def _add(cloud, row):
    cloud.db.add(row)
    cloud.db.flush()
    return row


def _workspace(cloud):
    ws = uuid.uuid4()
    cloud.db.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd270w2 export')"),
                     {"id": str(ws)})
    return ws


def _agent(cloud, **columns):
    """A marketplace agent K9 exports unless ``columns`` say otherwise."""
    return _add(cloud, Agent(**{
        "name": f"Agent {uuid.uuid4().hex[:8]}", "description": "An agent of the export test.",
        "agent_type": "custom", "status": "active", "owner_type": "marketplace", "owner_id": "marketplace",
        "is_approved": True, "version": "1.0.0", "created_at": WHEN, **columns,
    }))


def _playbook(cloud, **columns):
    """A marketplace Playbook K9 exports unless ``columns`` say otherwise."""
    return _add(cloud, WorkflowRecipe(**{
        "template_id": f"prd270w2-{uuid.uuid4().hex[:12]}", "name": f"Playbook {uuid.uuid4().hex[:8]}",
        "description": "A Playbook of the export test.", "owner_type": "marketplace", "owner_id": "marketplace",
        "is_approved": True, "is_public": True, "is_system": False, "version": "1.0", "created_at": WHEN,
        "created_by": "Test Publisher", "template_definition": {"steps": []}, "steps": [], **columns,
    }))


def _skill(cloud, name, package=None, workspace_id=None):
    return _add(cloud, Skill(name=name, skill_type="technical", package_slug=package, workspace_id=workspace_id))


def _category(cloud, slug, icon):
    """The plugin category ``slug`` (made here when the database has none), its icon ``icon`` in this session."""
    category = cloud.db.query(PluginCategory).filter(PluginCategory.slug == slug).first()
    if category is None:
        category = PluginCategory(slug=slug, name=slug.title())
    category.icon = icon
    return _add(cloud, category)


def _scan(cloud, slug, version, **columns):
    """A scan of ``slug`` at ``version`` that says what the scanner says of a safe plugin."""
    return _add(cloud, PluginSecurityScan(**{
        "plugin_slug": slug, "plugin_version": version, "static_scan_status": "passed", "static_findings": [],
        "llm_scan_status": "passed", "llm_risk_score": 3, "llm_findings": [], "llm_summary": "No issues found.",
        "overall_verdict": "safe", "scanned_at": WHEN, "scanned_by": f"scanner-{MADE_UP}@prd270w2.test",
        **columns,
    }))


def _plugin(cloud, scan=None, **columns):
    """An approved, active plugin whose scan said safe, which K9 exports unless ``columns`` or ``scan`` say
    otherwise; its storage, approver and source are made up so the bytes can be searched for them."""
    slug, version = columns.pop("slug", f"test-plugin-{cloud.tag}-{uuid.uuid4().hex[:6]}"), columns.pop("version", "1.0.0")
    scan_row = _scan(cloud, slug, version, **(scan or {}))
    return _add(cloud, MarketplacePlugin(**{
        "slug": slug, "name": f"Plugin {slug}", "version": version, "description": "A plugin of the export test.",
        "approval_status": "approved", "is_active": True, "security_status": "safe",
        "security_scan_id": scan_row.id, "s3_bucket": f"bucket-{MADE_UP}", "s3_path": f"plugins/{slug}/{version}/",
        "approved_by": f"approver-{MADE_UP}@prd270w2.test", "source_url": f"https://source.test/{MADE_UP}",
        "source_repo": f"repo-{MADE_UP}", **columns,
    }))


def _package(cloud, members, **columns):
    return _add(cloud, MarketplacePackage(**{
        "slug": f"test-pack-{cloud.tag}-{uuid.uuid4().hex[:6]}", "name": "Test Pack",
        "description": "A package of the export test.", "vertical_tags": ["test"],
        "matching": {"platforms": [], "url_patterns": [], "vocabulary": []}, "members": members,
        "setup_manifest": {}, **columns,
    }))


def _llm(model="test-vendor/test-model-1"):
    return {"type": "llm", "ref": model, "name": "Test Model", "description": ""}


# --- reads ---------------------------------------------------------------------------------------


def _export(cloud):
    agents, plugins = exported_agents(cloud.db), exported_plugins(cloud.db)
    playbooks = exported_playbooks(cloud.db, agents)
    packages = exported_packages(cloud.db, agents, playbooks, plugin_items=plugins)
    return NS(agents=agents, playbooks=playbooks, packages={item.entry.external_id: item for item in packages},
              plugins={item.entry.external_id: item for item in plugins})


def _package_id_of(package):
    return external_id("package", f"row:{package.slug}")


def _plugin_id_of(plugin):
    return external_id("plugin", f"row:{plugin.slug}")


def _fixture(name):
    return (FIXTURES / f"{name}.json").read_bytes().rstrip(b"\n")


# --- without a database ------------------------------------------------------------------------


def test_the_seeded_packages_are_the_seed_packages():
    from core.seeds.seed_packages import PACKAGES

    assert seeded_package_slugs() == {spec["slug"] for spec in PACKAGES}


def test_the_fixtures_external_ids_are_the_hash_of_their_slugs():
    assert external_id("package", f"row:{FIXTURE_PACKAGE_SLUG}") == json.loads(_fixture("package"))["externalId"]
    assert external_id("plugin", f"row:{FIXTURE_PLUGIN_SLUG}") == json.loads(_fixture("plugin"))["externalId"]


def test_an_author_is_a_name_never_an_email():
    assert author_of("Test Publisher") == "Test Publisher"
    assert author_of("Test Publisher <test@prd270w2.test>") == "Test Publisher"
    for author in ("test@prd270w2.test", "<test@prd270w2.test>", None, 7):
        assert author_of(author) == ""
    assert author_of("x" * 300) == "x" * 255


def test_counts_and_models_are_fitted_to_the_format():
    assert bounded(600, 500) == 500 and bounded(12, 500) == 12
    for value in (-1, None, True, "3", 2.5):
        assert bounded(value, 500) == 0
    models = ["test-vendor/m", "not a model", "test-vendor/m", 7, "other/m"] + [f"v/m{n}" for n in range(30)]
    assert models_of(models) == ["test-vendor/m", "other/m"] + [f"v/m{n}" for n in range(18)]
    assert models_of(None) == [] and models_of("test-vendor/m") == []


def test_a_scan_report_keeps_only_the_formats_fields():
    static = [
        {"type": "file_write", "severity": "low", "file": "a.md", "line": n, "pattern": "p", "matched_text": "m",
         "description": "d"}
        for n in range(55)
    ] + [{"type": "network_access", "severity": "high", "file": "b.py", "line": 1, "pattern": "p",
          "matched_text": "m", "description": "x" * 3000}]
    llm = [
        {"category": "c", "severity": " HIGH ", "description": "one", "file": "f", "evidence": "e"},
        {"severity": "info", "description": "two"}, {"severity": "critical", "description": 7}, "not an object",
    ]
    row = NS(overall_verdict="safe", static_scan_status="flagged", static_findings=static, llm_scan_status="passed",
             llm_risk_score=-2, llm_summary=None, llm_findings=llm, scanned_at=WHEN)
    report = scan_report(row)
    assert len(report["staticFindings"]) == 50
    assert report["staticFindings"][0] == {"type": "network_access", "severity": "high", "file": "b.py", "line": 1,
                                           "description": "x" * 2000}
    assert all(set(finding) == {"type", "severity", "file", "line", "description"}
               for finding in report["staticFindings"])
    assert report["findings"] == [{"severity": "critical", "description": ""}, {"severity": "high", "description": "one"},
                                  {"severity": "medium", "description": "two"}]
    assert report["riskScore"] == 0 and report["summary"] == "" and report["scannedAt"] == "2026-10-30T09:15:00"
    assert static[0]["matched_text"] == "m" and llm[0]["evidence"] == "e"  # the row's findings are untouched


def test_the_setup_and_matching_keep_only_the_formats_keys():
    manifest = {
        "questions": [{"id": "q", "prompt": "Which?", "api_key": MADE_UP, "setting": None}],
        "required_connects": [{"app_name": "SLACK", "webhook_url": f"https://hooks.test/{MADE_UP}"}],
        "secrets": [MADE_UP],
    }
    assert export_packages._setup(manifest) == {
        "questions": [{"id": "q", "prompt": "Which?"}], "required_connects": [{"app_name": "SLACK"}],
        "guide_steps": [], "report_templates": [],
    }
    assert export_packages._setup(None) == {"questions": [], "required_connects": [], "guide_steps": [],
                                            "report_templates": []}
    assert export_packages._matching({"platforms": ["shopify"], "token": MADE_UP}) == {
        "platforms": ["shopify"], "url_patterns": [], "vocabulary": [],
    }
    assert manifest["questions"][0]["api_key"] == MADE_UP  # the row's manifest is untouched


def _member_refs():
    """Marketplace rows as the export reads them, for each member type's reference."""
    seeded, community = NS(id=1, slug="seeded-agent", name="Seeded"), NS(id=2, slug=None, name="Community")
    unexported = NS(id=3, slug=None, name="Unexported")
    seeded_pb, community_pb = NS(id=10, template_id="seeded-pb", name="Seeded PB"), NS(id=11, template_id="pb", name="PB")
    plugin_id = uuid.UUID("00000000-0000-4000-8000-0000000000aa")
    return MemberRefs(
        agents=package_members._Rows({1: seeded, 2: community, 3: unexported},
                                     {"seeded-agent": seeded, "Community": community, "Unexported": unexported}),
        playbooks=package_members._Rows({10: seeded_pb, 11: community_pb}, {"seeded-pb": seeded_pb, "pb": community_pb}),
        skills={20: NS(id=20, name="cite", package_slug=None), 21: NS(id=21, name="sum", package_slug="kit"),
                22: NS(id=22, name="other", package_slug="unlisted")},
        plugins={plugin_id: package_members._Plugin("kit", "Kit")},
        agent_items={2: "a" * 32}, playbook_items={11: "b" * 32},
    )


def test_each_member_type_names_its_target_as_k8_says(monkeypatch):
    monkeypatch.setattr(package_members, "seeded_template_ids", lambda: frozenset({"seeded-pb"}))
    refs = _member_refs()
    assert refs.target_of("agent", "seeded-agent") == refs.target_of("agent", "1") == Target({"slug": "seeded-agent"},
                                                                                              "Seeded")
    assert refs.target_of("agent", "2") == refs.target_of("agent", "Community") == Target({"item": "a" * 32}, "Community")
    for unknown in ("3", "Unexported", "nobody", "99"):  # unexported, not found, an id looked up by id only
        assert refs.target_of("agent", unknown) is None
    assert refs.target_of("playbook", "seeded-pb") == Target({"templateId": "seeded-pb"}, "Seeded PB")
    assert refs.target_of("playbook", "11") == refs.target_of("playbook", "pb") == Target({"item": "b" * 32}, "PB")
    assert refs.target_of("skill", "20") == Target({"name": "cite"}, "cite")
    assert refs.target_of("skill", "21") == Target({"name": "sum", "package": "kit"}, "sum")
    assert refs.target_of("skill", "22") is None and refs.target_of("skill", "cite") is None  # an unlisted plugin's
    assert refs.target_of("plugin", "00000000-0000-4000-8000-0000000000aa") == Target({"slug": "kit"}, "Kit")
    assert refs.target_of("plugin", "kit") is None
    assert refs.target_of("llm", "test-vendor/m") == Target({"modelId": "test-vendor/m"}, "test-vendor/m")
    assert refs.target_of("tool", " slack ") == Target({"app": "SLACK"}, "SLACK")
    assert refs.target_of("theme", "x") is None and refs.target_of(None, "x") is None
    assert refs.target_of("agent", None) is None


def test_a_members_target_is_read_as_install_package_reads_it():
    assert raw_ref({"ref": 7}) == "7" and raw_ref({"id": "x"}) == "x" and raw_ref({"slug": "s"}) == "s"
    assert raw_ref({"ref": None, "slug": "s"}) == "s"
    assert raw_ref({"type": "agent"}) is None and raw_ref("agent") is None


# --- on Postgres: packages -------------------------------------------------------------------------


def test_a_seeded_package_never_exports(cloud, monkeypatch):
    seeded = _package(cloud, [_llm()])
    admin = _package(cloud, [_llm()])
    monkeypatch.setattr(export_packages, "seeded_package_slugs", lambda: frozenset({seeded.slug}))
    found = _export(cloud).packages
    assert _package_id_of(seeded) not in found
    item = found[_package_id_of(admin)]
    assert validate_bundle(item.bundle) is item.bundle and item.entry.version == "1.0.0"
    assert item.bundle["publisher"] == {"name": "Automatos"} and item.bundle["package"]["slug"] == admin.slug
    assert hashlib.sha256(canonical_bytes(item.bundle)).hexdigest() == item.revision


def test_an_admin_package_exports_each_member_by_its_k8_reference(cloud, monkeypatch):
    seeded_pb = _playbook(cloud)
    monkeypatch.setattr(package_members, "seeded_template_ids", lambda: frozenset({seeded_pb.template_id}))
    seeded, community, by_public_id = _agent(cloud, slug=f"seeded-{cloud.tag}"), _agent(cloud), _agent(cloud)
    playbook, plugin = _playbook(cloud), _plugin(cloud)
    skill, plain = _skill(cloud, f"skill-{cloud.tag}", package=plugin.slug), _skill(cloud, f"plain-{cloud.tag}")
    package = _package(cloud, [
        {"type": "agent", "ref": seeded.slug, "name": "Seeded helper", "description": "Seeded."},
        {"type": "agent", "ref": community.id},
        {"type": "agent", "ref": str(by_public_id.public_id), "name": "By public id"},
        {"type": "playbook", "ref": seeded_pb.template_id, "name": "Seeded Playbook"},
        {"type": "playbook", "ref": str(playbook.id), "name": "Community Playbook"},
        {"type": "skill", "ref": skill.id, "name": "Skill"},
        {"type": "skill", "ref": str(plain.id), "name": "Plain skill"},
        {"type": "plugin", "ref": str(plugin.id), "name": "Plugin"},
        _llm(),
        {"type": "tool", "ref": "slack", "name": "Slack"},
    ])
    members = _export(cloud).packages[_package_id_of(package)].bundle["package"]["members"]
    assert [(member["type"], member["ref"]) for member in members] == [
        ("agent", {"slug": seeded.slug}),
        ("agent", {"item": external_id("agent", f"row:{community.public_id}")}),
        ("agent", {"item": external_id("agent", f"row:{by_public_id.public_id}")}),
        ("playbook", {"templateId": seeded_pb.template_id}),
        ("playbook", {"item": external_id("playbook", f"row:{playbook.template_id}")}),
        ("skill", {"name": skill.name, "package": skill.package_slug}),
        ("skill", {"name": plain.name}),
        ("plugin", {"slug": plugin.slug}),
        ("llm", {"modelId": "test-vendor/test-model-1"}),
        ("tool", {"app": "SLACK"}),
    ]
    assert members[0]["name"] == "Seeded helper" and members[0]["description"] == "Seeded."
    assert members[1]["name"] == community.name and members[1]["description"] == ""


def test_a_member_without_an_allowed_reference_keeps_the_package_out(cloud, caplog):
    workspace_id = _workspace(cloud)
    own = _agent(cloud, owner_type="workspace", owner_id=str(workspace_id), workspace_id=workspace_id,
                 is_approved=False, name=f"Own private agent {cloud.tag}")
    own_skill = _skill(cloud, f"own-skill-{cloud.tag}", workspace_id=workspace_id)
    unapproved, pending = _agent(cloud, is_approved=False), _plugin(cloud, approval_status="pending")
    refused = _plugin(cloud, scan={"llm_risk_score": None})
    inactive = _add(cloud, Skill(name=f"inactive-{cloud.tag}", skill_type="technical", is_active=False))
    unlisted_skill = _skill(cloud, f"pending-made-{cloud.tag}", package=pending.slug)
    left_out = [
        _package(cloud, [_llm(), {"type": "agent", "ref": own.id, "name": "Helper"}]),
        _package(cloud, [{"type": "agent", "ref": unapproved.id, "name": "Unapproved"}]),
        _package(cloud, [{"type": "plugin", "ref": str(pending.id), "name": "Pending"}]),
        _package(cloud, [{"type": "plugin", "ref": str(refused.id), "name": "Refused"}]),
        _package(cloud, [{"type": "skill", "ref": own_skill.id, "name": "Own skill"}]),
        _package(cloud, [{"type": "skill", "ref": inactive.id, "name": "Inactive skill"}]),
        _package(cloud, [{"type": "skill", "ref": unlisted_skill.id, "name": "Unlisted plugin's skill"}]),
        _package(cloud, [{"type": "skill", "ref": "not-an-id", "name": "Skill"}]),
        _package(cloud, [{"type": "theme", "ref": "x", "name": "Theme"}]),
        _package(cloud, ["not an object"]),
    ]
    with caplog.at_level(logging.WARNING, logger="services.marketplace_bundles.export_packages"):
        found = _export(cloud)
    assert not {_package_id_of(package) for package in left_out} & set(found.packages)
    warned = [record.getMessage() for record in caplog.records
              if record.name == "services.marketplace_bundles.export_packages"]
    assert any(f"package row {left_out[0].id} is left out: its member 1 " in message for message in warned)
    for item in found.packages.values():
        data = canonical_bytes(item.bundle).decode("utf-8")
        assert own.name not in data and pending.slug not in data


# --- on Postgres: plugins -------------------------------------------------------------------------


def test_plugins_k9_leaves_out_never_export(cloud):
    left_out = [
        _plugin(cloud, approval_status="pending"), _plugin(cloud, approval_status="rejected"),
        _plugin(cloud, is_active=False), _plugin(cloud, security_status="review_required"),
        _plugin(cloud, security_status="blocked"), _plugin(cloud, security_status="pending"),
        _plugin(cloud, scan={"overall_verdict": "review_required"}), _plugin(cloud, scan={"overall_verdict": "blocked"}),
        _plugin(cloud, scan={"plugin_version": "0.9.0"}), _plugin(cloud, security_scan_id=None),
    ]
    kept = _plugin(cloud)
    found = _export(cloud).plugins
    assert not {_plugin_id_of(plugin) for plugin in left_out} & set(found)
    assert _plugin_id_of(kept) in found


def test_a_safe_approved_plugin_exports_with_its_report(cloud):
    category = _category(cloud, f"test-cat-{cloud.tag}", "🧪")
    plugin = _plugin(
        cloud, category_id=category.id, original_author="Test Author <author@prd270w2.test>", license="MIT",
        tags=["test"], skills_count=600, commands_count=1, token_estimate=-5,
        recommended_models=["test-vendor/m", "not a model", "test-vendor/m"],
        scan={
            "static_scan_status": "flagged", "llm_risk_score": 7, "llm_summary": "Fine.",
            "static_findings": [{"type": "file_write", "severity": "medium", "file": "skills/x/SKILL.md", "line": 3,
                                 "pattern": "open", "matched_text": "open('x', 'a')",
                                 "description": "Opens file in append mode"}],
            "llm_findings": [{"category": "fs", "severity": "Low", "description": "Writes a note.", "file": "f",
                              "evidence": "open"}],
        },
    )
    item = _export(cloud).plugins[_plugin_id_of(plugin)]
    bundle, body = item.bundle, item.bundle["plugin"]
    assert validate_bundle(bundle) is bundle and item.entry.version == "1.0.0" and body["slug"] == plugin.slug
    assert bundle["category"] == category.slug and bundle["icon"] == "🧪" and bundle["publisher"] == {"name": "Automatos"}
    assert body["author"] == "Test Author" and body["contents"] == {"skills": 500, "commands": 1, "agents": 0, "hooks": 0}
    assert body["tokenEstimate"] == 0 and body["recommendedModels"] == ["test-vendor/m"]
    assert body["scan"] == {
        "verdict": "safe", "staticStatus": "flagged", "llmStatus": "passed", "riskScore": 7, "summary": "Fine.",
        "staticFindings": [{"type": "file_write", "severity": "medium", "file": "skills/x/SKILL.md", "line": 3,
                            "description": "Opens file in append mode"}],
        "findings": [{"severity": "low", "description": "Writes a note."}], "scannedAt": WHEN.isoformat(),
    }
    assert hashlib.sha256(canonical_bytes(bundle)).hexdigest() == item.revision


def test_a_plugin_whose_report_the_format_refuses_is_logged_and_left_out(cloud, caplog):
    critical = _plugin(cloud, scan={"static_findings": [
        {"type": "dangerous_call", "severity": "critical", "file": "a.py", "line": 1, "pattern": "eval",
         "matched_text": f"eval({MADE_UP})", "description": "Uses eval()"},
    ]})
    unscored = _plugin(cloud, scan={"llm_risk_score": None})
    with caplog.at_level(logging.ERROR, logger="services.marketplace_bundles.exported"):
        found = _export(cloud).plugins
    assert _plugin_id_of(critical) not in found and _plugin_id_of(unscored) not in found
    refused = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert any(f"plugin row {critical.id} is left out" in message for message in refused)
    assert any(f"plugin row {unscored.id} is left out" in message for message in refused)
    assert not any(MADE_UP in message for message in refused)


def test_nothing_private_reaches_the_bundle_bytes(cloud):
    plugin = _plugin(cloud, scan={
        "static_findings": [{"type": "file_write", "severity": "low", "file": "a.md", "line": 1,
                             "pattern": f"pattern-{MADE_UP}", "matched_text": f"matched-{MADE_UP}", "description": "d"}],
        "llm_findings": [{"severity": "low", "description": "d", "evidence": f"evidence-{MADE_UP}",
                          "file": f"file-{MADE_UP}", "category": f"category-{MADE_UP}"}],
    })
    package = _package(
        cloud, [_llm()], matching={"platforms": ["test"], "url_patterns": [], "vocabulary": [], "token": MADE_UP},
        setup_manifest={"questions": [{"id": "q", "prompt": "Which?", "api_key": MADE_UP}],
                        "required_connects": [{"app_name": "SLACK", "webhook_url": f"https://hooks.test/{MADE_UP}"}],
                        "secrets": [MADE_UP]},
    )
    found = _export(cloud)
    ours = (found.plugins[_plugin_id_of(plugin)], found.packages[_package_id_of(package)])
    for item in (*found.plugins.values(), *found.packages.values()):
        data = canonical_bytes(item.bundle).decode("utf-8")
        assert MADE_UP not in data and plugin.s3_path not in data
    for item in ours:
        data = canonical_bytes(item.bundle).decode("utf-8")
        for private in ("s3_path", "s3_bucket", "bucket", "approved_by", "matched_text", "evidence", "scanned_by",
                        "source.test", "webhook", "api_key", "secrets", str(plugin.id), str(package.id)):
            assert private not in data


# --- the round trip --------------------------------------------------------------------------------


def _fixture_plugin(cloud):
    """Rows built from plugin.json, with what never travels beside them (patterns, matched text, evidence)."""
    bundle = json.loads(_fixture("plugin"))
    body, scan = bundle["plugin"], bundle["plugin"]["scan"]
    category = _category(cloud, bundle["category"], bundle["icon"])
    return _plugin(
        cloud, slug=body["slug"], version=bundle["version"], name=bundle["name"], description=bundle["description"],
        long_description=body["longDescription"], tags=bundle["tags"], category_id=category.id,
        original_author=body["author"], license=body["license"], recommended_models=body["recommendedModels"],
        token_estimate=body["tokenEstimate"], **{f"{kind}_count": count for kind, count in body["contents"].items()},
        scan={
            "static_scan_status": scan["staticStatus"], "llm_scan_status": scan["llmStatus"],
            "llm_risk_score": scan["riskScore"], "llm_summary": scan["summary"],
            "scanned_at": datetime.fromisoformat(scan["scannedAt"]),
            "static_findings": [{**finding, "pattern": r"\bopen\s*\(", "matched_text": "open('notes.md', 'a')"}
                                for finding in scan["staticFindings"]],
            "llm_findings": [{**finding, "category": "file_access", "file": "commands/save-notes.md",
                              "evidence": "open("} for finding in scan["findings"]],
        },
    )


def _fixture_package(cloud, plugin):
    """Rows built from package.json: each member's target, stored as an admin's package names it (an agent
    by id or slug, a Playbook by template id, a skill and a plugin by id, a model by id, a tool by app)."""
    bundle = json.loads(_fixture("package"))
    body = bundle["package"]
    stored = [
        _agent(cloud, public_id=uuid.UUID(FIXTURE_AGENT_PUBLIC_ID)).id,
        _agent(cloud, slug="content-strategist", name=f"Seeded {cloud.tag}").slug,
        _playbook(cloud, template_id=FIXTURE_TEMPLATE_ID).template_id,
        _playbook(cloud, template_id=FIXTURE_SEEDED_TEMPLATE_ID).template_id,
        _skill(cloud, "summarise-sources", package=FIXTURE_PLUGIN_SLUG).id,
        _skill(cloud, "cite-sources").id,
        str(plugin.id),
        "test-vendor/test-model-1",
        "slack",
    ]
    members = [{**{key: value for key, value in member.items() if key != "ref"}, "ref": ref}
               for member, ref in zip(body["members"], stored)]
    assert [member["type"] for member in members] == [member["type"] for member in body["members"]]
    return _package(
        cloud, members, slug=body["slug"], name=bundle["name"], description=bundle["description"],
        vertical_tags=body["verticalTags"], matching={**body["matching"], "token": MADE_UP},
        setup_manifest={**body["setup"], "secrets": [MADE_UP]},
    )


def test_the_fixtures_round_trip(cloud, monkeypatch):
    monkeypatch.setattr(package_members, "seeded_template_ids", lambda: frozenset({FIXTURE_SEEDED_TEMPLATE_ID}))
    _fixture_package(cloud, _fixture_plugin(cloud))
    found = _export(cloud)
    plugin_item = found.plugins[json.loads(_fixture("plugin"))["externalId"]]
    assert canonical_bytes(plugin_item.bundle) == _fixture("plugin")
    assert plugin_item.revision == hashlib.sha256(_fixture("plugin")).hexdigest()
    expected = canonical_bytes({**json.loads(_fixture("package")), **PACKAGE_ENVELOPE})
    package_item = found.packages[json.loads(_fixture("package"))["externalId"]]
    assert canonical_bytes(package_item.bundle) == expected
    assert package_item.revision == hashlib.sha256(expected).hexdigest()
    # Called as the AC names it, the export lists the plugins itself: the same package.
    alone = exported_packages(cloud.db, found.agents, found.playbooks)
    assert [canonical_bytes(item.bundle) for item in alone if item.entry.external_id == package_item.entry.external_id] \
        == [expected]
