"""PRD-183 S5 (F076) — generic VerticalProvisioner abstraction.

Only the widget READ path went through PRD-141; the provision/sync/webhook
WRITE path stayed Shopify-shaped, so vertical #2 would fork ``api/shopify.py``.
S5 extracts a generic ``VerticalProvisioner`` interface + a
``PROVISIONER_REGISTRY``, so provisioning a workspace for any vertical is a
registry lookup + a generic flow — not a copy of the Shopify routes.

These tests pin the contract:

  * ``VerticalProvisioner`` is a structural interface; Shopify's provisioner
    is registered under "shopify" and declares its roster, widget defaults,
    key permissions, ops-manager slug, and site type.
  * ``provision_vertical`` (the generic flow) provisions a **mock** vertical
    end-to-end through the registry — no import of ``api.shopify`` — creating
    the workspace with the vertical stamped, seeding the declared roster, and
    minting a key with the declared permissions.
  * catalog/orders graph-source mappers are reachable through the registry,
    not hardcoded in generic code.

Pure: the DB, ApiKeyService, and agent-seeding are faked at the boundary.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

from integrations.provisioning import (  # noqa: E402
    PROVISIONER_REGISTRY,
    VerticalProvisioner,
    provision_vertical,
    get_graph_source_mapper,
)
# Importing the shopify package must self-register its provisioner (as it does
# its widget plugin).
import integrations.shopify  # noqa: E402,F401


# ------------------------------------------------------------------
# Registry + interface
# ------------------------------------------------------------------


def test_shopify_provisioner_registered():
    assert "shopify" in PROVISIONER_REGISTRY
    prov = PROVISIONER_REGISTRY["shopify"]
    assert isinstance(prov, VerticalProvisioner)
    # Declares its roster + key perms + widget defaults + ops manager slug.
    assert "shopify-ops" in prov.agent_slugs
    assert prov.ops_manager_slug == "shopify-ops"
    assert "chat" in prov.key_permissions
    assert prov.default_widget_config.get("enabled") is False
    assert prov.site_type == "shopify"


def test_shopify_allowed_domains_delegates():
    prov = PROVISIONER_REGISTRY["shopify"]
    domains = prov.allowed_domains("store.myshopify.com", {})
    assert "https://store.myshopify.com" in domains
    assert "https://*.myshopify.com" in domains


# ------------------------------------------------------------------
# Graph-source mappers behind the registry
# ------------------------------------------------------------------


def test_graph_source_mappers_reachable_via_registry():
    """The catalog + orders mappers are looked up through the vertical, not
    hardcoded in generic code."""
    catalog = get_graph_source_mapper("shopify", "catalog")
    orders = get_graph_source_mapper("shopify", "orders")
    assert callable(catalog)
    assert callable(orders)
    # It is the real graph_extraction mapper (identity check via name).
    assert catalog.__name__ == "map_shopify_catalog"
    assert orders.__name__ == "map_shopify_orders"


# ------------------------------------------------------------------
# Generic provision flow with a MOCK vertical (no api.shopify)
# ------------------------------------------------------------------


class _FakeQuery:
    def filter(self, *a, **k):
        return self

    def first(self):
        return None  # no existing workspace → create path

    def count(self):
        return 0  # no existing agents → seed path


class _FakeDb:
    def __init__(self):
        self.added = []
        self.committed = False

    def query(self, *a, **k):
        return _FakeQuery()

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        pass

    def commit(self):
        self.committed = True


def test_vertical_provision_generic(monkeypatch):
    """A second (mock) vertical provisions through the generic path.

    Registers a throwaway 'demo' provisioner and drives ``provision_vertical``.
    Asserts: workspace created with vertical='demo', the declared roster seeded,
    and a key minted with the declared permissions — all without touching
    ``api.shopify``.
    """
    seeded = {}
    minted = {}

    class _DemoProvisioner:
        vertical = "demo"
        agent_slugs = ["demo-ops", "demo-helper"]
        ops_manager_slug = "demo-ops"
        default_widget_config = {"enabled": False, "page_types": ["home"]}
        key_permissions = ["chat", "agents:read"]
        key_type = "public"
        site_type = None

        def allowed_domains(self, external_id, metadata):
            return [f"https://{external_id}"]

        def on_provisioned(self, db, workspace):
            seeded["hook_ran"] = True

    demo = _DemoProvisioner()
    monkeypatch.setitem(PROVISIONER_REGISTRY, "demo", demo)

    # Fake the generic building blocks the flow leans on.
    import integrations.provisioning as prov_mod

    def _fake_seed(db, workspace_id, provisioner):
        seeded["workspace_id"] = workspace_id
        seeded["slugs"] = list(provisioner.agent_slugs)
        return len(provisioner.agent_slugs)

    monkeypatch.setattr(prov_mod, "_seed_roster", _fake_seed)

    def _fake_create_key(**kw):
        minted.update(kw)
        return {"key": "pk_demo_123", "key_prefix": "pk_demo"}

    monkeypatch.setattr(prov_mod, "_create_widget_key", _fake_create_key)

    db = _FakeDb()
    result = provision_vertical(
        db=db,
        vertical="demo",
        external_id="demo-store.example.com",
        name="Demo Store",
        metadata={},
    )

    # Workspace created + stamped with the vertical (the real Workspace model
    # constructs fine without a DB session).
    from core.models.workspaces import Workspace as _RealWorkspace
    created = [o for o in db.added if isinstance(o, _RealWorkspace)]
    assert created, "workspace was not created"
    ws = created[0]
    assert ws.settings["vertical"] == "demo"
    assert ws.settings["widget_proactive"]["page_types"] == ["home"]

    # Roster seeded from the provisioner's declared slugs.
    assert seeded["slugs"] == ["demo-ops", "demo-helper"]

    # Key minted with the provisioner's declared permissions + domains.
    assert minted["permissions"] == ["chat", "agents:read"]
    assert minted["allowed_domains"] == ["https://demo-store.example.com"]

    # Post-provision hook ran.
    assert seeded.get("hook_ran") is True

    assert result["agents_installed"] == 2
    assert result["api_key"] == "pk_demo_123"
    assert result["is_new"] is True


def test_unknown_vertical_rejected():
    db = _FakeDb()
    try:
        provision_vertical(db=db, vertical="nope", external_id="x", name="X", metadata={})
        assert False, "expected a rejection for an unknown vertical"
    except (KeyError, ValueError) as e:
        assert "nope" in str(e) or "vertical" in str(e).lower()
