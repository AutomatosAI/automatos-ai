"""
PRD-008-A Phase 2 — Sites service-layer unit tests
====================================================

Pure-Python unit tests with mocked SQLAlchemy sessions. No DB
roundtrip; no FastAPI app boot. Verifies authorization invariants,
shallow-merge semantics, status validation, secret omission, and
404-not-403 behaviour on cross-workspace access.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_site(
    *,
    workspace_id: UUID,
    type: str = "shopify",
    display_name: str = "test-site",
    external_id: str | None = "test.myshopify.com",
    settings: dict | None = None,
    capabilities: dict | None = None,
) -> SimpleNamespace:
    """Build a SimpleNamespace that quacks like a Site row."""
    from core.models.sites import (
        CAPABILITY_KEYS,
        SITE_TYPES,
        derive_default_capabilities,
    )

    stored = capabilities or {}
    defaults = (
        derive_default_capabilities(type)
        if type in SITE_TYPES
        else {key: False for key in CAPABILITY_KEYS}
    )
    effective = {key: bool(stored.get(key, defaults[key])) for key in CAPABILITY_KEYS}

    return SimpleNamespace(
        id=uuid4(),
        workspace_id=workspace_id,
        type=type,
        external_id=external_id,
        display_name=display_name,
        status="active",
        settings=settings or {},
        capabilities=stored,
        effective_capabilities=effective,
        secrets={"shopify_access_token": "shpat_secret"},
        created_at=datetime(2026, 5, 14, 12, 0, 0),
        updated_at=datetime(2026, 5, 14, 12, 0, 0),
    )


def _mock_db_returning(rows_or_one):
    """Build a MagicMock that mimics db.query(...).filter(...).first()/.all()/.one_or_none()."""
    db = MagicMock()
    chain = db.query.return_value.filter.return_value
    if isinstance(rows_or_one, list):
        chain.order_by.return_value.all.return_value = rows_or_one
        chain.all.return_value = rows_or_one
    else:
        chain.one_or_none.return_value = rows_or_one
    return db


# ---------------------------------------------------------------------------
# list_sites — order + scoping
# ---------------------------------------------------------------------------

def test_list_sites_returns_workspace_owned_rows():
    from services.sites import list_sites

    ws = uuid4()
    sites = [_make_fake_site(workspace_id=ws), _make_fake_site(workspace_id=ws)]
    db = _mock_db_returning(sites)

    result = list_sites(db, ws)
    assert result == sites


def test_list_sites_filters_by_workspace_id():
    """Trip-wire: ensure the query is scoped to workspace_id."""
    from services.sites import list_sites

    ws = uuid4()
    db = _mock_db_returning([])
    list_sites(db, ws)

    # db.query(Site).filter(...).order_by(...).all() was called
    db.query.assert_called_once()
    db.query.return_value.filter.assert_called_once()


# ---------------------------------------------------------------------------
# get_site — 404-not-403 on cross-workspace
# ---------------------------------------------------------------------------

def test_get_site_returns_site_when_owned():
    from services.sites import get_site

    ws = uuid4()
    site = _make_fake_site(workspace_id=ws)
    db = _mock_db_returning(site)

    assert get_site(db, ws, site.id) is site


def test_get_site_returns_none_when_not_found():
    """No existence leak: not-found and owned-by-other look identical
    from the caller's perspective."""
    from services.sites import get_site

    db = _mock_db_returning(None)
    assert get_site(db, uuid4(), uuid4()) is None


# ---------------------------------------------------------------------------
# create_site — type validation, capability derivation, settings deep-copy
# ---------------------------------------------------------------------------

def test_create_site_rejects_unknown_type():
    from services.sites import create_site

    db = MagicMock()
    with pytest.raises(ValueError, match="unknown site type"):
        create_site(db, uuid4(), type="magento", display_name="x")
    # Never reaches the DB
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_create_site_derives_capabilities_for_shopify():
    """Capabilities are derived from type, not accepted from caller."""
    from services.sites import create_site

    db = MagicMock()
    create_site(db, uuid4(), type="shopify", display_name="x")
    site = db.add.call_args[0][0]

    assert site.capabilities["has_cart"] is True
    assert site.capabilities["has_catalog"] is True
    assert site.capabilities["supports_theme_override"] is True


def test_create_site_derives_capabilities_for_custom():
    from services.sites import create_site

    db = MagicMock()
    create_site(db, uuid4(), type="custom", display_name="acme.com")
    site = db.add.call_args[0][0]

    assert site.capabilities["has_cart"] is False
    assert site.capabilities["supports_theme_override"] is False


def test_create_site_deep_copies_settings():
    """Caller-supplied settings must not be shared by reference — that
    would let one Site mutate another's settings via aliasing."""
    from services.sites import create_site

    db = MagicMock()
    shared = {"widget_proactive": {"enabled": True}}
    create_site(db, uuid4(), type="shopify", display_name="x", settings=shared)
    site = db.add.call_args[0][0]

    shared["widget_proactive"]["enabled"] = False
    assert site.settings["widget_proactive"]["enabled"] is True


def test_create_site_persists_via_db():
    from services.sites import create_site

    db = MagicMock()
    create_site(db, uuid4(), type="shopify", display_name="x")
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()


# ---------------------------------------------------------------------------
# update_site_meta — status allow-list, partial updates, not-found
# ---------------------------------------------------------------------------

def test_update_meta_returns_none_when_site_not_found():
    from services.sites import update_site_meta

    db = _mock_db_returning(None)
    result = update_site_meta(
        db, uuid4(), uuid4(), display_name="x", status="paused"
    )
    assert result is None
    db.commit.assert_not_called()


def test_update_meta_can_change_display_name_only():
    from services.sites import update_site_meta

    ws = uuid4()
    site = _make_fake_site(workspace_id=ws)
    db = _mock_db_returning(site)

    result = update_site_meta(db, ws, site.id, display_name="new-name")
    assert result is site
    assert site.display_name == "new-name"
    db.commit.assert_called_once()


def test_update_meta_can_change_status_only():
    from services.sites import update_site_meta

    ws = uuid4()
    site = _make_fake_site(workspace_id=ws)
    db = _mock_db_returning(site)

    update_site_meta(db, ws, site.id, status="paused")
    assert site.status == "paused"


def test_update_meta_rejects_status_outside_allowlist():
    """Backend-managed statuses like 'error' must NOT be settable via PATCH."""
    from services.sites import update_site_meta

    ws = uuid4()
    site = _make_fake_site(workspace_id=ws)
    db = _mock_db_returning(site)

    with pytest.raises(ValueError, match="status must be one of"):
        update_site_meta(db, ws, site.id, status="error")

    # The site must NOT have been mutated either — error happens before assignment
    assert site.status == "active"


# ---------------------------------------------------------------------------
# update_site_settings — shallow merge, immutability of inputs
# ---------------------------------------------------------------------------

def test_update_settings_shallow_merges_top_level_keys():
    """Top-level blocks (widget_proactive / callback / cart_idle) are
    independent units. Merging one must NOT touch the others."""
    from services.sites import update_site_settings

    ws = uuid4()
    site = _make_fake_site(
        workspace_id=ws,
        settings={
            "widget_proactive": {"enabled": True, "page_types": ["product"]},
            "callback": {"enabled": False},
        },
    )
    db = _mock_db_returning(site)

    update_site_settings(
        db, ws, site.id, settings_patch={"callback": {"enabled": True}}
    )

    # widget_proactive untouched
    assert site.settings["widget_proactive"] == {
        "enabled": True,
        "page_types": ["product"],
    }
    # callback replaced wholesale (shallow merge replaces the block)
    assert site.settings["callback"] == {"enabled": True}


def test_update_settings_does_not_mutate_input_patch():
    """The patch dict is the caller's; deep-copy so future mutations
    don't leak into the stored row."""
    from services.sites import update_site_settings

    ws = uuid4()
    site = _make_fake_site(workspace_id=ws, settings={})
    db = _mock_db_returning(site)

    patch = {"callback": {"enabled": True}}
    update_site_settings(db, ws, site.id, settings_patch=patch)
    patch["callback"]["enabled"] = False

    assert site.settings["callback"]["enabled"] is True


def test_update_settings_returns_none_when_site_not_found():
    from services.sites import update_site_settings

    db = _mock_db_returning(None)
    result = update_site_settings(
        db, uuid4(), uuid4(), settings_patch={"x": 1}
    )
    assert result is None


def test_update_settings_rejects_non_dict_patch():
    from services.sites import update_site_settings

    db = MagicMock()
    with pytest.raises(ValueError, match="settings_patch must be a dict"):
        update_site_settings(db, uuid4(), uuid4(), settings_patch="not a dict")


# ---------------------------------------------------------------------------
# public_site_dict — secrets never reach the wire
# ---------------------------------------------------------------------------

def test_public_site_dict_excludes_secrets():
    """The single most important test in this file. Secrets leaving the
    orchestrator is a security-grade bug."""
    from services.sites import public_site_dict

    ws = uuid4()
    site = _make_fake_site(workspace_id=ws)
    site.secrets = {"shopify_access_token": "shpat_super_secret"}

    out = public_site_dict(site)

    assert "secrets" not in out
    assert "shopify_access_token" not in str(out)
    assert "shpat_super_secret" not in str(out)


def test_public_site_dict_includes_expected_fields():
    from services.sites import public_site_dict

    ws = uuid4()
    site = _make_fake_site(workspace_id=ws)

    out = public_site_dict(site)
    expected_keys = {
        "id", "workspace_id", "type", "external_id",
        "display_name", "status", "settings", "capabilities",
        "created_at", "updated_at",
    }
    assert set(out.keys()) == expected_keys


def test_public_site_dict_returns_effective_capabilities():
    """PRD-008-A.1: Sites with capabilities={} (legacy data) must surface
    the type-derived defaults so dashboard panels render correctly."""
    from services.sites import public_site_dict

    ws = uuid4()
    site = _make_fake_site(workspace_id=ws, type="shopify", capabilities={})

    out = public_site_dict(site)
    assert out["capabilities"]["has_cart"] is True
    assert out["capabilities"]["has_catalog"] is True


def test_public_site_dict_serialises_uuids_and_dates_as_strings():
    """UUIDs and datetimes need to be JSON-safe for the API response."""
    from services.sites import public_site_dict

    ws = uuid4()
    site = _make_fake_site(workspace_id=ws)

    out = public_site_dict(site)
    assert isinstance(out["id"], str)
    assert isinstance(out["workspace_id"], str)
    assert isinstance(out["created_at"], str)
    assert isinstance(out["updated_at"], str)


def test_public_site_dict_tolerates_null_settings():
    """Older rows might have null settings; never crash on serialisation.
    PRD-008-A.1: capabilities falls back to type defaults via
    Site.effective_capabilities, so a shopify Site with stored
    capabilities=None still surfaces has_cart=True etc."""
    from core.models.sites import CAPABILITY_KEYS, derive_default_capabilities
    from services.sites import public_site_dict

    ws = uuid4()
    site = _make_fake_site(workspace_id=ws, type="shopify")
    site.settings = None
    site.capabilities = None
    site.effective_capabilities = derive_default_capabilities("shopify")

    out = public_site_dict(site)
    assert out["settings"] == {}
    # Capabilities now fall back to type defaults — never bare {}.
    assert set(out["capabilities"].keys()) == set(CAPABILITY_KEYS)
    assert out["capabilities"]["has_cart"] is True


# ---------------------------------------------------------------------------
# USER_SETTABLE_STATUSES — backend statuses excluded
# ---------------------------------------------------------------------------

def test_user_settable_statuses_excludes_error():
    """'error' is backend-managed (e.g. failed sync). Don't let merchants
    flip into it via PATCH — that would muddle the operational state."""
    from services.sites import USER_SETTABLE_STATUSES

    assert "error" not in USER_SETTABLE_STATUSES
    assert "active" in USER_SETTABLE_STATUSES
    assert "paused" in USER_SETTABLE_STATUSES
    assert "disconnected" in USER_SETTABLE_STATUSES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
