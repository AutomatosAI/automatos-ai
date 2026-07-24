"""
PRD-008-A Phase 3 — widget-config resolver unit tests
========================================================

Verifies that PRD-007 widget endpoints now read from the workspace's
default Site (PRD-008-A), with a workspace.settings fallback during the
migration transition window.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# Triggers ``load_dotenv`` so ``core.database.database`` can resolve
# Postgres creds at import time, regardless of test run order.
import config  # noqa: E402,F401


def _make_site(settings: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        workspace_id=uuid4(),
        type="shopify",
        external_id="test.myshopify.com",
        display_name="test",
        status="active",
        settings=settings or {},
        capabilities={},
        secrets=None,
        created_at=datetime(2026, 5, 14, 12, 0, 0),
        updated_at=datetime(2026, 5, 14, 12, 0, 0),
    )


def _make_workspace(settings: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        settings=settings or {},
    )


def _db_returning(site_result, workspace_result):
    """Build a db mock that returns ``site_result`` for the Site query
    and ``workspace_result`` for the Workspace query."""
    db = MagicMock()

    def query_dispatcher(model):
        chain = MagicMock()
        if "Site" in repr(model):
            chain.filter.return_value.order_by.return_value.first.return_value = site_result
        else:
            chain.filter.return_value.first.return_value = workspace_result
        return chain

    db.query.side_effect = query_dispatcher
    return db


# ---------------------------------------------------------------------------
# resolve_widget_settings_dict — Site preferred
# ---------------------------------------------------------------------------

def test_resolver_reads_from_site_when_one_exists():
    from api.widgets.config import resolve_widget_settings_dict

    site = _make_site(settings={"widget_proactive": {"enabled": True}})
    workspace = _make_workspace(settings={"widget_proactive": {"enabled": False}})
    db = _db_returning(site_result=site, workspace_result=workspace)

    result = resolve_widget_settings_dict(db, uuid4())
    assert result == {"widget_proactive": {"enabled": True}}


def test_resolver_falls_back_to_workspace_when_no_site():
    """Migration transition window: pre-migration workspaces still serve
    PRD-007 config correctly."""
    from api.widgets.config import resolve_widget_settings_dict

    workspace = _make_workspace(
        settings={"widget_proactive": {"enabled": True, "page_types": ["product"]}}
    )
    db = _db_returning(site_result=None, workspace_result=workspace)

    result = resolve_widget_settings_dict(db, uuid4())
    assert result == {"widget_proactive": {"enabled": True, "page_types": ["product"]}}


def test_resolver_falls_back_to_workspace_when_site_settings_empty():
    """An orphaned Site shouldn't mask a workspace that still has
    valid config — fall back if site.settings is empty."""
    from api.widgets.config import resolve_widget_settings_dict

    site = _make_site(settings={})
    workspace = _make_workspace(settings={"widget_proactive": {"enabled": True}})
    db = _db_returning(site_result=site, workspace_result=workspace)

    result = resolve_widget_settings_dict(db, uuid4())
    assert result == {"widget_proactive": {"enabled": True}}


def test_resolver_returns_none_when_both_missing():
    from api.widgets.config import resolve_widget_settings_dict

    db = _db_returning(site_result=None, workspace_result=None)
    assert resolve_widget_settings_dict(db, uuid4()) is None


# ---------------------------------------------------------------------------
# resolve_widget_config — public projection
# ---------------------------------------------------------------------------

def test_resolve_widget_config_strips_internal_keys():
    """Internal keys (shopify_access_token etc.) must never leak through
    the public widget config endpoint."""
    from api.widgets.config import resolve_widget_config

    site = _make_site(
        settings={
            "widget_proactive": {"enabled": True},
            "shopify_access_token": "shpat_should_NOT_leak",
            "shopify_domain": "test.myshopify.com",
        }
    )
    db = _db_returning(site_result=site, workspace_result=None)

    result = resolve_widget_config(db, uuid4())
    assert result == {"widget_proactive": {"enabled": True}}
    assert "shopify_access_token" not in str(result)
    assert "shopify_domain" not in str(result)


def test_resolve_widget_config_returns_none_for_empty_settings():
    """No public keys configured → None (so the SDK doesn't get an
    empty config object it didn't expect)."""
    from api.widgets.config import resolve_widget_config

    site = _make_site(settings={"shopify_access_token": "x"})  # only internal
    db = _db_returning(site_result=site, workspace_result=None)

    assert resolve_widget_config(db, uuid4()) is None


def test_resolve_widget_config_returns_none_when_no_source():
    from api.widgets.config import resolve_widget_config

    db = _db_returning(site_result=None, workspace_result=None)
    assert resolve_widget_config(db, uuid4()) is None


# ---------------------------------------------------------------------------
# Backward-compat: legacy build_widget_config still works on Workspace
# ---------------------------------------------------------------------------

def test_legacy_build_widget_config_still_works_on_workspace():
    """PRD-007 tests pass a Workspace; the function keeps that contract."""
    from api.widgets.config import build_widget_config

    workspace = _make_workspace(
        settings={"widget_proactive": {"enabled": True}, "internal_key": "x"}
    )
    result = build_widget_config(workspace)
    assert result == {"widget_proactive": {"enabled": True}}


def test_legacy_build_widget_config_returns_none_for_none_workspace():
    from api.widgets.config import build_widget_config

    assert build_widget_config(None) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
