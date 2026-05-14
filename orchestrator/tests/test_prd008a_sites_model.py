"""
PRD-008-A Sites model — unit tests
====================================

Pure-Python unit tests for the new ``Site`` ORM model. No DB roundtrip.

Covers:
  1. Site can be instantiated with the required fields.
  2. Default ``status`` is ``"active"``.
  3. Default ``settings`` and ``capabilities`` are empty dicts (not shared).
  4. ``derive_default_capabilities`` returns the expected shape per Site type.
  5. The model exposes the expected SQLAlchemy table name + indexed columns.
  6. The model's ``__repr__`` is informative (debug-friendly).
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------

def test_site_can_be_instantiated_with_required_fields():
    from core.models.sites import Site

    workspace_id = uuid4()
    site = Site(
        workspace_id=workspace_id,
        type="shopify",
        external_id="besafe-ltd.myshopify.com",
        display_name="besafe-ltd.myshopify.com",
    )
    assert site.workspace_id == workspace_id
    assert site.type == "shopify"
    assert site.external_id == "besafe-ltd.myshopify.com"
    assert site.display_name == "besafe-ltd.myshopify.com"


def test_site_supports_all_documented_types():
    """PRD-008-A allow-list: shopify, wix, woocommerce, custom."""
    from core.models.sites import SITE_TYPES

    assert "shopify" in SITE_TYPES
    assert "wix" in SITE_TYPES
    assert "woocommerce" in SITE_TYPES
    assert "custom" in SITE_TYPES


# ---------------------------------------------------------------------------
# 2/3. Defaults — never share mutable state across rows
# ---------------------------------------------------------------------------

def test_default_settings_is_an_empty_dict():
    from core.models.sites import Site

    # Each new Site must get an independent empty dict — not a shared
    # module-level one. Both the Python-side default (for unflushed
    # instances) and the server-side default (for direct SQL inserts)
    # must produce ``{}``.
    default = Site.__table__.c.settings.default
    assert default is not None and callable(default.arg)
    assert Site.__table__.c.settings.server_default.arg == "{}"


def test_default_capabilities_is_an_empty_dict():
    from core.models.sites import Site

    default = Site.__table__.c.capabilities.default
    assert default is not None and callable(default.arg)
    assert Site.__table__.c.capabilities.server_default.arg == "{}"


# ---------------------------------------------------------------------------
# 4. derive_default_capabilities — capability gating relies on this
# ---------------------------------------------------------------------------

def test_derive_default_capabilities_shopify():
    from core.models.sites import derive_default_capabilities

    caps = derive_default_capabilities("shopify")
    assert caps["has_cart"] is True
    assert caps["has_catalog"] is True
    assert caps["has_customer_records"] is True
    assert caps["has_working_hours_source"] is True
    assert caps["supports_theme_override"] is True
    # Volume discounts depend on the merchant having configured them, so this
    # is False by default — the connector flips it when scopes confirm.
    assert caps["has_volume_discounts"] is False


def test_derive_default_capabilities_custom_embed():
    """A pure <script> embed has no platform integration — UI components
    that depend on cart/catalog must hide themselves for this Site type."""
    from core.models.sites import derive_default_capabilities

    caps = derive_default_capabilities("custom")
    assert caps["has_cart"] is False
    assert caps["has_catalog"] is False
    assert caps["has_volume_discounts"] is False
    assert caps["has_customer_records"] is False
    assert caps["has_working_hours_source"] is False
    assert caps["supports_theme_override"] is False


def test_derive_default_capabilities_wix_stub():
    """Wix is in the type allow-list but not yet wired — capabilities
    default to off until the Wix adapter ships."""
    from core.models.sites import derive_default_capabilities

    caps = derive_default_capabilities("wix")
    assert caps["has_cart"] is False
    assert caps["supports_theme_override"] is False


def test_derive_default_capabilities_unknown_type_raises():
    from core.models.sites import derive_default_capabilities

    with pytest.raises(ValueError, match="unknown site type"):
        derive_default_capabilities("magento")


def test_capability_keys_are_stable():
    """Adding a new capability key requires a deliberate change here. Trip-wire
    so frontend components that branch on capabilities don't silently break."""
    from core.models.sites import CAPABILITY_KEYS

    assert CAPABILITY_KEYS == (
        "has_cart",
        "has_catalog",
        "has_volume_discounts",
        "has_customer_records",
        "has_working_hours_source",
        "supports_theme_override",
    )


# ---------------------------------------------------------------------------
# 5. Table metadata
# ---------------------------------------------------------------------------

def test_table_name_is_sites():
    from core.models.sites import Site

    assert Site.__tablename__ == "sites"


def test_workspace_id_is_indexed():
    from core.models.sites import Site

    indexed_cols = {idx.columns.keys()[0] for idx in Site.__table__.indexes}
    assert "workspace_id" in indexed_cols


def test_status_defaults_to_active():
    from core.models.sites import Site

    assert Site.__table__.c.status.server_default.arg == "active"


# ---------------------------------------------------------------------------
# 6. Debug-friendly repr
# ---------------------------------------------------------------------------

def test_repr_includes_type_and_external_id():
    from core.models.sites import Site

    site = Site(
        id=uuid4(),
        workspace_id=uuid4(),
        type="shopify",
        external_id="besafe-ltd.myshopify.com",
        display_name="besafe-ltd",
    )
    r = repr(site)
    assert "shopify" in r
    assert "besafe-ltd.myshopify.com" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
