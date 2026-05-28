"""
PRD-007 unit tests
==================

Pure-Python unit tests covering:

1. ``DEFAULT_WIDGET_PROACTIVE_CONFIG`` shape (workspace seeder default).
2. ``build_widget_config`` projection from ``workspace.settings``.
3. ``_build_proactive_opener_message`` synthesis from page context.
4. ``WidgetChatRequest`` accepts new optional fields and stays
   backwards-compatible.
5. ``SessionTokenResponse`` accepts the new ``widget_config`` field.

No FastAPI app boot, no DB, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make ``orchestrator/`` imports resolve when tests run from repo root.
ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))


# ---------------------------------------------------------------------------
# 1. Default workspace-seeder config (Shopify provision step)
# ---------------------------------------------------------------------------

def test_default_widget_proactive_shape():
    from api.shopify import DEFAULT_WIDGET_PROACTIVE_CONFIG as cfg

    # Locked-defaults contract per PRD-007 v0.2 — keep the schema explicit
    # so accidental drift breaks this test, not production widgets.
    assert cfg["enabled"] is False, "must default to OFF — opt-in"
    assert cfg["page_types"] == ["product"]
    assert cfg["triggers"] == [{"type": "time_on_page", "seconds": 20}]
    assert cfg["frequency_cap"] == {"scope": "session", "max_pops": 1}
    assert cfg["greeting_source"] == "agent_with_canned_fallback"
    assert cfg["canned_fallback"] == "Need a hand finding the right product?"
    assert cfg["agent_timeout_ms"] == 1500
    assert cfg["popup_style"] == "corner_bubble"
    assert cfg["respect_consent"] is True
    assert cfg["dismissal_persistence"] == "session"


def test_default_config_is_copyable():
    """Each new workspace must get its own dict — not the shared module-level
    object — so per-merchant edits don't bleed across tenants.
    """
    from api.shopify import DEFAULT_WIDGET_PROACTIVE_CONFIG

    a = dict(DEFAULT_WIDGET_PROACTIVE_CONFIG)
    b = dict(DEFAULT_WIDGET_PROACTIVE_CONFIG)
    a["enabled"] = True
    assert b["enabled"] is False
    assert DEFAULT_WIDGET_PROACTIVE_CONFIG["enabled"] is False


# ---------------------------------------------------------------------------
# 2. ``build_widget_config`` — public projection from workspace.settings
# ---------------------------------------------------------------------------

def test_build_widget_config_returns_none_for_missing_workspace():
    from api.widgets.config import build_widget_config

    assert build_widget_config(None) is None


def test_build_widget_config_returns_none_when_no_settings():
    from api.widgets.config import build_widget_config

    ws = SimpleNamespace(settings=None)
    assert build_widget_config(ws) is None

    ws_empty = SimpleNamespace(settings={})
    assert build_widget_config(ws_empty) is None


def test_build_widget_config_projects_only_public_keys():
    """Internal keys (e.g. shopify_access_token) must never reach the browser."""
    from api.widgets.config import build_widget_config

    ws = SimpleNamespace(
        settings={
            "shopify_access_token": "shpat_secret_should_NOT_leak",
            "shopify_domain": "example.myshopify.com",
            "widget_proactive": {"enabled": True, "page_types": ["product"]},
        }
    )
    result = build_widget_config(ws)
    assert result == {"widget_proactive": {"enabled": True, "page_types": ["product"]}}
    assert "shopify_access_token" not in result
    assert "shopify_domain" not in result


def test_build_widget_config_returns_none_when_no_public_keys_set():
    from api.widgets.config import build_widget_config

    ws = SimpleNamespace(settings={"shopify_domain": "x.myshopify.com"})
    assert build_widget_config(ws) is None


# ---------------------------------------------------------------------------
# 3. ``_build_proactive_opener_message`` — page-context → directive
# ---------------------------------------------------------------------------

def test_opener_message_includes_product_title_when_present():
    from integrations.shopify.widget_proactive import _build_proactive_opener_message

    msg = _build_proactive_opener_message({
        "pageType": "product",
        "productTitle": "EN 12101-9 Control Panel",
        "productType": "Control Panels",
    })
    assert msg.startswith("[PROACTIVE_OPENER]")
    assert "EN 12101-9 Control Panel" in msg
    assert "Control Panels" in msg
    assert "page_type=product" in msg


def test_opener_message_falls_back_to_handle_without_title():
    from integrations.shopify.widget_proactive import _build_proactive_opener_message

    msg = _build_proactive_opener_message({
        "pageType": "product",
        "productHandle": "en-12101-control-panel",
    })
    assert "product_handle=en-12101-control-panel" in msg


def test_opener_message_handles_collection_pages():
    from integrations.shopify.widget_proactive import _build_proactive_opener_message

    msg = _build_proactive_opener_message({
        "pageType": "collection",
        "collectionTitle": "Fans & Ventilation",
    })
    assert "page_type=collection" in msg
    assert "Fans & Ventilation" in msg


def test_opener_message_includes_full_grounding_context():
    """PRD-007 v0.4: rich page context grounds the agent so it stops
    inventing facts. Every populated field should reach the directive."""
    from integrations.shopify.widget_proactive import _build_proactive_opener_message

    msg = _build_proactive_opener_message({
        "pageType": "product",
        "productTitle": "Actulux SVM 4 amp Micro 24V Basic",
        "productType": "Smoke Control",
        "productVendor": "Actulux",
        "productPrice": "362.18",
        "productAvailable": True,
        "productHandle": "actulux-svm-basic",
        "shopDomain": "inbuilduk.com",
        "shopCurrency": "GBP",
        "cartItemCount": 0,
    })
    # All the rich facts the agent would otherwise invent
    assert "Actulux SVM 4 amp Micro 24V Basic" in msg
    assert "Smoke Control" in msg
    assert "Actulux" in msg
    assert "362.18" in msg
    assert "in_stock=True" in msg
    assert "GBP" in msg
    # Anti-fabrication directive
    assert "do NOT invent" in msg


def test_opener_message_skips_empty_zero_false_fields():
    """Don't pollute the directive with empty / zero / false signals."""
    from integrations.shopify.widget_proactive import _build_proactive_opener_message

    msg = _build_proactive_opener_message({
        "pageType": "product",
        "productTitle": "Widget",
        "productAvailable": False,        # should NOT appear
        "cartItemCount": 0,                # should NOT appear
        "productVendor": "",               # should NOT appear
        "customerId": None,                # should NOT appear
    })
    assert "in_stock" not in msg
    assert "cart_item_count" not in msg
    assert "vendor" not in msg
    assert "logged_in_customer_id" not in msg


def test_opener_message_quotes_values_with_spaces():
    """Multi-word values (like product titles) need quoting so the agent
    parses them as single tokens."""
    from integrations.shopify.widget_proactive import _build_proactive_opener_message

    msg = _build_proactive_opener_message({
        "pageType": "product",
        "productTitle": "EN 12101-9 Control Panel",
    })
    assert 'product="EN 12101-9 Control Panel"' in msg


def test_opener_message_handles_empty_context():
    from integrations.shopify.widget_proactive import _build_proactive_opener_message

    msg = _build_proactive_opener_message({})
    assert msg.startswith("[PROACTIVE_OPENER]")
    assert "no context" in msg


# ---------------------------------------------------------------------------
# 4. ``WidgetChatRequest`` — backwards-compatible new fields
# ---------------------------------------------------------------------------

def test_chat_request_accepts_legacy_payload():
    """Existing clients sending only ``message`` must keep working."""
    from api.widgets.chat import WidgetChatRequest

    body = WidgetChatRequest(message="hello")
    assert body.message == "hello"
    assert body.page_context is None
    assert body.trigger_reason is None


def test_chat_request_accepts_proactive_payload():
    from api.widgets.chat import WidgetChatRequest

    body = WidgetChatRequest(
        message="",  # widget sends empty for proactive
        trigger_reason="proactive_opener",
        page_context={"pageType": "product", "productHandle": "x"},
    )
    assert body.trigger_reason == "proactive_opener"
    assert body.page_context == {"pageType": "product", "productHandle": "x"}


def test_proactive_trigger_reason_constant_includes_proactive_opener():
    from api.widgets.chat import PROACTIVE_TRIGGER_REASONS

    assert "proactive_opener" in PROACTIVE_TRIGGER_REASONS


def test_proactive_trigger_reason_constant_includes_cart_idle():
    """PRD-008-B Feature C2 — cart-idle is a proactive trigger and must
    flow through the same opener-rewrite branch as the product-page one."""
    from api.widgets.chat import PROACTIVE_TRIGGER_REASONS

    assert "cart_idle" in PROACTIVE_TRIGGER_REASONS


# ---------------------------------------------------------------------------
# 4b. Cart-idle opener (PRD-008-B Feature C2)
# ---------------------------------------------------------------------------

def test_cart_idle_message_includes_cart_summary():
    from integrations.shopify.widget_proactive import _build_cart_idle_opener_message

    msg = _build_cart_idle_opener_message(
        {"cartItemCount": 3, "cartTotalPrice": 14299, "shopCurrency": "GBP"},
        recommendations=None,
    )
    assert "[CART_IDLE]" in msg
    assert "cart_item_count=3" in msg
    assert "142.99" in msg  # 14299 minor units → 142.99
    assert "GBP" in msg


def test_cart_idle_message_renders_recommendations_with_provenance():
    from integrations.shopify.widget_proactive import _build_cart_idle_opener_message

    recs = [
        {"label": "Elta 4A Actuator", "paired_with_count": 2, "score": 24, "total_orders": 87},
        {"label": "Cable Kit 5m", "paired_with_count": 1, "score": 8, "total_orders": 87},
    ]
    msg = _build_cart_idle_opener_message(
        {"cartItemCount": 2, "shopCurrency": "GBP"},
        recommendations=recs,
    )
    # First rec paired with multiple cart items uses cross-cutting phrasing
    assert 'Elta 4A Actuator' in msg
    assert 'bought with 2 of the items' in msg
    # Second rec paired with only 1 cart item uses raw co_count/total_orders
    assert 'Cable Kit 5m' in msg
    assert '8 of 87 orders' in msg


def test_cart_idle_message_falls_back_when_no_recommendations():
    from integrations.shopify.widget_proactive import _build_cart_idle_opener_message

    msg = _build_cart_idle_opener_message({"cartItemCount": 1}, recommendations=[])
    # Directive still tells the agent to nudge toward checkout without
    # fabricating products.
    assert "[CART_IDLE]" in msg
    assert "do NOT invent" in msg.lower() or "do not invent" in msg.lower()


def test_cart_idle_message_skips_zero_cart_total():
    from integrations.shopify.widget_proactive import _build_cart_idle_opener_message

    msg = _build_cart_idle_opener_message({"cartItemCount": 1}, recommendations=None)
    assert "cart_total" not in msg


# ---------------------------------------------------------------------------
# 5. ``SessionTokenResponse`` — new ``widget_config`` field
# ---------------------------------------------------------------------------

def test_session_response_widget_config_optional():
    from api.widgets.session import SessionTokenResponse

    resp = SessionTokenResponse(
        session_token="jwt",
        expires_at="2026-05-13T00:00:00",
        permissions=["chat"],
        workspace_id="00000000-0000-0000-0000-000000000000",
    )
    assert resp.widget_config is None


def test_session_response_widget_config_roundtrip():
    from api.widgets.session import SessionTokenResponse

    cfg = {"widget_proactive": {"enabled": True}}
    resp = SessionTokenResponse(
        session_token="jwt",
        expires_at="2026-05-13T00:00:00",
        permissions=["chat"],
        workspace_id="00000000-0000-0000-0000-000000000000",
        widget_config=cfg,
    )
    assert resp.widget_config == cfg


# ---------------------------------------------------------------------------
# 6. Whitelist guard — adding a new public key requires a deliberate change
# ---------------------------------------------------------------------------

def test_public_widget_config_keys_whitelist_is_minimal():
    """Trip-wire: if you add a public key, update this test consciously.

    Keeps the surface area visible during code review so nobody silently
    leaks an internal setting.
    """
    from api.widgets.config import PUBLIC_WIDGET_CONFIG_KEYS

    assert PUBLIC_WIDGET_CONFIG_KEYS == ("widget_proactive",)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
