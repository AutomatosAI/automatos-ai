"""PRD-141 US-003 — Shopify shim plugin behaviour.

This test file pins three things:

* Registration — importing ``integrations`` populates
  ``PLUGIN_REGISTRY["shopify"]`` with this module.
* Pass-through gating — the shim matches the existing chat.py
  ``is_proactive`` guard: anything other than
  ``trigger_reason in ("proactive_opener", "cart_idle")`` AND
  ``page_context is not None`` returns the message unchanged.
* Delegation contract — when the shim DOES rewrite, it calls the
  matching chat.py helper with the right arguments and returns the
  builder's output verbatim as ``message``. Combined with the fact
  that the chat.py builders themselves are unchanged in US-003, this
  is the transitive equivalence guarantee that US-011 will then pin
  byte-for-byte against captured INBUILD fixtures.

The delegation tests inject a **fake** ``api.widgets.chat`` module
into ``sys.modules`` via ``monkeypatch`` rather than loading the real
one. The real module pulls in the entire FastAPI / RAG / multimodal
dependency tree which is overkill for a unit test of a 60-line shim.
US-011 will exercise the real path against captured production
fixtures — that is where byte-equivalence is enforced.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from integrations import PLUGIN_REGISTRY, WidgetPluginResult
from integrations.shopify import widget_proactive


@pytest.fixture
def db():
    return MagicMock()


@pytest.fixture
def workspace_id():
    return uuid4()


@pytest.fixture
def fake_chat(monkeypatch):
    """Replace ``api.widgets.chat`` in ``sys.modules`` with a fake.

    The fake records every call so tests can assert the shim forwards
    ``workspace_id`` and ``page_context`` correctly, and returns a
    sentinel message so tests can assert the shim wires the builder's
    output into ``WidgetPluginResult.message`` unmodified.
    """
    fake = types.ModuleType("api.widgets.chat")

    calls: dict[str, list] = {
        "resolve_products": [],
        "resolve_cart": [],
        "build_product": [],
        "build_cart": [],
    }

    async def fake_resolve_products(workspace_id, page_context, **kwargs):
        calls["resolve_products"].append({
            "workspace_id": workspace_id,
            "page_context": page_context,
            "kwargs": kwargs,
        })
        return [{"label": "fake-related", "relation": "in_collection"}]

    async def fake_resolve_cart(workspace_id, page_context, **kwargs):
        calls["resolve_cart"].append({
            "workspace_id": workspace_id,
            "page_context": page_context,
            "kwargs": kwargs,
        })
        return [
            {"label": "fake-rec", "score": 5, "paired_with_count": 2, "total_orders": 10},
            {"label": "another", "score": 2, "paired_with_count": 1, "total_orders": 7},
        ]

    def fake_build_product(page_context, related_products=None):
        calls["build_product"].append({
            "page_context": dict(page_context),
            "related_products": related_products,
        })
        return "FAKE_PRODUCT_OPENER_MESSAGE"

    def fake_build_cart(page_context, recommendations=None):
        calls["build_cart"].append({
            "page_context": dict(page_context),
            "recommendations": recommendations,
        })
        return "FAKE_CART_IDLE_OPENER_MESSAGE"

    fake._resolve_graph_related_products = fake_resolve_products
    fake._resolve_cart_recommendations = fake_resolve_cart
    fake._build_proactive_opener_message = fake_build_product
    fake._build_cart_idle_opener_message = fake_build_cart

    monkeypatch.setitem(sys.modules, "api.widgets.chat", fake)
    return calls


# ---- Registration ------------------------------------------------------------


def test_shopify_plugin_is_registered():
    assert "shopify" in PLUGIN_REGISTRY
    assert PLUGIN_REGISTRY["shopify"] is widget_proactive


# ---- Pass-through cases (no chat.py touch) -----------------------------------


@pytest.mark.asyncio
async def test_no_trigger_no_context_passes_through(db, workspace_id):
    result = await widget_proactive.handle_widget_message(
        message="hello",
        page_context=None,
        trigger_reason=None,
        workspace_id=workspace_id,
        db=db,
    )
    assert isinstance(result, WidgetPluginResult)
    assert result.message == "hello"
    assert result.context_note is None
    assert result.telemetry == {}


@pytest.mark.asyncio
async def test_no_trigger_with_context_passes_through(db, workspace_id):
    # Mid-conversation message on a Shopify workspace: the Shopify
    # plugin must NOT prepend an opaque "(Context: ...)" JSON block.
    # That behaviour is owned by the generic plugin only — Shopify's
    # context-handling lives in the skill prompt + proactive directive.
    result = await widget_proactive.handle_widget_message(
        message="Tell me about Hochiki detectors",
        page_context={"productHandle": "hochiki-aln", "productTitle": "Hochiki ALN"},
        trigger_reason=None,
        workspace_id=workspace_id,
        db=db,
    )
    assert result.message == "Tell me about Hochiki detectors"
    assert result.context_note is None


@pytest.mark.asyncio
async def test_unknown_trigger_passes_through(db, workspace_id):
    result = await widget_proactive.handle_widget_message(
        message="agent will see this verbatim",
        page_context={"pageType": "product"},
        trigger_reason="not_a_real_trigger",
        workspace_id=workspace_id,
        db=db,
    )
    assert result.message == "agent will see this verbatim"
    assert result.context_note is None


@pytest.mark.asyncio
async def test_proactive_trigger_with_none_context_passes_through(db, workspace_id):
    # Mirrors chat.py's ``is_proactive`` guard: trigger_reason in the
    # set AND page_context is not None. None context skips entirely.
    result = await widget_proactive.handle_widget_message(
        message="placeholder",
        page_context=None,
        trigger_reason="proactive_opener",
        workspace_id=workspace_id,
        db=db,
    )
    assert result.message == "placeholder"
    assert result.context_note is None


@pytest.mark.asyncio
async def test_cart_idle_trigger_with_none_context_passes_through(db, workspace_id):
    result = await widget_proactive.handle_widget_message(
        message="placeholder",
        page_context=None,
        trigger_reason="cart_idle",
        workspace_id=workspace_id,
        db=db,
    )
    assert result.message == "placeholder"
    assert result.context_note is None


# ---- Delegation contract (chat.py faked) -------------------------------------


@pytest.mark.asyncio
async def test_proactive_opener_delegates_to_chat_helpers(fake_chat, db, workspace_id):
    page_context = {"pageType": "product", "productHandle": "hochiki-aln"}
    result = await widget_proactive.handle_widget_message(
        message="(synthesized by SDK — replaced)",
        page_context=page_context,
        trigger_reason="proactive_opener",
        workspace_id=workspace_id,
        db=db,
    )

    # 1. Resolver called with str(workspace_id) and the page context.
    assert len(fake_chat["resolve_products"]) == 1
    rp = fake_chat["resolve_products"][0]
    assert rp["workspace_id"] == str(workspace_id)
    assert rp["page_context"] == page_context

    # 2. Builder called with page context + resolver's return value.
    assert len(fake_chat["build_product"]) == 1
    bp = fake_chat["build_product"][0]
    assert bp["page_context"] == page_context
    assert bp["related_products"] == [
        {"label": "fake-related", "relation": "in_collection"},
    ]

    # 3. Cart-idle helpers were NOT touched on the product path.
    assert fake_chat["resolve_cart"] == []
    assert fake_chat["build_cart"] == []

    # 4. Result wires the builder output verbatim into message.
    assert result.message == "FAKE_PRODUCT_OPENER_MESSAGE"
    assert result.context_note == "shopify shim: proactive_opener rewrite"
    assert result.telemetry == {
        "trigger_reason": "proactive_opener",
        "related_count": 1,
    }


@pytest.mark.asyncio
async def test_cart_idle_delegates_to_chat_helpers(fake_chat, db, workspace_id):
    page_context = {
        "cartItemCount": 3,
        "cartTotalPrice": 12000,
        "shopCurrency": "GBP",
    }
    result = await widget_proactive.handle_widget_message(
        message="(synthesized by SDK — replaced)",
        page_context=page_context,
        trigger_reason="cart_idle",
        workspace_id=workspace_id,
        db=db,
    )

    assert len(fake_chat["resolve_cart"]) == 1
    rc = fake_chat["resolve_cart"][0]
    assert rc["workspace_id"] == str(workspace_id)
    assert rc["page_context"] == page_context

    assert len(fake_chat["build_cart"]) == 1
    bc = fake_chat["build_cart"][0]
    assert bc["page_context"] == page_context
    assert bc["recommendations"] == [
        {"label": "fake-rec", "score": 5, "paired_with_count": 2, "total_orders": 10},
        {"label": "another", "score": 2, "paired_with_count": 1, "total_orders": 7},
    ]

    # Product-path helpers untouched on the cart path.
    assert fake_chat["resolve_products"] == []
    assert fake_chat["build_product"] == []

    assert result.message == "FAKE_CART_IDLE_OPENER_MESSAGE"
    assert result.context_note == "shopify shim: cart_idle rewrite"
    assert result.telemetry == {
        "trigger_reason": "cart_idle",
        "related_count": 2,
    }
