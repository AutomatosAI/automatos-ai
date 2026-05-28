"""PRD-141 — Shopify plugin behaviour.

This test file pins three things:

* Registration — importing ``integrations`` populates
  ``PLUGIN_REGISTRY["shopify"]`` with this module.
* Pass-through gating — the plugin matches chat.py's old
  ``is_proactive`` guard: anything other than
  ``trigger_reason in ("proactive_opener", "cart_idle")`` AND
  ``page_context is not None`` returns the message unchanged.
* Wiring contract — when the plugin DOES rewrite, it calls the
  matching resolver + builder with the right arguments and returns the
  builder's output verbatim as ``message``. All four helpers live in
  :mod:`integrations.shopify.widget_proactive`, so the wiring contract
  is a pure intra-module check; US-011's snapshot tests pin the actual
  builder output byte-for-byte against the US-004 INBUILD fixtures.

The wiring tests monkey-patch the four helpers directly on
:mod:`widget_proactive` rather than running the real graph/builder
code. The real resolvers traverse the Shopify graph and the real
builders close over the context-field mapping — overkill for a unit
test of the dispatch contract. US-011 exercises the real path against
the deterministic fixture graph; that is where byte-equivalence is
enforced.
"""

from __future__ import annotations

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
    """Replace the plugin's downstream helpers with sentinels.

    All four helpers live on :mod:`widget_proactive` itself:

    * ``_resolve_graph_related_products`` (local since US-006)
    * ``_resolve_cart_recommendations`` (local since US-007)
    * ``_build_proactive_opener_message`` (local since US-008)
    * ``_build_cart_idle_opener_message`` (local since US-008)

    Each is patched in place via ``monkeypatch.setattr``. The fake
    records every call so tests can assert the plugin forwards
    ``workspace_id`` and ``page_context`` correctly, and returns a
    sentinel message so tests can assert the plugin wires the builder's
    output into ``WidgetPluginResult.message`` unmodified.

    Fixture name kept as ``fake_chat`` for git-history continuity
    through the Phase 1 lift; the historical "chat" reference now means
    "the proactive helpers that used to live in chat.py".
    """
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

    monkeypatch.setattr(
        widget_proactive,
        "_resolve_graph_related_products",
        fake_resolve_products,
    )
    monkeypatch.setattr(
        widget_proactive,
        "_resolve_cart_recommendations",
        fake_resolve_cart,
    )
    monkeypatch.setattr(
        widget_proactive,
        "_build_proactive_opener_message",
        fake_build_product,
    )
    monkeypatch.setattr(
        widget_proactive,
        "_build_cart_idle_opener_message",
        fake_build_cart,
    )
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
    assert result.context_note == "shopify: proactive_opener rewrite"
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
    assert result.context_note == "shopify: cart_idle rewrite"
    assert result.telemetry == {
        "trigger_reason": "cart_idle",
        "related_count": 2,
    }


# ---- US-011 snapshot equivalence (PRD-007 + PRD-008-B byte-equality) --------
#
# These tests are the byte-equality safety net that gates every Phase 1
# lift (US-005/006/007/008/010). After US-008 all four proactive helpers
# live in :mod:`integrations.shopify.widget_proactive` and the snapshot
# tests call ``handle_widget_message`` directly. The
# ``real_chat_with_graph`` fixture in ``conftest.py`` provides only one
# piece of indirection — a fixture-bound ``GraphifyService`` stub that
# returns the US-004 synthetic INBUILD graph — so the test exercises
# the exact production code path while remaining deterministic.
#
# These tests must KEEP PASSING through every remaining lift (US-010 in
# particular rewires chat.py to dispatch via the registry — same plugin
# entry point, same expected output). If one fails, the lift broke
# equivalence — fix the lift, NOT the golden fixture (per US-011 notes).


@pytest.mark.asyncio
async def test_product_page_opener_byte_equality(
    real_chat_with_graph,
    product_page_context,
    expected_product_page_opener,
    db,
    workspace_id,
):
    """PRD-007 product-page opener — byte-equal to the US-004 fixture.

    Exercises the proactive_opener path end-to-end: plugin gates on
    (trigger_reason, page_context), calls ``_resolve_graph_related_products``
    against the fixture graph, calls ``_build_proactive_opener_message``,
    returns the directive as ``result.message``.

    The fixture graph encodes the by_vendor-overrides-FBT quirk
    (NetworkX undirected storage) documented in ``fixtures/README.md`` —
    that's WHY the expected opener mentions "Hochiki Banshee Wall Sounder"
    as a same-vendor sibling rather than as an FBT pair.
    """
    result = await widget_proactive.handle_widget_message(
        message="(placeholder synthesized by SDK)",
        page_context=product_page_context,
        trigger_reason="proactive_opener",
        workspace_id=workspace_id,
        db=db,
    )

    assert result.message == expected_product_page_opener
    assert result.context_note == "shopify: proactive_opener rewrite"


@pytest.mark.asyncio
async def test_cart_idle_opener_byte_equality(
    real_chat_with_graph,
    cart_idle_context,
    expected_cart_idle_opener,
    db,
    workspace_id,
):
    """PRD-008-B cart-idle opener — byte-equal to the US-004 fixture.

    Exercises the cart_idle path end-to-end: multi-seed FBT walk across
    every cart line item, aggregation by (paired_with_count, score),
    top-3 recommendations rendered into the cart-idle directive.

    The fixture's three cart items (hochiki-aln/-acb/-atg) produce a
    deterministic top-3 of (base-ybn, mxpro5, banshee). Banshee is in
    the cart-idle output via the elif branch (added together in 15 of
    31 orders) because it pairs with only ONE cart item — the aln-banshee
    FBT edge was overwritten by by_vendor in the synthetic graph (a
    quirk of NetworkX undirected storage; see fixtures/README.md).
    """
    result = await widget_proactive.handle_widget_message(
        message="(placeholder synthesized by SDK)",
        page_context=cart_idle_context,
        trigger_reason="cart_idle",
        workspace_id=workspace_id,
        db=db,
    )

    assert result.message == expected_cart_idle_opener
    assert result.context_note == "shopify: cart_idle rewrite"


@pytest.mark.parametrize(
    "trigger",
    [None, "proactive_opener", "cart_idle", "unknown_trigger"],
    ids=["no_trigger", "proactive_opener", "cart_idle", "unknown"],
)
@pytest.mark.asyncio
async def test_no_context_no_rewrite(trigger, db, workspace_id):
    """US-011 AC test 4: page_context=None ⇒ message returned unchanged.

    Parametrised across every trigger value (including ``None`` and an
    unknown trigger) because the plugin's gate is symmetric — either side
    short-circuits the rewrite. This is a regression guard for the
    chat.py ``is_proactive`` semantics now owned by the plugin.
    """
    result = await widget_proactive.handle_widget_message(
        message="user message verbatim",
        page_context=None,
        trigger_reason=trigger,
        workspace_id=workspace_id,
        db=db,
    )

    assert result.message == "user message verbatim"
    assert result.context_note is None
    assert result.telemetry == {}
