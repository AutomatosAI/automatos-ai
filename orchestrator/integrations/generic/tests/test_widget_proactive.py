"""PRD-141 US-002 — generic widget plugin behaviour.

Covers the three documented paths:

* empty / missing ``page_context`` → no rewrite
* populated ``page_context`` + no ``trigger_reason`` → JSON prefix prepended
* any ``trigger_reason`` set → no rewrite (proactive directives are
  vertical-specific; generic stays opaque)

Also pins down the registration side effect — importing
``integrations`` must populate ``PLUGIN_REGISTRY["generic"]``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from integrations import PLUGIN_REGISTRY, WidgetPluginResult
from integrations.generic import widget_proactive


@pytest.fixture
def db():
    return MagicMock()


@pytest.fixture
def workspace_id():
    return uuid4()


# ---- Registration ------------------------------------------------------------


def test_generic_plugin_is_registered():
    assert "generic" in PLUGIN_REGISTRY
    assert PLUGIN_REGISTRY["generic"] is widget_proactive


# ---- Behaviour ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_context_returns_message_unchanged(db, workspace_id):
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


@pytest.mark.asyncio
async def test_empty_dict_context_returns_message_unchanged(db, workspace_id):
    result = await widget_proactive.handle_widget_message(
        message="hello",
        page_context={},
        trigger_reason=None,
        workspace_id=workspace_id,
        db=db,
    )
    assert result.message == "hello"
    assert result.context_note is None


@pytest.mark.asyncio
async def test_populated_context_prepends_json_prefix(db, workspace_id):
    context = {"pageType": "home", "locale": "en-GB"}
    result = await widget_proactive.handle_widget_message(
        message="Tell me about your services",
        page_context=context,
        trigger_reason=None,
        workspace_id=workspace_id,
        db=db,
    )
    expected_prefix = f"(Context: {json.dumps(context, sort_keys=True)})\n\n"
    assert result.message == f"{expected_prefix}Tell me about your services"
    assert result.context_note == "generic: prepended opaque page_context"


@pytest.mark.asyncio
async def test_json_prefix_uses_sort_keys_for_determinism(db, workspace_id):
    # Same data, different insertion orders — output must be identical so
    # snapshot tests downstream don't flake on dict iteration order.
    ordered_one = {"b": 2, "a": 1, "c": 3}
    ordered_two = {"a": 1, "c": 3, "b": 2}
    result_one = await widget_proactive.handle_widget_message(
        message="hi",
        page_context=ordered_one,
        trigger_reason=None,
        workspace_id=workspace_id,
        db=db,
    )
    result_two = await widget_proactive.handle_widget_message(
        message="hi",
        page_context=ordered_two,
        trigger_reason=None,
        workspace_id=workspace_id,
        db=db,
    )
    assert result_one.message == result_two.message
    assert '"a": 1, "b": 2, "c": 3' in result_one.message


@pytest.mark.asyncio
async def test_proactive_trigger_returns_message_unchanged(db, workspace_id):
    # Even with a populated context, a proactive trigger must NOT cause
    # the generic plugin to invent an opener — that's vertical-specific.
    result = await widget_proactive.handle_widget_message(
        message="agent will see this verbatim",
        page_context={"pageType": "product", "productHandle": "xyz"},
        trigger_reason="proactive_opener",
        workspace_id=workspace_id,
        db=db,
    )
    assert result.message == "agent will see this verbatim"
    assert result.context_note is None


@pytest.mark.asyncio
async def test_cart_idle_trigger_also_passes_through(db, workspace_id):
    result = await widget_proactive.handle_widget_message(
        message="placeholder",
        page_context={"cartItems": []},
        trigger_reason="cart_idle",
        workspace_id=workspace_id,
        db=db,
    )
    assert result.message == "placeholder"
