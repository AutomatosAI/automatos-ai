"""PRD-141 US-001 — Plugin registry contract.

Asserts the shape of `orchestrator.integrations` (PLUGIN_REGISTRY, the
`WidgetPlugin` protocol, the `WidgetPluginResult` dataclass) and — once
later Phase 0 stories land — that `"generic"` and `"shopify"` keys are
present.

The two registration checks are intentionally `skip`-tolerant so this
file lands cleanly in US-001 before US-002/US-003 wire their plugins in.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from typing import get_type_hints

import pytest

import integrations
from integrations import (
    PLUGIN_REGISTRY,
    WidgetPlugin,
    WidgetPluginResult,
)


# ---- Module surface ----------------------------------------------------------


def test_plugin_registry_is_a_dict():
    assert isinstance(PLUGIN_REGISTRY, dict)


def test_module_exports_expected_names():
    assert hasattr(integrations, "PLUGIN_REGISTRY")
    assert hasattr(integrations, "WidgetPlugin")
    assert hasattr(integrations, "WidgetPluginResult")


# ---- WidgetPluginResult dataclass --------------------------------------------


def test_widget_plugin_result_is_a_dataclass():
    assert is_dataclass(WidgetPluginResult)


def test_widget_plugin_result_field_names():
    names = {f.name for f in fields(WidgetPluginResult)}
    assert names == {"message", "context_note", "system_preamble", "telemetry"}


def test_widget_plugin_result_defaults():
    result = WidgetPluginResult(message="hello")
    assert result.message == "hello"
    assert result.context_note is None
    assert result.system_preamble is None
    assert result.telemetry == {}


def test_widget_plugin_result_accepts_full_construction():
    result = WidgetPluginResult(
        message="rewritten",
        context_note="applied shopify rewrite",
        system_preamble="[PAGE_CONTEXT] Currently viewing: product=Widget.",
        telemetry={"related": 3},
    )
    assert result.context_note == "applied shopify rewrite"
    assert result.system_preamble == "[PAGE_CONTEXT] Currently viewing: product=Widget."
    assert result.telemetry == {"related": 3}


# ---- WidgetPlugin protocol ---------------------------------------------------


def test_widget_plugin_is_a_protocol():
    # Protocols are still classes; the marker we rely on is the presence of
    # the `handle_widget_message` attribute and that it is callable.
    assert hasattr(WidgetPlugin, "handle_widget_message")


def test_widget_plugin_handle_widget_message_signature():
    sig = inspect.signature(WidgetPlugin.handle_widget_message)
    params = sig.parameters
    # `self` plus the five documented keyword-only args.
    assert list(params.keys()) == [
        "self",
        "message",
        "page_context",
        "trigger_reason",
        "workspace_id",
        "db",
    ]
    for name in ("message", "page_context", "trigger_reason", "workspace_id", "db"):
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only"
        )


def test_widget_plugin_handle_widget_message_annotations():
    hints = get_type_hints(WidgetPlugin.handle_widget_message)
    assert hints["message"] is str
    assert hints["return"] is WidgetPluginResult


# ---- Phase 0 follow-on stories (US-002 / US-003) -----------------------------


def test_generic_plugin_registered_or_pending():
    if "generic" not in PLUGIN_REGISTRY:
        pytest.skip("generic plugin not yet registered (US-002 lands it)")
    plugin = PLUGIN_REGISTRY["generic"]
    assert hasattr(plugin, "handle_widget_message")
    assert callable(plugin.handle_widget_message)


def test_shopify_plugin_registered_or_pending():
    # US-003 has landed — Phase 0 is complete. The skip-tolerance
    # remains in case a downstream story restructures the package, but
    # the assertion side is what gates Phase 1.
    if "shopify" not in PLUGIN_REGISTRY:
        pytest.skip("shopify plugin not yet registered (US-003 lands it)")
    plugin = PLUGIN_REGISTRY["shopify"]
    assert hasattr(plugin, "handle_widget_message")
    assert callable(plugin.handle_widget_message)
