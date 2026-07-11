"""PRD-193 S3 (P2-12) — the in-chat tool-approval card (backend payload).

A confirmation ask that carries a grant must surface a ``tool_approval``
card payload from ``format_for_frontend`` — grant id, action, human-readable
params digest, permission level, and the AI-Act oversight fields — mirroring
the PRD-163/181 ``mission_approval`` card. The S15 prose stays the model's
view; the card is the human's.

Result-shape driven, not tool_name driven: direct ``platform_*`` calls and
the ``platform_execute`` meta-dispatch both get the card. No grant ⇒ no card
(prose-only, exactly today's behaviour). The frontend half
(ToolApprovalWidget) is covered by a vitest test.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))


def _ask_result(**overrides):
    base = {
        "success": False,
        "requires_confirmation": True,
        "action": "platform_delete_document",
        "permission_level": "destructive",
        "message": (
            "This action (destructive) requires confirmation. "
            "Action: platform_delete_document — Delete a document permanently."
        ),
        "params": {"document_id": 7, "_agent_id": 9, "_agent_name": "Scribe"},
        # attached by the S1 gate
        "grant_id": 42,
        "risk_class": "destructive",
        "risk_tier": "human_in_the_loop",
        "oversight_rationale": "Destructive: deletes data. A human must approve before it runs.",
        "requires_approval": True,
    }
    base.update(overrides)
    return base


def test_formatter_emits_tool_approval_card():
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    fd = ToolResultFormatter.format_for_frontend(_ask_result(), "platform_delete_document")

    card = fd["tool_approval"]
    assert card["grant_id"] == 42
    assert card["action"] == "platform_delete_document"
    assert card["permission_level"] == "destructive"
    assert card["message"].startswith("This action (destructive)")
    # The digest shows the model-provided params; server plumbing is stripped.
    assert card["params"] == {"document_id": 7}
    # AI-Act oversight fields for the approver.
    assert card["risk_class"] == "destructive"
    assert card["risk_tier"] == "human_in_the_loop"
    assert "human must approve" in card["oversight_rationale"]
    assert card["requires_approval"] is True


def test_card_works_for_platform_execute_dispatch():
    """The meta-dispatch lane reports tool_name=platform_execute — the card
    still names the REAL action (carried on the ask result)."""
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    fd = ToolResultFormatter.format_for_frontend(_ask_result(), "platform_execute")
    assert fd["tool_approval"]["action"] == "platform_delete_document"
    assert fd["tool_approval"]["grant_id"] == 42


def test_no_card_without_grant():
    """A grant-less ask (grant machinery faulted — the fail-safe floor) stays
    prose-only: no half-card the human cannot act on."""
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    ask = _ask_result()
    for k in ("grant_id", "risk_class", "risk_tier", "oversight_rationale"):
        ask.pop(k, None)
    fd = ToolResultFormatter.format_for_frontend(ask, "platform_delete_document")
    assert "tool_approval" not in fd


def test_no_card_on_success():
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    fd = ToolResultFormatter.format_for_frontend(
        {"success": True, "documents": []}, "platform_delete_document"
    )
    assert "tool_approval" not in fd


def test_oversight_fields_floor_fail_safe():
    """A card missing its oversight fields (older/degraded ask shape) must
    still never imply 'no human oversight'."""
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    ask = _ask_result()
    ask.pop("risk_class", None)
    ask.pop("risk_tier", None)
    ask.pop("oversight_rationale", None)
    fd = ToolResultFormatter.format_for_frontend(ask, "platform_delete_document")
    card = fd["tool_approval"]
    assert card["risk_tier"] == "human_in_the_loop"
    assert card["oversight_rationale"]
    assert card["requires_approval"] is True


def test_params_digest_truncates_long_values():
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    digest = ToolResultFormatter._tool_params_digest(
        {"key": "x" * 500, "n": 3, "flag": True, "none": None, "_agent_id": 9}
    )
    assert digest["n"] == 3 and digest["flag"] is True and digest["none"] is None
    assert "_agent_id" not in digest
    assert len(digest["key"]) <= 120


@pytest.mark.asyncio
async def test_router_failure_envelope_carries_ask_card():
    """The tool-router failure path emits the card's frontend_data for an ask
    that carries a grant — and stays empty for grant-less failures. The S15
    prose remains the model's channel either way."""
    from modules.tools.tool_router import ToolRouter

    router = ToolRouter()
    ask = _ask_result()

    with patch(
        "modules.tools.tool_router.execute_tool", new=AsyncMock(return_value=ask)
    ), patch.object(ToolRouter, "_record_tool_signal", lambda *a, **k: None):
        envelope = await router.execute_and_format(
            "platform_delete_document", {"document_id": 7}, agent_id=1
        )

    assert envelope["success"] is False
    assert envelope["frontend_data"]["tool_approval"]["grant_id"] == 42
    # S15 stays: the ask message reaches the model as ever.
    assert "requires confirmation" in envelope["llm_context"].lower()

    # Grant-less ask (fail-safe floor) ⇒ prose-only, no card payload.
    bare = {k: v for k, v in ask.items() if k != "grant_id"}
    with patch(
        "modules.tools.tool_router.execute_tool", new=AsyncMock(return_value=bare)
    ), patch.object(ToolRouter, "_record_tool_signal", lambda *a, **k: None):
        envelope2 = await router.execute_and_format(
            "platform_delete_document", {"document_id": 7}, agent_id=1
        )
    assert envelope2["frontend_data"] == {}
    assert "requires confirmation" in envelope2["llm_context"].lower()
