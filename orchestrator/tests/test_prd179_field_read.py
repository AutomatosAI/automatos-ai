"""PRD-179 S1 (F021 read-half) — field/durable memory digest in the HEARTBEAT
and PLANNING context modes.

PRD-164 wired planning to documents + the knowledge graph but never the field,
and heartbeat agents were memory-blind by design (`modes.py:76-88`, F021
CONFIRMED). This adds a workspace-scoped field digest to both modes by reading
what Wave 8 promotes into the workspace-persistent field
(`VectorFieldSharedContext.query_workspace`) and rendering it through the ONE
existing digest builder (`field_scoring.budget_results` + `format_digest`) — no
second builder.

`test_planning_reads_field` is the W9 acceptance gate: a completed mission's
field distillation appears in the next code-touching mission's planning pack. It
spans Wave 8's promotion (mocked here at the workspace-read boundary) and this
digest. Pure — the field backend is mocked, no Qdrant / DB.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config as _config  # noqa: E402

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"):
    if not getattr(_config, _k, None):
        setattr(_config, _k, os.environ[_k])

from modules.context.modes import MODE_CONFIGS, ContextMode  # noqa: E402
from modules.context.sections import SECTION_REGISTRY  # noqa: E402
from modules.context.sections.base import SectionContext  # noqa: E402
from modules.context.sections.field_memory import FieldMemorySection  # noqa: E402

WS = "11111111-1111-1111-1111-111111111111"

# What Wave 8 promotes into the workspace-persistent field: a completed
# mission's distilled finding, keyed by the task/mission it came from.
PROMOTED_FINDING = {
    "key": "Refactor auth module",
    "value": "The auth module's token refresh must run BEFORE the request retry; "
             "reversing them double-charges the rate limiter (learned last mission).",
    "score": 0.91,
}


def _fake_field(rows: List[Dict[str, Any]]):
    """A stand-in shared-context backend whose workspace read returns *rows*.

    Mirrors the instrumentation wrapper shape: the real workspace-scoped method
    lives on the inner adapter, reached via ``getattr(field, '_inner', field)``.
    """
    inner = SimpleNamespace(query_workspace=AsyncMock(return_value=rows))
    return SimpleNamespace(_inner=inner)


# ---------------------------------------------------------------------------
# S1a — the section itself renders the promoted finding through the ONE builder
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_field_section_renders_workspace_digest():
    section = FieldMemorySection()
    ctx = SectionContext(
        agent=None,
        workspace_id=WS,
        task_description="Refactor the auth module for the retry path",
    )
    with patch(
        "modules.context.sections.field_memory.get_shared_context",
        return_value=_fake_field([PROMOTED_FINDING]),
    ):
        out = await section.render(ctx)

    assert PROMOTED_FINDING["value"] in out, "promoted finding not surfaced in digest"
    assert "Field memory" in out, "digest not built by the shared format_digest builder"


@pytest.mark.asyncio
async def test_field_section_empty_when_no_patterns():
    """No promoted patterns → empty string (never a misleading 'no memory' block)."""
    section = FieldMemorySection()
    ctx = SectionContext(agent=None, workspace_id=WS, task_description="anything")
    with patch(
        "modules.context.sections.field_memory.get_shared_context",
        return_value=_fake_field([]),
    ):
        out = await section.render(ctx)
    assert out == ""


@pytest.mark.asyncio
async def test_field_section_workspace_scoped_read():
    """The section reads the WORKSPACE-scoped field (query_workspace), not a
    mission-scoped one — so only this workspace's promoted patterns can appear."""
    section = FieldMemorySection()
    field = _fake_field([PROMOTED_FINDING])
    ctx = SectionContext(agent=None, workspace_id=WS, task_description="refactor auth")
    with patch(
        "modules.context.sections.field_memory.get_shared_context",
        return_value=field,
    ):
        await section.render(ctx)
    field._inner.query_workspace.assert_awaited_once()
    _, kwargs = field._inner.query_workspace.call_args
    passed_ws = kwargs.get("workspace_id") or field._inner.query_workspace.call_args.args[0]
    assert str(passed_ws) == WS


@pytest.mark.asyncio
async def test_field_section_never_raises():
    """A backend explosion degrades to '' — memory read must never crash a prompt."""
    section = FieldMemorySection()
    ctx = SectionContext(agent=None, workspace_id=WS, task_description="x")
    boom = SimpleNamespace(_inner=SimpleNamespace(
        query_workspace=AsyncMock(side_effect=RuntimeError("qdrant down"))
    ))
    with patch(
        "modules.context.sections.field_memory.get_shared_context",
        return_value=boom,
    ):
        out = await section.render(ctx)
    assert out == ""


# ---------------------------------------------------------------------------
# S1b — both modes actually include the section (the wiring)
# ---------------------------------------------------------------------------

def test_field_memory_registered():
    assert "field_memory" in SECTION_REGISTRY
    assert SECTION_REGISTRY["field_memory"] is FieldMemorySection


def test_heartbeat_mode_includes_field_memory():
    assert "field_memory" in MODE_CONFIGS[ContextMode.HEARTBEAT_AGENT].sections


def test_planning_mode_includes_field_memory():
    assert "field_memory" in MODE_CONFIGS[ContextMode.PLANNING].sections


def test_field_memory_not_added_to_chatbot_region():
    """W7 owns the CHATBOT / chain-hints region — W9 must not touch it. CHATBOT
    already carries the 'memory' section; it must NOT gain 'field_memory'."""
    assert "field_memory" not in MODE_CONFIGS[ContextMode.CHATBOT].sections


# ---------------------------------------------------------------------------
# S1c — W9 GATE: a completed mission's distillation reaches the planning pack
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_planning_reads_field():
    """W9 ACCEPTANCE GATE. Build the ONE planning pack for a code-touching goal;
    the workspace field (holding a prior completed mission's promoted finding —
    what Wave 8 writes) must appear in the assembled pack.

    Wave 8's promotion is mocked at the workspace-read boundary; this proves the
    W9 half — the pack now consults the field — so the two meet at merge.
    """
    from modules.context.service import ContextService

    field = _fake_field([PROMOTED_FINDING])
    svc = ContextService(db_session=None)

    with patch(
        "modules.context.sections.field_memory.get_shared_context",
        return_value=field,
    ):
        pack = await svc.build_planning_context(
            goal="Refactor the auth module retry path",
            workspace_id=WS,
            include_roster=False,
        )

    assert PROMOTED_FINDING["value"] in pack.content, (
        "completed mission's field distillation absent from the planning pack"
    )
    assert "field_memory" in pack.sections_included, (
        f"planning pack did not include the field section (got {pack.sections_included})"
    )
