"""PRD-268 P268W1-RVW-2 (§6.2.3, FR-1): building a check-in's context changes nothing in the field.

``ContextMode.CHECK_IN`` includes the ``field_memory`` section, which reads the workspace's
field through ``VectorFieldSharedContext.query_workspace``. That read reinforced every
pattern it returned (Hebbian ``_reinforce_batch``: access_count, last_accessed and
strength) on every check-in, in ``ContextService.build_context``, before the lane saw any
call, so the patterns a check-in read stopped decaying. A check-in's context now reads the
field with ``record_access=False``; HEARTBEAT_AGENT and PLANNING reinforce what they read,
as before.

The field is the real adapter over a mocked Qdrant holding two patterns (the PRD-178
harness), with its reads and its reinforcement spied on. No database.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from modules.context import field_scoring
from modules.context.modes import MODE_CONFIGS, ContextMode
from modules.context.sections import SECTION_REGISTRY
from modules.context.sections.tools import ToolsSection
from modules.context.service import ContextService
from tests.test_prd178_field_trace_readonly import _adapter_with_client, _client_with_hits

WS = "6d1f0b2a-3c54-4b9f-8d72-1e8f3c0a5b21"
TASK = "What changed in the workspace since the last check-in?"
FIELD_SECTION = "field_memory"


class _Quiet:
    """A section that renders nothing, so a build reads only the field."""

    name = "quiet"
    priority = 10
    max_tokens = None

    async def render(self, ctx):
        return ""


@pytest.fixture
def no_telemetry(monkeypatch):
    """The field read's substrate event goes nowhere: these tests watch the field alone."""
    monkeypatch.setattr("core.observability.substrate_metrics.record_substrate_search_nowait", lambda **kw: None)


@pytest.fixture
def field(monkeypatch, no_telemetry):
    """The field the section reaches: the real adapter, its reads and reinforcement spied on."""
    adapter = _adapter_with_client(_client_with_hits())
    adapter.query_workspace = AsyncMock(side_effect=adapter.query_workspace)
    adapter._reinforce_batch = AsyncMock(side_effect=adapter._reinforce_batch)
    monkeypatch.setattr("modules.context.sections.field_memory.get_shared_context", lambda: adapter)
    return adapter


@pytest.fixture
def only_the_field(monkeypatch):
    """Every other section of CHECK_IN and HEARTBEAT_AGENT renders nothing, and no tools load."""
    for mode in (ContextMode.CHECK_IN, ContextMode.HEARTBEAT_AGENT):
        for name in MODE_CONFIGS[mode].sections:
            if name != FIELD_SECTION:
                monkeypatch.setitem(SECTION_REGISTRY, name, type(f"_Quiet_{name}", (_Quiet,), {"name": name}))
    monkeypatch.setattr(ToolsSection, "load_tools", AsyncMock(return_value=([], "auto")))
    monkeypatch.setattr("modules.context.service.with_rule_for_mode", lambda prompt, *args: prompt)


def _build(mode):
    return asyncio.run(ContextService(db_session=None).build_context(
        mode=mode, agent=None, workspace_id=WS, task_description=TASK,
    ))


def _record_access(field) -> bool:
    """What the section passed for record_access: it always passes it, so a dropped kwarg fails."""
    return field.query_workspace.await_args.kwargs["record_access"]


def test_a_check_in_context_reads_the_field_without_reinforcing_it(field, only_the_field):
    result = _build(ContextMode.CHECK_IN)

    assert FIELD_SECTION in result.sections_included, "the check-in still reads the field"
    field.query_workspace.assert_awaited_once()
    assert _record_access(field) is False
    field._reinforce_batch.assert_not_awaited()
    assert field._client.set_payload.await_count == 0


def test_a_heartbeat_context_still_reinforces_what_it_reads(field, only_the_field):
    result = _build(ContextMode.HEARTBEAT_AGENT)

    assert FIELD_SECTION in result.sections_included
    assert _record_access(field) is True
    field._reinforce_batch.assert_awaited_once()
    assert field._client.set_payload.await_count >= 1


def test_a_planning_pack_still_reinforces_what_it_reads(field):
    pack = asyncio.run(ContextService(db_session=None).build_planning_context(
        goal=TASK, workspace_id=WS, include_roster=False,
    ))

    assert FIELD_SECTION in pack.sections_included
    assert _record_access(field) is True
    field._reinforce_batch.assert_awaited_once()


def test_query_workspace_reinforces_only_when_it_records_access(no_telemetry):
    client = _client_with_hits()
    adapter = _adapter_with_client(client)

    rows = asyncio.run(adapter.query_workspace(workspace_id=WS, query=TASK, record_access=False))
    assert rows and client.set_payload.await_count == 0

    rows = asyncio.run(adapter.query_workspace(workspace_id=WS, query=TASK))
    assert rows and client.set_payload.await_count == len(rows)


def test_the_co_access_bonus_kept_its_curve_when_it_moved_to_field_scoring():
    assert field_scoring.co_accessed_strength(1.0, 1, 2.0) == 1.0
    assert field_scoring.co_accessed_strength(1.0, 3, 2.0) == pytest.approx(1.0 + 2 * field_scoring.CO_ACCESS_BONUS)
    assert field_scoring.co_accessed_strength(0.5, 200, 2.0) == pytest.approx(1.0)
