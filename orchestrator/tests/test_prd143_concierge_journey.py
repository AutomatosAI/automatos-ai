"""PRD-143 S15 — Concierge MVP golden journey: "set up my workspace/agents".

US-007 / WS-D: the user says "set up my workspace" and Auto plans and executes
it end-to-end under ``autonomy=full`` with no confirmation — every step
audited, never touching an su tool. Journey #1 is workspace/agents
(pure-platform, PRD-143 Open Q3 recommendation).

This drives the EXISTING Arc — no new planner (PRD-143 §9):

    StreamingChatService._stream_tool_loop (consumers/chatbot/service.py)
      → ToolLoopExecutor (the converged spine)
      → _tool_callback → ToolRouter.execute_and_format (tool_router.py)
      → execute_tool → UnifiedToolExecutor.execute_tool (platform_execute
        dispatcher + first-class platform_* routing, S14 selection telemetry)
      → exec_platform → PlatformActionExecutor.execute (REAL gates: su lock,
        confirmation/autonomy dial, PRD-140 hierarchy check, destructive
        backstop, S8 autonomous audit marker)
      → leaf handlers (stubbed = the mocked externals; everything above the
        leaves is the real production path).

The journey (operator tools from S10/S11 + existing actions; the LLM is
scripted — its "plan" mirrors how Auto drives the real surface, promoted
actions first-class + everything else through the platform_execute enum):

    1. platform_create_agent      (first-class, promoted)   create the agent
    2. platform_update_agent      (first-class, promoted)   configure persona/model
    3. platform_set_power_mode    (dispatcher)              power config
    4. platform_connect_channel   (dispatcher)              channel (driver mocked)
    5. platform_invite_member     (dispatcher)              admin operator surface —
    6. platform_set_member_role   (dispatcher)              the Rev 2 inversion in
                                                            action (destructive +
                                                            requires_confirmation)
    7. platform_execute_playbook  (dispatcher)              launch starter playbook
    8. final round                                          recommend next steps

Idioms: S8's fake-POSTGRES preamble + real-catalogue registry fixture +
autonomy/hierarchy/rate-limit patches; S14's selection stash priming
(record_selection → router_decision->'selection' on dispatcher audit rows).
No DB, no network, no real LLM.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import os
import sys
import types
import uuid
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.pop("HARNESS_SELF_MANAGEMENT_ENABLED", None)
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


# Lean-venv shim: the chatbot service import chain pulls modules.rag's
# ingestion stack (camelot at module top). Stub the missing leaf only when
# truly absent (mirrors S8).
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import consumers.chatbot.service as chat_mod  # noqa: E402
import modules.tools.discovery.platform_executor as pe  # noqa: E402
import modules.tools.execution.unified_executor as ue_mod  # noqa: E402
import modules.tools.tool_router as tr  # noqa: E402
from consumers.chatbot.service import StreamingChatService  # noqa: E402
from core.security.hierarchy_permissions import (  # noqa: E402
    PermissionDecision,
    TARGET_AGENT,
)
from modules.tools.discovery.action_registry import ActionRegistry  # noqa: E402
from modules.tools.discovery.platform_actions import register_all_actions  # noqa: E402
from modules.tools.discovery.signal_recorder import (  # noqa: E402
    get_tool_signal_recorder,
)
from modules.tools.execution.telemetry import write_telemetry  # noqa: E402
from modules.tools.execution.unified_executor import UnifiedToolExecutor  # noqa: E402

pytestmark = pytest.mark.asyncio

_WS = "11111111-1111-1111-1111-000000000015"
_AGENT_ID = 9
_NEW_AGENT_ID = 101
_MEMBER_ID = 55
_SU_PROBE = "platform_get_system_health"  # su tier since S4 — must never run

# Dispatcher-routed journey actions (non-promoted → platform_execute enum).
_DISPATCH_ACTIONS = [
    "platform_set_power_mode",
    "platform_connect_channel",
    "platform_invite_member",
    "platform_set_member_role",
    "platform_execute_playbook",
]

# The full ordered journey identity sequence (audit assertion).
_JOURNEY_SEQUENCE = [
    "platform_create_agent",
    "platform_update_agent",
    "platform_set_power_mode",
    "platform_connect_channel",
    "platform_invite_member",
    "platform_set_member_role",
    "platform_execute_playbook",
]


@pytest.fixture(scope="module")
def real_registry() -> ActionRegistry:
    """The REAL catalogue, registered directly (not the singleton)."""
    reg = ActionRegistry()
    register_all_actions(reg)
    reg._initialized = True
    return reg


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB) — pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 10)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


def _su_names(reg: ActionRegistry) -> set:
    """The su tier FROM THE REGISTRY (single source of truth, mirrors S16)."""
    names = {a.name for a in reg.get_all() if a.super_admin_only}
    assert names, "su tier unexpectedly empty — registry fixture broken"
    return names


# ---------------------------------------------------------------------------
# Scripted-LLM + journey harness
# ---------------------------------------------------------------------------

def _resp(content: str = "", tool_calls: Optional[List[Dict[str, Any]]] = None):
    return SimpleNamespace(
        content=content, tool_calls=tool_calls, usage=None,
        finish_reason="stop", model="scripted", provider="test",
    )


def _tc(call_id: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _dispatch(call_id: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return _tc(call_id, "platform_execute", {"action": action, "params": params})


_FINAL_RECOMMENDATION = (
    "Your workspace is set up: Atlas Research Agent is configured on gpt-4o, "
    "power mode is standard, Telegram is connected, your co-founder is invited "
    "as admin and the Starter Daily Digest playbook is running. Next I "
    "recommend connecting Slack and uploading your knowledge documents."
)


def _full_journey_script():
    """initial response + the LLM rounds for the complete journey."""
    initial = [_tc("c1", "platform_create_agent", {
        "name": "Atlas Research Agent",
        "agent_type": "researcher",
        "model_id": "gpt-4o",
    })]
    rounds = [
        # Round 2 — the script "read" the create result: configure agent 101.
        _resp(tool_calls=[_tc("c2", "platform_update_agent", {
            "agent_id": _NEW_AGENT_ID,
            "system_prompt": "You are Atlas, the workspace research agent.",
        })]),
        # Round 3 — power + channel batched (dispatcher).
        _resp(tool_calls=[
            _dispatch("c3", "platform_set_power_mode", {"power_mode": "standard"}),
            _dispatch("c4", "platform_connect_channel", {
                "platform": "telegram", "config": {"bot_token": "tok-test-123"},
            }),
        ]),
        # Round 4 — the administrative operator surface (Rev 2): invite.
        _resp(tool_calls=[_dispatch("c5", "platform_invite_member", {
            "email": "cofounder@example.com", "role": "editor",
        })]),
        # Round 5 — grant admin (destructive + requires_confirmation=True —
        # the step the autonomy dial governs).
        _resp(tool_calls=[_dispatch("c6", "platform_set_member_role", {
            "member_id": _MEMBER_ID, "role": "admin",
        })]),
        # Round 6 — launch the starter playbook.
        _resp(tool_calls=[_dispatch("c7", "platform_execute_playbook", {
            "playbook_name": "Starter Daily Digest",
        })]),
        # Round 7 — recommend next steps (no tools → loop ends).
        _resp(content=_FINAL_RECOMMENDATION),
    ]
    return initial, rounds


def _make_runtime(rounds) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=_AGENT_ID,
        llm_manager=SimpleNamespace(
            generate_response=AsyncMock(side_effect=list(rounds)),
        ),
    )


def _make_service() -> StreamingChatService:
    """The real service minus __init__'s AgentFactory/DB wiring — every
    method the tool loop touches is the real production code."""
    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db = MagicMock()
    svc.workspace_id = _WS
    svc.widget_mode = False
    svc.tool_router = tr.get_tool_router()
    svc.streaming_handler = chat_mod.get_streaming_handler()
    return svc


def _journey_surface(reg: ActionRegistry) -> List[Dict[str, Any]]:
    """The operator tool surface exactly as production shapes it: promoted
    first-class schemas + the platform_execute dispatcher, su fail-closed."""
    return reg.to_first_class_schemas(include_super_admin=False) + [
        reg.to_dispatcher_schema(include_super_admin=False)
    ]


def _prime_selection_stash() -> None:
    """Mirror the production surface step (S14): get_tools_for_agent records
    the narrowed selection; the dispatcher peeks it so audit rows carry
    router_decision->'selection'->'action'."""
    get_tool_signal_recorder().record_selection(
        workspace_id=_WS,
        agent_id=_AGENT_ID,
        narrowed=True,
        reason=None,
        allowed_names=list(_DISPATCH_ACTIONS),
    )


@contextmanager
def _arc(real_registry: ActionRegistry, *, full_autonomy: bool):
    """Patch the Arc's leaves only — gates, routing and formatting are real.

    - registry → the real catalogue fixture
    - SessionLocal/_get_executor_for_request → no real DB
    - fire_telemetry → captured (replayed through the REAL write_telemetry)
    - _full_autonomy → the dial under test
    - can_actor_modify → allow (consultation still asserted)
    - rate limiter → no-op
    - leaf handlers → AsyncMock externals (the "mocked externals")
    """
    handlers = {
        "create_agent": AsyncMock(return_value={
            "success": True,
            "agent": {"id": _NEW_AGENT_ID, "name": "Atlas Research Agent"},
        }),
        "update_agent": AsyncMock(return_value={
            "success": True,
            "agent": {"id": _NEW_AGENT_ID, "model_id": "gpt-4o"},
        }),
        "set_power_mode": AsyncMock(return_value={
            "success": True, "power_mode": "standard",
        }),
        "connect_channel": AsyncMock(return_value={
            "success": True,
            "channel": {"id": "ch-1", "platform": "telegram", "status": "active"},
        }),
        "invite_member": AsyncMock(return_value={
            "success": True,
            "member": {"member_id": _MEMBER_ID, "email": "cofounder@example.com",
                       "role": "editor"},
        }),
        "set_member_role": AsyncMock(return_value={
            "success": True,
            "member": {"member_id": _MEMBER_ID, "role": "admin"},
        }),
        "execute_playbook": AsyncMock(return_value={
            "success": True, "execution_id": 777, "status": "running",
        }),
        "get_system_health": AsyncMock(return_value={"success": True}),
    }
    telemetry: List[Dict[str, Any]] = []
    router_db = MagicMock()
    router_db.query.return_value.filter.return_value.first.return_value = None

    with ExitStack() as stack:
        stack.enter_context(patch(
            "modules.tools.discovery.get_action_registry",
            return_value=real_registry,
        ))
        stack.enter_context(patch(
            "core.database.database.SessionLocal", return_value=router_db,
        ))
        stack.enter_context(patch.object(
            tr, "_get_executor_for_request",
            side_effect=lambda db: UnifiedToolExecutor(db, registry=MagicMock()),
        ))
        stack.enter_context(patch.object(
            ue_mod, "fire_telemetry",
            side_effect=lambda db, **kw: telemetry.append(kw),
        ))
        stack.enter_context(patch.object(
            pe.PlatformActionExecutor, "_full_autonomy",
            return_value=full_autonomy,
        ))
        gate = stack.enter_context(patch.object(
            pe, "can_actor_modify",
            return_value=PermissionDecision(allowed=True, reason="s15_allow"),
        ))
        stack.enter_context(patch(
            "core.security.rate_limiter.check_rate_limit",
            new=AsyncMock(return_value=None),
        ))
        for fn_name, stub in handlers.items():
            stack.enter_context(patch.object(pe, fn_name, stub))
        yield SimpleNamespace(handlers=handlers, telemetry=telemetry, gate=gate)


async def _run_journey(initial, rounds):
    """Drive the real _stream_tool_loop; return (final_response, llm_messages,
    runtime)."""
    _prime_selection_stash()
    svc = _make_service()
    runtime = _make_runtime(rounds)
    llm_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are Auto, the workspace operator."},
        {"role": "user", "content": "set up my workspace"},
    ]
    final = None
    async for item in svc._stream_tool_loop(
        _resp(tool_calls=initial), llm_messages, runtime, {}, None,
    ):
        if isinstance(item, dict) and "_final_response" in item:
            final = item["_final_response"]
    assert final is not None, "tool loop never yielded a final response"
    return final, llm_messages, runtime


async def _audit_rows(telemetry: List[Dict[str, Any]]) -> list:
    """Replay every captured universal-telemetry call through the REAL
    write_telemetry and return the ToolExecutionLog rows it builds."""
    rows = []
    for kw in telemetry:
        db = MagicMock()
        await write_telemetry(db, **kw)
        assert db.add.call_count == 1, f"telemetry write failed for {kw['tool_name']}"
        rows.append(db.add.call_args[0][0])
        db.commit.assert_called_once()
    return rows


def _identity(row) -> Optional[str]:
    """The queryable action identity of an audit row: action_name for
    first-class calls, router_decision->'selection'->'action' for dispatches."""
    if row.action_name != "platform_execute":
        return row.action_name
    return ((row.router_decision or {}).get("selection") or {}).get("action")


# ---------------------------------------------------------------------------
# 1. End-to-end with mocked externals: every setup step takes effect
# ---------------------------------------------------------------------------

async def test_journey_end_to_end_with_mocked_externals(real_registry):
    initial, rounds = _full_journey_script()
    with _arc(real_registry, full_autonomy=True) as arc:
        final, llm_messages, runtime = await _run_journey(initial, rounds)

    # The journey completed and ends with the next-step recommendation.
    assert final.content == _FINAL_RECOMMENDATION
    assert "recommend" in final.content

    # Each setup step's effect, workspace-scoped, with the scripted plan's
    # params — and the create→configure chain carried the new agent's id.
    h = arc.handlers
    for name in ("create_agent", "update_agent", "set_power_mode",
                 "connect_channel", "invite_member", "set_member_role",
                 "execute_playbook"):
        h[name].assert_awaited_once()
        _db, ws, params = h[name].await_args.args
        assert ws == _WS, f"{name} not workspace-scoped"
        assert params.get("_agent_id") == _AGENT_ID, f"{name} lost actor identity"

    assert h["create_agent"].await_args.args[2]["name"] == "Atlas Research Agent"
    assert h["update_agent"].await_args.args[2]["agent_id"] == _NEW_AGENT_ID
    assert h["set_power_mode"].await_args.args[2]["power_mode"] == "standard"
    assert h["connect_channel"].await_args.args[2]["platform"] == "telegram"
    assert h["invite_member"].await_args.args[2]["email"] == "cofounder@example.com"
    assert h["set_member_role"].await_args.args[2].items() >= {
        "member_id": _MEMBER_ID, "role": "admin",
    }.items()
    assert (
        h["execute_playbook"].await_args.args[2]["playbook_name"]
        == "Starter Daily Digest"
    )

    # No confirmation stop anywhere: under autonomy=full the destructive
    # role-grant ran end-to-end (the S8 positive contract, now in a journey).
    tool_contents = [
        str(m.get("content") or "") for m in llm_messages if m.get("role") == "tool"
    ]
    assert tool_contents, "no tool results reached the LLM transcript"
    assert not any("requires confirmation" in c.lower() for c in tool_contents)

    # The PRD-140 hierarchy gate was CONSULTED for the agent edit (the
    # safety net replacing exclusion — open ≠ unguarded).
    update_calls = [
        c for c in arc.gate.call_args_list
        if c.kwargs.get("target_type") == TARGET_AGENT
        and c.kwargs.get("target_id") == _NEW_AGENT_ID
    ]
    assert update_calls, "hierarchy gate never consulted for platform_update_agent"
    assert update_calls[0].kwargs["actor_agent_id"] == _AGENT_ID
    assert update_calls[0].kwargs["change_type"] == "update"
    assert update_calls[0].kwargs["source"] == "platform_tool"

    # The executed sequence matches the plan, in order (telemetry capture).
    executed = [
        _identity(r) for r in await _audit_rows(arc.telemetry)
    ]
    assert executed == _JOURNEY_SEQUENCE


# ---------------------------------------------------------------------------
# 2. Every step audited: distinct, queryable Wave 4 rows
# ---------------------------------------------------------------------------

async def test_every_step_audited(real_registry):
    initial, rounds = _full_journey_script()
    with _arc(real_registry, full_autonomy=True) as arc:
        await _run_journey(initial, rounds)

    rows = await _audit_rows(arc.telemetry)

    # One row per journey step — none missing, none merged.
    assert len(rows) == len(_JOURNEY_SEQUENCE)
    identities = [_identity(r) for r in rows]
    assert identities == _JOURNEY_SEQUENCE
    assert len(set(identities)) == len(identities), "audit rows not distinct"

    # Who / what / where on every row.
    for row in rows:
        assert row.workspace_id == _WS
        assert row.agent_id == _AGENT_ID
        assert row.status == "success"

    # The autonomous marker is distinct and queryable: EXACTLY the
    # confirmation-skipped administrative step carries it
    # (router_decision->>'autonomous'), nothing else does.
    autonomous = [
        r for r in rows if (r.router_decision or {}).get("autonomous") is True
    ]
    assert len(autonomous) == 1
    assert _identity(autonomous[0]) == "platform_set_member_role"

    # Dispatcher rows carry the S14 selection outcome (narrowed surface hit).
    role_row = autonomous[0]
    sel = (role_row.router_decision or {}).get("selection") or {}
    assert sel.get("narrowed") is True
    assert sel.get("hit") is True


# ---------------------------------------------------------------------------
# 3. The journey never touches the su tier — surface, execution, refusal
# ---------------------------------------------------------------------------

async def test_journey_never_invokes_su_tool(real_registry):
    su = _su_names(real_registry)
    assert _SU_PROBE in su  # guard: the probe really is su tier (S4)

    # (a) The journey surface itself is su-clean — and the Rev 2 inversion
    # holds: the administrative operator tools ARE offered.
    surface = _journey_surface(real_registry)
    first_class_names = {s["function"]["name"] for s in surface[:-1]}
    enum = set(
        surface[-1]["function"]["parameters"]["properties"]["action"]["enum"]
    )
    assert first_class_names & su == set()
    assert enum & su == set()
    assert "platform_create_agent" in first_class_names  # promoted, first-class
    assert "platform_set_member_role" in enum            # admin tool, operator tier
    assert "platform_invite_member" in enum

    # (b) The executed action list ∩ su tier == ∅ across the whole journey.
    initial, rounds = _full_journey_script()
    with _arc(real_registry, full_autonomy=True) as arc:
        await _run_journey(initial, rounds)
        executed = {_identity(r) for r in await _audit_rows(arc.telemetry)}
    assert executed & su == set()
    arc.handlers["get_system_health"].assert_not_awaited()

    # (c) Even a hallucinated su call mid-journey is refused fail-closed —
    # the chat path carries no super-admin principal (caller_context=None).
    probe_initial = [_dispatch("p1", _SU_PROBE, {})]
    probe_rounds = [_resp(content="I cannot access the observability tier.")]
    with _arc(real_registry, full_autonomy=True) as arc:
        final, llm_messages, _ = await _run_journey(probe_initial, probe_rounds)

    arc.handlers["get_system_health"].assert_not_awaited()
    probe_rows = await _audit_rows(arc.telemetry)
    assert len(probe_rows) == 1
    assert probe_rows[0].status == "error"
    assert "super admin" in (probe_rows[0].error_message or "").lower()


# ---------------------------------------------------------------------------
# 4. The dial governs the journey: standard autonomy requires confirmation
# ---------------------------------------------------------------------------

async def test_journey_requires_confirmation_at_standard_autonomy(real_registry):
    initial, full_rounds = _full_journey_script()
    # Same plan, but the scripted Auto reacts to the confirmation stop the
    # way production Auto does: relay the ask and halt the journey.
    rounds = full_rounds[:4] + [_resp(content=(
        "Granting admin needs your confirmation before I can continue the setup."
    ))]

    with _arc(real_registry, full_autonomy=False) as arc:
        final, llm_messages, _ = await _run_journey(initial, rounds)

    h = arc.handlers
    # Steps before the destructive role-grant ran (standard autonomy only
    # stops confirmation-bearing actions — the documented dial semantics)…
    for name in ("create_agent", "update_agent", "set_power_mode",
                 "connect_channel", "invite_member"):
        h[name].assert_awaited_once()
    # …the role-grant stopped at the confirmation gate, and the journey
    # never reached the playbook launch.
    h["set_member_role"].assert_not_awaited()
    h["execute_playbook"].assert_not_awaited()

    # The confirmation ask is VISIBLE to Auto/the user — the tool transcript
    # names the action and asks for confirmation (not a swallowed
    # "Unknown error").
    tool_contents = [
        str(m.get("content") or "") for m in llm_messages if m.get("role") == "tool"
    ]
    confirm_msgs = [c for c in tool_contents if "requires confirmation" in c.lower()]
    assert confirm_msgs, (
        "the requires_confirmation stop never surfaced in the LLM transcript"
    )
    assert any("platform_set_member_role" in c for c in confirm_msgs)
    assert final.content.lower().startswith("granting admin needs your confirmation")

    # No autonomous markers at standard — the dial really was off.
    rows = await _audit_rows(arc.telemetry)
    assert all(not (r.router_decision or {}).get("autonomous") for r in rows)
    # The halted dispatch is audited as a non-success row (the stop is on
    # the record too).
    role_rows = [r for r in rows if _identity(r) == "platform_set_member_role"]
    assert len(role_rows) == 1
    assert role_rows[0].status == "error"
