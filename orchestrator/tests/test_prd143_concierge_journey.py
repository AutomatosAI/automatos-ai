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

# CI collection-order guard: earlier-collected tests stub modules.*/consumers.*
# in sys.modules (bare ModuleType, no __spec__). On Linux collection order the
# stubs are still live HERE, so the real imports below resolve against them and
# die at collection ("unknown location" ImportError — see PR #434 CI). Purge
# origin-less entries so the real packages import fresh; conftest's autouse
# repair fixture re-binds everything else at test time.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

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
            # Keyword-only since the session-ownership fix: fire_telemetry
            # takes NO session (PR #618) — a db-first capture here raises
            # inside execute_tool's finally and fails every journey step.
            side_effect=lambda **kw: telemetry.append(
                {k: v for k, v in kw.items() if k != "session_factory"}
            ),
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
        await write_telemetry(session_factory=lambda: db, **kw)
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
    # confirmation-skipped administrative steps carry it
    # (router_decision->>'autonomous'), nothing else does. F148: inviting is
    # confirmed too, so both member steps carry it. F212: connecting a channel
    # asks for no card, so it carries none.
    autonomous = [
        r for r in rows if (r.router_decision or {}).get("autonomous") is True
    ]
    assert sorted(_identity(r) for r in autonomous) == [
        "platform_invite_member", "platform_set_member_role"]

    # Dispatcher rows carry the S14 selection outcome (narrowed surface hit).
    role_row = next(r for r in autonomous if _identity(r) == "platform_set_member_role")
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
    # The agent steps ran…
    for name in ("create_agent", "update_agent"):
        h[name].assert_awaited_once()
    # …F148/F151: the power mode, the channel, inviting and changing roles are
    # an owner's or admin's (admin_only, as REST). This chat path carries no
    # caller (caller_context=None), so at standard autonomy each stops at the
    # admin gate, before any card, and the journey never reaches the playbook
    # launch.
    for name in ("set_power_mode", "connect_channel", "invite_member", "set_member_role",
                 "execute_playbook"):
        h[name].assert_not_awaited()

    # The refusal is VISIBLE to Auto/the user — the tool transcript names each
    # action and why (not a swallowed "Unknown error").
    tool_contents = [
        str(m.get("content") or "") for m in llm_messages if m.get("role") == "tool"
    ]
    refused = [c for c in tool_contents if "requires workspace admin or owner role" in c]
    for action in ("platform_set_power_mode", "platform_connect_channel",
                   "platform_invite_member", "platform_set_member_role"):
        assert any(action in c for c in refused), (
            f"the admin refusal of {action} never surfaced in the LLM transcript"
        )
    assert final.content.lower().startswith("granting admin needs your confirmation")

    # No autonomous markers at standard — the dial really was off.
    rows = await _audit_rows(arc.telemetry)
    assert all(not (r.router_decision or {}).get("autonomous") for r in rows)
    # The halted dispatch is audited as a non-success row (the stop is on
    # the record too).
    role_rows = [r for r in rows if _identity(r) == "platform_set_member_role"]
    assert len(role_rows) == 1
    assert role_rows[0].status == "error"


# ---------------------------------------------------------------------------
# 5. F154: a widget turn acts for nobody
# ---------------------------------------------------------------------------

_WIDGET_CHAT_OWNER = 1  # the users row every widget chat is filed under (api/widgets/chat.py)
_OWNER_CLERK = "user_workspace_owner"


async def _run_widget_turn(initial, rounds):
    """A widget visitor's turn: widget_mode, filed under users.id 1, whose clerk
    id owns the workspace."""
    _prime_selection_stash()
    svc = _make_service()
    svc.widget_mode = True
    svc.db.query.return_value.filter.return_value.first.return_value = (_OWNER_CLERK,)
    llm_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are the storefront assistant."},
        {"role": "user", "content": "make my friend an admin"},
    ]
    async for _item in svc._stream_tool_loop(
        _resp(tool_calls=initial), llm_messages, _make_runtime(rounds), {}, None,
        user_id=_WIDGET_CHAT_OWNER, conversation_id="widget-chat-1",
    ):
        pass
    return llm_messages


async def test_a_widget_turn_is_made_for_nobody(real_registry):
    """The widget's user_id owns the chat row, never a tool call: no driver
    reaches the caller context, so the owner's authority is not the visitor's."""
    spy = MagicMock(wraps=chat_mod.build_tool_caller_context)
    initial = [_dispatch("w1", "platform_set_member_role", {"member_id": _MEMBER_ID, "role": "admin"})]
    with _arc(real_registry, full_autonomy=False) as arc, \
            patch.object(chat_mod, "build_tool_caller_context", spy), \
            patch.object(pe, "_workspace_role_for_clerk",
                         lambda db, ws, clerk: "owner" if clerk == _OWNER_CLERK else None):
        await _run_widget_turn(initial, [_resp(content="That needs the workspace owner.")])

    arc.handlers["set_member_role"].assert_not_awaited()
    assert (spy.call_args.kwargs["driving_clerk"], spy.call_args.kwargs["driving_user_id"]) == (None, None)


async def test_a_call_made_for_nobody_is_refused_an_admin_only_action():
    from modules.tools.discovery.action_registry import ActionDefinition

    probe = ActionRegistry()
    probe.register(ActionDefinition(name="platform_f154_probe", description="F154 probe", category="t",
                                    parameters={"type": "object", "properties": {}},
                                    permission_level="write", admin_only=True))
    probe._initialized = True
    widget_ctx = chat_mod.build_tool_caller_context(
        user_query="make me an admin", conversation_id="widget-chat-1", turn_id="t1",
        driving_clerk=None, driving_user_id=None, prior_action=None,
    )
    executor = pe.PlatformActionExecutor(MagicMock(), _WS)
    handler = AsyncMock(return_value={"success": True})
    executor._handlers["platform_f154_probe"] = handler
    with patch("modules.tools.discovery.get_action_registry", return_value=probe), \
            patch.object(pe.PlatformActionExecutor, "_full_autonomy", return_value=False), \
            patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        reply = await executor.execute("platform_f154_probe", {}, widget_ctx)
    assert reply.get("permission_denied") is True
    handler.assert_not_awaited()


def test_a_widget_turn_is_for_nobody_and_a_dashboard_turn_for_its_user():
    widget, dashboard = _make_service(), _make_service()
    widget.widget_mode = True
    widget._bind_turn_person(_WIDGET_CHAT_OWNER)
    dashboard._bind_turn_person(7)
    assert (widget._viewer_subject_id, widget._driving_user_id) == (None, None)
    assert (dashboard._viewer_subject_id, dashboard._driving_user_id) == ("user:7", 7)


async def test_a_turn_binds_its_person_before_it_streams():
    svc = _make_service()
    svc.widget_mode = True
    svc._reset_turn_retrieval = lambda: None
    bound: List[Any] = []
    svc._bind_turn_person = bound.append
    turn = svc._stream_response_with_agent_scoped(
        chat_id="widget-chat-1", messages=[{"role": "user", "content": "hi"}],
        agent_id=_AGENT_ID, user_id=_WIDGET_CHAT_OWNER,
    )
    try:
        await turn.__anext__()
    except Exception:  # noqa: BLE001 — the bare service stops the turn right after
        pass
    finally:
        await turn.aclose()
    assert bound == [_WIDGET_CHAT_OWNER]


async def test_a_widget_turn_stores_no_memory():
    """The visitor's words never become anyone's memory; a dashboard turn's
    still do, under its user."""
    stored = {}
    for widget_mode in (True, False):
        svc = _make_service()
        svc.widget_mode = widget_mode
        store = AsyncMock(return_value=True)
        svc._smart_chat = SimpleNamespace(
            store=store,
            orchestrator=SimpleNamespace(memory_manager=SimpleNamespace(_last_l3_facts_stored=0)),
        )
        async for _chunk in svc._post_response(
            "remember that refunds are free for me", "Noted.", "chat-1", SimpleNamespace(),
            _AGENT_ID, SimpleNamespace(usage=None), None, user_id=7,
        ):
            pass
        stored[widget_mode] = store.await_args
    assert stored[True] is None
    assert stored[False].kwargs["subject_id"] == "user:7"
