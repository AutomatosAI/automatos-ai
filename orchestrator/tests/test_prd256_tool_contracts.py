"""PRD-256 US-006 (FR-7): Auto's writes are first-class tools with strict schemas, and a
refused write is reported refused.

The commonest false claim of eleven customer nights was a call that failed on its arguments
and was then reported as done (F108: ``platform_approve_mission`` with ``params: {}``,
"Missing required params: mission_id", then "I've approved the mission. It's now running").
On 53% of dispatcher calls the action used sat outside the ranked top-15 of the enum, led by
these writes. They are now promoted and pinned (``TOOL_ROUTING_PROMOTION_PINS``), each its own
tool with a strict schema; the ATOM lane and the heartbeat's dispatcher-only load ship them
beside the dispatcher; a direct call is held to its required fields; and after a refused write
the loop's nudge and Auto's prompt carry one rule.
"""
from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest

from core.llm.clients.base import LLMResponse

ORCH = Path(__file__).resolve().parents[1]
CONTRACT = (
    "platform_create_task", "platform_update_task", "platform_update_task_status", "platform_assign_task",
    "platform_create_mission", "platform_execute_playbook", "platform_schedule_playbook",
    "platform_create_social_post", "platform_store_memory", "platform_get_task", "platform_query_data",
)
# Each keeps the permission level it had before it was promoted (check_hierarchy_gate reads them).
LEVELS = {name: "write" for name in CONTRACT} | {"platform_get_task": "read", "platform_query_data": "read"}
# The objects keyed by what only the run knows: a playbook's own inputs, a template's own
# fields (and the claims backing them), its aspect ratios and its footage slots. Every other
# object names its fields.
OPEN_MAPS = {
    "platform_execute_playbook": {"input_data"},
    "platform_create_social_post": {"variables", "sources", "media", "footage"},
}
CATCH_ALLS = {"params", "parameters", "args", "arguments", "data", "payload", "extra"}


def _registry():
    from modules.tools.discovery.action_registry import get_action_registry

    return get_action_registry()


def _pins():
    from modules.tools.tool_router import _promotion_pins

    return _promotion_pins()


def _names(schemas):
    return [s["function"]["name"] for s in schemas]


def _enum(dispatcher):
    return dispatcher["function"]["parameters"]["properties"]["action"].get("enum", [])


# ── promoted and pinned ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name", CONTRACT)
def test_each_write_is_promoted_pinned_and_keeps_its_permission_level(name):
    action = _registry().get(name)
    assert action is not None and action.promoted, f"{name} is not promoted"
    assert name in _pins(), f"{name} is not in TOOL_ROUTING_PROMOTION_PINS"
    assert action.permission_level == LEVELS[name]
    assert not action.admin_only and not action.super_admin_only


def test_the_first_class_builder_ships_them_outside_the_dispatchers_enum():
    from modules.tools.tool_router import _first_class_names

    registry = _registry()
    promoted = {a.name for a in registry.get_all() if a.promoted}
    first_class = _first_class_names(None, promoted)
    shipped = set(_names(registry.to_first_class_schemas(exclude_admin=True, first_class_names=first_class)))
    enum = set(_enum(registry.to_dispatcher_schema(exclude_admin=True, exclude_promoted=False,
                                                   exclude_names=first_class)))
    assert set(CONTRACT) <= shipped
    assert not set(CONTRACT) & enum


# ── strict schemas ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", CONTRACT)
def test_each_promoted_schema_is_strict(name):
    schema = _registry().get(name).to_openai_schema()["function"]["parameters"]
    props = schema["properties"]
    assert schema["type"] == "object"
    assert set(schema.get("required") or []) <= set(props), "a required field the schema does not declare"
    assert not CATCH_ALLS & set(props), f"{name} takes a free-form catch-all"
    for field, spec in props.items():
        assert spec.get("type"), f"{name}.{field} has no type"
        if spec["type"] == "object" and field not in OPEN_MAPS.get(name, set()):
            assert spec.get("properties"), f"{name}.{field} is an object that names no field"
    if "status" in props:
        assert props["status"].get("enum"), f"{name}.status is free text"


def test_the_open_maps_are_exactly_the_ones_keyed_by_the_run():
    registry = _registry()
    for name, fields in OPEN_MAPS.items():
        props = registry.get(name).parameters["properties"]
        assert {f for f, spec in props.items() if spec.get("type") == "object" and not spec.get("properties")} \
            == fields


def test_a_card_moves_status_is_the_boards_own_words():
    from api.board_tasks import VALID_STATUSES

    registry = _registry()
    for name in ("platform_update_task", "platform_update_task_status"):
        status = registry.get(name).parameters["properties"]["status"]
        assert status["enum"] == sorted(VALID_STATUSES), name
        assert {"done", "cancelled", "assigned"} <= set(status["enum"])


def test_required_fields_are_listed():
    registry = _registry()
    assert registry.get("platform_create_task").parameters["required"] == ["title", "description"]
    assert registry.get("platform_assign_task").parameters["required"] == ["task_id"]  # FX-012: agent_id or agent_name
    assert registry.get("platform_update_task_status").parameters["required"] == ["status"]
    assert registry.get("platform_create_mission").parameters["required"] == ["goal"]
    assert registry.get("platform_get_task").parameters["required"] == ["task_id"]


def test_a_missions_config_names_its_fields():
    config = _registry().get("platform_create_mission").parameters["properties"]["config"]
    assert config["type"] == "object"
    assert set(config["properties"]) == {"auto_approve", "max_retries", "category", "output_format", "publish",
                                         "check_each_step"}
    assert config["properties"]["check_each_step"]["type"] == "boolean"
    assert config["properties"]["output_format"]["enum"] == ["markdown", "json", "code"]


def test_platform_execute_still_reads_params_sent_as_json_text():
    """F181/F321: the dispatcher's params as JSON text still run as the object they hold."""
    from modules.tools.execution.params_text import nested_params_decoded

    sent = {"action": "platform_create_task", "params": json.dumps({"title": "Call Quay", "description": "Terms"})}
    assert nested_params_decoded("platform_execute", sent)["params"] == {"title": "Call Quay", "description": "Terms"}


# ── the lanes that ship the dispatcher alone ────────────────────────────────

def _dispatcher():
    return _registry().to_dispatcher_schema(exclude_admin=True)


def test_the_atom_surface_carries_the_promoted_tools_beside_the_dispatcher():
    from modules.tools.first_class_tools import with_first_class

    tools = with_first_class([_dispatcher()])
    names = _names(tools)
    assert names[0] == "platform_execute"
    assert set(CONTRACT) <= set(names)
    assert len(names) == len(set(names)), "a tool shipped twice"


def test_a_surface_with_no_tools_stays_empty():
    from modules.tools.first_class_tools import with_first_class

    assert with_first_class([]) == [] and with_first_class(None) == []


def test_a_tool_already_held_is_not_added_again():
    from modules.tools.first_class_tools import with_first_class

    held = _registry().get("platform_create_task").to_openai_schema()
    assert _names(with_first_class([_dispatcher(), held])).count("platform_create_task") == 1


def test_a_hidden_category_stays_hidden_on_the_atom_surface():
    from modules.tools.discovery.action_registry import hidden_scope
    from modules.tools.first_class_tools import with_first_class

    category = _registry().get("platform_create_social_post").category
    with hidden_scope((category,)):
        names = _names(with_first_class([_dispatcher()]))
    assert "platform_create_social_post" not in names and "platform_create_task" in names


def _function(path: Path, name: str) -> ast.AST:
    tree = ast.parse(path.read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def test_the_atom_branch_calls_the_same_first_class_builder():
    """The change in service.py is a call: the ATOM path's tools pass through with_first_class."""
    atom = ast.unparse(_function(ORCH / "consumers" / "chatbot" / "service.py", "_prepare_atom_path"))
    assert "with_first_class(atom_tools" in atom


@pytest.mark.asyncio
async def test_the_heartbeats_dispatcher_only_load_carries_them_too(monkeypatch):
    from modules.context.sections.tools import ToolLoadingStrategy, ToolsSection
    import modules.tools.tool_router as tr

    async def _ranked(query, is_admin, is_super_admin, **_scope):
        return ["platform_list_tasks", "platform_create_task"], "ranked", False

    monkeypatch.setattr(tr, "_narrow_dispatcher_actions_async", _ranked)
    tools, choice = await ToolsSection().load_tools(
        agent_id=None, workspace_id="ws-hb", strategy=ToolLoadingStrategy.DISPATCHER_ONLY,
        query="Check the board and file what is missing.")
    assert choice == "auto" and tools[0]["function"]["name"] == "platform_execute"
    assert "platform_create_task" not in _enum(tools[0])
    assert {"platform_create_task", "platform_update_task_status", "platform_assign_task"} <= set(_names(tools))


def test_the_tool_shadow_log_still_reports_shipped_and_would_be_first_class():
    shadow = ast.unparse(_function(ORCH / "modules" / "tools" / "tool_router.py", "_maybe_log_shadow_surface"))
    assert "[tool-shadow %s] shipped_tools=%d would_first_class=%d" in shadow


# ── a direct call is held to its required fields ────────────────────────────

def test_a_direct_call_missing_a_required_field_is_refused_in_its_own_form():
    from modules.tools.execution.unified_executor import (
        REFUSED_CALL_IS_YOURS, VIA_DIRECT_CALL, VIA_DISPATCHER, undeclared_params_refusal,
    )

    action = _registry().get("platform_create_task")
    refused = undeclared_params_refusal("platform_create_task", action, {"title": "Call Quay"}, "t", VIA_DIRECT_CALL)
    assert refused.startswith("Missing required params for 'platform_create_task': ['description'], "
                              "so nothing was done.")
    assert 'platform_create_task({"title": "<string>", "description": "<string>"})' in refused
    assert REFUSED_CALL_IS_YOURS in refused
    # The dispatcher checks its required fields itself, before this; its refusal is unchanged.
    assert undeclared_params_refusal("platform_create_task", action, {"title": "Call Quay"}, "t",
                                     VIA_DISPATCHER) is None


def test_a_complete_direct_call_is_not_refused():
    from modules.tools.execution.unified_executor import VIA_DIRECT_CALL, undeclared_params_refusal

    action = _registry().get("platform_update_task_status")
    assert undeclared_params_refusal("platform_update_task_status", action,
                                     {"task_id": "#0422", "status": "cancelled"}, "t", VIA_DIRECT_CALL) is None


def test_only_a_first_class_actions_direct_call_is_held_to_its_required_fields():
    from modules.tools.execution.direct_contract import missing_on_a_direct_call

    unpromoted = next(a for a in _registry().get_all() if not a.promoted and a.parameters.get("required"))
    assert missing_on_a_direct_call(unpromoted.name, unpromoted, {}) is None
    promoted = _registry().get("platform_get_task")
    assert "['task_id']" in missing_on_a_direct_call("platform_get_task", promoted, {})


def test_a_direct_call_missing_a_field_never_reaches_the_action(monkeypatch):
    from modules.tools.execution import unified_executor
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    ran = []

    async def _run(self, action_name, params, **kwargs):
        ran.append(action_name)
        return {"success": True}

    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedToolExecutor, "_execute_platform_action", _run)
    monkeypatch.setattr(unified_executor, "fire_telemetry", lambda **kwargs: None)
    executor = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    executor.composio_actions, executor.db = {}, None
    result = asyncio.run(executor.execute_tool("platform_assign_task", {"agent_name": "Content Creator"},
                                               agent_id=7, workspace_id="ws", trace_id="t-us006"))
    assert result["success"] is False and ran == []
    assert "Missing required params for 'platform_assign_task': ['task_id']" in result["error"]


# ── a refused write is reported refused ─────────────────────────────────────

class _Model:
    def __init__(self, *responses):
        self.queue = list(responses)

    async def __call__(self, messages, tools):
        return self.queue.pop(0)


async def _refuse_the_approval(name, args, call_id, workspace_id):
    return {"success": False, "error": "Missing required params for 'platform_approve_mission': ['mission_id']"}


def _approve_call():
    return {"id": "call_approve", "type": "function",
            "function": {"name": "platform_execute",
                         "arguments": json.dumps({"action": "platform_approve_mission", "params": {}})}}


def test_after_a_refused_write_the_nudge_says_it_was_refused_and_why():
    from modules.tools.execution.tool_loop import ToolLoopExecutor
    from services.brief_facts import REFUSED_WRITE_RULE

    model = _Model(LLMResponse(content="I've approved the mission. It's now running.", tool_calls=None),
                   LLMResponse(content="The approval didn't go through: it needs the mission's id.", tool_calls=None))
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=_refuse_the_approval, max_iterations=5)
    messages = [{"role": "user", "content": "approve the Harbourline mission"}]
    result = asyncio.run(executor.run(initial_response=LLMResponse(content="", tool_calls=[_approve_call()]),
                                      messages=messages, tools=[{"type": "function",
                                                                 "function": {"name": "platform_execute"}}],
                                      workspace_id="ws"))
    nudges = [m["content"] for m in messages
              if m["role"] == "user" and "A write in this turn was refused" in m["content"]]
    assert len(nudges) == 1
    assert ("platform_approve_mission (the tool said: \"Missing required params for 'platform_approve_mission': "
            "['mission_id']\")") in nudges[0]
    assert REFUSED_WRITE_RULE in nudges[0]
    assert "says something was approved" in nudges[0]          # F108's sentence, kept
    assert result.response.content.startswith("The approval didn't go through")


def test_a_claim_with_no_refused_write_keeps_f108s_nudge():
    from modules.tools.execution.nudges import CLAIMED_ACTION_RECOVERY_MSG, claimed_action_nudge

    plain = CLAIMED_ACTION_RECOVERY_MSG.format(claim="approved")
    assert claimed_action_nudge("approved", []) == plain
    # a refused read is not a refused write
    assert claimed_action_nudge("approved", [("platform_get_mission", {}, {"success": False})]) == plain
    # a write refused, then made good by the same action, is done
    later_done = [("platform_create_task", {}, {"success": False, "error": "no description"}),
                  ("platform_create_task", {}, {"success": True})]
    assert claimed_action_nudge("created", later_done) == CLAIMED_ACTION_RECOVERY_MSG.format(claim="created")


def test_the_refused_writes_are_named_with_what_refused_them():
    from modules.tools.execution.nudges import refused_writes

    outcomes = [("platform_update_task_status", {"task_id": "#0422"},
                 {"success": False, "error": "A running ticket can only be cancelled.\nPress Cancel."}),
                ("COMPOSIO_GMAIL_SEND_EMAIL", {}, {"successful": False, "message": "not connected"})]
    assert refused_writes(outcomes) == [
        'platform_update_task_status (the tool said: "A running ticket can only be cancelled. Press Cancel.")',
        'COMPOSIO_GMAIL_SEND_EMAIL (the tool said: "not connected")',
    ]


def test_a_write_held_for_the_owners_click_is_waiting_not_refused():
    """PRD-256 US-004's ask is success False with requires_confirmation: no refusal to report."""
    from modules.tools.execution.nudges import refused_writes

    ask = {"success": False, "requires_confirmation": True, "owner_only": True,
           "message": "Waiting for the owner's click: cancel #0422."}
    assert refused_writes([("platform_update_task_status", {"task_id": "#0422", "status": "cancelled"}, ask)]) == []


def test_autos_prompt_carries_the_same_one_rule():
    from consumers.chatbot.personality import AutomatosPersonality
    from services.brief_facts import AUTO_OWNER_RULES, REFUSED_WRITE_RULE

    assert REFUSED_WRITE_RULE in AUTO_OWNER_RULES
    assert REFUSED_WRITE_RULE in AutomatosPersonality.get_anti_patterns()
    # one rule, one home: the nudge reads it from the prompt's module
    nudges = (ORCH / "modules" / "tools" / "execution" / "nudges.py").read_text()
    assert "from services.brief_facts import REFUSED_WRITE_RULE" in nudges
