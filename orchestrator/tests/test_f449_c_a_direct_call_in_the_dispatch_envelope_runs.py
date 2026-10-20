"""P256-FIX-N17-10 (F449, night 17): a direct call sent in platform_execute's envelope runs.

17 direct platform_* calls came as ``tool({"action": tool, "params": {...}})`` (asks 14, 75×2, 89, 143,
163, 197, 217, 245, 265, 285, 293×3, 364, 369, 403), each refused as taking neither ``action`` nor
``params``, then retried. Such a call now runs with the object inside, when the action declares neither
key and takes every key inside it (``schema_reads.envelope_opened``). platform_execute is untouched:
F182's night-6 refusal of a playbook's inputs nested under ``params`` stands. The calls below are shaped
like the night's (its data stays local), one per tool the night sent so.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
GULL = {"usual_order": "6 kg of Harbour Blend and 2 kg of decaf a week", "draft_only": True,
        "contact_person": "Priya Shah", "cafe_name": "Gull & Anchor"}
# (the tool, what the envelope held, what the action runs with)
ENVELOPED = [
    ("platform_get_task", {"task_id": "#1751"}, {"task_id": "#1751"}),
    ("platform_board_snapshot", {"limit": 20}, {"limit": 20}),
    ("platform_board_summary", {}, {}),
    ("platform_list_tasks", {"status": "review", "search": "Guji quote"}, {"status": "review", "search": "Guji quote"}),
    ("platform_read_document", {"document": "Tidewater quote.pdf"}, {"filename": "Tidewater quote.pdf"}),  # ask 75
    ("platform_read_document", {"file_name": "Guji quote.pdf"}, {"filename": "Guji quote.pdf"}),
    ("platform_read_document", {"document_id": 412, "page": 1}, {"document_id": 412, "page": 1}),
    ("platform_recommend_agent", {"request": "Make the Tidewater quote shorter"},
     {"objective": "Make the Tidewater quote shorter"}),
    ("platform_recommend_agent", {"query": "Who writes our quotes?"}, {"objective": "Who writes our quotes?"}),
    ("platform_grep_documents", {"pattern": "fortnightly", "max_results": 20},
     {"pattern": "fortnightly", "limit": 20}),                                                    # ask 364
    ("platform_search_documents", {"query": "Guji quote"}, {"query": "Guji quote"}),
    ("platform_list_documents", {"search": "quote"}, {"search": "quote"}),
    ("platform_list_agents", {}, {}),
    ("platform_get_agent", {"agent_name": "WRITER"}, {"agent_name": "WRITER"}),
    ("platform_my_actions", {"window": "tonight"}, {"window": "tonight"}),
    ("platform_list_social_posts", {"status": "draft"}, {"status": "draft"}),
    ("platform_update_task_status", {"task_id": "#1751", "status": "assigned", "note": "8kg fortnightly instead"},
     {"task_id": "#1751", "status": "assigned", "note": "8kg fortnightly instead"}),
]


@pytest.fixture
def direct(monkeypatch):
    """F182's direct-call fixture: the platform action recorded, never run."""
    from modules.tools.execution import unified_executor
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    ran = []

    async def _run(self, action_name, params, **kwargs):
        ran.append((action_name, params))
        return {"success": True}

    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedToolExecutor, "_execute_platform_action", _run)
    monkeypatch.setattr(unified_executor, "fire_telemetry", lambda **kwargs: None)
    executor = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    executor.composio_actions, executor.db = {}, None

    def call(tool, params):
        return asyncio.run(executor.execute_tool(tool, params, agent_id=322, workspace_id=WS, trace_id="t-f449"))
    return SimpleNamespace(call=call, ran=ran)


def _without_server_keys(params):
    return {key: value for key, value in params.items() if not key.startswith("_")}


@pytest.mark.parametrize("tool, inner, runs", ENVELOPED, ids=[f"{t}-{i}" for i, (t, _p, _r) in enumerate(ENVELOPED)])
def test_a_direct_call_in_the_envelope_runs_with_what_it_holds(direct, tool, inner, runs, caplog):
    with caplog.at_level(logging.INFO, logger="modules.tools.execution.schema_reads"):
        result = direct.call(tool, {"action": tool, "params": inner})

    assert result["success"] is True, result
    ((ran_as, params),) = direct.ran
    assert ran_as == tool and _without_server_keys(params) == runs and "action" not in params
    assert any(r.getMessage().startswith(f"[F449] {tool} came in platform_execute's envelope") for r in caplog.records)


def test_the_envelope_text_and_the_name_without_its_prefix_open_too(direct):
    direct.call("platform_get_task", {"action": "get_task", "params": '{"task_id": "#1751"}'})
    assert direct.ran == [("platform_get_task", {"task_id": "#1751"})]


def test_the_document_key_is_what_makes_ask_75_run(direct, monkeypatch):
    """Without the KEYS_SENT entry the call is still refused, as it was on the night."""
    from modules.tools.execution import schema_reads

    trimmed = {action: keys for action, keys in schema_reads.KEYS_SENT.items() if action != schema_reads.READ_DOCUMENT}
    monkeypatch.setattr(schema_reads, "KEYS_SENT", trimmed)

    enveloped = direct.call("platform_read_document", {"action": "platform_read_document",
                                                       "params": {"document": "Tidewater quote.pdf"}})
    dispatched = direct.call("platform_execute", {"action": "platform_read_document",
                                                  "params": {"document": "Tidewater quote.pdf"}})

    assert enveloped["success"] is False and dispatched["success"] is False and direct.ran == []
    assert "'platform_read_document' does not take ['document']" in dispatched["error"]


@pytest.mark.parametrize("sent", [
    {"params": GULL},                                                         # no action: not the envelope
    {"action": "platform_execute_playbook", "params": GULL},                  # keys the action does not take
    {"action": "platform_execute_playbook", "params": GULL, "playbook_id": 102},   # a key beside the envelope
    {"action": "platform_get_task", "params": {"input_data": GULL}},          # another action's name
], ids=["no-action", "untaken-keys", "beside", "another-action"])
def test_an_envelope_that_is_not_this_calls_stays_refused(direct, sent):
    result = direct.call("platform_execute_playbook", sent)
    assert result["success"] is False and direct.ran == []


def test_an_action_that_declares_params_is_never_unwrapped(monkeypatch):
    from modules.tools.execution import schema_reads

    takes_params = SimpleNamespace(name="platform_takes_params", accepts=(), promoted=False,
                                   parameters={"properties": {"params": {"type": "object"}, "x": {"type": "string"}}})
    monkeypatch.setattr(schema_reads, "_action", lambda name: takes_params)
    sent = {"action": "platform_takes_params", "params": {"x": "y"}}

    assert schema_reads.envelope_opened("platform_takes_params", sent) is sent


def test_platform_execute_still_refuses_inputs_nested_under_params(direct):
    """F182, night 6: run 207 started with no inputs; the dispatcher's refusal is untouched."""
    result = direct.call("platform_execute", {"action": "platform_execute_playbook",
                                              "params": {"params": GULL, "playbook_id": 102}})

    assert result["success"] is False and direct.ran == []
    assert "'platform_execute_playbook' does not take ['params']" in result["error"]
