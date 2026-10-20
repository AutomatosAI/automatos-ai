"""P256-FIX-N17-10 (F449, night 17): the names Auto reaches for are read as the parameter they mean.

platform_read_document's ``file_name`` was refused 15 times; platform_recommend_agent got ``{request}`` or
``{query}`` 7 times, because the hand-off note and Auto's assign directive said "call
platform_recommend_agent with the request" while it takes only ``objective``; platform_my_actions got
``window: '1d'`` 11 times and ``'7d'`` 4 times. Each is read now (``schema_reads.KEYS_SENT``, through
platform_execute and directly; ``my_actions_window.WINDOWS_SAID``), and the two prompts name ``objective``.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
# (the action, what Auto sent, what the action runs with)
SENT = [
    ("platform_read_document", {"file_name": "Guji quote.pdf"}, {"filename": "Guji quote.pdf"}),
    ("platform_read_document", {"name": "Guji quote.pdf"}, {"filename": "Guji quote.pdf"}),
    ("platform_read_document", {"document": "Guji quote.pdf", "page": 2}, {"filename": "Guji quote.pdf", "page": 2}),
    ("platform_recommend_agent", {"request": "Shorten the Tidewater quote"}, {"objective": "Shorten the Tidewater quote"}),
    ("platform_recommend_agent", {"query": "Who writes quotes?"}, {"objective": "Who writes quotes?"}),
    ("platform_grep_documents", {"pattern": "8kg", "max_results": 5}, {"pattern": "8kg", "limit": 5}),
]


@pytest.fixture
def executor(monkeypatch):
    from modules.tools.execution import unified_executor
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    ran = []

    async def _run(self, action_name, params, **kwargs):
        ran.append((action_name, {k: v for k, v in params.items() if not k.startswith("_")}))
        return {"success": True}

    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedToolExecutor, "_execute_platform_action", _run)
    monkeypatch.setattr(unified_executor, "fire_telemetry", lambda **kwargs: None)
    made = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    made.composio_actions, made.db = {}, None

    def call(tool, params):
        return asyncio.run(made.execute_tool(tool, params, agent_id=322, workspace_id=WS, trace_id="t-f449"))
    return SimpleNamespace(call=call, ran=ran)


@pytest.mark.parametrize("action, sent, runs", SENT, ids=[f"{a}-{next(iter(s))}" for a, s, _r in SENT])
def test_the_key_is_read_through_the_dispatcher_and_directly(executor, action, sent, runs):
    dispatched = executor.call("platform_execute", {"action": action, "params": sent})
    direct = executor.call(action, sent)

    assert dispatched["success"] is True and direct["success"] is True, (dispatched, direct)
    assert executor.ran == [(action, runs), (action, runs)]


def test_a_key_the_call_also_sent_by_its_own_name_wins():
    from modules.tools.execution.schema_reads import keys_read

    assert keys_read("platform_read_document", {"filename": "a.pdf", "file_name": "b.pdf"}) == {
        "filename": "a.pdf", "file_name": "b.pdf"}


@pytest.mark.parametrize("said, window", [("1d", "in the last 24 hours"), ("24h", "in the last 24 hours"),
                                          ("7d", "this week"), ("week", "this week"), ("1D", "in the last 24 hours")])
def test_a_window_said_short_is_the_window_it_means(said, window):
    from modules.tools.discovery.my_actions_window import window_of

    found, refused = window_of({"window": said})
    assert refused is None and found.said == window


def test_a_window_no_one_names_is_still_refused():
    from modules.tools.discovery.my_actions_window import BAD_WINDOW, WINDOWS, window_of

    assert window_of({"window": "fortnight"}) == (None, BAD_WINDOW.format(windows=", ".join(WINDOWS)))


def _team_note():
    from consumers.chatbot.handoffs import TEAM_NOTE

    return TEAM_NOTE.format(article="a", kind="quote")


def test_the_hand_off_note_and_the_directive_name_the_objective():
    from consumers.chatbot.auto import build_assign_directive

    ask = build_assign_directive(target_agent_name=None, resolved=False, deferred=False)
    for said in (_team_note(), ask):
        assert "platform_recommend_agent {objective: the work in one or two sentences}" in said
        assert "recommend_agent with the request" not in said
    assert _team_note().startswith("The owner asked for a") and "platform_create_task" in _team_note()
