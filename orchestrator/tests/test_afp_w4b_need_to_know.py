"""Wave 4b: need-to-know. A fact labelled with an audience is read only by agents whose labels
meet it (PRD-274, "Automatos for products", §5.8).

The rule is held at the durable store by the run's reader (a context variable an agent's run
sets): every read leaves out what the reader may not see, the caches key on the reader, a
labelled agent's writes carry its labels, and consolidation never crosses audiences. A system
agent, and any read with no reader (the owner's pages, exports, jobs), sees everything. Off
unless ``MEMORY_AUDIENCE_ENABLED``. The memory service runs over the eval's in-memory store.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from core.security import memory_audience as ma

FINANCE, OPS = frozenset({"finance"}), frozenset({"ops"})


@pytest.fixture
def audience_on(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "MEMORY_AUDIENCE_ENABLED", True)


def _row(text, audience=None):
    return {"id": text, "memory": text, "metadata": {"audience": audience} if audience is not None else {}}


# --- the rule --------------------------------------------------------------------------------

def test_a_labelled_fact_is_read_only_where_the_labels_meet():
    unlabelled, finance = _row("plan"), _row("payroll", ["Finance "])
    assert ma.visible_to(unlabelled, OPS) and ma.visible_to(unlabelled, frozenset())
    assert ma.visible_to(finance, FINANCE) and ma.visible_to(finance, FINANCE | OPS)
    assert not ma.visible_to(finance, OPS) and not ma.visible_to(finance, frozenset())
    assert ma.visible_to(finance, None)  # a run that reads everything


def test_audiences_are_short_lists_of_labels():
    assert ma.audience_errors(None) == [] and ma.audience_errors(["finance", "ops"]) == []
    for bad in ("finance", [""], ["x" * 65], ["a"] * 11, [3]):
        assert ma.audience_errors(bad), bad


def test_an_agents_reader_is_its_labels_and_a_system_agent_reads_everything(audience_on, monkeypatch):
    from config import config

    finance = SimpleNamespace(is_system_agent=False, configuration={"audience": ["finance"]})
    assert ma.reader_for(finance) == FINANCE
    assert ma.reader_for(SimpleNamespace(is_system_agent=False, configuration={})) == frozenset()
    assert ma.reader_for(SimpleNamespace(is_system_agent=True, configuration={"audience": ["x"]})) is None
    monkeypatch.setattr(config, "MEMORY_AUDIENCE_ENABLED", False)
    assert ma.reader_for(finance) is None


def test_saving_an_agent_or_a_memory_with_a_bad_audience_is_refused():
    from api.agents import _reject_invalid_runtime
    from api.widget_memory import MemoryCreate

    _reject_invalid_runtime({"audience": ["finance"]})
    with pytest.raises(HTTPException):
        _reject_invalid_runtime({"audience": "finance"})
    assert MemoryCreate(content="Q3 payroll runs on the 25th.", metadata={"audience": ["finance"]})
    with pytest.raises(ValidationError):
        MemoryCreate(content="Q3 payroll runs on the 25th.", metadata={"audience": "finance"})


# --- the store, by the run's reader ---------------------------------------------------------------

def _service():
    from evals.memory_pipeline import offline_service

    return offline_service()


async def _seed(service, ws):
    for text, audience in (("The plan renews on 1 March.", None),
                           ("Payroll for Q3 runs on the 25th.", ["finance"]),
                           ("The on-call rota changes on Monday.", ["ops"])):
        stored = await service.store_long_term(workspace_id=ws, content=text,
                                               metadata={"audience": audience} if audience else {})
        assert stored.get("success"), stored


def _contents(rows):
    return sorted(r.get("memory") or r.get("content") for r in rows)


def test_every_read_leaves_out_what_the_reader_may_not_see():
    service, ws = _service(), str(uuid4())

    async def scenario():
        await _seed(service, ws)
        seen = {}
        for name, reader in (("all", None), ("finance", FINANCE), ("unlabelled", frozenset())):
            with ma.memory_reader(reader):
                seen[name] = (_contents(await service.search_long_term(workspace_id=ws, query="payroll rota plan",
                                                                       limit=10)),
                              _contents(await service.get_all_memories(ws, limit=10)))
        return seen

    seen = asyncio.run(scenario())
    everything = ["Payroll for Q3 runs on the 25th.", "The on-call rota changes on Monday.", "The plan renews on 1 March."]
    assert seen["all"][1] == everything
    assert seen["finance"][1] == ["Payroll for Q3 runs on the 25th.", "The plan renews on 1 March."]
    assert seen["unlabelled"][1] == ["The plan renews on 1 March."]
    assert "The on-call rota changes on Monday." not in seen["finance"][0]  # the search too
    assert "Payroll for Q3 runs on the 25th." not in seen["unlabelled"][0]


def test_a_labelled_agents_write_carries_its_labels_unless_it_names_its_own():
    service, ws = _service(), str(uuid4())

    async def scenario():
        with ma.memory_reader(FINANCE):
            await service.store_long_term(workspace_id=ws, content="The auditors arrive on 3 December.")
            await service.store_long_term(workspace_id=ws, content="The office closes on 24 December.",
                                          metadata={"audience": ["everyone-in-ops"]})
        with ma.memory_reader(None):
            await service.store_long_term(workspace_id=ws, content="The plan renews on 1 March.")
            return {r["memory"]: (r.get("metadata") or {}).get("audience") for r in await service.get_all_memories(ws)}

    by_text = asyncio.run(scenario())
    assert by_text["The auditors arrive on 3 December."] == ["finance"]
    assert by_text["The office closes on 24 December."] == ["everyone-in-ops"]
    assert by_text["The plan renews on 1 March."] is None


def test_the_caches_never_serve_one_readers_result_to_another():
    from consumers.chatbot.smart_memory import SmartMemoryManager
    from modules.memory.unified_memory_service import MemoryNamespace

    ns = MemoryNamespace("ws")
    manager = object.__new__(SmartMemoryManager)
    keys = []
    for reader in (None, FINANCE, OPS):
        with ma.memory_reader(reader):
            keys.append((ns.cache_key(None, "h"), manager._get_cache_key("ws", None, "q")))
    assert len({k for k, _ in keys}) == 3 and len({k for _, k in keys}) == 3
    assert keys[0][0] == "mem:cache:ws:global:h"  # no reader: the key it always was


# --- the runs that set the reader -----------------------------------------------------------------

def test_an_agents_task_reads_memory_as_its_agent(monkeypatch):
    seen = []
    monkeypatch.setattr(ma, "reader_for_agent_id", lambda db, agent_id: FINANCE if agent_id == 7 else None)

    @ma.reads_memory_as_its_agent
    async def execute(factory, agent, prompt):
        seen.append(ma.current_reader())

    asyncio.run(execute(SimpleNamespace(db_session=None), 7, "close the month"))
    asyncio.run(execute(SimpleNamespace(db_session=None), SimpleNamespace(agent_id=8), "x"))
    assert seen == [FINANCE, None] and ma.current_reader() is None


def test_the_factory_and_the_chat_turn_set_the_reader():
    import inspect

    from consumers.chatbot.service import StreamingChatService
    from modules.agents.factory.agent_factory import AgentFactory

    assert "memory_reader(reader)" in inspect.getsource(StreamingChatService.stream_response_with_agent)
    wrappers, fn = [], AgentFactory.execute_with_prompt
    while fn is not None:
        wrappers.append(fn.__code__.co_qualname)
        fn = getattr(fn, "__wrapped__", None)
    assert "reads_memory_as_its_agent.<locals>.wrapped" in wrappers


# --- consolidation ---------------------------------------------------------------------------------

def test_consolidation_never_crosses_audiences():
    from modules.memory.operations.contradiction import plan_consolidation

    def fact(i, text, audience=None):
        return {"id": str(i), "memory": text, "created_at": f"2026-11-0{i}T00:00:00Z",
                "metadata": {"importance": 0.6, **({"audience": audience} if audience else {})}}

    same = "Payroll for the third quarter runs on the 25th of the month."
    crossed = plan_consolidation([fact(1, same), fact(2, same, ["finance"])])
    assert not crossed.merges  # the same words, for different readers: both kept
    within = plan_consolidation([fact(1, same, ["finance"]), fact(2, same, ["finance"])])
    assert [len(m.merged_from) for m in within.merges] == [1]


def test_the_memory_page_merges_only_one_audiences_memories():
    @ma.one_audience_only
    async def find(*args):
        return list(args)

    with pytest.raises(ValueError, match="different audiences"):
        asyncio.run(find(_row("a", ["finance"]), _row("b")))
    rows = asyncio.run(find(_row("a", ["finance"]), _row("b", ["finance"])))
    assert ma.shared_audience(rows) == {"audience": ["finance"]} and ma.shared_audience([_row("a")]) == {}
