"""A widened memory search reads only namespaces of the searcher's workspace (CodeRabbit on #1434).

``DurableMemoryStore.search(also_namespaces=…)`` reads more namespaces in one query, so an agent's
chat turn reads its workspace's facts too. The only caller passes its own workspace's namespace,
but nothing held it there: a namespace of another workspace would have been searched. Tenant
isolation is a hard rule, so the filter refuses any namespace that is not of the same workspace.
"""
from __future__ import annotations

import asyncio

import pytest

from modules.memory.durable_store import DurableMemoryStore

AGENT = "mem:ws-acme:agent:7"


def test_the_workspaces_own_namespaces_widen_the_search():
    flt = DurableMemoryStore._namespaces_filter(AGENT, ("mem:ws-acme", "mem:ws-acme:daily"))
    assert flt.must[0].match.any == [AGENT, "mem:ws-acme", "mem:ws-acme:daily"]
    assert DurableMemoryStore._namespaces_filter(AGENT, ()).must[0].match.value == AGENT


@pytest.mark.parametrize("stranger", ["mem:ws-globex", "mem:ws-globex:agent:7", "not-a-namespace", ""])
def test_another_workspaces_namespace_is_refused(stranger):
    with pytest.raises(ValueError, match="workspace"):
        DurableMemoryStore._namespaces_filter(AGENT, ("mem:ws-acme", stranger))


def test_a_refused_widening_never_reaches_the_store():
    store = object.__new__(DurableMemoryStore)
    store._enabled, store._bootstrap_done = True, True
    queried = []

    async def _query(query, flt, limit):
        queried.append(flt)
        return []

    store._query = _query
    store._record_search = lambda *args, **kwargs: None
    with pytest.raises(ValueError):
        asyncio.run(store.search("payroll", AGENT, also_namespaces=("mem:ws-globex",)))
    assert queried == []
