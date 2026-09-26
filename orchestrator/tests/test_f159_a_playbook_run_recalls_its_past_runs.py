"""F159: a playbook run recalls its playbook's past runs.

Before its first step, a run asks PlaybookMemoryService.retrieve_relevant_memories
for what earlier runs of the playbook learned, and step 1's prompt carries it
("Learnings from Previous Runs"). The call passed recipe_id=, but the method
takes playbook_id. It raised TypeError on every run, the except logged it at
INFO as "Mem0 memory retrieval skipped", and no run ever recalled anything.
The call now passes playbook_id, and a recall that really fails is logged as a
failure (WARNING, with the error). A run a widget turn started still recalls
nothing (F155).
"""
from __future__ import annotations

import logging

from tests import helpers_playbook_run as helper

LEARNED = {"summary": "Last run: the club list had two duplicate members.", "total_memories": 1}


def test_step_one_is_given_what_earlier_runs_learned(monkeypatch):
    asked = []

    async def _recall(self, playbook_id, context=None):
        asked.append(playbook_id)
        return LEARNED

    monkeypatch.setattr(helper._Memory, "retrieve_relevant_memories", _recall)
    calls = []
    helper.run_playbook(monkeypatch, outcomes=[helper.done("Listed."), helper.done("Sent.")], step_seconds=1,
                        exec_config={}, calls=calls)
    assert asked == [79]
    assert [call.get("recipe_memories") for call in calls] == [LEARNED, None]


def test_a_recall_that_fails_is_logged_as_a_failure(monkeypatch, caplog):
    async def _recall(self, playbook_id, context=None):
        raise RuntimeError("the durable memory store is unreachable")

    monkeypatch.setattr(helper._Memory, "retrieve_relevant_memories", _recall)
    with caplog.at_level(logging.WARNING, logger="api.recipe_executor"):
        execution, _card = helper.run_playbook(monkeypatch, outcomes=[helper.done("Listed."), helper.done("Sent.")],
                                               step_seconds=1, exec_config={})
    assert any(record.levelno == logging.WARNING and "durable memory store is unreachable" in record.getMessage()
               for record in caplog.records)
    assert execution.status == "completed"  # a failed recall never stops the run


def test_a_recall_that_names_no_workspace_recalls_nothing(db_session, seed_workspace, monkeypatch):
    """The memories are the calling run's workspace's. Guessing the workspace
    from the playbook row could recall another workspace's runs (a shared
    playbook has none, or not the caller's)."""
    import asyncio
    from uuid import UUID

    from core.models.core import WorkflowTemplate
    from core.services import playbook_memory_service as memory

    searched = []

    async def _search(**kwargs):
        searched.append(kwargs.get("workspace_id"))
        return []

    fake = type("Unified", (), {"search_long_term_scoped": staticmethod(_search)})()
    monkeypatch.setattr(memory, "get_unified_memory_service", lambda: fake)
    ws = UUID(seed_workspace())
    playbook = WorkflowTemplate(template_id="f159-club-newsletter", name="Club newsletter", description="d",
                                workspace_id=ws, template_definition={"steps": []}, steps=[], created_by="f159")
    db_session.add(playbook)
    db_session.flush()
    service = memory.PlaybookMemoryService(db=db_session)

    none = asyncio.run(service.retrieve_relevant_memories(playbook_id=playbook.id, context={}))
    assert (none["total_memories"], searched) == (0, [])
    asyncio.run(service.retrieve_relevant_memories(playbook_id=playbook.id, context={"workspace_id": str(ws)}))
    assert searched == [str(ws)]
