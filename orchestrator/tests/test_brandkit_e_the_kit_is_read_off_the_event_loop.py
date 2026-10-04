"""Brand kit at generation (E): a run's kit read never waits for a connection on the event loop.

F105 and F330: a sync pool wait on the event loop froze the whole process. The brand kit
is read where a card's answer is finished, where a playbook step runs, where a document
is signed and where generate_document names the banned words, all inside coroutines on
the loop. Each read now runs on a worker thread; a fresh cached read needs no thread.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace as NS

import pytest

from core.models.workspaces import Workspace

WS = "6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e61"
SIGNED = "Gerard, Harbourline Coffee Roasters"
KIT = {"name": "Harbourline Coffee Roasters",   # 3 to 5 tone words, or the read keeps the default voice
       "voice": {"tone": ["warm", "plain", "local"], "banned_phrases": ["exquisite"], "sign_off": SIGNED}}


@pytest.fixture(autouse=True)
def fresh_kits():
    from services.brand_rules import forget_cached_kits

    forget_cached_kits()
    yield
    forget_cached_kits()


class _Session:
    """A session that records the thread of every read."""

    def __init__(self):
        self.read_on: list = []

    def get(self, model, *_a, **_k):
        self.read_on.append(threading.get_ident())
        return NS(settings={"brand_kit": KIT}) if model is Workspace else None

    def query(self, *_a, **_k):
        self.read_on.append(threading.get_ident())
        return NS(filter=lambda *_a: NS(first=lambda: NS(workspace_id=WS)))


def _on_the_loop(coro_factory):
    """Run a coroutine; return its result and the loop's thread."""
    async def main():
        return await coro_factory(), threading.get_ident()
    return asyncio.run(main())


def test_a_cards_answer_reads_the_kit_off_the_loop():
    from services.brand_hooks import a_cards_answer_is_on_brand

    seen = {}

    @a_cards_answer_is_on_brand
    async def finalize(db, **kwargs):
        seen.update(kwargs)

    db = _Session()
    _, loop_thread = _on_the_loop(lambda: finalize(db, workspace_id=WS,
                                                   exec_result={"status": "success", "result": "Best,\n[Your name]"}))
    assert seen["exec_result"]["result"].endswith(SIGNED)
    assert db.read_on and loop_thread not in db.read_on


def test_a_playbook_step_reads_the_kit_and_the_agent_off_the_loop(monkeypatch):
    import services.brand_hooks as bh

    asked_on = []
    monkeypatch.setattr(bh, "_runs_in_a_session", lambda db, agent_id: asked_on.append(threading.get_ident()))

    @bh.a_playbook_step_is_on_brand
    async def execute_step(**kwargs):
        return {"status": "success", "result": f"{kwargs['clean_prompt']}\n\nBest,\n[Your name]"}

    db = _Session()
    result, loop_thread = _on_the_loop(lambda: execute_step(db=db, workspace_id=WS, agent=NS(id=7),
                                                            clean_prompt="Write the club note."))
    assert "## The brand's rules" in result["result"] and result["result"].endswith(SIGNED)
    assert asked_on and loop_thread not in asked_on
    assert db.read_on and loop_thread not in db.read_on


def test_generate_document_reads_the_agents_workspace_and_kit_off_the_loop():
    from modules.tools.execution.document_brand_check import a_documents_banned_words_are_said

    @a_documents_banned_words_are_said
    async def execute(executor, tool_name, parameters, agent_id, workspace_id=None, trace_id=None):
        return {"success": True, "results": [{"filename": "box.pdf"}]}

    db = _Session()
    params = {"title": "October", "data": {"body": "Our exquisite Guji."}}
    result, loop_thread = _on_the_loop(lambda: execute(NS(db=db), "generate_document", params, 7))
    assert '"exquisite"' in result["results"][0]["brand_check"]
    assert len(db.read_on) == 2 and loop_thread not in db.read_on   # the agent, then the kit


def test_a_fresh_cached_kit_is_read_without_a_thread(monkeypatch):
    from services import brand_rules as br

    db = _Session()
    br.stored_kit(db, WS)                      # warms the cache
    hops = []
    monkeypatch.setattr(br.asyncio, "to_thread", lambda *a, **k: hops.append(a))
    kit = asyncio.run(br.kit_off_loop(db, WS))
    assert kit and kit["name"] == KIT["name"] and hops == [] and len(db.read_on) == 1


def test_a_kit_read_never_flushes_the_callers_pending_changes():
    from contextlib import contextmanager

    from services import brand_rules as br

    class _Pending(_Session):
        def __init__(self):
            super().__init__()
            self.flush_held, self.read_while_held = False, []

        @property
        @contextmanager
        def no_autoflush(self):
            self.flush_held = True
            try:
                yield self
            finally:
                self.flush_held = False

        def get(self, model, *a, **k):
            self.read_while_held.append(self.flush_held)
            return super().get(model, *a, **k)

    db = _Pending()
    assert br.stored_kit(db, WS)["voice"]["sign_off"] == SIGNED
    assert db.read_while_held == [True]
