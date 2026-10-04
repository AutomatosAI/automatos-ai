"""Brand kit at generation (C, night 9b → night 10): a banned word is said plainly, never rewritten.

Night 9b: #1982 (run 3) was right in every fact and said "delightful notes"; #1986 said
"delightful"; Auto's PDF 49c0c2b1 said "our exquisite Guji coffee". The brand voice bans
both, and nothing on the card or in the tool's answer said so. The platform does not
rewrite the owner's words for them: the card's answer keeps the word and ends with a
plain note naming it, once, and an agent's generate_document answer carries the same
note (an API agent reads it in the tool summary). Without a kit nothing is added.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

import api.board_tasks as bt
from core.models.workspaces import Workspace

WS = "6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e60"
KIT = {"name": "Harbourline Coffee Roasters",
       "voice": {"tone": ["warm", "plain", "local"], "banned_phrases": ["delightful", "exquisite"],
                 "sign_off": "Gerard, Harbourline Coffee Roasters"}}
NOTE_1982 = ("Hello club,\n\nThis month: Guji Shakiso and Nariño Buesaco, with delightful notes of blueberry and "
             "caramel, roasted on Tuesday.\n\nGerard, Harbourline Coffee Roasters")
NOTE = 'Check before using this answer: it uses words the brand kit bans: "delightful". Change them before it goes out.'


@pytest.fixture(autouse=True)
def fresh_kits():
    from services.brand_rules import forget_cached_kits

    forget_cached_kits()
    yield
    forget_cached_kits()


class _Task:
    def __init__(self):
        self.id, self.status, self.result, self.error_message = 1982, "in_progress", None, None
        self.completed_at = self.lease_until = None
        self.runtime_ref = None
        self.title, self.description = "Club box note", "85 to 95 words for the October box."


class _Session:
    def __init__(self, task, settings):
        self.task, self.workspace = task, NS(settings=settings)

    def query(self, *_a, **_k):
        return self

    def get(self, model, *_a, **_k):
        return self.workspace if model is Workspace else self.task

    def commit(self):
        pass


@pytest.fixture
def finalize(monkeypatch):
    async def _noop(*_a, **_k):
        return None

    for name in ("_dispatch_task_complete", "_dispatch_task_failed", "_auto_create_task_report"):
        monkeypatch.setattr(bt, name, _noop)
    monkeypatch.setattr("services.result_files.check_named_files", _noop)

    def run(text, settings):
        task = _Task()
        asyncio.run(bt.finalize_board_task_run(_Session(task, settings), task_id=task.id, workspace_id=WS,
                                               agent_id=7, exec_result={"status": "success", "result": text}))
        return task.result
    return run


def test_the_card_keeps_the_word_and_says_the_kit_bans_it(finalize):
    result = finalize(NOTE_1982, {"brand_kit": KIT})
    assert result.startswith(NOTE_1982)                         # not rewritten
    assert result.endswith(NOTE) and result.count("Check before using this answer") == 1


def test_the_note_is_added_once_however_often_the_answer_is_checked():
    from services.brand_rules import on_brand_text

    once = on_brand_text(NOTE_1982, KIT)
    assert on_brand_text(once, KIT) == once


def test_a_word_inside_another_word_is_not_a_banned_word():
    from services.brand_rules import banned_found

    assert banned_found("Delightfully plain, and EXQUISITE.", ["delightful", "exquisite"]) == ["exquisite"]


def test_without_a_kit_the_card_says_nothing_more(finalize):
    assert finalize(NOTE_1982, {}) == NOTE_1982


def test_an_agents_document_with_a_banned_word_comes_back_saying_so():
    from modules.tools.execution import exec_document
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    made = {"success": True, "results": [{"status": "success", "filename": "club-box.pdf", "format": "pdf",
                                          "download_url": "/api/documents/generated/club-box.pdf", "size_kb": 12}]}

    async def execute_tool(**_kwargs):
        return made

    db = NS(get=lambda model, key: NS(settings={"brand_kit": KIT}), query=lambda *a, **k: None)
    executor = NS(db=db, platform_tools=NS(execute_tool=execute_tool))
    params = {"title": "Two Bags of Guji!", "format": "pdf",
              "data": {"sections": [{"title": "October", "content": "Our exquisite Guji coffee."}]}}
    result = asyncio.run(exec_document.execute_generate_document(executor, "generate_document", params, 7,
                                                                 workspace_id=WS))
    said = result["results"][0]["brand_check"]
    assert '"exquisite"' in said and said.startswith("Check before using this answer")
    assert said in ToolResultFormatter.format_for_llm(result, "generate_document")

    plain = {**params, "data": {"sections": [{"title": "October", "content": "Our Guji coffee."}]}}
    clean = asyncio.run(exec_document.execute_generate_document(executor, "generate_document", plain, 7,
                                                                workspace_id=WS))
    assert "brand_check" not in clean["results"][0]
