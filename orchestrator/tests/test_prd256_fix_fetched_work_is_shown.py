"""PRD-256 P256-FIX-N13-6 (F413, night 13): fetched work is shown in the reply.

"show me the Guji blog draft" ran platform_get_deliverable, the receipt said done, and the
reply held none of the draft (12 times on Haiku 5.5). Now, in the owner's chat, the fetch's
answer says first that its text is the owner's to see, cut at DELIVERABLE_SHOWN_MAX_CHARS
with the link to the rest; and a final reply that quotes none of a text the owner asked to
see is nudged once (modules/tools/execution/fetched_work.shows_fetched_work, on the loop's
final-reply check). An agent's run reads the whole text with no note, as before.
"""
from __future__ import annotations

import asyncio
import json
from urllib.parse import urlparse

import pytest

from core.llm.clients.base import LLMResponse
from core.llm.usage_context import LANE_CHAT, usage_scope
from modules.tools.discovery import handlers_deliverables
from modules.tools.execution import fetched_work
from modules.tools.execution.nudges import is_nudge
from modules.tools.execution.tool_loop import ToolLoopExecutor
from tests.test_prd256_fix_waiting_receipt import _chats, _routed

GET = "platform_get_deliverable"
DELIVERABLE_ID = "5b0c1f3e-0d4f-4a52-9a43-3c8f7d2e9a10"
TITLE = "Guji blog draft"
DRAFT = (
    "# Guji: a harvest worth the wait\n\n"
    "This month's box opens with a Guji from the Hambela slopes at 2,100 metres.\n"
    "Expect bergamot and white peach up front, with a black-tea finish that lingers.\n"
)
SHOW = "show me the Guji blog draft"
DESCRIBED = "Your Guji post covers the harvest, the tasting notes and the finish."
QUOTED = "Here it is:\n\n" + DRAFT
TOOLS = [{"type": "function", "function": {"name": "platform_execute"}}]


class _Service:
    """DeliverableService as the handler reads it: one deliverable, its text read from the workspace."""

    text = DRAFT

    def __init__(self, db, workspace_id):
        self.workspace_id = workspace_id

    async def get_deliverable(self, deliverable_id, include_content=False):
        row = {"id": deliverable_id, "title": TITLE, "artifact_type": "report", "source_type": "agent_output",
               "file_path": "reports/guji-blog-draft.md", "status": "draft"}
        content = {"content": self.text, "content_truncated": False} if include_content else {}
        return {"success": True, "deliverable": {**row, **content}}


@pytest.fixture(autouse=True)
def _workspace(monkeypatch):
    import services.deliverable_service as service
    from config import config

    monkeypatch.setattr(service, "DeliverableService", _Service)
    monkeypatch.setattr(_Service, "text", DRAFT)
    monkeypatch.setattr(config, "FRONTEND_URL", "https://app.example.test")


def _fetch(include_content=True, *, lane=LANE_CHAT):
    """The real handler's answer, in ``lane`` (the chat books its whole turn to LANE_CHAT)."""
    params = {"deliverable_id": DELIVERABLE_ID, "include_content": include_content}
    with usage_scope(request_type=lane):
        return asyncio.run(handlers_deliverables.get_deliverable(None, "ws", params))


def _get_call(call_id="call_get", include_content=True):
    args = {"action": GET, "params": {"deliverable_id": DELIVERABLE_ID, "include_content": include_content}}
    return {"id": call_id, "type": "function", "function": {"name": "platform_execute", "arguments": json.dumps(args)}}


class _Model:
    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = 0

    async def __call__(self, messages, tools):
        self.calls += 1
        return self.queue.pop(0)


def _run(*replies, said=SHOW, promises=True, envelopes=None, first=None):
    """The loop over the chat-shaped result of the fetch (built before the loop: ``_routed`` runs
    its own event loop), from a first response that makes the fetch."""
    queue = list(envelopes or [_chats(_routed(_fetch(), tool=GET))])
    model = _Model(*replies)

    async def tools(name, args, call_id, workspace_id):
        return {**queue.pop(0)}

    executor = ToolLoopExecutor(llm_callback=model, tool_callback=tools, max_iterations=5, promises=promises)
    messages = [{"role": "user", "content": said}]
    with usage_scope(request_type=LANE_CHAT if promises else "agent"):
        result = asyncio.run(executor.run(initial_response=first or LLMResponse(content="", tool_calls=[_get_call()]),
                                          messages=messages, tools=TOOLS, workspace_id="ws"))
    return result, model, [m["content"] for m in messages if is_nudge(m)]


# ── the tool's answer: the note first, the text, cut with the link ───────────────────

def test_the_fetch_tells_the_model_the_text_is_the_owners_to_see_before_the_text():
    found = _fetch()["deliverable"]
    assert found["content"] == DRAFT and found["show_the_owner"] == handlers_deliverables.SHOW_IT_NOTE
    assert list(found).index("show_the_owner") < list(found).index("content")

    seen = _routed(_fetch(), tool=GET)["llm_context"]       # what the model reads
    assert "The owner sees none of this text until you put it in your reply" in seen
    assert seen.index("The owner sees none") < seen.index("Hambela slopes")


def test_a_long_file_comes_back_cut_at_the_configured_length_with_the_link(monkeypatch):
    from config import config
    from modules.tools.execution.generate_document_tool import deliverable_open_url

    monkeypatch.setattr(config, "DELIVERABLE_SHOWN_MAX_CHARS", 60)
    monkeypatch.setattr(_Service, "text", DRAFT * 20)
    found = _fetch()["deliverable"]
    link = deliverable_open_url(DELIVERABLE_ID)

    assert found["content"] == (DRAFT * 20)[:60]
    assert (found["content_truncated"], found["content_chars"], found["rest_at"]) == (True, len(DRAFT * 20), link)
    assert f"This is its first 60 of {len(DRAFT * 20):,} characters" in found["show_the_owner"]
    assert found["show_the_owner"].endswith(link)
    opened = urlparse(link)
    assert (opened.scheme, opened.hostname) == ("https", "app.example.test") and f"deliverable={DELIVERABLE_ID}" in opened.query


def test_an_agents_run_reads_the_whole_text_with_no_note(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "DELIVERABLE_SHOWN_MAX_CHARS", 60)
    found = _fetch(lane="agent")["deliverable"]
    assert found["content"] == DRAFT and found["content_truncated"] is False
    assert "show_the_owner" not in found and "rest_at" not in found


def test_details_only_carry_no_text_and_no_note():
    found = _fetch(include_content=False)["deliverable"]
    assert "content" not in found and "show_the_owner" not in found


# ── the reply: it holds the text ───────────────────────────────────────────────────

def test_show_me_the_draft_the_reply_holds_the_text():
    result, model, nudges = _run(LLMResponse(content=QUOTED, tool_calls=None))
    assert nudges == [] and model.calls == 1
    assert "Hambela slopes at 2,100 metres" in result.response.content


def test_a_reply_that_quotes_none_of_it_is_retried_once_then_holds_the_text():
    result, model, nudges = _run(LLMResponse(content=DESCRIBED, tool_calls=None),
                                 LLMResponse(content=QUOTED, tool_calls=None))
    assert model.calls == 2 and len(nudges) == 1
    assert f'The owner asked to see "{TITLE}" and you fetched it, but your reply shows none of its text' in nudges[0]
    assert "Hambela slopes at 2,100 metres" in result.response.content


def test_the_retry_is_asked_once_only():
    result, model, nudges = _run(LLMResponse(content=DESCRIBED, tool_calls=None),
                                 LLMResponse(content="It is a lovely draft.", tool_calls=None))
    assert model.calls == 2 and len(nudges) == 1
    assert result.response.content == "It is a lovely draft."


def test_a_long_file_the_reply_holds_the_first_n_characters_and_the_link(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "DELIVERABLE_SHOWN_MAX_CHARS", 60)
    monkeypatch.setattr(_Service, "text", DRAFT * 20)
    found = _fetch()["deliverable"]
    shown, link = found["content"], found["rest_at"]
    result, model, nudges = _run(LLMResponse(content=DESCRIBED, tool_calls=None),
                                 LLMResponse(content=f"{shown}\n\nThe rest: {link}", tool_calls=None))

    assert len(nudges) == 1 and f"its first 60 characters, then the link to the rest ({link})" in nudges[0]
    assert shown in result.response.content and link in result.response.content


def test_a_fetch_of_its_details_only_is_asked_for_the_text_and_the_retry_fetches_it():
    details = _chats(_routed(_fetch(include_content=False), tool=GET))
    text = _chats(_routed(_fetch(), tool=GET))
    result, model, nudges = _run(LLMResponse(content=DESCRIBED, tool_calls=None),
                                 LLMResponse(content="", tool_calls=[_get_call("call_text")]),
                                 LLMResponse(content=QUOTED, tool_calls=None),
                                 envelopes=[details, text], first=LLMResponse(content="", tool_calls=[
                                     _get_call(include_content=False)]))

    assert len(nudges) == 1 and "fetched only its details, not its text" in nudges[0]
    assert "include_content=true" in nudges[0]
    assert "Hambela slopes at 2,100 metres" in result.response.content


# ── what is left alone ─────────────────────────────────────────────────────────────

def test_an_ask_for_something_else_from_it_is_not_nudged():
    result, model, nudges = _run(LLMResponse(content="Two points: the altitude and the finish.", tool_calls=None),
                                 said="summarise the Guji blog draft in two bullets")
    assert nudges == [] and model.calls == 1


def test_an_agents_run_is_not_nudged():
    result, model, nudges = _run(LLMResponse(content=DESCRIBED, tool_calls=None), promises=False)
    assert nudges == [] and model.calls == 1


def test_a_reformatted_quote_counts_as_shown():
    reformatted = "**Guji: a harvest worth the wait**\n\n> This month's box opens with a Guji"
    assert fetched_work.quotes_any(reformatted, DRAFT)
    assert not fetched_work.quotes_any(DESCRIBED, DRAFT)
    assert fetched_work.quotes_any("anything", "\n---\n")                  # nothing in it to show


def test_the_owners_words_are_the_persons_never_a_nudge():
    from modules.tools.execution.nudges import nudge

    messages = [{"role": "user", "content": SHOW}, {"role": "assistant", "content": DESCRIBED}, nudge("check")]
    assert fetched_work.owners_words(messages) == SHOW
    assert fetched_work.owners_words([{"role": "user", "content": [{"type": "text", "text": SHOW}]}]) == SHOW
