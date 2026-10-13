"""PRD-256 P256-FIX-N14-6 (F420, night 14): a file Auto reads for the owner is shown in the reply.

Night 14 (12–13 Oct, Auto on Haiku 5.5): 3 of 3 "show me the file" misses, one with the
receipt workspace_read_file · guji-shakiso-natural-blog-draft.md · done and no text. N13-6
covered platform_get_deliverable only. Now one rule (modules/tools/execution/shown_reads)
covers every read the owner may ask to see, workspace_read_file, platform_read_document and
platform_get_deliverable: in the owner's chat the read's answer says first that its text is
the owner's to see, cut at DELIVERABLE_SHOWN_MAX_CHARS with the link to the rest; and a reply
that quotes none of a text the owner asked to see is nudged once (fetched_work, the FX-017
retry seam). An agent's run reads the whole file with no note, as before.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import pytest

from core.llm.clients.base import LLMResponse
from core.llm.usage_context import LANE_CHAT, usage_scope
from modules.tools.discovery import handlers_deliverables, handlers_documents
from modules.tools.execution import exec_workspace, fetched_work, shown_reads
from modules.tools.execution.nudges import is_nudge
from modules.tools.execution.tool_loop import ToolLoopExecutor
from tests.test_f327_a_read_document_reads_exactly_the_document_asked_for import CLUB_BOX, db, shelf  # noqa: F401
from tests.test_prd256_fix_waiting_receipt import _chats, _routed

READ = "workspace_read_file"
PATH = "drafts/guji-shakiso-natural-blog-draft.md"
NAME = "guji-shakiso-natural-blog-draft.md"
DRAFT = (
    "# Guji Shakiso Natural: a harvest worth the wait\n\n"
    "This month's box opens with a natural Guji from the Shakiso slopes at 2,100 metres.\n"
    "Expect blueberry and jasmine up front, with a dark-chocolate finish that lingers.\n"
)
SHOW = "show me the Guji blog draft"
DESCRIBED = "Your Guji post covers the harvest, the tasting notes and the finish."
QUOTED = "Here it is:\n\n" + DRAFT
# What the worker holds at PATH; a test that needs a long file sets it through monkeypatch.
_FILE = {"text": DRAFT}
TOOLS = [{"type": "function", "function": {"name": READ}}, {"type": "function", "function": {"name": "platform_execute"}}]


@pytest.fixture(autouse=True)
def _worker(monkeypatch):
    """The workspace worker as the client reads it: the Guji draft at PATH."""
    import core.workspace_client as wc
    from config import config

    async def read_file(self, path):
        text = _FILE["text"]
        return {"path": path, "name": path.rsplit("/", 1)[-1], "content": text, "size": len(text),
                "language": "markdown", "mime_type": "text/markdown", "success": True}

    monkeypatch.setitem(_FILE, "text", DRAFT)
    monkeypatch.setattr(wc.WorkspaceClient, "read_file", read_file)
    monkeypatch.setattr(config, "FRONTEND_URL", "https://app.example.test")


def _read(*, lane=LANE_CHAT, by_keyword=False):
    """The real workspace executor's answer to a read of PATH, in ``lane``."""
    async def go():
        if by_keyword:
            return await exec_workspace.execute_workspace_action(
                executor=None, tool_name=READ, parameters={"path": PATH}, workspace_id=uuid4())
        return await exec_workspace.execute_workspace_action(None, READ, {"path": PATH}, workspace_id=uuid4())

    with usage_scope(request_type=lane):
        return asyncio.run(go())


def _read_call(call_id="call_read"):
    return {"id": call_id, "type": "function", "function": {"name": READ, "arguments": json.dumps({"path": PATH})}}


class _Model:
    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = 0

    async def __call__(self, messages, tools):
        self.calls += 1
        return self.queue.pop(0)


def _run(*replies, said=SHOW, promises=True, envelope=None, first=None):
    """The loop over the chat-shaped result of the read (built before the loop: ``_routed`` runs
    its own event loop), from a first response that makes the read."""
    queue = [envelope or _chats(_routed(_read(), tool=READ))]
    model = _Model(*replies)

    async def tools(name, args, call_id, workspace_id):
        return {**queue.pop(0)}

    executor = ToolLoopExecutor(llm_callback=model, tool_callback=tools, max_iterations=5, promises=promises)
    messages = [{"role": "user", "content": said}]
    with usage_scope(request_type=LANE_CHAT if promises else "agent"):
        result = asyncio.run(executor.run(initial_response=first or LLMResponse(content="", tool_calls=[_read_call()]),
                                          messages=messages, tools=TOOLS, workspace_id="ws"))
    return result, model, [m["content"] for m in messages if is_nudge(m)]


# ── one rule for every read ───────────────────────────────────────────────────────

def test_every_read_the_owner_may_ask_to_see_is_in_the_one_table():
    assert {"workspace_read_file", "platform_read_document", "platform_get_deliverable"} <= set(shown_reads.READS)
    assert "SHOW_IT_NOTE" not in vars(handlers_deliverables) and "text_to_show" in vars(handlers_deliverables)
    assert handlers_deliverables.text_to_show is shown_reads.text_to_show       # one seam, not a copy


def test_the_reads_are_wired_through_the_seam():
    assert exec_workspace.execute_workspace_action.__code__.co_qualname == "shows_a_read_file.<locals>.run"
    assert inspect.unwrap(exec_workspace.execute_workspace_action).__name__ == "execute_workspace_action"
    wrapped = handlers_documents.read_document
    chain = []
    while wrapped is not None:
        chain.append(wrapped.__code__.co_qualname)
        wrapped = getattr(wrapped, "__wrapped__", None)
    assert "shows_the_owner.<locals>.wrap.<locals>.run" in chain


# ── workspace_read_file: the note first, the text, cut with the link ──────────────────

@pytest.mark.parametrize("by_keyword", [False, True])
def test_a_file_read_tells_the_model_the_text_is_the_owners_to_see_before_the_text(by_keyword):
    found = _read(by_keyword=by_keyword)
    assert found["success"] is True and found["content"] == DRAFT
    assert found["show_the_owner"] == shown_reads.SHOW_IT_NOTE
    assert list(found).index("show_the_owner") < list(found).index("content")

    seen = _routed(_read(), tool=READ)["llm_context"]                 # what the model reads
    assert seen.index("The owner sees none of this text") < seen.index("Shakiso slopes")


def test_a_long_file_comes_back_cut_at_the_configured_length_with_the_explorer_link(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "DELIVERABLE_SHOWN_MAX_CHARS", 60)
    monkeypatch.setitem(_FILE, "text", DRAFT * 20)
    found = _read()

    assert found["content"] == (DRAFT * 20)[:60]
    assert (found["content_truncated"], found["content_chars"]) == (True, len(DRAFT * 20))
    opened = urlparse(found["rest_at"])
    assert (opened.scheme, opened.hostname, opened.path) == ("https", "app.example.test", "/deliverables/explorer")
    assert parse_qs(opened.query) == {"path": [PATH]}
    assert f"This is its first 60 of {len(DRAFT * 20):,} characters" in found["show_the_owner"]
    assert found["show_the_owner"].endswith(found["rest_at"])


def test_an_agents_run_reads_the_whole_file_with_no_note(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "DELIVERABLE_SHOWN_MAX_CHARS", 60)
    found = _read(lane="agent")
    assert found["content"] == DRAFT and "show_the_owner" not in found and "rest_at" not in found


def test_a_failed_or_empty_read_passes_as_it_is():
    with usage_scope(request_type=LANE_CHAT):
        failed = {"success": False, "error": "File not found", "content": DRAFT}
        assert shown_reads.shown_read(failed, shown_reads.file_link) == failed
        assert shown_reads.shown_read({"success": True, "content": "  "}, shown_reads.file_link) == {
            "success": True, "content": "  "}
        assert shown_reads.shown_read(None, shown_reads.file_link) is None


def test_another_workspace_tool_passes_as_it_is(monkeypatch):
    import core.workspace_client as wc

    async def list_dir(self, path="."):
        return {"path": path, "entries": [{"name": NAME, "type": "file"}], "content": DRAFT}

    monkeypatch.setattr(wc.WorkspaceClient, "list_dir", list_dir)
    with usage_scope(request_type=LANE_CHAT):
        listed = asyncio.run(exec_workspace.execute_workspace_action(None, "workspace_list_dir", {"path": "drafts"},
                                                                     workspace_id=uuid4()))
    assert "show_the_owner" not in listed and listed["success"] is True


# ── platform_read_document: the same rule, through the real handler ─────────────────

def test_a_document_read_in_the_owners_chat_carries_the_note_first(db, shelf):  # noqa: F811
    with usage_scope(request_type=LANE_CHAT):
        found = asyncio.run(handlers_documents.read_document(db, shelf, {"filename": CLUB_BOX}))
    assert found["success"] is True and found["content"].startswith("October club box")
    assert found["show_the_owner"] == shown_reads.SHOW_IT_NOTE
    assert list(found)[0] == "show_the_owner"

    agents = asyncio.run(handlers_documents.read_document(db, shelf, {"filename": CLUB_BOX}))
    assert "show_the_owner" not in agents


def test_a_long_document_is_cut_with_the_documents_link(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "DELIVERABLE_SHOWN_MAX_CHARS", 60)

    @shown_reads.shows_the_owner(shown_reads.document_link)
    async def read(_db, _ws, _params):
        return {"success": True, "document_id": 1518, "filename": CLUB_BOX, "content": DRAFT * 20, "has_more": False}

    with usage_scope(request_type=LANE_CHAT):
        found = asyncio.run(read(None, None, {}))
    assert found["content"] == (DRAFT * 20)[:60] and found["rest_at"] == "https://app.example.test/documents"
    assert found["filename"] == CLUB_BOX and found["document_id"] == 1518


# ── the reply: it holds the text ───────────────────────────────────────────────────

def test_show_me_the_guji_draft_the_reply_holds_the_text():
    result, model, nudges = _run(LLMResponse(content=QUOTED, tool_calls=None))
    assert nudges == [] and model.calls == 1
    assert "Shakiso slopes at 2,100 metres" in result.response.content


def test_a_reply_that_omits_it_is_retried_once_then_holds_the_text():
    result, model, nudges = _run(LLMResponse(content=DESCRIBED, tool_calls=None),
                                 LLMResponse(content=QUOTED, tool_calls=None))
    assert model.calls == 2 and len(nudges) == 1
    assert f'The owner asked to see "{NAME}" and you fetched it, but your reply shows none of its text' in nudges[0]
    assert "Shakiso slopes at 2,100 metres" in result.response.content


def test_a_long_file_the_retry_is_told_the_first_n_characters_and_the_link(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "DELIVERABLE_SHOWN_MAX_CHARS", 60)
    monkeypatch.setitem(_FILE, "text", DRAFT * 20)
    found = _read()
    shown, link = found["content"], found["rest_at"]
    result, model, nudges = _run(LLMResponse(content=DESCRIBED, tool_calls=None),
                                 LLMResponse(content=f"{shown}\n\nThe rest: {link}", tool_calls=None),
                                 envelope=_chats(_routed(found, tool=READ)))

    assert len(nudges) == 1 and f"its first 60 characters, then the link to the rest ({link})" in nudges[0]
    assert shown in result.response.content and link in result.response.content


def test_a_document_read_through_platform_execute_is_nudged_the_same_way():
    document = {"success": True, "document_id": 1518, "filename": CLUB_BOX, "content": DRAFT, "has_more": False}
    call = {"id": "call_doc", "type": "function", "function": {"name": "platform_execute", "arguments": json.dumps(
        {"action": "platform_read_document", "params": {"filename": CLUB_BOX}})}}
    result, model, nudges = _run(LLMResponse(content=DESCRIBED, tool_calls=None),
                                 LLMResponse(content=QUOTED, tool_calls=None),
                                 envelope=_chats(_routed(document, tool="platform_read_document")),
                                 first=LLMResponse(content="", tool_calls=[call]), said=f"show me {CLUB_BOX}")
    assert len(nudges) == 1 and f'The owner asked to see "{CLUB_BOX}"' in nudges[0]
    assert "Shakiso slopes at 2,100 metres" in result.response.content


# ── what is left alone ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("said", ["read the Guji blog draft and tell me its word count",
                                  "see if the Guji draft mentions jasmine",
                                  "summarise the Guji blog draft in two bullets"])
def test_work_on_the_file_is_no_ask_to_see_it(said):
    result, model, nudges = _run(LLMResponse(content="It runs to 38 words.", tool_calls=None), said=said)
    assert nudges == [] and model.calls == 1


def test_an_agents_run_is_not_nudged():
    result, model, nudges = _run(LLMResponse(content=DESCRIBED, tool_calls=None), promises=False)
    assert nudges == [] and model.calls == 1


def test_the_owner_still_asks_to_see_it_in_their_own_words():
    for said in ("show me the file", "can I see it", "read me the post", "read it to me", "open the report"):
        assert fetched_work._ASKS_TO_SEE.search(said), said
