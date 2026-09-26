"""F099 (night 3) — a reply may not give as its source a tool that did not run.

Chat c7206157 (20:37, 16 s, no tools) answered the persona's five questions
character-for-character as chat 607fd898 had at 19:03 after five searches,
each line still ending "(Source: search_knowledge, ticket #213 and #282)". The
turn's context carried the earlier answer as a recalled memory, and the model
repeated it as if it had just searched. Now the loop nudges once — search, or
say the answer is from an earlier conversation — and the chat says so where the
owner can see it when a reply still names a source that did not run.
"""
from __future__ import annotations

import asyncio

from core.llm.clients.base import LLMResponse
from modules.tools.execution.tool_loop import (
    UNRUN_SOURCE_NOTICE,
    ToolLoopExecutor,
    cited_tool_not_run,
    offered_tool_names,
)

REPLAYED_2037 = (
    "1. The Guji Hambela Alaka is not mentioned with a per-kilo price or stock levels in the provided "
    "documents. I only have information for the Guji Shakiso Natural at £10.50 for a 250g retail bag "
    "(Source: search_knowledge, ticket #213 and #282)."
)
SEARCH = {"type": "function", "function": {"name": "search_knowledge", "parameters": {"type": "object"}}}
OFFERED = offered_tool_names([SEARCH, {"type": "function", "function": {"name": "platform_list_agents"}}])


def test_the_replayed_reply_names_a_source_that_did_not_run():
    assert OFFERED == {"search_knowledge", "platform_list_agents"}
    assert cited_tool_not_run(REPLAYED_2037, OFFERED) == "search_knowledge"
    assert cited_tool_not_run("Sources: search_knowledge — wholesale-terms.md", OFFERED) == "search_knowledge"


def test_a_source_that_ran_a_mention_or_a_file_name_is_not_flagged():
    assert cited_tool_not_run(REPLAYED_2037, OFFERED, ran={"search_knowledge"}) is None
    assert cited_tool_not_run("I can look that up with search_knowledge if you like.", OFFERED) is None
    assert cited_tool_not_run("(Source: green_coffee_list_autumn.csv)", OFFERED) is None
    assert cited_tool_not_run("", OFFERED) is None


# ── the loop's one nudge ────────────────────────────────────────────────────

class _Model:
    def __init__(self, *responses):
        self.queue, self.calls = list(responses), []

    async def __call__(self, messages, tools):
        self.calls.append([dict(m) for m in messages])
        return self.queue.pop(0)


class _Tools:
    def __init__(self):
        self.executed = []

    async def __call__(self, name, args, call_id, workspace_id):
        self.executed.append(name)
        return {"success": True, "results": [{"content": "Hambela Alaka £8.62/kg, 143.5 kg"}]}


def _search_call():
    return {"id": "call_1", "type": "function",
            "function": {"name": "search_knowledge", "arguments": "{\"query\": \"Hambela Alaka\"}"}}


def test_the_replayed_reply_is_nudged_once_and_the_retry_searches():
    model = _Model(LLMResponse(content="", tool_calls=[_search_call()]),
                   LLMResponse(content="Hambela Alaka: £8.62/kg, 143.5 kg in stock.", tool_calls=None))
    tools = _Tools()
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=tools, max_iterations=5)
    messages = [{"role": "user", "content": "Quick ones — answer each from what I've given you."}]
    result = asyncio.run(executor.run(initial_response=LLMResponse(content=REPLAYED_2037, tool_calls=None),
                                      messages=messages, tools=[SEARCH], workspace_id="ws"))
    assert tools.executed == ["search_knowledge"]
    assert result.response.content.startswith("Hambela Alaka: £8.62/kg")
    nudge = model.calls[0][-1]
    assert nudge["role"] == "system" and "gives search_knowledge as its source, but no tool ran" in nudge["content"]


def test_the_notice_names_the_tool():
    assert UNRUN_SOURCE_NOTICE.format(tool="search_knowledge").startswith(
        "No search ran for this reply — it gives search_knowledge as its source")


# ── both reply paths ask the same question ──────────────────────────────────

def test_the_notice_covers_a_source_that_did_not_run_and_narrated_actions():
    from consumers.chatbot.service import NARRATED_ACTIONS_NOTICE, unexecuted_claims_notice

    assert unexecuted_claims_notice(REPLAYED_2037, [SEARCH], set(), any_tool_ran=False).startswith(
        "No search ran for this reply — it gives search_knowledge as its source")
    assert unexecuted_claims_notice(REPLAYED_2037, [SEARCH], {"search_knowledge"}, any_tool_ran=True) is None
    narrated = "Now let me create OPS and TRACKER. Good — both created."
    assert unexecuted_claims_notice(narrated, [SEARCH], set(), any_tool_ran=False) == NARRATED_ACTIONS_NOTICE
    assert unexecuted_claims_notice(narrated, [SEARCH], {"platform_create_agent"}, any_tool_ran=True) is None
    assert unexecuted_claims_notice(REPLAYED_2037, None, set(), any_tool_ran=False) is None      # no tools offered


def test_a_first_reply_with_no_tool_call_is_checked_too():
    """Night 3's replayed answer was a first reply with no tool call: it never
    entered the tool loop, where the notice used to live."""
    import inspect

    from consumers.chatbot import service

    source = inspect.getsource(service.StreamingChatService._stream_response_with_agent_scoped)
    branch = source[source.index('final_text = response.content or ""'):]
    assert branch.index("unexecuted_claims_notice(") < branch.index("split_reply(")   # before the reply is made
    assert "unexecuted_claims_notice(" in inspect.getsource(service.StreamingChatService._stream_tool_loop)
