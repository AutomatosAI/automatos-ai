"""F351 (night 10b, build 19): a document conversation keeps its document tools on a follow-up.

Turn 1 of a chat made the PDF with generate_document. Turn 2 ("Yes, go ahead and try again.",
"Please just make it.", "Yes, issue 43. Go ahead.") read on its own as chat: the ATOM lane
shipped the dispatcher alone, or the router's "no tools" verdict left only the platform_*
tools. Auto then called invented actions (platform_generate_document, platform_create_document,
a "Generate PDF from template" playbook) and sent the owner to Proposify, Venngage, Canva and
Visme. Chats 2fec4fd9, 0e417f99, 43c5e928, a71ae29b, 1e439ba6, bfbc4118.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

TEMPLATE_ACTIONS = ("platform_list_templates", "platform_get_template_schema")
DOCUMENT_TOOLS = {"generate_document", *TEMPLATE_ACTIONS}

# A document conversation as the chat stores it (parts) and as the router gets it (content).
DOCUMENT_CHAT = [
    {"role": "user", "parts": [{"type": "text", "text": 'Write the Quay letter on my "Harbourline Letter" template.'}]},
    {"role": "assistant", "parts": [{"type": "text", "text": "Here's the letter as a PDF: Quay letter.pdf"}]},
]
COFFEE_CHAT = [
    {"role": "user", "parts": [{"type": "text", "text": "How much Guji have we got for October's club boxes?"}]},
    {"role": "assistant", "parts": [{"type": "text", "text": "38 kg, enough for about 150 boxes."}]},
]


def _content(chat):
    return [{"role": "system", "content": "Use the document templates and generate_document."}] + [
        {"role": m["role"], "content": m["parts"][0]["text"]} for m in chat
    ]


def _tool(name, description=""):
    return {"type": "function", "function": {"name": name, "description": description or f"{name} desc"}}


# The surface the chat entrypoint builds: the dispatcher, generate_document, a promoted pin,
# first-class schemas for the template actions, and one unrelated app tool.
SURFACE = [
    _tool("platform_execute"), _tool("generate_document"), _tool("platform_list_agents"),
    _tool("platform_list_templates"), _tool("platform_get_template_schema"),
    _tool("gmail_send", "send an email through Gmail"), _tool("unrelated_tool"),
]


def _names(result):
    return {t["function"]["name"] for t in result.filtered_tools}


def _route(query, conversation, *, classifier=None, tool_hints=None):
    from consumers.chatbot.smart_tool_router import SmartToolRouter

    router = SmartToolRouter()
    if classifier is not None:
        router.classifier = classifier
    return asyncio.run(router.route(query=query, available_tools=SURFACE, conversation_context=conversation,
                                    tool_hints=tool_hints, agent_id=7, workspace_id="ws"))


def _intent(primary_name, requires_tools=True):
    from consumers.chatbot.intent_classifier import Intent, IntentResult

    result = IntentResult(primary_intent=Intent[primary_name], confidence=0.9, requires_tools=requires_tools,
                          requires_memory=False, suggested_tools=[], reasoning="stub", is_simple=False)
    return SimpleNamespace(classify=lambda query, context=None: result)


@pytest.mark.parametrize("follow_up", ["make it a bit shorter", "use my logo too", "Yes, go ahead and try again."])
def test_a_no_tools_follow_up_of_a_document_conversation_keeps_the_document_tools(follow_up, monkeypatch):
    from config import config

    monkeypatch.setattr(config, "TOOL_ROUTING_GRAPH", False)
    result = _route(follow_up, _content(DOCUMENT_CHAT))     # the real classifier: no tools on its own
    assert result.should_include_tools is True
    assert DOCUMENT_TOOLS <= _names(result)
    assert "platform_execute" in _names(result)            # the door to every other action
    assert result.tool_choice == "auto"                   # kept within reach, never forced
    assert "unrelated_tool" not in _names(result)


def test_the_hint_branch_keeps_the_document_tools():
    result = _route("make it a bit shorter", _content(DOCUMENT_CHAT), tool_hints=["email"])
    assert "gmail_send" in _names(result) and DOCUMENT_TOOLS <= _names(result)


def test_the_category_branch_keeps_the_document_tools(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "TOOL_ROUTING_GRAPH", False)
    result = _route("add the total at the bottom", _content(DOCUMENT_CHAT), classifier=_intent("DATA_QUERY"))
    assert DOCUMENT_TOOLS <= _names(result)


def test_the_graph_branch_keeps_the_document_tools(monkeypatch):
    from config import config

    async def rank_chains(**kwargs):
        return [("platform_list_agents", 0.9, ["platform_list_agents"])]

    monkeypatch.setattr(config, "TOOL_ROUTING_GRAPH", True)
    monkeypatch.setattr("modules.tools.discovery.graph_router.get_graph_router",
                        lambda: SimpleNamespace(rank_chains=rank_chains))
    result = _route("use my logo too", _content(DOCUMENT_CHAT), classifier=_intent("CREATION"))
    assert result.reasoning.startswith("Graph routing")
    assert DOCUMENT_TOOLS <= _names(result)


def test_a_conversation_that_never_touched_documents_is_routed_as_before(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "TOOL_ROUTING_GRAPH", False)
    result = _route("make it a bit shorter", _content(COFFEE_CHAT))
    assert result.should_include_tools is False and result.filtered_tools == []
    hinted = _route("make it a bit shorter", _content(COFFEE_CHAT), tool_hints=["email"])
    assert not {"platform_list_templates", "platform_get_template_schema"} & _names(hinted)


def test_only_the_latest_few_spoken_messages_count():
    from consumers.chatbot.document_conversation import LOOKBACK_MESSAGES, about_a_document

    assert about_a_document(DOCUMENT_CHAT) and about_a_document(_content(DOCUMENT_CHAT))
    assert not about_a_document(_content(COFFEE_CHAT))      # the system prompt's words never count
    assert about_a_document(COFFEE_CHAT, query="Put it on the Branded Page template")   # this turn's own ask
    long_ago = DOCUMENT_CHAT + COFFEE_CHAT * LOOKBACK_MESSAGES
    assert not about_a_document(long_ago)


def _assessment(complexity):
    from consumers.chatbot.auto import Action, ComplexityAssessment

    return ComplexityAssessment(complexity=complexity, action=Action.RESPOND, reasoning="Greeting or chitchat")


def _lane_taken(messages, assessment, *, widget_mode=False):
    """The complexity the turn runs with, from stream_response_with_agent's own entry."""
    from consumers.chatbot.service import StreamingChatService

    service = StreamingChatService.__new__(StreamingChatService)
    service.widget_mode = widget_mode
    service.widget_scopes, service.widget_team, service.widget_agent_lock = (), None, None

    async def _turn(*args, **kwargs):
        yield kwargs["complexity_assessment"].complexity.value

    service._stream_response_with_agent_scoped = _turn

    async def _chat():
        return [chunk async for chunk in service.stream_response_with_agent(
            chat_id="chat-1", messages=messages, agent_id=9, user_id=1, complexity_assessment=assessment)]

    return asyncio.run(_chat())[0], service


def test_a_document_follow_up_takes_the_full_path_not_the_atom_lane():
    from consumers.chatbot.auto import Complexity

    follow_up = DOCUMENT_CHAT + [{"role": "user", "parts": [{"type": "text", "text": "Yes, issue 43. Go ahead."}]}]
    atom = _assessment(Complexity.ATOM)
    lane, service = _lane_taken(follow_up, atom)
    assert lane == "molecule" and service._document_turn is True
    assert atom.complexity is Complexity.ATOM                 # AutoBrain's verdict is not mutated

    other = COFFEE_CHAT + [{"role": "user", "parts": [{"type": "text", "text": "Yes, go ahead."}]}]
    assert _lane_taken(other, _assessment(Complexity.ATOM))[0] == "atom"
    assert _lane_taken(follow_up, _assessment(Complexity.ATOM), widget_mode=True)[0] == "atom"
    assert _lane_taken(follow_up, _assessment(Complexity.CELL))[0] == "cell"


def test_a_proactive_opener_stays_text_only():
    from consumers.chatbot.auto import Complexity
    from consumers.chatbot.document_conversation import full_path_for_documents

    atom = _assessment(Complexity.ATOM)
    assert full_path_for_documents(atom, True, force_text_only=True) is atom


def _page_actions_shipped(monkeypatch, document_turn, page_actions=None):
    """What ``_get_tools`` folds into the dispatcher's ranked enum for this turn."""
    from consumers.chatbot import service as chat_service

    seen = {}

    async def _tools(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(chat_service, "get_tools_for_agent_async", _tools)
    service = chat_service.StreamingChatService.__new__(chat_service.StreamingChatService)
    service.db, service.workspace_id, service._document_turn = None, "ws", document_turn
    asyncio.run(service._get_tools(7, None, query="Yes, issue 43. Go ahead.", page_actions=page_actions))
    return seen["page_actions"]


def test_the_template_actions_are_folded_into_a_document_turns_ranked_enum(monkeypatch):
    from modules.tools.tool_router import _apply_page_prior

    shipped = _page_actions_shipped(monkeypatch, True, page_actions=["platform_list_agents"])
    assert shipped == [*TEMPLATE_ACTIONS, "platform_list_agents"]
    ranked = (["platform_execute_playbook", "platform_list_agents"], None, False)   # "go ahead" ranks no template
    allowed, _reason, _pins = _apply_page_prior(ranked, shipped, is_admin=False, is_super_admin=False)
    assert set(TEMPLATE_ACTIONS) <= set(allowed)               # the owner's own role clears their gate
    assert _page_actions_shipped(monkeypatch, False) is None   # any other turn: the page's actions only


def test_autos_rules_never_send_the_owner_to_an_outside_product_for_a_document():
    from consumers.chatbot.personality import AutomatosPersonality

    guidance = AutomatosPersonality.get_tool_guidance_prompt(has_tools=True)
    assert ("Send the owner to an outside product (Canva, Proposify, Venngage, Visme or any other) for a document "
            "my templates and generate_document can make") in guidance
    assert "If generating it fails, I say what failed and offer to try again" in guidance
    rules = guidance.split("### What I Never Do with Tools", 1)[1]
    assert "outside product" in rules
