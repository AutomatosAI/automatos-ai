"""F188 (night 6) — Auto says what is connected and where files go.

(a) At 02:03:19 platform_assign_tool_to_agent put GMAIL on agent 323 and said
"Tool 'GMAIL' assigned", with no app connected in the workspace. Auto told the
owner Gmail was ready, and pointed at a "Settings → Integrations" page that
does not exist. The assignment now says whether the app is connected, and when
it is not, says so and where to connect it.
(c) At 02:07:41 Auto asked the owner to paste their files into the chat.
Nothing in its prompt named the upload control, and "Upload documents to the
knowledge base" read as something Auto does. Its prompt now names Knowledge
Base → Upload Documents, and the upload tool says it is for text Auto wrote.
The checklist says "Connect an app" while none is connected, not "a second app".
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from modules.tools.discovery import handlers_assignments

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
NOT_CONNECTED = ("Tool 'GMAIL' assigned to agent 'Shopify Support Agent', but GMAIL is NOT connected for this "
                 "workspace, so the agent cannot use it yet. Tell the owner it is assigned but not connected — "
                 "never that it is ready. They connect it on Tools & Integrations → Gmail → Connect.")


class _Query:
    def __init__(self, found):
        self.found = found

    def filter(self, *a, **k):
        return self

    def first(self):
        return self.found


class _Db:
    def __init__(self, existing=None):
        self.existing, self.added = existing, []

    def query(self, *a):
        return _Query(self.existing)

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        pass


@pytest.fixture
def assign(monkeypatch):
    import core.composio.entity_manager as em

    agent = NS(id=323, name="Shopify Support Agent")
    monkeypatch.setattr(handlers_assignments, "resolve_agent", lambda db, ws, params: (agent, None))

    def run(connected, existing=None):
        monkeypatch.setattr(em.EntityManager, "get_connected_apps", lambda self, ws: list(connected))
        return asyncio.run(handlers_assignments.assign_tool_to_agent(
            _Db(existing), WS, {"agent_name": "Shopify Support Agent", "app_name": "gmail"}))
    return run


@pytest.mark.parametrize("existing", [None, NS(is_active=True), NS(is_active=False)],
                         ids=["new", "already-assigned", "reactivated"])
def test_an_app_that_is_not_connected_is_never_ready(assign, existing):
    result = assign([], existing)

    assert result["success"] is True and result["connected"] is False
    assert result["message"].endswith(NOT_CONNECTED[NOT_CONNECTED.index("GMAIL is NOT connected"):])


def test_a_connected_app_says_so(assign):
    result = assign(["GMAIL", "SLACK"])
    assert result["connected"] is True
    assert result["message"] == "Tool 'GMAIL' assigned to agent 'Shopify Support Agent'."


def test_the_prompt_names_the_owners_upload_control():
    from consumers.chatbot import personality

    text = " ".join(open(personality.__file__).read().split())
    assert ("Read the files the owner adds: they upload them on **Knowledge Base → Upload Documents** (PDF, Word, "
            "Excel, CSV, text); for a file needed only in this chat, the paperclip in the message box. Never ask "
            "them to paste a file's contents into chat.") in text
    assert "Upload documents to the knowledge base for semantic search" not in text


def test_the_upload_tool_is_for_text_auto_wrote():
    from modules.tools.discovery import get_action_registry

    description = get_action_registry().get("platform_upload_document").description
    assert description.endswith("Only for text you wrote — never ask the owner to paste a file; their files go "
                                "through Knowledge Base → Upload Documents.")


@pytest.mark.parametrize("connections, label", [(0, "Connect an app"), (1, "Connect a second app"),
                                                (2, "Connect a second app")])
def test_the_checklist_asks_for_an_app_before_a_second_one(connections, label):
    from services.onboarding_state import build_checklist

    item = build_checklist(connections_count=connections, missions_count=0, members_count=1,
                           plan_seats=1)["items"][0]
    assert item["label"] == label and item["done"] is (connections >= 2)
