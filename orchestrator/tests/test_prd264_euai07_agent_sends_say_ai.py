"""PRD-264 EUAI-07 (Guidelines ¶36): an agent's email or message says at the top that an AI wrote it, and for whom.

Every Composio send or publish an agent starts (``owner_only.is_composio_send``) carries, as the
first line of its body, "Written and sent by an AI agent on behalf of {principal}." (FR-15: the
workspace's public name, else "... of the sender."); an HTML body gets it as its first
paragraph. The line goes on before the owner's card is built, so the card shows it and the
grant's params hash covers it; the click runs that exact call and never removes the line (D9).
A person's own send through the platform's email routes is unchanged, unless an agent's
``http_request`` tool made it. A send the executor resolves from another name can't take the
line, so one with words of its own is refused.

The gate here is the real ``owner_only.asks_before_a_send`` over the executor's own reading of a
call (``UnifiedToolExecutor._resolve_effective_call``); the executor beneath it records what each
send went out with. The resolved-send case runs the real chain through
``ComposioToolExecutor.execute`` (the RVW-3 fixture).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from typing import Any, List, Optional
from uuid import uuid4

import pytest

from modules.policy.ai_disclosure import (
    SEND_BODY_PARAMS,
    is_html_body,
    send_action_key,
    send_body_key,
    send_notice,
    split_send_notice,
    with_send_notice,
)
from modules.socials.capabilities import SEEDED_CHANNELS
from modules.tools.discovery import owner_only
from modules.tools.discovery.card_question import send_question
from modules.tools.discovery.owner_only import asks_before_a_send, is_composio_send
from modules.tools.discovery.send_notice import RESOLVED_SEND_UNTOLD, told_call, untold_resolved_send
from modules.tools.execution.agent_request import AGENT_REQUEST_HEADER, marked_request, marks_an_agent_request
from modules.tools.execution.composio_params import sent_params
from modules.tools.execution.unified_executor import UnifiedToolExecutor
from tests import test_prd256_fix_orders_and_resolved_sends_wait_for_the_click as resolved

desk = resolved.desk   # the RVW-3 fixture: the real chain through ComposioToolExecutor.execute

WS = str(uuid4())
HARBOUR = "Written and sent by an AI agent on behalf of Harbour Coffee."
UNNAMED = "Written and sent by an AI agent on behalf of the sender."
ANA = "ana@harbourline.test"
LETTER = "Hi Ana,\nPlease send 12 kilos of the Guji."
EMAIL = {"recipient_email": ANA, "subject": "Spring order", "body": LETTER}
SEND = {"action": "GMAIL_SEND_EMAIL", "params": EMAIL}
OWNERS_CHAT = {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": "chat-1", "turn_id": "t-1"}
AGENTS_RUN: dict = {}                                    # an agent's own run, on a ticket a person wrote
PERSONS_OWN = {"actor_type": "user_direct", "user_id": "user_owner", "surface": "widget_email"}

# The Composio actions the send gate's own tests check, the platform's email routes run and the
# messaging and commerce toolkits agents call: every one the gate accepts names its body.
KNOWN_SLUGS = (
    "GMAIL_SEND_EMAIL", "gmail-send-email", "GMAIL_REPLY_TO_THREAD", "GMAIL_REPLY_TO_EMAIL", "GMAIL_FORWARD_MESSAGE",
    "GMAIL_SEND_DRAFT", "GMAIL_FETCH_EMAILS", "GMAIL_CREATE_EMAIL_DRAFT", "GMAIL_UPDATE_DRAFT",
    "OUTLOOK_SEND_EMAIL", "OUTLOOK_REPLY_EMAIL", "OUTLOOK_SEND_DRAFT", "OUTLOOK_CREATE_DRAFT_MESSAGE",
    "OUTLOOK_CREATE_DRAFT_REPLY",
    "SLACK_SEND_MESSAGE", "SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL", "SLACK_CHAT_POST_MESSAGE",
    "SLACK_SCHEDULE_MESSAGE", "SLACK_LIST_SCHEDULED_MESSAGES", "composio_SLACK_SEND_MESSAGE",
    "DISCORDBOT_CREATE_MESSAGE", "TELEGRAM_SEND_MESSAGE", "WHATSAPP_SEND_MESSAGE", "TWILIO_CREATE_MESSAGE",
    "TWILIO_SEND_SMS", "TWITTER_CREATE_A_NEW_DM_CONVERSATION", "TWITTER_CREATE_TWEET", "TWITTER_CREATION_OF_A_POST",
    "TWITTER_POST_DELETE_BY_POST_ID", "TWITTER_POST_LOOKUP_BY_POST_ID", "GITHUB_CREATE_AN_ISSUE_COMMENT",
    "GITHUB_LIST_ISSUE_COMMENTS", "FACEBOOK_REPLY_TO_POST_COMMENT", "PUSHOVER_NOTIFY_USER", "PUSHOVER_LIST_NOTIFY_RULES",
    "LINKEDIN_CREATE_LINKED_IN_POST", "LINKEDIN_CREATE_POST", "LINKEDIN_CREATE_IMAGE_POST", "LINKEDIN_CREATE_VIDEO_POST",
    "LINKEDIN_CREATE_SHARE", "LINKEDIN_CREATE_ARTICLE_OR_URL_SHARE", "LINKEDIN_DELETE_POST", "LINKEDIN_GET_POST",
    "INSTAGRAM_CREATE_POST", "INSTAGRAM_POST_IG_USER_MEDIA", "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH",
    "INSTAGRAM_MEDIA_PUBLISH", "INSTAGRAM_PUBLISH_MEDIA", "THREADS_PUBLISH_MEDIA", "TIKTOK_UPLOAD_VIDEO",
    "TIKTOK_PUBLISH_VIDEO", "TIKTOK_POST_PHOTO", "YOUTUBE_UPLOAD_VIDEO", "YOUTUBE_UPDATE_THUMBNAIL",
    "REDDIT_CREATE_REDDIT_POST", "MASTODON_CREATE_POST", "MAILCHIMP_SEND_CAMPAIGN", "MAILCHIMP_SCHEDULE_CAMPAIGN",
    "SHOPIFY_CREATE_ORDER", "SHOPIFY_CANCEL_ORDER", "SHOPIFY_CREATE_REFUND", "SHOPIFY_GET_ORDER", "SHOPIFY_LIST_ORDERS",
    "SQUARE_PLACE_ORDER", "AMAZON_PURCHASE_ITEM", "STRIPE_CREATE_PAYMENT_INTENT", "STRIPE_CREATE_INVOICE",
    "STRIPE_CREATE_CHARGE", "STRIPE_CREATE_REFUND", "STRIPE_CREATE_PAYOUT", "STRIPE_CREATE_TRANSFER",
    "STRIPE_LIST_REFUNDS", "STRIPE_RETRIEVE_INVOICE", "PAYPAL_CREATE_PAYOUT", "CALENDLY_BOOK_MEETING",
    "CALCOM_CREATE_BOOKING",
)
COPY = "$copy"   # the seeded channel adapters' source for the post's words


# ── A session that answers the two FR-15 reads ────────────────────────────────────────

class _Result:
    def __init__(self, row: Any) -> None:
        self._row = row

    def fetchone(self) -> Any:
        return self._row


class _Savepoint:
    def __enter__(self) -> "_Savepoint":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class FakeDb:
    """Answers the business profile's company name and the workspace's name; counts each read."""

    def __init__(self, company: Any = None, workspace_name: Any = None) -> None:
        self.company, self.workspace_name = company, workspace_name
        self.reads: List[str] = []

    def execute(self, statement: Any, params: Any = None) -> _Result:
        sql = str(statement)
        self.reads = [*self.reads, sql]
        if "business_profiles" in sql:
            return _Result(None if self.company is None else (self.company,))
        if "FROM workspaces" in sql:
            return _Result(None if self.workspace_name is None else (self.workspace_name,))
        return _Result(None)

    def begin_nested(self) -> _Savepoint:
        return _Savepoint()


class _Sender:
    """UnifiedToolExecutor's shape: the real owner's-click gate over the executor's own reading
    of a call; beneath it, each call that ran is recorded as it went out."""

    composio_actions = {"GMAIL_SEND_EMAIL": "GMAIL"}
    tool_routes: dict = {}
    _resolve_effective_call = UnifiedToolExecutor._resolve_effective_call
    _tool_is_composio = UnifiedToolExecutor._tool_is_composio

    def __init__(self, db: Any) -> None:
        self.db = db
        self.ran: List[Any] = []

    @asks_before_a_send
    async def execute_tool(self, tool_name, parameters, agent_id=0, tenant_id=None, workspace_id=None,
                           trace_id=None, caller_context=None):
        self.ran = [*self.ran, (tool_name, parameters)]
        return {"success": True, "data": {"id": "msg-1"}}


def _send(sender: _Sender, call: Any, context: Any, tool: str = "composio_execute") -> Any:
    return asyncio.run(sender.execute_tool(tool, call, workspace_id=WS, caller_context=context))


def _went_out(sender: _Sender) -> List[dict]:
    """What each call that ran sent to the action, as exec_composio sends it."""
    return [sent_params(params) if tool == "composio_execute" else params for tool, params in sender.ran]


@pytest.fixture
def card(monkeypatch):
    """The owner's card: no click yet; what reaches the grant is kept."""
    import modules.tools.execution.tool_grants as tool_grants

    stored: dict = {}
    monkeypatch.setattr(owner_only, "_the_click", lambda db, ws, action, params: None)
    monkeypatch.setattr(tool_grants, "attach_ask_grant",
                        lambda db, ws, **kw: stored.update(kw) or {**kw["ask"], "grant_id": 41})
    return stored


@pytest.fixture
def clicked(monkeypatch):
    """The owner clicked: every claim of the grant is kept with the params it was claimed for."""
    import modules.tools.execution.tool_grants as tool_grants

    claims: List[Any] = []

    def claim(db, ws, action, params):
        claims.append(params)
        return NS(id=42)

    monkeypatch.setattr(owner_only, "_the_click", claim)
    monkeypatch.setattr(tool_grants, "give_back_unused", lambda db, grant_id, result: None)
    return claims


# ── The line, plain and HTML ──────────────────────────────────────────────────────────

def test_the_line_names_the_principal_or_says_the_sender():
    assert send_notice("Harbour Coffee") == HARBOUR
    assert send_notice(None) == UNNAMED
    assert send_notice("") == UNNAMED


def test_a_plain_body_gets_the_line_as_its_first_line():
    assert with_send_notice(LETTER, HARBOUR, False) == f"{HARBOUR}\n\n{LETTER}"


def test_an_html_body_gets_the_line_as_its_first_paragraph():
    assert with_send_notice("<p>Hi Ana,</p>", HARBOUR, True) == f"<p>{HARBOUR}</p><p>Hi Ana,</p>"
    document = '<html><body class="mail"><p>Hi Ana,</p></body></html>'
    assert with_send_notice(document, HARBOUR, True) == (
        f'<html><body class="mail"><p>{HARBOUR}</p><p>Hi Ana,</p></body></html>')
    ben = send_notice("Ben & Jerry's")
    assert with_send_notice("<p>Hi</p>", ben, True).startswith(
        "<p>Written and sent by an AI agent on behalf of Ben &amp; Jerry's.</p>")


def test_the_line_is_never_doubled_and_a_line_naming_someone_else_gets_ours_on_top():
    told = with_send_notice(LETTER, HARBOUR, False)
    assert with_send_notice(told, HARBOUR, False) == told
    html = with_send_notice("<p>Hi</p>", HARBOUR, True)
    assert with_send_notice(html, HARBOUR, True) == html
    forged = "Written and sent by an AI agent on behalf of Kerbside Ltd.\n\nHi Ana,"
    assert with_send_notice(forged, HARBOUR, False) == f"{HARBOUR}\n\n{forged}"


@pytest.mark.parametrize("text, params, html", [
    ("Hi Ana", {"is_html": True}, True),
    ("<p>Hi Ana</p>", {"is_html": False}, False),
    ("<p>Hi Ana</p>", {}, True),
    ("<div>Hi</div>", {}, True),
    ("Margins are < 5% and > 2%", {}, False),
])
def test_a_body_is_html_by_its_flag_else_by_its_tags(text, params, html):
    assert is_html_body(text, params) is html


@pytest.mark.parametrize("slug, params, key", [
    ("GMAIL_SEND_EMAIL", {"body": "Hi"}, "body"),
    ("gmail-send-email", {"body": "Hi"}, "body"),
    ("composio_SLACK_SEND_MESSAGE", {"channel": "#orders", "text": "Hi"}, "text"),
    ("GMAIL_REPLY_TO_THREAD", {"message_body": "Hi", "thread_id": "t-1"}, "message_body"),
    ("GMAIL_SEND_EMAIL", {"message": "Hi"}, "message"),            # the body under another name: still told
    ("ACME_SEND_NOTE", {"text": "Hi"}, "text"),                    # a send the map does not name: still told
    ("SHOPIFY_CREATE_ORDER", {"note": "Gift wrap", "message": "x"}, None),   # an order sends no words of its own
    ("GMAIL_SEND_EMAIL", {"body": "   "}, None),
    ("GMAIL_SEND_EMAIL", {"body": ["Hi"]}, None),
    ("GMAIL_SEND_EMAIL", None, None),
])
def test_the_body_key_comes_from_the_map_then_the_usual_names(slug, params, key):
    assert send_body_key(slug, params) == key


def test_a_card_shows_the_line_on_its_own_and_the_letters_first_line():
    told = with_send_notice(LETTER, HARBOUR, False)
    lines = send_question("send, publish or order through GMAIL_SEND_EMAIL", {**EMAIL, "body": told}).splitlines()

    assert f"- AI notice: {HARBOUR}" in lines and "- first line: Hi Ana," in lines
    assert lines.index(f"- AI notice: {HARBOUR}") + 1 == lines.index("- first line: Hi Ana,")
    assert split_send_notice(f"<p>{HARBOUR}</p><p>Hi</p>") == (HARBOUR, "<p>Hi</p>")
    assert split_send_notice(LETTER) == (None, LETTER)


# ── The body map is complete ──────────────────────────────────────────────────────────

def _seeded_steps():
    return [step for adapter in SEEDED_CHANNELS.values() for sequence in adapter.kinds.values() for step in sequence]


def test_every_send_the_gate_accepts_names_its_body_parameter():
    seeded = {step.action for step in _seeded_steps()}
    publishes = {slug for adapter in SEEDED_CHANNELS.values() for slug in adapter.publish_actions}
    known = sorted({*KNOWN_SLUGS, *seeded, *publishes})

    missing = [slug for slug in known if is_composio_send(slug) and send_action_key(slug) not in SEND_BODY_PARAMS]
    assert missing == []


def test_a_seeded_channels_post_is_told_in_the_parameter_that_carries_its_copy():
    for step in _seeded_steps():
        copy = [name for name, source in step.params.items() if source == COPY]
        if copy and is_composio_send(step.action):
            assert SEND_BODY_PARAMS[step.action] == copy[0], step.action


def test_the_map_is_keyed_by_the_action_and_names_one_parameter_or_none():
    for action, key in SEND_BODY_PARAMS.items():
        assert send_action_key(action) == action
        assert key is None or (isinstance(key, str) and key.strip() == key and key)


# ── An agent's send, through the gate ─────────────────────────────────────────────────

def test_an_agents_email_goes_out_with_the_line_at_the_top():
    sender = _Sender(FakeDb(company="Harbour Coffee"))

    reply = _send(sender, SEND, AGENTS_RUN)

    assert reply["success"] is True
    (sent,) = _went_out(sender)
    assert sent["body"] == f"{HARBOUR}\n\n{LETTER}"
    assert sent["recipient_email"] == ANA and sent["subject"] == "Spring order"
    assert SEND["params"]["body"] == LETTER                 # the model's call is never changed in place


def test_an_agents_html_email_goes_out_with_the_line_as_its_first_paragraph():
    sender = _Sender(FakeDb(company="Harbour Coffee"))

    _send(sender, {"action": "GMAIL_SEND_EMAIL", "params": {**EMAIL, "body": "<p>Hi Ana,</p>", "is_html": True}},
          AGENTS_RUN)

    (sent,) = _went_out(sender)
    assert sent["body"] == f"<p>{HARBOUR}</p><p>Hi Ana,</p>" and sent["is_html"] is True


@pytest.mark.parametrize("call, tool", [
    (EMAIL, "GMAIL_SEND_EMAIL"),                                                       # a per-action tool
    ({"action": "GMAIL_SEND_EMAIL", "parameters": EMAIL}, "composio_execute"),         # under "parameters"
    ({"action": "GMAIL_SEND_EMAIL", "params": {"recipient_email": ANA}, "body": LETTER}, "composio_execute"),  # beside
], ids=["per-action", "under-parameters", "body-beside-params"])
def test_the_line_goes_where_the_executor_reads_the_body(call, tool):
    sender = _Sender(FakeDb(company="Harbour Coffee"))

    _send(sender, call, AGENTS_RUN, tool=tool)

    (sent,) = _went_out(sender)
    assert sent["body"] == f"{HARBOUR}\n\n{LETTER}"


def test_in_the_owners_chat_the_card_shows_the_line_and_the_grant_covers_it(card):
    sender = _Sender(FakeDb(company="Harbour Coffee"))

    reply = _send(sender, SEND, OWNERS_CHAT)

    assert sender.ran == [] and reply["requires_confirmation"] is True     # nothing before the click
    lines = reply["question_md"].splitlines()
    assert f"- AI notice: {HARBOUR}" in lines and "- first line: Hi Ana," in lines
    assert card["params"] == {"action": "GMAIL_SEND_EMAIL", "params": {**EMAIL, "body": f"{HARBOUR}\n\n{LETTER}"}}
    assert card["question_md"] == reply["question_md"]


def test_the_click_sends_the_line_and_the_call_sent_again_keeps_one(clicked):
    """D9: the owner's approval never removes the line. The click's run sends the stored call,
    which already carries it: the same params are claimed and nothing is added twice."""
    sender = _Sender(FakeDb(company="Harbour Coffee"))

    _send(sender, SEND, OWNERS_CHAT)
    stored = clicked[0]
    _send(sender, stored, OWNERS_CHAT)                       # the resume runs the grant's own call

    first, again = _went_out(sender)
    assert first["body"] == again["body"] == f"{HARBOUR}\n\n{LETTER}"
    assert clicked == [stored, stored]                       # one params hash for the card and the click


def test_a_persons_own_send_is_unchanged_and_reads_no_name():
    db = FakeDb(company="Harbour Coffee")
    sender = _Sender(db)

    _send(sender, SEND, PERSONS_OWN)

    (sent,) = _went_out(sender)
    assert sent == EMAIL and db.reads == []


def test_an_order_carries_no_line_and_reads_no_name():
    db = FakeDb(company="Harbour Coffee")
    sender = _Sender(db)
    order = {"action": "SHOPIFY_CREATE_ORDER", "params": {"email": ANA, "note": "Gift wrap", "line_items": []}}

    _send(sender, order, AGENTS_RUN)

    assert _went_out(sender) == [order["params"]] and db.reads == []


@pytest.mark.parametrize("company, workspace_name, line", [
    ("Harbour Coffee", "Ada Lovelace's Workspace", HARBOUR),                         # the company name first
    (None, "Harbour Coffee", HARBOUR),                                               # a workspace name chosen
    ("owner@harbour.example", "Harbour Coffee", HARBOUR),                            # an "@" is never a name
    (None, "Ada Lovelace's Workspace", UNNAMED),                                     # the auto-created form
    (None, "ada@harbour.example's Workspace", UNNAMED),
    (None, "ada@harbour.example", UNNAMED),
    (None, None, UNNAMED),
], ids=["company", "chosen-workspace-name", "company-is-an-email", "auto-created-workspace",
        "auto-created-from-an-email", "workspace-is-an-email", "no-name"])
def test_the_line_names_only_a_public_name_fr15(company, workspace_name, line):
    told = told_call(FakeDb(company, workspace_name), WS, "composio_execute", SEND, "GMAIL_SEND_EMAIL", AGENTS_RUN)

    assert told["params"]["body"] == f"{line}\n\n{LETTER}"


def test_a_session_that_cant_be_read_names_no_one():
    told = told_call(None, WS, "composio_execute", SEND, "GMAIL_SEND_EMAIL", AGENTS_RUN)

    assert told["params"]["body"] == f"{UNNAMED}\n\n{LETTER}"


# ── A send resolved from another name ────────────────────────────────────────────────

def test_a_resolved_send_with_words_of_its_own_is_refused():
    refusal = untold_resolved_send("GMAIL_SNED_EMAIL", "GMAIL_SEND_EMAIL", EMAIL, AGENTS_RUN)

    assert refusal == RESOLVED_SEND_UNTOLD.format(asked="GMAIL_SNED_EMAIL", action="GMAIL_SEND_EMAIL")
    assert "Call GMAIL_SEND_EMAIL by that name" in refusal
    assert untold_resolved_send("GMAIL_SNED_EMAIL", "GMAIL_SEND_EMAIL", {"params": EMAIL}, AGENTS_RUN) == refusal
    assert untold_resolved_send("GMAIL_SNED_EMAIL", "GMAIL_SEND_EMAIL", {"recipient_email": ANA}, AGENTS_RUN) is None
    assert untold_resolved_send("GMAIL_SNED_EMAIL", "GMAIL_SEND_EMAIL", EMAIL, PERSONS_OWN) is None
    assert untold_resolved_send("SHOPIFY_ORDR", "SHOPIFY_CREATE_ORDER", {"note": "x"}, AGENTS_RUN) is None


@pytest.mark.parametrize("where", ["autos-ticket", "a-persons-ticket", "owners-chat"])
def test_a_near_miss_onto_a_send_with_a_body_runs_nothing_and_raises_no_card(desk, where):
    context = {"autos-ticket": resolved._session(desk), "a-persons-ticket": {"session_task_id": desk.theirs.id},
               "owners-chat": OWNERS_CHAT}[where]

    reply = resolved._call(desk, context, resolved.NEAR_MISS, resolved.EMAIL)

    assert resolved._ran(desk) == [] and resolved._grants(desk) == []
    assert reply["success"] is False and not reply.get("requires_confirmation")
    assert reply["error"] == RESOLVED_SEND_UNTOLD.format(asked=resolved.NEAR_MISS, action="GMAIL_SEND_EMAIL")


def test_a_send_asked_for_by_its_name_on_autos_ticket_carries_the_line_on_the_click(desk):
    """The real chain: the card raised for the slug form, the click, the send that runs."""
    from core.public_principal import workspace_principal_name

    resolved._call(desk, resolved._session(desk), "gmail-send-email", resolved.EMAIL, "GMAIL")
    (grant,) = resolved._grants(desk)
    resolved._click(desk, grant)

    assert resolved._ran(desk) == ["GMAIL_SEND_EMAIL"]
    sent = desk.client.execute_action.call_args.kwargs["params"]
    line = send_notice(workspace_principal_name(desk.db, desk.ws))
    assert sent["body"] == f"{line}\n\n{resolved.EMAIL['body']}"
    assert grant.details["params"]["params"]["body"] == sent["body"]


# ── The platform's email routes: a person's own Send, or an agent's http_request ─────

def _route(monkeypatch):
    import api.widget_email as widget_email

    calls: List[dict] = []

    async def spine(tool_name, tool_args, agent_id=0, **kw):
        calls.append(tool_args)
        return {"success": True, "data": {"id": "m-1"}, "error": None}

    monkeypatch.setattr("modules.tools.tool_router.execute_tool", spine)
    monkeypatch.setattr(widget_email, "_get_entity_id", lambda db, workspace_id: "entity-1")
    ctx = NS(workspace_id=WS, user=NS(id="user_owner"), auth_type="clerk")
    return widget_email, ctx, calls


def _headers(agent: bool):
    from starlette.datastructures import Headers

    return Headers(headers={AGENT_REQUEST_HEADER.lower(): "1"} if agent else {})


@pytest.mark.parametrize("agent, body", [(False, LETTER), (True, f"{HARBOUR}\n\n{LETTER}")],
                         ids=["a-persons-send", "an-agents-http-request"])
def test_the_send_route_tells_only_an_agents_request(monkeypatch, agent, body):
    widget_email, ctx, calls = _route(monkeypatch)
    payload = widget_email.SendEmailRequest(to=[ANA], subject="Spring order", body=LETTER)

    result = asyncio.run(widget_email.send_email(payload, NS(headers=_headers(agent)), ctx,
                                                 FakeDb(company="Harbour Coffee")))

    assert result.success is True
    (call,) = calls
    assert call["action"] == "GMAIL_SEND_EMAIL" and call["params"]["body"] == body


@pytest.mark.parametrize("agent, body", [(False, "<p>Thanks</p>"), (True, f"<p>{HARBOUR}</p><p>Thanks</p>")],
                         ids=["a-persons-reply", "an-agents-http-request"])
def test_the_reply_route_tells_only_an_agents_request(monkeypatch, agent, body):
    widget_email, ctx, calls = _route(monkeypatch)
    payload = widget_email.ReplyEmailRequest(body="<p>Thanks</p>", is_html=True)

    result = asyncio.run(widget_email.reply_to_email("18f2a9c4d1e0b7a3", payload, NS(headers=_headers(agent)), ctx,
                                                     FakeDb(company="Harbour Coffee")))

    assert result.success is True
    (call,) = calls
    assert call["action"] == "GMAIL_REPLY_TO_EMAIL" and call["params"]["body"] == body


def test_every_http_request_an_agent_makes_says_so_whatever_headers_it_wrote():
    seen: List[Any] = []

    @marks_an_agent_request
    async def execute(executor, tool_name, parameters, agent_id, workspace_id: Optional[str] = None, trace_id=None):
        seen.append(parameters)
        return {"success": True}

    forged = {"url": "http://localhost:8000/api/emails", "method": "POST",
              "headers": {"x-automatos-agent-request": "", "Accept": "application/json"}}
    asyncio.run(execute(None, "http_request", forged, 7, workspace_id=WS))

    assert seen == [{**forged, "headers": {"Accept": "application/json", AGENT_REQUEST_HEADER: "1"}}]
    assert forged["headers"]["x-automatos-agent-request"] == ""            # the model's call is not changed in place
    assert marked_request({"url": "u"})["headers"] == {AGENT_REQUEST_HEADER: "1"}
    assert marked_request({"url": "u", "headers": [["Accept", "text/plain"]]})["headers"] == {
        "Accept": "text/plain", AGENT_REQUEST_HEADER: "1"}


def test_the_http_request_tool_is_the_marked_one():
    from modules.tools.execution import exec_shell

    assert exec_shell.execute_http_request.__code__.co_qualname == "marks_an_agent_request.<locals>.wrapped"
