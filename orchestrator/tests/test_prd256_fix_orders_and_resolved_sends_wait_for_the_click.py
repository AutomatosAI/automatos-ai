"""PRD-256 P256-FIX-RVW-3 (Decision D7): an order, and a send the executor resolves from another name, wait for the click.

D7: an agent's Composio send, publish or ORDER on a ticket Auto wrote waits for the owner's
click. The send words held messages and posts only, so SHOPIFY_CREATE_ORDER on Auto's
"Confirm the order with the supplier" ticket ran with no card. And the gate classified the
requested slug split on '_' only, while ``ComposioToolExecutor.execute`` resolves a slug form
('gmail-send-email') or a near-miss it auto-maps onto another action, and
re-checked only the deny list and the Socials post gate on the action that runs.

Now the order, booking and payment words join the send words (one list, shared with the
brief's, ``send_words``), the slug splits on any non-alphanumeric, and the executor's re-check
of a resolved action carries the click gate's (``core/composio/resolved_action``).

The chain is real from the gate through ``ComposioToolExecutor.execute`` (its validation, the
action cache, the auto-map) to the SDK call, which is recorded; the deny list and the Socials
gate say yes, the connections and the entity are stubbed.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from modules.tools.discovery.agent_sends import AGENT_SEND
from modules.tools.discovery.owner_only import asks_before_a_send, is_composio_send
from tests.test_prd256_fix_agent_sends_wait_for_the_click import _click, _grants, _session, _ticket

SUPPLIER = "orders@kerbside.example"
EMAIL = {"recipient_email": SUPPLIER, "subject": "Order confirmation", "body": "Hi Kerbside,\nPlease confirm."}
ORDER = {"email": SUPPLIER, "title": "40 Christmas boxes", "line_items": [{"variant_id": 1, "quantity": 40}]}
FETCH = {"query": "from:kerbside"}
NEAR_MISS = "GMAIL_SNED_EMAIL"   # no send word: the executor auto-maps it onto GMAIL_SEND_EMAIL
CACHED = (
    ("GMAIL", "GMAIL_SEND_EMAIL", "Send an email to a recipient"),
    ("GMAIL", "GMAIL_FETCH_EMAILS", "Fetch emails from the inbox"),
    ("SHOPIFY", "SHOPIFY_CREATE_ORDER", "Create an order"),
)


class _Executor:
    """UnifiedToolExecutor's shape: the real owner's-click gate over the real
    ComposioToolExecutor.execute, as composio_execute calls it (the name upper-cased)."""

    composio = None

    def __init__(self, db):
        self.db = db

    def _resolve_effective_call(self, tool_name, parameters):
        return str(parameters.get("action")).upper().strip(), parameters.get("params"), True

    @asks_before_a_send
    async def execute_tool(self, tool_name, parameters, agent_id=0, tenant_id=None, workspace_id=None,
                           trace_id=None, caller_context=None):
        return await _Executor.composio.execute(
            action=str(parameters["action"]).upper().strip(), params=dict(parameters["params"]),
            agent_id=agent_id, workspace_id=workspace_id, app_name=parameters.get("app_name"),
        )


def _cache(db):
    """The action cache as the sync leaves it, for the apps the workspace connected."""
    from core.models.composio_cache import ComposioActionCache, ComposioAppCache

    for app in sorted({app for app, _slug, _desc in CACHED}):
        if db.query(ComposioAppCache).filter(ComposioAppCache.app_name == app).first() is None:
            db.add(ComposioAppCache(app_name=app, app_slug=app.lower(), display_name=app.title(),
                                    categories=[], auth_schemes=[]))
    db.flush()
    for app, slug, desc in CACHED:
        if db.query(ComposioActionCache).filter(ComposioActionCache.action_name == slug).first() is None:
            db.add(ComposioActionCache(app_name=app, action_name=slug, action_slug=slug.lower().replace("_", "-"),
                                       display_name=slug.replace("_", " ").title(), description=desc))
    db.flush()


@pytest.fixture
def desk(db_session, seed_workspace, monkeypatch):
    """A workspace with Auto, the agent that buys, a ticket Auto wrote, and Gmail and Shopify connected."""
    import core.composio.entity_manager as entity_manager
    import core.composio.post_gate as post_gate
    import core.composio.tool_executor as tool_executor
    import modules.tools.execution.tool_grants as tool_grants
    import modules.tools.execution.unified_executor as unified_executor
    import services.cli_host_service as host
    from core.models.core import Agent

    monkeypatch.setattr(tool_executor, "composio_action_denial_async", AsyncMock(return_value=None))
    monkeypatch.setattr(post_gate, "post_action_refusal", AsyncMock(return_value=None))   # Socials off
    monkeypatch.setattr(entity_manager.EntityManager, "get_connected_apps", lambda self, ws: ["GMAIL", "SHOPIFY"])
    monkeypatch.setattr(tool_grants, "_notify_approval_pending", lambda grant, ws: None)
    monkeypatch.setattr(host, "publish_note_line", lambda ws, task_id, note: None)
    monkeypatch.setattr(unified_executor, "UnifiedToolExecutor", _Executor)
    _cache(db_session)
    client = MagicMock(name="client")
    client.execute_action.return_value = {"success": True, "data": None, "error": None}
    composio = tool_executor.ComposioToolExecutor(db=db_session, client=client)
    monkeypatch.setattr(composio, "get_entity_for_workspace", lambda ws: {"composio_entity_id": "entity-1"})
    monkeypatch.setattr(composio, "_resolve_file_uploads",
                        AsyncMock(side_effect=lambda action, params, workspace_id: (params, [])))
    monkeypatch.setattr(_Executor, "composio", composio)

    ws = UUID(seed_workspace())
    common = dict(description="", status="active", configuration={}, workspace_id=ws, created_by="test",
                  owner_type="workspace", owner_id=str(ws))
    auto = Agent(name="Auto", slug=f"auto-{ws}", agent_type="system", is_system_agent=True, **common)
    buyer = Agent(name="CHRISTMAS BOX", agent_type="chatbot", **common)
    db_session.add_all([auto, buyer])
    db_session.flush()
    ticket = _ticket(db_session, ws, created_by_type="agent", created_by_id=str(auto.id), agent=buyer.id)
    theirs = _ticket(db_session, ws, created_by_type="user", created_by_id="7", agent=buyer.id)
    return NS(db=db_session, ws=ws, buyer=buyer, ticket=ticket, theirs=theirs, client=client)


def _call(desk, context, action, params, app_name=None):
    parameters = {"action": action, "params": params, **({"app_name": app_name} if app_name else {})}
    return asyncio.run(_Executor(desk.db).execute_tool(
        "composio_execute", parameters, agent_id=desk.buyer.id, workspace_id=desk.ws, caller_context=context))


def _ran(desk):
    return [call.kwargs["action"] for call in desk.client.execute_action.call_args_list]


CALLS = [
    pytest.param("SHOPIFY_CREATE_ORDER", ORDER, None, "SHOPIFY_CREATE_ORDER", id="an order"),
    pytest.param("gmail-send-email", EMAIL, "GMAIL", "GMAIL_SEND_EMAIL", id="a slug form"),
    pytest.param(NEAR_MISS, EMAIL, None, "GMAIL_SEND_EMAIL", id="a near-miss the executor auto-maps"),
]


# ── On Auto's ticket each raises the card; nothing runs ──────────────────────────────

@pytest.mark.parametrize("action, params, app_name, runs_as", CALLS)
def test_on_autos_ticket_it_raises_the_card_and_nothing_runs(desk, action, params, app_name, runs_as):
    reply = _call(desk, _session(desk), action, params, app_name)

    assert _ran(desk) == []
    assert reply["requires_confirmation"] is True and reply["message"].startswith("Card raised: ")
    (grant,) = _grants(desk)
    assert grant.details[AGENT_SEND]["task_id"] == desk.ticket.id
    assert SUPPLIER in grant.question_md


def test_a_near_miss_card_names_the_action_that_would_run(desk):
    """The gate let the near-miss by (no send word); the executor's re-check of the action it
    resolved to raised the card, so the card names GMAIL_SEND_EMAIL, the action the click runs."""
    assert not is_composio_send(NEAR_MISS)
    _call(desk, _session(desk), NEAR_MISS, EMAIL)

    (grant,) = _grants(desk)
    assert "GMAIL_SEND_EMAIL" in grant.question_md
    assert grant.details["params"] == {"action": NEAR_MISS, "params": EMAIL}


def test_the_click_on_a_near_miss_card_runs_the_resolved_send_once(desk):
    from api.approval_grants import _resume_tool_call

    _call(desk, _session(desk), NEAR_MISS, EMAIL)
    (grant,) = _grants(desk)
    _click(desk, grant)

    assert _ran(desk) == ["GMAIL_SEND_EMAIL"]
    assert grant.details["executed_result"]["success"] is True
    asyncio.run(_resume_tool_call(desk.db, grant))       # the same click replayed: nothing runs again
    assert _ran(desk) == ["GMAIL_SEND_EMAIL"] and len(_grants(desk)) == 1


def test_a_near_miss_in_a_persons_chat_asks_the_owner(desk):
    reply = _call(desk, {"driving_user_id": "7", "conversation_id": "chat-1"}, NEAR_MISS, EMAIL)

    assert _ran(desk) == [] and reply["requires_confirmation"] is True and reply["owner_only"] is True
    assert len(_grants(desk)) == 1


# ── A read never asks; a person's ticket runs each directly ───────────────────────────

def test_a_fetch_never_asks(desk):
    reply = _call(desk, _session(desk), "GMAIL_FETCH_EMAILS", FETCH)

    assert reply["success"] is True and _ran(desk) == ["GMAIL_FETCH_EMAILS"]
    assert _grants(desk) == []


@pytest.mark.parametrize("action, params, app_name, runs_as", CALLS)
def test_on_a_ticket_a_person_wrote_it_runs_directly(desk, action, params, app_name, runs_as):
    reply = _call(desk, {"session_task_id": desk.theirs.id}, action, params, app_name)

    assert reply["success"] is True and _ran(desk) == [runs_as]
    assert _grants(desk) == []


# ── The words, and the seam ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("slug, sends", [
    ("SHOPIFY_CREATE_ORDER", True), ("SQUARE_PLACE_ORDER", True), ("STRIPE_CREATE_PAYMENT_INTENT", True),
    ("STRIPE_CREATE_INVOICE", True), ("STRIPE_CREATE_CHARGE", True), ("AMAZON_PURCHASE_ITEM", True),
    ("CALENDLY_BOOK_MEETING", True), ("gmail-send-email", True), ("Send Email", True),
    ("SHOPIFY_GET_ORDER", False), ("SHOPIFY_LIST_ORDERS", False), ("STRIPE_RETRIEVE_INVOICE", False),
    ("GMAIL_FETCH_EMAILS", False), ("gmail-fetch-emails", False), (NEAR_MISS, False),
])
def test_the_send_words_cover_an_order_a_booking_and_a_payment(slug, sends):
    assert is_composio_send(slug) is sends


def test_the_brief_and_the_gate_share_one_list_of_order_words():
    from modules.tools.discovery import owner_only
    from modules.tools.discovery.brief_sends import SENDS_WORDS, sends_or_orders
    from modules.tools.discovery.send_words import ORDER_WORDS

    assert ORDER_WORDS <= SENDS_WORDS
    assert {word.upper() for word in ORDER_WORDS} <= owner_only.COMPOSIO_SEND_WORDS
    assert sends_or_orders("Pay the Kerbside invoice") is True
    assert sends_or_orders("Purchase 40 Christmas boxes") is True
    assert sends_or_orders("Draft a friendly reminder for invoice HL-2291") is False   # a brief that only drafts


def test_the_executors_re_check_of_a_resolved_action_is_the_seam():
    """execute() calls ``post_action_refusal`` on the action it resolved to (beside the deny
    list's re-check); that name is the seam that carries the click gate's check."""
    import core.composio.resolved_action as resolved_action
    import core.composio.tool_executor as tool_executor

    assert tool_executor.post_action_refusal is resolved_action.post_action_refusal
    tree = ast.parse(textwrap.dedent(inspect.getsource(tool_executor.ComposioToolExecutor.execute)))
    called = [node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "post_action_refusal"]
    assert len(called) == 2   # the name asked for, then the name resolved


def test_the_socials_refusal_wins_and_outside_a_watch_the_gate_is_the_socials_one(monkeypatch):
    import core.composio.post_gate as post_gate
    from core.composio.resolved_action import checks_the_resolved_action, post_action_refusal

    check = AsyncMock(return_value="card raised")
    monkeypatch.setattr(post_gate, "post_action_refusal", AsyncMock(return_value="Socials is on"))
    with checks_the_resolved_action(check):
        assert asyncio.run(post_action_refusal("LINKEDIN_CREATE_LINKED_IN_POST", "ws")) == "Socials is on"
    check.assert_not_called()

    monkeypatch.setattr(post_gate, "post_action_refusal", AsyncMock(return_value=None))
    assert asyncio.run(post_action_refusal("GMAIL_SEND_EMAIL", "ws")) is None
    check.assert_not_called()
    with checks_the_resolved_action(check):
        assert asyncio.run(post_action_refusal("GMAIL_SEND_EMAIL", "ws")) == "card raised"
    check.assert_awaited_once_with("GMAIL_SEND_EMAIL")
