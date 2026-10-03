"""F197, the persona's #1 after night 6 (2 Oct): "During the outage nothing told me,
and nothing said when it was back."

- In chat, the credit refusal arrived wrapped in another error, and the reply
  showed it raw: "Auto could not finish this reply: Error code: 402 - {...
  'user_id': ...". It now reads as the provider's own 402 does: plain words,
  no payload.
- When credit comes back, the bell says so once, with how many of the reports
  that stopped are running again. Before, the reruns started silently.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
# Night 6's chat failure (the account id is made up here).
WRAPPED_402 = ("Task execution failed after 2 attempts: Error code: 402 - {'error': {'message': 'This request "
               "would exceed your available credits given your current in-flight requests.', 'code': 402}, "
               "'user_id': 'user_2mYFakeAccount0000'}")


@pytest.fixture(autouse=True)
def _fresh_outages():
    from core.llm import credit

    credit.reset()
    yield
    credit.reset()


def test_a_wrapped_credit_refusal_in_chat_is_said_plainly():
    from consumers.chatbot.turn_errors import CODE_PROVIDER_FAILED, describe_turn_error

    said = describe_turn_error(RuntimeError(WRAPPED_402), agent_name="Auto")
    assert said.code == CODE_PROVIDER_FAILED
    assert "out of credits" in said.message and "Top up the provider account" in said.message
    assert "402" not in said.message and "user_" not in said.message    # night 6: the raw payload


def test_another_wrapped_error_keeps_its_own_words():
    from consumers.chatbot.turn_errors import CODE_TURN_FAILED, describe_turn_error

    said = describe_turn_error(RuntimeError("the knowledge base is still indexing"), agent_name="Auto")
    assert said.code == CODE_TURN_FAILED and "still indexing" in said.message


class _Db:
    def close(self):
        pass


@pytest.fixture
def bell(monkeypatch):
    """The bell's dispatcher and the session it is given, recorded instead of written."""
    import core.database.database as database
    import core.services.notification_dispatcher as dispatcher

    calls = []

    class _Bell:
        def __init__(self, db, workspace_id):
            self.workspace_id = workspace_id

        async def dispatch(self, **kwargs):
            calls.append((self.workspace_id, kwargs))
            return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(dispatcher, "NotificationDispatcher", _Bell)
    monkeypatch.setattr(database, "SessionLocal", _Db)
    monkeypatch.setattr("services.watch_rerun.launch_execution", lambda rerun: None)
    return calls


def test_the_bell_says_credit_is_back_once_with_the_reports_running_again(bell, monkeypatch):
    from core.llm import credit

    reruns = [NS(retry_of="exec-1160", execution_id="exec-a"), NS(retry_of="exec-1171", execution_id="exec-b")]
    monkeypatch.setattr(credit, "_stage_reruns", lambda workspace_id: reruns)
    credit.note_refused(WS, 8000)                       # the outage
    assert credit._credit_is_back(WS, 8000)             # a call as big as the refused ones works again

    asyncio.run(credit._rerun_marked(WS))
    asyncio.run(credit._rerun_marked(WS))               # nothing more to say the second time

    assert bell == [(WS, {"event_type": credit.CREDIT_BACK_EVENT, "title": "AI credit is back",
                          "message": credit.credit_back_notice(2)})]


def test_no_outage_means_no_notice(bell, monkeypatch):
    from core.llm import credit

    monkeypatch.setattr(credit, "_stage_reruns", lambda workspace_id: [])
    assert credit._credit_is_back(WS, 8000)             # the first big success since start, no outage
    asyncio.run(credit._rerun_marked(WS))
    assert bell == []


def test_the_notice_counts_the_reports_running_again():
    from core.llm import credit

    assert credit.credit_back_notice(0) == credit.CREDIT_BACK_NOTICE
    assert credit.credit_back_notice(1).endswith("One scheduled report that stopped is running again.")
    assert "3 scheduled reports that stopped are running again." in credit.credit_back_notice(3)


def test_the_owner_can_route_it_like_any_bell_event():
    from core.llm.credit import CREDIT_BACK_EVENT
    from core.services.notification_dispatcher import VALID_EVENT_TYPES

    assert CREDIT_BACK_EVENT in VALID_EVENT_TYPES
