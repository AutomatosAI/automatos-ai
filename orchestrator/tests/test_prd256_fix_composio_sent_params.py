"""PRD-256 P256-FIX-RVW-31: one reading of what a composio_execute call sends.

The owner's-click card was built from ``params`` alone while exec_composio also sent the
``parameters`` alias and every stray top-level key: a ``bcc`` beside ``params`` went out on
the click unshown. ``composio_params.sent_params`` is now the one reading; exec_composio
sends it and the executor's ``_resolve_effective_call`` (the card's source) returns it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

from modules.tools.execution import exec_composio
from modules.tools.execution.composio_params import sent_params, stray_params
from modules.tools.execution.unified_executor import UnifiedToolExecutor

TO, BCC = "orders@kerbside.example", "x@other.test"

SHAPES = [
    ({"action": "GMAIL_SEND_EMAIL", "params": {"recipient_email": TO}, "bcc": BCC},
     {"recipient_email": TO, "bcc": BCC}),
    ({"action": "GMAIL_SEND_EMAIL", "parameters": {"recipient_email": TO, "bcc": BCC}},
     {"recipient_email": TO, "bcc": BCC}),
    ({"action": "GMAIL_SEND_EMAIL", "recipient_email": TO, "app_name": "gmail"}, {"recipient_email": TO}),
    ({"action": "GMAIL_SEND_EMAIL", "params": {"recipient_email": TO}, "recipient_email": "decoy@other.test"},
     {"recipient_email": TO}),                                       # the explicit param wins
    ({"action": "GMAIL_SEND_EMAIL", "params": {"recipient_email": TO}, "parameters": {"bcc": BCC}},
     {"recipient_email": TO}),                                       # params first, as exec_composio read it
    ({"action": "GMAIL_SEND_EMAIL", "params": "not an object", "bcc": BCC}, {"bcc": BCC}),
]


@pytest.mark.parametrize("call, sent", SHAPES)
def test_sent_params_reads_params_or_parameters_and_the_strays(call, sent):
    assert sent_params(call) == sent


def test_sent_params_never_mutates_the_call_and_reads_no_meta_key_as_a_param():
    call = {"action": "X", "action_name": "X", "app": "GMAIL", "app_name": "GMAIL", "params": {"to": TO}, "cc": BCC}
    before = {**call, "params": dict(call["params"])}

    assert sent_params(call) == {"to": TO, "cc": BCC} and stray_params(call) == {"cc": BCC}
    assert call == before
    assert sent_params(None) == {} and sent_params("text") == {}


class _Composio:
    def __init__(self):
        self.calls = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return {"success": False, "error": "fake"}


@pytest.mark.parametrize("call, sent", SHAPES)
def test_exec_composio_sends_exactly_sent_params(call, sent):
    composio = _Composio()
    executor = NS(composio_executor=composio, db=None)

    asyncio.run(exec_composio.execute_composio_execute(executor, "composio_execute", call, 1, workspace_id=uuid4()))

    assert [c["params"] for c in composio.calls] == [sent] and composio.calls[0]["action"] == "GMAIL_SEND_EMAIL"


@pytest.mark.parametrize("call, sent", SHAPES)
def test_the_executor_hands_the_card_the_params_the_send_carries(call, sent):
    name, params, composio = UnifiedToolExecutor._resolve_effective_call(NS(), "composio_execute", call)

    assert (name, params, composio) == ("GMAIL_SEND_EMAIL", sent, True)
