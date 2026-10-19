"""PRD-263 US-013 / US-019 — a failed tool run is classed auth, connection, rate_limit or other.

``services.harness_metrics.classify_tool_error`` reads the run's ``error_code`` first
(``RECONNECT_CODES``) and otherwise the first of ``RECONNECT_PATTERNS`` to match the code
and message together, so a row written before US-018 stored codes (its ``error_code``
always NULL) still classes. Auth and connection failures become "reconnect <app>" in the
HARNESS report, never a bad tool (US-019).
"""
from __future__ import annotations

import pytest

from services.harness_metrics import (
    ERROR_CLASS_AUTH,
    ERROR_CLASS_CONNECTION,
    ERROR_CLASS_OTHER,
    ERROR_CLASS_RATE_LIMIT,
    RECONNECT_CLASSES,
    RECONNECT_CODES,
    classify_tool_error,
)

# Composio's words for a connected account whose sign-in has lapsed.
COMPOSIO_EXPIRED = "Error executing action: Connected account ca_8Hk2 has expired. Reconnect it to continue."


def test_a_composio_expired_auth_message_with_no_code_is_auth():
    assert classify_tool_error(None, COMPOSIO_EXPIRED) == ERROR_CLASS_AUTH


def test_a_connection_refused_message_is_connection():
    assert classify_tool_error(None, "httpx.ConnectError: [Errno 111] Connection refused") == ERROR_CLASS_CONNECTION


def test_integrations_unavailable_is_connection_by_its_code_and_by_its_message():
    from modules.tools import tool_router

    refused = tool_router._integrations_unavailable_result("composio_execute", "trace-1")
    assert classify_tool_error(refused["error_code"], refused["error"]) == ERROR_CLASS_CONNECTION
    assert RECONNECT_CODES[tool_router.INTEGRATIONS_UNAVAILABLE] == ERROR_CLASS_CONNECTION
    # A row written before US-018 kept the message only.
    assert classify_tool_error(None, refused["error"]) == ERROR_CLASS_CONNECTION


def test_an_unrelated_message_is_other():
    assert classify_tool_error(None, "channel_not_found") == ERROR_CLASS_OTHER
    assert classify_tool_error(None, None) == ERROR_CLASS_OTHER
    assert classify_tool_error("", "max tokens exceeded for this model") == ERROR_CLASS_OTHER


@pytest.mark.parametrize("code, message, expected", [
    (" 401 ", None, ERROR_CLASS_AUTH),
    (None, "401 Client Error: Unauthorized for url", ERROR_CLASS_AUTH),
    (None, "LinkedIn access token may be expired and no refresh_token is set.", ERROR_CLASS_AUTH),
    (None, "'SLACK' is assigned to agent 12 but is not connected for this workspace. Connect it first, then retry.",
     ERROR_CLASS_CONNECTION),
    (None, "No connected account found for user default", ERROR_CLASS_CONNECTION),
    (None, "429 Too Many Requests", ERROR_CLASS_RATE_LIMIT),
    # A known code decides, whatever the message says.
    ("integrations_unavailable", "401 Unauthorized", ERROR_CLASS_CONNECTION),
])
def test_codes_and_messages_class_as_the_code_that_writes_them_means(code, message, expected):
    assert classify_tool_error(code, message) == expected


def test_only_auth_and_connection_are_reconnect_classes():
    assert RECONNECT_CLASSES == {ERROR_CLASS_AUTH, ERROR_CLASS_CONNECTION}
