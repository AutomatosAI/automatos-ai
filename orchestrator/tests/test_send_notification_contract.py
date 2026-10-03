"""platform_send_notification offers only values its handler accepts.

Its severity enum offered "warning", which the handler refuses, and left out
"approval" and "task", which it takes (the Monday HARNESS cadence asks for
severity=approval). The description asks for approval requests, but
approval_pending was not an event_type it offered, and the title and message
the owner reads carried no description.
"""
from core.services.notification_dispatcher import VALID_EVENT_TYPES
from modules.tools.discovery.action_registry import get_action_registry
from modules.tools.discovery.handlers_auto_reporting import _VALID_SEVERITIES, _VALID_STATUSES


def _properties():
    return get_action_registry().get("platform_send_notification").parameters["properties"]


def test_every_offered_value_is_one_the_handler_accepts():
    props = _properties()
    assert set(props["severity"]["enum"]) == set(_VALID_SEVERITIES)
    assert set(props["status"]["enum"]) <= set(_VALID_STATUSES)
    assert set(props["event_type"]["enum"]) <= set(VALID_EVENT_TYPES)
    assert "approval_pending" in props["event_type"]["enum"]


def test_the_content_fields_say_what_they_carry():
    props = _properties()
    assert props["title"].get("description") and props["message"].get("description")
