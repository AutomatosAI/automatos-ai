"""Composio catalogue payloads -> the records the catalogue sync stores.

The toolkit and bulk-action mapping moved out of ``ComposioClient`` into
``core.composio.catalog_records`` so the client's fetches fit the code-shape
rules. These pin the field fallbacks the move had to keep: where a toolkit's
triggers, description, logo and categories come from, and which bulk ``/tools``
items become actions, under which name and toolkit. Pure: no DB, no network.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.composio.catalog_records import (
    bulk_actions_from_items,
    normalize_triggers,
    toolkit_app_record,
)


def test_toolkit_with_an_sdk_meta_model_keeps_every_field():
    github = SimpleNamespace(
        slug="github",
        name="GitHub",
        auth_schemes=["OAUTH2"],
        categories=["developer"],
        meta=SimpleNamespace(
            description="Code hosting",
            tools_count=42,
            logo="https://logo/github.png",
            triggers=[{"slug": "GITHUB_PUSH_EVENT", "name": "Push"}],
        ),
    )

    assert toolkit_app_record(github, trigger_map={}) == {
        "name": "github",
        "display_name": "GitHub",
        "description": "Code hosting",
        "logo_url": "https://logo/github.png",
        "categories": ["developer"],
        "auth_schemes": ["OAUTH2"],
        "action_count": 42,
        "triggers": [{"name": "Push", "display_name": "Push", "description": ""}],
        "trigger_count": 1,
    }


def test_toolkit_with_a_dict_meta_falls_back_to_the_trigger_map_and_meta_categories():
    slack = SimpleNamespace(
        slug="Slack",
        name="Slack",
        auth_schemes=None,
        categories=None,
        meta={"logo": "https://logo/slack.png", "categories": [SimpleNamespace(name="Communication"), "chat"]},
    )
    trigger_map = {"slack": [{"name": "SLACK_NEW_MESSAGE", "display_name": "New message", "description": "A message"}]}

    assert toolkit_app_record(slack, trigger_map) == {
        "name": "Slack",
        "display_name": "Slack",
        "description": "",
        "logo_url": "https://logo/slack.png",
        "categories": ["Communication", "chat"],
        "auth_schemes": [],
        "action_count": 0,
        "triggers": [{"name": "SLACK_NEW_MESSAGE", "display_name": "New message", "description": "A message"}],
        "trigger_count": 1,
    }


def test_toolkit_without_meta_has_no_logo_count_or_categories():
    bare = SimpleNamespace(slug="notion", name="Notion", auth_schemes=[], description="Docs")

    record = toolkit_app_record(bare, trigger_map={})

    assert record["description"] == "Docs"
    assert record["logo_url"] is None
    assert record["action_count"] == 0
    assert record["categories"] == []
    assert record["triggers"] == [] and record["trigger_count"] == 0


def test_normalize_triggers_accepts_names_and_dicts_and_drops_the_rest():
    assert normalize_triggers(None) == []
    assert normalize_triggers("not a list") == []
    assert normalize_triggers(["PUSH", {"slug": "STAR", "trigger_name": "ignored"}, 7]) == [
        {"name": "PUSH", "display_name": "PUSH", "description": ""},
        {"name": "STAR", "display_name": "STAR", "description": ""},
    ]


def test_bulk_items_become_actions_named_by_enum_slug_or_display_name():
    circular: dict = {}
    circular["self"] = circular
    items = [
        {
            "slug": "gmail-fetch-emails",
            "name": "Fetch emails",
            "toolkit": {"slug": "gmail"},
            "description": "Fetch",
            "version": "20260101_00",
        },
        {"type": "mcp", "name": "not a function tool"},
        {
            "enum": "slack_send_message",
            "function": {"name": "Send message", "description": "Send", "parameters": {"type": "object"}},
            "toolkit_slug": "slack",
            "tags": ["msg"],
            "is_premium": True,
        },
        {"name": "NOTION_CREATE_PAGE"},
        {"name": "orphan"},
        "not a dict",
        {"slug": "issue-create", "toolkit": "linear", "circular": circular},
    ]

    actions = bulk_actions_from_items(items)

    assert [(a["app_name"], a["name"]) for a in actions] == [
        ("GMAIL", "GMAIL_FETCH_EMAILS"),
        ("SLACK", "SLACK_SEND_MESSAGE"),
        ("NOTION", "NOTION_CREATE_PAGE"),
        ("LINEAR", "ISSUE_CREATE"),
    ]
    gmail, slack, notion, linear = actions
    assert gmail == {
        "app_name": "GMAIL",
        "name": "GMAIL_FETCH_EMAILS",
        "display_name": "Fetch emails",
        "description": "Fetch",
        "parameters": {},
        "response_schema": {},
        "tags": [],
        "category": None,
        "is_premium": False,
        "metadata": {"version": "20260101_00"},
    }
    assert slack["display_name"] == "Send message"
    assert slack["description"] == "Send"
    assert slack["parameters"] == {"type": "object"}
    assert slack["tags"] == ["msg"] and slack["is_premium"] is True
    assert slack["metadata"] == {"enum": "slack_send_message"}
    assert notion["metadata"] == {}
    # A value that cannot be serialised (a circular reference) is left out of the metadata.
    assert linear["metadata"] == {}
