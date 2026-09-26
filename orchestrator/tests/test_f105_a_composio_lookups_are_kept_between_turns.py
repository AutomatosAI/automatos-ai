"""F105 (26 Sep): a Composio lookup a turn has just made is answered from memory.

Every chat turn with Composio apps asked the SDK again for answers it had just
given: the hint enrichment fetched each matched app's whole action list
(tools.get, 1.4-5.6 s a turn on 26 Sep, every turn), and the tool search asked
the same question again. Both are kept for COMPOSIO_LOOKUP_CACHE_TTL_SECONDS:
an app's actions by app (fetched as a placeholder user, the same for every
workspace), a search by the workspace's Composio entity and everything it asked.

Fake SDK clients count the calls. No database, no SDK.
"""
import time
from types import SimpleNamespace

from config import config
from core.composio.client import ComposioClient
from modules.tools.services.composio_hint_service import ComposioHintService

SEND_EMAIL = {"name": "GMAIL_SEND_EMAIL", "parameters": {
    "type": "object", "properties": {"to": {"type": "string", "description": "Recipient address"}},
    "required": ["to"],
}}


class _AppActions:
    """The client's get_app_actions, counted."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def get_app_actions(self, app, **kwargs):
        self.asked.append(app)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def _turn(monkeypatch, client, app="GMAIL"):
    """One turn's hint enrichment; returns the parameter hints it produced."""
    monkeypatch.setattr("core.composio.client.get_composio_client", lambda: client)
    params = {}
    ComposioHintService(db=None)._enrich_params_from_sdk([(app, ["GMAIL_SEND_EMAIL"])], params)
    return params


def test_two_turns_fetch_an_apps_actions_from_the_sdk_once(monkeypatch):
    client = _AppActions([SEND_EMAIL])

    first, second = _turn(monkeypatch, client), _turn(monkeypatch, client)

    assert client.asked == ["GMAIL"]
    assert first == second and "to" in first["GMAIL_SEND_EMAIL"]


def test_an_empty_action_list_is_asked_for_again(monkeypatch):
    """get_app_actions answers [] when the SDK call fails, so [] is never kept."""
    client = _AppActions([], [SEND_EMAIL])

    first, second = _turn(monkeypatch, client), _turn(monkeypatch, client)

    assert client.asked == ["GMAIL", "GMAIL"]
    assert first == {} and "GMAIL_SEND_EMAIL" in second


class _Search:
    """The SDK's tools.get for a step search, counted; fails while ``failing``."""

    def __init__(self, failing=0):
        self.failing = failing
        self.asked = []

    def get(self, **kwargs):
        self.asked.append(kwargs)
        if self.failing:
            self.failing -= 1
            raise RuntimeError("Composio: 502")
        return [{"type": "function", "function": {"name": "GMAIL_SEND_EMAIL", "parameters": {}}}]


def _client(search):
    client = ComposioClient.__new__(ComposioClient)  # no key, no SDK
    client._toolset = SimpleNamespace(tools=search)  # what the lazy `toolset` property hands out
    return client


def _search(client, query="Email the café its invoice.", entity="ws-c1", apps=("gmail",)):
    return client.search_actions_for_step(query, list(apps), entity, limit=5)


def test_the_same_step_search_asks_the_sdk_once_per_workspace():
    search = _Search()
    client = _client(search)

    first, again = _search(client), _search(client)
    other_workspace = _search(client, entity="ws-other")
    other_question = _search(client, query="Draft the newsletter.")

    assert [(ask["user_id"], ask["search"]) for ask in search.asked] == [
        ("ws-c1", "Email the café its invoice."),
        ("ws-other", "Email the café its invoice."),
        ("ws-c1", "Draft the newsletter."),
    ]
    assert first == again == other_workspace == other_question


def test_a_kept_search_is_handed_out_as_a_copy():
    client = _client(_Search())
    first = _search(client)
    first[0]["schema"]["function"]["name"] = "EDITED_BY_ONE_TURN"

    assert _search(client)[0]["schema"]["function"]["name"] == "GMAIL_SEND_EMAIL"


def test_a_failed_search_is_asked_again():
    search = _Search(failing=1)
    client = _client(search)

    assert _search(client) == []
    assert [r["action_name"] for r in _search(client)] == ["GMAIL_SEND_EMAIL"]
    assert len(search.asked) == 2


def test_a_kept_answer_lasts_the_ttl(monkeypatch):
    monkeypatch.setattr(config, "COMPOSIO_LOOKUP_CACHE_TTL_SECONDS", 0.3, raising=False)
    search = _Search()
    client = _client(search)

    _search(client), _search(client)
    assert len(search.asked) == 1
    time.sleep(0.4)
    _search(client)
    assert len(search.asked) == 2
