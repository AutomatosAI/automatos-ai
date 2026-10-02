"""PRD-251 Wave 2, US-205 (S2.1) — global search finds a post: ``q`` on GET /api/socials/posts.

Global search lists a Socials post by its title: it calls the posts list with
``q``, which keeps the posts whose title or brief contains the text. Pinned on
the Wave 0 API harness (the real router, gate and permission check over SQLite):

* ``q`` matches the title and the brief, case-insensitively, anywhere in them;
* it is literal: ``%``, ``_`` and ``\\`` in it match themselves, never as LIKE
  wildcards (``modules/socials/text_search.py``);
* it stays in the caller's workspace: another workspace's post with the same
  title is never listed;
* a blank ``q`` lists every post, ``q`` combines with ``status``, and a ``q``
  longer than ``text_search.QUERY_MAX_CHARS`` is 422;
* the route is a plain ``def``: it touches the database and awaits nothing (F105);
* the source picker searches through the same helper, whose escaping moved there.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials as socials_api  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from modules.socials import service, sources, text_search  # noqa: E402
from tests.test_prd251_api import WS_A, WS_B, _create, _ctx, _post  # noqa: E402

# The Wave 0 API harness: the real router and gate over in-memory SQLite, both switches on.
api = api_harness.api

ROUTE = "/api/socials/posts"


def _titles(api, **params):
    resp = api.client.get(ROUTE, params=params)
    assert resp.status_code == 200, resp.text
    return sorted(p["title"] for p in resp.json()["posts"])


def test_q_matches_the_title_case_insensitively_anywhere(api):
    _create(api, title="Harvest Club launch", brief=None)
    _create(api, title="Spring menu", brief=None)

    assert _titles(api, q="harvest") == ["Harvest Club launch"]
    assert _titles(api, q="CLUB LAUNCH") == ["Harvest Club launch"]
    # Anywhere in the title, and the text is trimmed first.
    assert _titles(api, q="  vest cl  ") == ["Harvest Club launch"]
    assert _titles(api, q="autumn") == []


def test_q_matches_the_brief_case_insensitively(api):
    _create(api, title="Countdown", brief="Three weeks to Lisbon")
    _create(api, title="Spring menu", brief=None)

    assert _titles(api, q="lisbon") == ["Countdown"]
    assert _titles(api, q="WEEKS TO") == ["Countdown"]


def test_q_is_literal_percent_underscore_and_backslash_are_not_wildcards(api):
    for title in (
        "100% organic", "1000 new members",
        "snake_case tips", "snakeXcase tips",
        "Folder C:\\drafts", "Folder C:drafts",
    ):
        _create(api, title=title, brief=None)

    # Unescaped, "%100%%" would list both 100… titles and "%%%" every post.
    assert _titles(api, q="100%") == ["100% organic"]
    assert _titles(api, q="%") == ["100% organic"]
    # Unescaped, "_" matches any one character.
    assert _titles(api, q="e_c") == ["snake_case tips"]
    assert _titles(api, q="_") == ["snake_case tips"]
    # The escape character itself is escaped: ":\" is a colon and a backslash.
    assert _titles(api, q=":\\") == ["Folder C:\\drafts"]


def test_q_stays_in_the_callers_workspace(api):
    api.ctx = _ctx(WS_B)
    _create(api, title="Harvest Club launch", brief="Their brief")
    api.ctx = _ctx(WS_A)
    mine = _create(api, title="Harvest Club recap", brief=None)

    listed = api.client.get(ROUTE, params={"q": "harvest club"}).json()
    assert [p["id"] for p in listed["posts"]] == [mine["id"]] and listed["total"] == 1
    assert _titles(api, q="their brief") == []
    assert [p.title for p in service.list_posts(api.session, WS_B, q="harvest")] == ["Harvest Club launch"]


def test_a_blank_q_lists_every_post_and_q_combines_with_status(api):
    _create(api, title="Harvest draft", brief=None)
    waiting = _create(api, title="Harvest review", brief=None)
    assert _post(api, waiting["id"], "submit").status_code == 200
    _create(api, title="Winter menu", brief=None)

    assert _titles(api, q="   ") == ["Harvest draft", "Harvest review", "Winter menu"]
    assert _titles(api, q="harvest", status="needs_approval") == ["Harvest review"]
    assert _titles(api, q="harvest", status="draft") == ["Harvest draft"]


def test_a_q_longer_than_the_limit_is_refused(api):
    _create(api)
    assert api.client.get(ROUTE, params={"q": "x" * text_search.QUERY_MAX_CHARS}).status_code == 200
    assert api.client.get(ROUTE, params={"q": "x" * (text_search.QUERY_MAX_CHARS + 1)}).status_code == 422


def test_the_service_takes_q_title_or_brief(api):
    _create(api, title="Harvest Club launch", brief=None)
    _create(api, title="Weekend", brief="The HARVEST is in")
    _create(api, title="Winter menu", brief=None)

    assert sorted(p.title for p in service.list_posts(api.session, WS_A, q="harvest")) == [
        "Harvest Club launch", "Weekend",
    ]
    assert len(service.list_posts(api.session, WS_A, q=None)) == 3


def test_the_list_route_is_a_plain_def():
    assert not inspect.iscoroutinefunction(socials_api.list_social_posts)


@pytest.mark.parametrize(
    "text, pattern",
    [
        ("100%", "%100\\%%"),
        ("snake_case", "%snake\\_case%"),
        ("C:\\drafts", "%c:\\\\drafts%"),
        ("MiXeD Case", "%mixed case%"),
    ],
)
def test_contains_pattern_escapes_and_folds(text, pattern):
    assert text_search.contains_pattern(text) == pattern


def test_the_source_picker_searches_through_the_same_helper():
    assert sources.contains_pattern is text_search.contains_pattern
    assert sources.escape_like is text_search.escape_like
    assert sources.LIKE_ESCAPE == text_search.LIKE_ESCAPE
    assert not hasattr(sources, "_contains") and not hasattr(sources, "_escape_like")
