"""PRD-251 Wave 3, US-309 (S3.5-check, D14) — one way out, end to end.

Now that the publisher has a way through the post gate, the other ways stay shut.
On the Wave 1 harness (``tests/test_prd251w1_post_gate.py``: the real settings reads
on SQLite; the Composio SDK, the entity and the LinkedIn direct API mocked) with every
seeded channel connected in a Socials-on workspace:

* every publish action of every seeded channel (the ones its sequences run, and the
  ones it never offers) is refused to an agent on both executor paths: the name asked
  for (``tool_executor.py``, before access validation) and the name validation
  resolved it to (after it); no Composio call is made;
* the publisher's own call (``execute_with_uploads`` with ``PLATFORM_PUBLISHER``)
  passes the same gate, and the same call without it is refused;
* ``PLATFORM_PUBLISHER`` is named only where it is defined and by the publisher's
  step runner, and only the step runner passes a way through;
* no Socials agent tool, and nothing they run, publishes, schedules or approves.
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w1_post_gate as gate_harness  # noqa: E402
from core.composio.post_gate import PLATFORM_PUBLISHER  # noqa: E402
from modules.socials.capabilities import SEEDED_CHANNELS  # noqa: E402
from tests.test_prd251w1_post_gate import WS_ON, _Agent, _assert_refused  # noqa: E402
from tests.test_prd251w2_channel_gate import _connect  # noqa: E402

env = gate_harness.env

PUBLISH_ACTIONS = sorted(
    (toolkit, slug) for toolkit, adapter in SEEDED_CHANNELS.items() for slug in adapter.publish_actions
)
PUBLISH_CODE = {"modules/socials/publish_steps.py"}
DEFINED_IN = "core/composio/post_gate.py"


def _connect_every_channel(env):
    _connect(env, *(toolkit.upper() for toolkit in SEEDED_CHANNELS))


def test_every_seeded_channel_has_publish_actions_to_check():
    assert {toolkit for toolkit, _ in PUBLISH_ACTIONS} == set(SEEDED_CHANNELS)
    # The sequences' own publish steps, and the stale or URL-pull ones they never offer.
    assert {"LINKEDIN_CREATE_LINKED_IN_POST", "TWITTER_CREATION_OF_A_POST", "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH",
            "TIKTOK_UPLOAD_VIDEO", "YOUTUBE_UPLOAD_VIDEO", "TWITTER_CREATE_TWEET", "TIKTOK_PUBLISH_VIDEO",
            "TIKTOK_POST_PHOTO", "INSTAGRAM_CREATE_POST"} <= {slug for _, slug in PUBLISH_ACTIONS}


@pytest.mark.asyncio
@pytest.mark.parametrize("toolkit, slug", PUBLISH_ACTIONS)
async def test_an_agent_asking_for_a_publish_action_is_refused_before_validation(env, monkeypatch, toolkit, slug):
    _connect_every_channel(env)
    agent = _Agent(monkeypatch)

    result = await agent.execute(slug, WS_ON, {"text": "Launch"})

    _assert_refused(result, slug)
    agent.db.query.assert_not_called()  # refused before access validation
    agent.assert_nothing_ran()


def _alias(slug, toolkit):
    """A name that is not itself refused but that validation resolves onto ``slug``."""
    return slug[len(toolkit) + 1:].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("toolkit, slug", PUBLISH_ACTIONS)
async def test_an_agent_whose_name_validation_resolves_onto_a_publish_action_is_refused(env, monkeypatch, toolkit, slug):
    from core.composio.post_gate import post_action_refusal

    _connect_every_channel(env)
    agent = _Agent(monkeypatch)
    agent.validates(slug, toolkit.upper())
    asked = _alias(slug, toolkit)
    assert await post_action_refusal(asked, WS_ON) is None  # the first path lets the alias by

    result = await agent.execute(asked, WS_ON, {"text": "Launch"}, app_name=toolkit.upper())

    _assert_refused(result, slug)
    agent.assert_nothing_ran()


@pytest.mark.asyncio
@pytest.mark.parametrize("toolkit, slug", [
    ("linkedin", "LINKEDIN_CREATE_VIDEO_POST"),
    ("tiktok", "TIKTOK_UPLOAD_VIDEO"),
    ("youtube", "YOUTUBE_UPLOAD_VIDEO"),
])
async def test_the_publishers_call_passes_the_gate_and_the_same_call_without_it_is_refused(env, monkeypatch, toolkit, slug):
    _connect_every_channel(env)
    agent = _Agent(monkeypatch)

    refused = await agent.executor.execute_with_uploads(slug, {"text": "x"}, agent_id=0, workspace_id=WS_ON, app_name=toolkit.upper())
    _assert_refused(refused, slug)
    agent.assert_nothing_ran()

    once = MagicMock(name="ComposioSendOnce")  # the handle with the SDK's re-send off
    once.tools.execute.return_value = {"successful": True, "data": {"ok": True}}
    agent.executor.client._composio_once = once
    passed = await agent.executor.execute_with_uploads(
        slug, {"text": "x"}, agent_id=0, workspace_id=WS_ON, app_name=toolkit.upper(), way_through=PLATFORM_PUBLISHER,
    )
    assert passed["success"] is True
    once.tools.execute.assert_called_once()  # sent once: never through the re-sending handle
    agent.sdk.tools.execute.assert_not_called()


def _source_files():
    for path in (_ORCH).rglob("*.py"):
        rel = path.relative_to(_ORCH).as_posix()
        if rel.startswith(("tests/", "alembic/", ".venv/")) or "/tests/" in rel:
            continue
        yield rel, path


def test_the_way_through_is_named_only_where_it_is_defined_and_by_the_publisher():
    holders = {rel for rel, path in _source_files() if "PLATFORM_PUBLISHER" in path.read_text(encoding="utf-8")}
    assert holders == {DEFINED_IN} | PUBLISH_CODE


def test_only_the_publisher_passes_a_way_through():
    passers = set()
    for rel, path in _source_files():
        source = path.read_text(encoding="utf-8")
        if "way_through" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            for keyword in getattr(node, "keywords", ()) if isinstance(node, ast.Call) else ():
                forwarded = isinstance(keyword.value, ast.Name) and keyword.value.id == "way_through"
                if keyword.arg == "way_through" and not forwarded and not (
                    isinstance(keyword.value, ast.Constant) and keyword.value.value is None
                ):
                    passers.add(rel)
    assert passers == PUBLISH_CODE


# ---------------------------------------------------------------------------
# The Socials agent tools stay draft-only
# ---------------------------------------------------------------------------

WAY_OUT = re.compile(r"publisher|publishing|publish_steps|publish_records|schedule_jobs|begin_publish|begin_retry|"
                     r"PLATFORM_PUBLISHER|approve|schedule\b|unschedule", re.IGNORECASE)


def test_no_socials_agent_tool_publishes_schedules_or_approves():
    from api import socials as socials_api
    from modules.tools.discovery import handlers_socials
    from modules.tools.discovery.action_registry import get_action_registry

    tools = {a.name: a for a in get_action_registry().get_all() if a.category == "socials"}
    assert set(tools) == {
        "platform_create_social_post", "platform_update_social_post", "platform_submit_social_post",
        "platform_get_social_post", "platform_list_social_posts",
    }
    for action in tools.values():
        assert not re.search(r"approv|schedul|publish", action.name, re.IGNORECASE)
    # The handlers, and the flows they share with the routes, never reach the way out.
    code = inspect.getsource(handlers_socials)
    for flow in (socials_api.create_post, socials_api.edit_post, socials_api.submit_post, socials_api.render_post):
        code += inspect.getsource(flow)
    imported = {node.module for node in ast.walk(ast.parse(inspect.getsource(handlers_socials)))
                if isinstance(node, ast.ImportFrom) and node.module}
    assert not {m for m in imported if WAY_OUT.search(m)}
    names = {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Attribute)
    }
    assert {n for n in names if WAY_OUT.fullmatch(n) or n in {"begin_publish", "begin_retry", "launch", "run_publish"}} == set()
