"""PRD-251 Wave 2, US-203 (D14 completed, S3.5) — one way out: the post gate refuses
what the channel registry classes as publishing to a connected channel.

The Wave 1 gate (``core/composio/post_gate.py``, US-118) refused the actions on its
data list (``socials.post_actions``). It now ALSO refuses any action the Socials
registry (``modules/socials/capabilities.py``) classes as ``publish`` for a channel
connected in the workspace, at the same call sites, after the Wave 0 deny list.
Pinned on Wave 1's harness (``tests/test_prd251w1_post_gate.py``: the real settings
reads on SQLite, with the Composio SDK, the LinkedIn direct API and a Playbook step's
spine mocked), the workspace's connections and cached actions in the registry's
own tables:

* TikTok connected: an agent's TIKTOK_UPLOAD_VIDEO, which is not on the list, is
  refused with the message and no Composio call, through the executor, both agent
  tool paths and a Playbook step; so are YouTube's upload and the publish actions
  the data never offers; taking a registry publish action off the list lets nothing
  through;
* the deny list still wins; a Socials-off workspace (either switch) runs as before;
  a channel not connected in the workspace, and a channel's upload, read and status
  steps, are not the registry's to refuse;
* the generic adapter: a connected toolkit's qualifying post action is refused; a
  post-named action whose schema does not qualify (Slack's message) runs;
* fail closed: a registry read that cannot complete refuses a candidate (ERROR);
* the way through: only the platform publisher passes;
* F105: the registry is read in a worker thread, once, only for a candidate the
  list does not hold, and only where Socials is not off.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from sqlalchemy.orm import sessionmaker  # noqa: E402

import core.composio.deny_list as deny_list  # noqa: E402
import core.composio.post_gate as post_gate  # noqa: E402
import modules.socials.capabilities as capabilities  # noqa: E402
import tests.test_prd251w1_post_gate as gate_harness  # noqa: E402
from core.models.composio import ComposioConnection, ComposioEntity  # noqa: E402
from core.models.composio_cache import ComposioActionCache  # noqa: E402
from modules.socials.capabilities import GENERIC, SEEDED, publish_candidate  # noqa: E402
from tests.test_prd251w1_post_gate import REFUSAL, WS_OFF, WS_ON, _Agent, _assert_refused, _set, _Step  # noqa: E402

env = gate_harness.env  # the Wave 1 harness's fixture (pytest finds it by this name)

TIKTOK_UPLOAD = "TIKTOK_UPLOAD_VIDEO"
REDDIT_POST = "REDDIT_CREATE_REDDIT_POST"
REDDIT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "video_file": {"type": "object", "file_uploadable": True}},
}
SLACK_MESSAGE = "SLACK_CHAT_POST_MESSAGE"
SLACK_SCHEMA = {  # a post word in its name, but no media field: not a channel's post action
    "type": "object",
    "properties": {"channel": {"type": "string"}, "text": {"type": "string"}, "icon_url": {"type": "string"}},
}
BLOCKED = "This action is blocked in Automatos: "


def _connect(env, *apps, workspace=WS_ON):
    db = sessionmaker(bind=env.engine)()
    try:
        entity = db.query(ComposioEntity).filter(ComposioEntity.workspace_id == workspace).first()
        if entity is None:
            entity = ComposioEntity(workspace_id=workspace, composio_entity_id=str(workspace))
            db.add(entity)
            db.flush()
        for app in apps:
            db.add(ComposioConnection(entity_id=entity.id, app_name=app, status="active", connection_id=f"ca_{app.lower()}"))
        db.commit()
    finally:
        db.close()


def _cache(env, app, slug, parameters):
    db = sessionmaker(bind=env.engine)()
    try:
        db.add(ComposioActionCache(
            app_name=app, action_name=slug, action_slug=slug.lower().replace("_", "-"),
            display_name=slug.replace("_", " ").title(), parameters=parameters,
        ))
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# A registry publish action of a connected channel: refused, with no Composio call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_agents_tiktok_upload_is_refused_in_a_socials_workspace_with_tiktok_connected(env, monkeypatch):
    _connect(env, "TIKTOK")
    assert TIKTOK_UPLOAD not in gate_harness.POST_SEED  # not on the list: the registry refuses it
    agent = _Agent(monkeypatch)

    result = await agent.execute(TIKTOK_UPLOAD, WS_ON, {"file_to_upload": "/workspace/launch.mp4", "caption": "Hi"})

    _assert_refused(result, TIKTOK_UPLOAD)
    agent.db.query.assert_not_called()  # refused before access validation
    agent.assert_nothing_ran()


@pytest.mark.asyncio
@pytest.mark.parametrize("slug, app", [
    (TIKTOK_UPLOAD, "TIKTOK"),
    ("YOUTUBE_UPLOAD_VIDEO", "YOUTUBE"),
    ("TIKTOK_PUBLISH_VIDEO", "TIKTOK"),  # never offered (URL pull), refused all the same
    ("TWITTER_CREATE_TWEET", "TWITTER"),  # never offered (stale), refused all the same
])
async def test_both_agent_tool_paths_refuse_a_registry_publish_action(env, monkeypatch, slug, app):
    from modules.tools.execution import exec_composio

    _connect(env, app)
    agent = _Agent(monkeypatch)
    per_action = await exec_composio.execute_composio_tool(
        agent.tools(), SimpleNamespace(name=f"composio_{slug}", metadata={"action": slug}), {},
        agent_id=7, workspace_id=WS_ON,
    )
    meta_tool = await exec_composio.execute_composio_execute(
        agent.tools(), "composio_execute", {"action": slug.lower(), "params": {"caption": "hi"}},
        agent_id=7, workspace_id=WS_ON,
    )

    for result in (per_action, meta_tool):
        _assert_refused(result, slug)
    agent.assert_nothing_ran()


@pytest.mark.asyncio
async def test_a_playbook_step_is_refused_before_uploads_or_the_spine(env, monkeypatch):
    _connect(env, "TIKTOK")
    step = _Step(monkeypatch, TIKTOK_UPLOAD, {"file_to_upload": "/workspace/launch.mp4"})

    result = await step.run(WS_ON)

    (call,) = result["execution"]["tool_calls"]
    assert call["result"] == f"Error executing {TIKTOK_UPLOAD}: {REFUSAL}"
    step.spine.execute_and_format.assert_not_called()
    step.resolve_uploads.assert_not_called()
    step.get_client.assert_not_called()


@pytest.mark.asyncio
async def test_taking_a_registry_publish_action_off_the_list_does_not_let_it_through(env):
    _connect(env, "INSTAGRAM")
    _set(env.engine, "socials", "post_actions", [slug for slug in gate_harness.POST_SEED if slug != gate_harness.PUBLISH])

    assert await post_gate.post_action_refusal(gate_harness.PUBLISH, WS_ON) == REFUSAL


# ---------------------------------------------------------------------------
# The deny list wins; Socials off, not connected, and the non-publish steps run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_deny_list_still_wins(env, monkeypatch):
    _connect(env, "TIKTOK")
    _set(env.engine, "composio", "denied_actions", [*gate_harness.DENIED_SEED, TIKTOK_UPLOAD])
    agent = _Agent(monkeypatch)

    result = await agent.execute(TIKTOK_UPLOAD, WS_ON)

    assert result["error_type"] == "action_denied"
    assert result["error"] == BLOCKED + deny_list.DENIED_REASON.format(slug=TIKTOK_UPLOAD)
    agent.assert_nothing_ran()


@pytest.mark.asyncio
async def test_a_socials_off_workspace_runs_a_registry_publish_action_as_before(env, monkeypatch):
    _connect(env, "TIKTOK", workspace=WS_OFF)
    agent = _Agent(monkeypatch)

    result = await agent.execute(TIKTOK_UPLOAD, WS_OFF, {"caption": "Hi"}, skip_validation=True)

    assert result["success"] is True, result
    assert agent.sdk.tools.execute.call_args.kwargs["slug"] == TIKTOK_UPLOAD
    _connect(env, "TIKTOK")
    assert await post_gate.post_action_refusal(TIKTOK_UPLOAD, WS_ON) == REFUSAL
    _set(env.engine, "socials", "enabled", "false")  # the platform master switch
    assert await post_gate.post_action_refusal(TIKTOK_UPLOAD, WS_ON) is None


@pytest.mark.asyncio
async def test_a_channel_not_connected_in_the_workspace_is_not_the_registrys_to_refuse(env):
    _connect(env, "TIKTOK", workspace=WS_OFF)  # connected elsewhere only

    # The executor's own validation refuses it here: the app is not connected.
    assert await post_gate.post_action_refusal(TIKTOK_UPLOAD, WS_ON) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("slug, app", [
    ("INSTAGRAM_POST_IG_USER_MEDIA", "INSTAGRAM"),  # a container: upload
    ("LINKEDIN_UPLOAD_VIDEO", "LINKEDIN"),
    ("TWITTER_UPLOAD_LARGE_MEDIA", "TWITTER"),
    ("TIKTOK_QUERY_CREATOR_INFO", "TIKTOK"),  # read
    ("TIKTOK_FETCH_PUBLISH_STATUS", "TIKTOK"),  # status
    ("YOUTUBE_UPDATE_THUMBNAIL", "YOUTUBE"),
])
async def test_a_channels_upload_read_and_status_steps_still_run(env, monkeypatch, slug, app):
    from modules.tools.execution import exec_composio

    _connect(env, app)
    agent = _Agent(monkeypatch)
    agent.validates(slug, app)

    result = await exec_composio.execute_composio_execute(
        agent.tools(), "composio_execute", {"action": slug, "params": {"caption": "hi"}},
        agent_id=7, workspace_id=WS_ON,
    )

    assert result["success"] is True, result
    assert agent.sdk.tools.execute.call_args.kwargs["slug"] == slug


# ---------------------------------------------------------------------------
# The generic adapter's post actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_connected_toolkits_generic_post_action_is_refused_and_a_plain_message_runs(env, monkeypatch):
    _cache(env, "REDDIT", REDDIT_POST, REDDIT_SCHEMA)
    _cache(env, "SLACK", SLACK_MESSAGE, SLACK_SCHEMA)
    _connect(env, "REDDIT", "SLACK")
    agent = _Agent(monkeypatch)

    _assert_refused(await agent.execute(REDDIT_POST, WS_ON, {"text": "Launch day"}), REDDIT_POST)
    agent.assert_nothing_ran()
    assert await post_gate.post_action_refusal(SLACK_MESSAGE, WS_ON) is None
    assert await post_gate.post_action_refusal(REDDIT_POST, WS_OFF) is None  # Socials off


@pytest.mark.asyncio
async def test_a_generic_post_action_of_a_toolkit_not_connected_here_runs(env):
    _cache(env, "REDDIT", REDDIT_POST, REDDIT_SCHEMA)
    _connect(env, "REDDIT", workspace=WS_OFF)

    assert await post_gate.post_action_refusal(REDDIT_POST, WS_ON) is None


def test_whether_an_action_may_publish_is_answered_from_the_data_alone():
    assert publish_candidate(TIKTOK_UPLOAD.lower()) == SEEDED
    assert publish_candidate(" twitter_create_tweet ") == SEEDED
    assert publish_candidate(REDDIT_POST) == GENERIC
    assert publish_candidate(SLACK_MESSAGE) == GENERIC  # a name that may post: its schema decides
    for slug in (
        "TWITTER_UPLOAD_MEDIA", "LINKEDIN_GET_MY_INFO", "LINKEDIN_DELETE_POST", "INSTAGRAM_POST_IG_USER_MEDIA",
        "REDDIT_GET_POSTS", "FACEBOOK_REPLY_TO_POST_COMMENT", "GMAIL_SEND_EMAIL", "", None,
    ):
        assert publish_candidate(slug) is None, slug


# ---------------------------------------------------------------------------
# Fails closed; the way through; F105
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_registry_that_cannot_be_read_refuses_a_candidate_and_logs(env, monkeypatch):
    logger = MagicMock(name="logger")
    monkeypatch.setattr(post_gate, "logger", logger)

    def unreadable(db, workspace_id, slug):
        raise gate_harness.READ_FAILURES[1]

    monkeypatch.setattr(capabilities, "channel_publish_action", unreadable)
    agent = _Agent(monkeypatch)

    result = await agent.execute(TIKTOK_UPLOAD, WS_ON)

    _assert_refused(result, TIKTOK_UPLOAD, post_gate.REGISTRY_UNREAD_REFUSAL.format(slug=TIKTOK_UPLOAD))
    agent.assert_nothing_ran()
    assert "could not tell whether" in logger.error.call_args.args[0]
    assert logger.error.call_args.kwargs["exc_info"] is True
    assert await post_gate.post_action_refusal(gate_harness.READ, WS_ON) is None  # not a candidate: no read
    assert await post_gate.post_action_refusal(TIKTOK_UPLOAD, WS_OFF) is None  # Socials off: no read


@pytest.mark.asyncio
async def test_only_the_platform_publisher_passes_a_registry_publish_action(env):
    _connect(env, "TIKTOK")

    for impostor in (True, "platform_publisher", object()):
        assert await post_gate.post_action_refusal(TIKTOK_UPLOAD, WS_ON, way_through=impostor) == REFUSAL
    assert await post_gate.post_action_refusal(TIKTOK_UPLOAD, WS_ON, way_through=post_gate.PLATFORM_PUBLISHER) is None


@pytest.mark.asyncio
async def test_the_registry_is_read_off_the_loop_once_and_only_for_a_candidate_the_list_lacks(env, monkeypatch):
    _connect(env, "TIKTOK")
    loop_thread = threading.get_ident()
    threads = []
    real = post_gate._registry_publish

    def read(workspace_id, slug):
        threads.append(threading.get_ident())
        return real(workspace_id, slug)

    monkeypatch.setattr(post_gate, "_registry_publish", read)

    assert await post_gate.post_action_refusal(TIKTOK_UPLOAD, WS_ON) == REFUSAL  # a candidate: one read
    assert await post_gate.post_action_refusal(gate_harness.PUBLISH, WS_ON) == REFUSAL  # on the list: none
    assert await post_gate.post_action_refusal(gate_harness.READ, WS_ON) is None  # not a candidate: none
    assert await post_gate.post_action_refusal(TIKTOK_UPLOAD, WS_OFF) is None  # Socials off: none

    assert len(threads) == 1 and loop_thread not in threads
