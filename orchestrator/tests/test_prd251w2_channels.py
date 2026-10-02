"""PRD-251 Wave 2, US-203 (S3.2, D8, D16) — one registry: the channels a workspace
can post to, and ``GET /api/socials/channels``.

``modules/socials/capabilities.py`` (the Wave 1 registry, extended with the
``publish`` class) resolves each connected social toolkit's post kinds from the
seeded adapter data (``modules/socials/channel_adapters.py``) or the generic
adapter. Pinned on the real read paths, over SQLite copies of the tables: the
connections through ``EntityManager.get_connected_apps``, the cached actions in
``composio_actions_cache`` (their ``parameters`` EMPTY, as the bulk sync leaves
them), the Wave 0 deny list through ``read_system_setting``, and the published
targets in ``social_post_targets``; the route behind the real Socials gate.

* LinkedIn, X and Instagram connected, with only their slugs cached: all three are
  listed with their post kinds, and each kind carries the documented parameters;
* a connected toolkit with no post action (Gmail) is not listed; a seeded kind whose
  action the cache lacks is unavailable, naming it; a denied action is never offered;
* the generic adapter offers a connected toolkit outside the data only when a cached
  post action's schema has a text field and a media field, as an "unverified channel"
  until one of the workspace's own targets has published;
* ``needs_public_storage``: a step that takes only a link, without public storage;
* the data: checked when the module loads, the stale and URL-pull slugs never usable,
  every media parameter a file or a declared link (file-first), the global
  ``UPLOAD_ACTIONS`` not widened, and no channel slug literal in ``modules/socials``,
  ``core/composio`` or the Socials API outside the adapter data (a scan);
* the route: a plain ``def``, gated, in the committed route manifest.

The post gate's half of the story (D14 completed) is ``test_prd251w2_channel_gate.py``.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import re
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.dialects.postgresql import ARRAY, JSONB  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every mapper)
import api.socials as socials_api  # noqa: E402
import core.composio.deny_list as deny_list  # noqa: E402
import core.database.database as database_mod  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.composio.tool_executor import UPLOAD_ACTIONS  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.composio import ComposioConnection, ComposioEntity  # noqa: E402
from core.models.composio_cache import ComposioActionCache  # noqa: E402
from core.models.socials import SOCIAL_TARGET_STATUSES, SocialPost, SocialPostTarget  # noqa: E402
from core.models.system_settings import SystemSetting  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.socials import capabilities, media_urls  # noqa: E402
from modules.socials.capabilities import (  # noqa: E402
    PUBLISH,
    PUBLISHED_TARGET,
    SEEDED_CHANNELS,
    STEP_CLASSES,
    parse_channel_adapters,
    social_channels,
)

VERSIONS = _ORCH / "alembic" / "versions"
MANIFEST = _ORCH / "reports" / "route-manifest.json"
WS = uuid.UUID("00000000-0000-0000-0000-0000000203a1")
OTHER_WS = uuid.UUID("00000000-0000-0000-0000-0000000203a2")
WS_OFF = uuid.UUID("00000000-0000-0000-0000-0000000203a3")
CREATED = datetime(2026, 9, 29, 9, 0)
X_NOTE = "Composio removed its managed X credentials in February 2026 — connect X with your own X API app in Composio"
# The stale and URL-pull slugs the story names: never usable, only ever refused.
NEVER_USABLE = (
    "TWITTER_CREATE_TWEET", "INSTAGRAM_CREATE_MEDIA_CONTAINER", "INSTAGRAM_CREATE_POST",
    "INSTAGRAM_GET_POST_STATUS", "TIKTOK_PUBLISH_VIDEO", "TIKTOK_POST_PHOTO",
)
# A generic post action's schema: a text field, a media link and a media file.
REDDIT_POST = "REDDIT_CREATE_REDDIT_POST"
REDDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "text": {"type": "string"},
        "image_url": {"type": "string"},
        "video_file": {"type": "object", "file_uploadable": True},
    },
    "required": ["title"],
}
GMAIL_SEND_SCHEMA = {
    "type": "object",
    "properties": {"recipient_email": {"type": "string"}, "body": {"type": "string"}, "attachment": {"file_uploadable": True}},
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WAVE0 = _load(VERSIONS / "prd251_socials.py", "prd251_socials_migration_channels")
WAVE1 = _load(VERSIONS / "prd251_wave1.py", "prd251_wave1_migration_channels")
DENIED_SEED = list(WAVE0.COMPOSIO_DENIED_ACTIONS_SEED)


def _adapter_slugs(toolkit: str) -> list:
    return sorted({step.action for steps in SEEDED_CHANNELS[toolkit].kinds.values() for step in steps})


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def _portable(col_type):
    if isinstance(col_type, (JSONB, ARRAY)):
        return sa.JSON()
    if isinstance(col_type, sa.Uuid) or type(col_type).__name__.upper() == "UUID":
        return sa.Uuid()
    return col_type


def _sqlite_copy(table: sa.Table, metadata: sa.MetaData) -> sa.Table:
    """A column-for-column copy SQLite can build (JSONB → JSON, UUID → CHAR(32))."""
    columns = [sa.Column(col.name, _portable(col.type), primary_key=col.primary_key) for col in table.columns]
    return sa.Table(table.name, metadata, *columns)


_TABLES = sa.MetaData()
for _table in (Workspace.__table__, ComposioEntity.__table__, ComposioConnection.__table__, ComposioActionCache.__table__):
    _sqlite_copy(_table, _TABLES)


def _ctx(workspace_id):
    return RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id="member-1", clerk_user_id="clerk-member-1", system_role="user"),
        auth_type="clerk",
    )


@pytest.fixture
def channels(monkeypatch):
    """One in-memory database behind SessionLocal and the route's session: the
    Wave 0 deny list and Wave 1's settings as the migrations seed them, the master
    switch on, two Socials-on workspaces and a Socials-off one, the Composio
    tables, and the social posts and targets. Public storage is off."""
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    _TABLES.create_all(engine)
    SystemSetting.__table__.create(bind=engine)
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])
    with engine.begin() as conn:
        WAVE0._seed_settings(conn, WAVE0._composio_settings_seed())
        WAVE1.seed_settings(conn, WAVE1.settings_seed())
        conn.execute(SystemSetting.__table__.insert().values(
            category="socials", key="enabled", value="true", value_type="boolean", created_by="prd251",
        ))
    factory = sessionmaker(bind=engine)
    session = factory()
    for ws_id, settings in ((WS, {"socials": {"enabled": True}}), (OTHER_WS, {"socials": {"enabled": True}}), (WS_OFF, {})):
        session.add(Workspace(
            id=ws_id, name=f"ws-{ws_id.hex[-2:]}", plan="basic", plan_limits={},
            settings=settings, onboarding={}, created_at=CREATED, updated_at=CREATED,
        ))
    session.commit()
    monkeypatch.setattr(database_mod, "SessionLocal", factory)
    monkeypatch.setattr(media_urls, "media_public_url_available", lambda: False)
    deny_list.reset_cache()
    state = SimpleNamespace(engine=engine, session=session, ctx=_ctx(WS))
    app = FastAPI()
    app.include_router(socials_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        deny_list.reset_cache()
        engine.dispose()


def _cache(env, app: str, *slugs: str, parameters=None) -> None:
    """Cached actions as the bulk sync leaves them: ``parameters`` empty unless given."""
    for slug in slugs:
        env.session.add(ComposioActionCache(
            app_name=app, action_name=slug, action_slug=slug.lower().replace("_", "-"),
            display_name=slug.replace("_", " ").title(), parameters=parameters if parameters is not None else {},
        ))
    env.session.commit()


def _cache_channel(env, toolkit: str, *, without=()) -> None:
    _cache(env, toolkit.upper(), *[slug for slug in _adapter_slugs(toolkit) if slug not in without])


def _connect(env, *apps: str, workspace=WS, status="active") -> None:
    entity = env.session.query(ComposioEntity).filter(ComposioEntity.workspace_id == workspace).first()
    if entity is None:
        entity = ComposioEntity(workspace_id=workspace, composio_entity_id=str(workspace))
        env.session.add(entity)
        env.session.flush()
    for app in apps:
        env.session.add(ComposioConnection(entity_id=entity.id, app_name=app, status=status, connection_id=f"ca_{app.lower()}"))
    env.session.commit()


def _deny(env, *slugs: str) -> None:
    table = SystemSetting.__table__
    with env.engine.begin() as conn:
        conn.execute(table.update().where(table.c.category == "composio", table.c.key == "denied_actions")
                     .values(value=json.dumps(DENIED_SEED + list(slugs))))
    deny_list.reset_cache()


def _published(env, toolkit: str, *, workspace=WS, status=PUBLISHED_TARGET) -> None:
    post = SocialPost(workspace_id=workspace, title="Launch", created_by="member-1", content_hash="0" * 64)
    env.session.add(post)
    env.session.flush()
    env.session.add(SocialPostTarget(
        post_id=post.id, toolkit=toolkit, post_kind="image", idempotency_key=f"sp:{post.id}:{toolkit}:image", status=status,
    ))
    env.session.commit()


def _listed(env) -> dict:
    resp = env.client.get("/api/socials/channels")
    assert resp.status_code == 200, resp.text
    return {channel["toolkit"]: channel for channel in resp.json()}


def _kinds(channel: dict) -> dict:
    return {kind["kind"]: kind for kind in channel["post_kinds"]}


# ---------------------------------------------------------------------------
# AC1 — LinkedIn, X and Instagram connected: listed with their post kinds
# ---------------------------------------------------------------------------


def test_linkedin_x_and_instagram_connected_are_listed_with_their_post_kinds(channels):
    for toolkit in ("linkedin", "twitter", "instagram"):
        _cache_channel(channels, toolkit)
    _connect(channels, "LINKEDIN", "TWITTER", "INSTAGRAM")

    listed = _listed(channels)

    assert list(listed) == ["linkedin", "twitter", "instagram"]  # the data's order
    assert {toolkit: list(_kinds(channel)) for toolkit, channel in listed.items()} == {
        "linkedin": ["text", "image", "video"],
        "twitter": ["text", "image", "video"],
        "instagram": ["image", "reel", "carousel"],
    }
    for channel in listed.values():
        assert channel["verified"] is True
        # US-207: and the channel's copy limits, which the composer counts against.
        assert set(channel) == {"toolkit", "label", "post_kinds", "verified", "setup_note", "copy_limits"}
        for kind in channel["post_kinds"]:
            assert kind == {"kind": kind["kind"], "available": True, "reason": None, "needs_public_storage": False}
    assert (listed["linkedin"]["label"], listed["twitter"]["label"], listed["instagram"]["label"]) == (
        "LinkedIn", "X", "Instagram",
    )
    assert listed["twitter"]["copy_limits"] == {"text": 280}
    assert listed["instagram"]["copy_limits"] == {"text": 2200, "hashtags": 30}
    assert listed["twitter"]["setup_note"] == X_NOTE
    assert listed["linkedin"]["setup_note"] is None and listed["instagram"]["setup_note"] is None


def test_a_seeded_adapter_needs_only_its_slugs_in_the_cache_and_carries_the_parameters(channels):
    _cache_channel(channels, "twitter")  # every cached `parameters` is {}
    _connect(channels, "TWITTER")

    (twitter,) = social_channels(channels.session, WS)
    video = next(kind for kind in twitter.post_kinds if kind.kind == "video")

    assert video.available
    assert [(step.action, step.step_class) for step in video.steps] == [
        ("TWITTER_UPLOAD_LARGE_MEDIA", "upload"),  # X's chunked upload, run by Composio
        ("TWITTER_GET_MEDIA_UPLOAD_STATUS", "status"),
        ("TWITTER_CREATION_OF_A_POST", PUBLISH),
    ]
    upload, _status, post = video.steps
    assert (upload.files, dict(post.params)) == (("media",), {"text": "$copy", "media_media_ids": ["$steps.media"]})


def test_only_connected_toolkits_are_listed_and_another_workspace_sees_its_own(channels):
    for toolkit in ("linkedin", "tiktok"):
        _cache_channel(channels, toolkit)
    _connect(channels, "LINKEDIN")
    _connect(channels, "TIKTOK", workspace=OTHER_WS)
    _connect(channels, "TWITTER", status="error")  # not connected

    assert list(_listed(channels)) == ["linkedin"]
    channels.ctx = _ctx(OTHER_WS)
    assert list(_listed(channels)) == ["tiktok"]


# ---------------------------------------------------------------------------
# AC2 — no post action: not listed; a missing slug: that kind unavailable; denied: never offered
# ---------------------------------------------------------------------------


def test_a_connected_toolkit_with_no_post_action_is_not_listed(channels):
    _cache(channels, "GMAIL", "GMAIL_SEND_EMAIL", "GMAIL_FETCH_EMAILS", parameters=GMAIL_SEND_SCHEMA)
    _cache_channel(channels, "linkedin")
    _connect(channels, "GMAIL", "LINKEDIN")

    assert list(_listed(channels)) == ["linkedin"]


def test_a_seeded_kind_whose_action_the_cache_lacks_is_unavailable_naming_it(channels):
    _cache_channel(channels, "instagram", without=("INSTAGRAM_CREATE_CAROUSEL_CONTAINER",))
    _connect(channels, "INSTAGRAM")

    kinds = _kinds(_listed(channels)["instagram"])

    assert (kinds["image"]["available"], kinds["reel"]["available"]) == (True, True)
    carousel = kinds["carousel"]
    assert carousel["available"] is False
    assert carousel["reason"] == capabilities.MISSING_ACTION.format(slug="INSTAGRAM_CREATE_CAROUSEL_CONTAINER")
    assert "POST /api/tools/sync" in carousel["reason"] and "not when an app is connected" in carousel["reason"]


def test_every_kind_needing_a_missing_publish_action_names_it(channels):
    _cache_channel(channels, "instagram", without=("INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH",))
    _connect(channels, "INSTAGRAM")

    kinds = _kinds(_listed(channels)["instagram"])

    assert {kind: (k["available"], "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH" in k["reason"]) for kind, k in kinds.items()} == {
        "image": (False, True), "reel": (False, True), "carousel": (False, True),
    }


def test_a_denied_slug_is_never_offered(channels):
    _cache_channel(channels, "tiktok")
    _cache(channels, "REDDIT", REDDIT_POST, parameters=REDDIT_SCHEMA)
    _connect(channels, "TIKTOK", "REDDIT")
    _deny(channels, "TIKTOK_UPLOAD_VIDEO", REDDIT_POST)

    listed = _listed(channels)

    video = _kinds(listed["tiktok"])["video"]
    assert video["available"] is False
    assert video["reason"] == deny_list.BLOCKED_PREFIX + deny_list.DENIED_REASON.format(slug="TIKTOK_UPLOAD_VIDEO")
    assert "reddit" not in listed  # its one post action is denied: the generic adapter offers nothing


def test_a_deny_list_that_cannot_be_used_offers_no_kind(channels):
    _cache_channel(channels, "linkedin")
    _connect(channels, "LINKEDIN")
    table = SystemSetting.__table__
    with channels.engine.begin() as conn:
        conn.execute(table.update().where(table.c.category == "composio", table.c.key == "denied_actions")
                     .values(value="not json"))
    deny_list.reset_cache()

    kinds = _kinds(_listed(channels)["linkedin"])

    assert all(kind["available"] is False for kind in kinds.values())
    assert {kind["reason"] for kind in kinds.values()} == {deny_list.BLOCKED_PREFIX + deny_list.UNREADABLE_REASON}


# ---------------------------------------------------------------------------
# AC3 — the generic adapter: text + media, an "unverified channel" until it has published
# ---------------------------------------------------------------------------


def test_the_generic_adapter_offers_a_connected_toolkit_whose_post_action_has_text_and_media(channels):
    _cache(channels, "REDDIT", REDDIT_POST, parameters=REDDIT_SCHEMA)
    _cache(channels, "REDDIT", "REDDIT_GET_SUBREDDIT_POSTS", parameters=REDDIT_SCHEMA)  # a read, never offered
    _connect(channels, "REDDIT")

    reddit = _listed(channels)["reddit"]

    assert (reddit["label"], reddit["verified"], reddit["setup_note"]) == ("Reddit (unverified channel)", False, None)
    kinds = _kinds(reddit)
    assert list(kinds) == ["text", "image", "video"]
    assert kinds["text"] == {"kind": "text", "available": True, "reason": None, "needs_public_storage": False}
    assert kinds["video"] == {"kind": "video", "available": True, "reason": None, "needs_public_storage": False}
    image = kinds["image"]  # carried only by image_url: a link the platform fetches
    assert (image["available"], image["needs_public_storage"]) == (False, True)
    assert image["reason"].startswith(media_urls.NEEDS_PUBLIC_STORAGE) and REDDIT_POST in image["reason"]


def test_a_generic_media_link_is_available_with_public_storage_and_a_file_goes_first(channels, monkeypatch):
    _cache(channels, "REDDIT", REDDIT_POST, parameters=REDDIT_SCHEMA)
    _connect(channels, "REDDIT")
    monkeypatch.setattr(media_urls, "media_public_url_available", lambda: True)

    (reddit,) = social_channels(channels.session, WS)
    kinds = {kind.kind: kind for kind in reddit.post_kinds}

    assert all(kind.available and not kind.needs_public_storage for kind in kinds.values())
    (step,) = kinds["image"].steps
    assert (step.action, step.step_class, step.urls, dict(step.params)) == (
        REDDIT_POST, PUBLISH, ("image_url",), {"text": "$copy", "image_url": "$media"},
    )
    assert kinds["video"].steps[0].files == ("video_file",)  # the file-marked field: a file


def test_it_is_an_unverified_channel_until_one_of_the_workspaces_targets_has_published(channels):
    _cache(channels, "REDDIT", REDDIT_POST, parameters=REDDIT_SCHEMA)
    _connect(channels, "REDDIT")
    assert PUBLISHED_TARGET in SOCIAL_TARGET_STATUSES

    _published(channels, "reddit", status="pending")
    _published(channels, "reddit", status="failed")
    _published(channels, "reddit", workspace=OTHER_WS)  # another workspace's publish verifies nothing here
    assert (_listed(channels)["reddit"]["verified"], _listed(channels)["reddit"]["label"]) == (
        False, "Reddit (unverified channel)",
    )

    _published(channels, "reddit")
    assert (_listed(channels)["reddit"]["verified"], _listed(channels)["reddit"]["label"]) == (True, "Reddit")


@pytest.mark.parametrize("parameters", [
    {},  # the bulk sync's empty schema
    {"type": "object", "properties": {}},
    {"type": "object", "properties": {"text": {"type": "string"}, "link": {"type": "string"}}},  # no media field
    {"type": "object", "properties": {"image_url": {"type": "string"}}},  # no text field
])
def test_a_toolkit_with_no_usable_schema_offers_nothing(channels, parameters):
    _cache(channels, "MASTODON", "MASTODON_CREATE_POST", parameters=parameters)
    _connect(channels, "MASTODON")

    assert _listed(channels) == {}


def test_a_post_action_that_requires_media_offers_no_text_kind(channels):
    schema = {**REDDIT_SCHEMA, "required": ["video_file"]}
    _cache(channels, "THREADS", "THREADS_PUBLISH_MEDIA", parameters=schema)
    _connect(channels, "THREADS")

    assert list(_kinds(_listed(channels)["threads"])) == ["image", "video"]


def test_a_youtube_thumbnail_needs_public_storage_and_is_skipped_without_it(channels, monkeypatch):
    _cache_channel(channels, "youtube")
    _connect(channels, "YOUTUBE")

    video = _kinds(_listed(channels)["youtube"])["video"]
    assert video == {"kind": "video", "available": True, "reason": None, "needs_public_storage": True}

    monkeypatch.setattr(media_urls, "media_public_url_available", lambda: True)
    assert _kinds(_listed(channels)["youtube"])["video"]["needs_public_storage"] is False


def test_an_optional_step_that_cannot_run_is_skipped_and_leaves_its_kind_available(channels):
    _cache_channel(channels, "youtube", without=("YOUTUBE_UPDATE_THUMBNAIL",))
    _connect(channels, "YOUTUBE")
    missing = _kinds(_listed(channels)["youtube"])["video"]
    assert (missing["available"], missing["reason"]) == (True, None)

    _cache(channels, "YOUTUBE", "YOUTUBE_UPDATE_THUMBNAIL")
    _deny(channels, "YOUTUBE_UPDATE_THUMBNAIL")
    refused = _kinds(_listed(channels)["youtube"])["video"]
    assert (refused["available"], refused["reason"]) == (True, None)

    _deny(channels, "YOUTUBE_UPLOAD_VIDEO")  # a step the kind must run: the deny list wins
    blocked = _kinds(_listed(channels)["youtube"])["video"]
    assert blocked["available"] is False
    assert blocked["reason"] == deny_list.BLOCKED_PREFIX + deny_list.DENIED_REASON.format(slug="YOUTUBE_UPLOAD_VIDEO")


# ---------------------------------------------------------------------------
# The data: checked, file-first, never the stale or URL-pull slugs, no literals elsewhere
# ---------------------------------------------------------------------------


def test_the_registry_carries_the_publish_class_and_every_kind_publishes():
    assert PUBLISH in STEP_CLASSES
    assert set(SEEDED_CHANNELS) == {"linkedin", "twitter", "instagram", "tiktok", "youtube"}
    for adapter in SEEDED_CHANNELS.values():
        for kind, steps in adapter.kinds.items():
            assert any(step.step_class == PUBLISH for step in steps), (adapter.toolkit, kind)


@pytest.mark.parametrize("broken", [
    {"x": {"label": "X", "kinds": {"text": [{"id": "a", "action": "Y_POST", "class": "publish"}]}}},
    {"x": {"label": "X", "kinds": {"text": [{"id": "a", "action": "X_POST", "class": "upload"}]}}},
    {"x": {"label": "X", "kinds": {"text": [{"id": "a", "action": "X_POST", "class": "publish", "params": {"t": "$nope"}}]}}},
    {"x": {"label": "X", "kinds": {"text": [{"id": "a", "action": "X_POST", "class": "publish", "params": {"t": "$steps.b"}}]}}},
    {"x": {"label": "X", "kinds": {"text": [{"id": "a", "action": "X_POST", "class": "publish", "files": ["f"]}]}}},
    {"x": {"label": "X", "kinds": {"banner": [{"id": "a", "action": "X_POST", "class": "publish"}]}}},
    {"x": {"label": "", "kinds": {"text": [{"id": "a", "action": "X_POST", "class": "publish"}]}}},
    {"x": {"label": "X", "kinds": {"text": [{"id": "a", "action": "X_POST", "class": "publish"}]}, "never_offered": ["Y_Z"]}},
    {"X": {"label": "X", "kinds": {"text": [{"id": "a", "action": "X_POST", "class": "publish"}]}}},
])
def test_the_adapter_data_is_checked_when_the_module_loads(broken):
    with pytest.raises(ValueError):
        parse_channel_adapters(broken)


def test_the_data_never_lists_the_stale_or_url_pull_slugs_as_usable():
    usable = {step.action for adapter in SEEDED_CHANNELS.values() for steps in adapter.kinds.values() for step in steps}
    assert usable.isdisjoint(NEVER_USABLE)
    refused = {slug for adapter in SEEDED_CHANNELS.values() for slug in adapter.never_offered}
    assert refused <= set(NEVER_USABLE)


def test_publishing_is_file_first_and_the_global_upload_list_is_not_widened():
    # The executor's list, exactly as before the registry: Instagram, TikTok and YouTube stay out.
    assert UPLOAD_ACTIONS == {
        "TWITTER_UPLOAD_MEDIA", "TWITTER_INITIALIZE_MEDIA_UPLOAD", "TWITTER_UPLOAD_LARGE_MEDIA",
        "TWITTER_APPEND_MEDIA_UPLOAD", "LINKEDIN_CREATE_LINKED_IN_POST", "LINKEDIN_CREATE_IMAGE_POST",
        "LINKEDIN_CREATE_SHARE", "LINKEDIN_INITIALIZE_IMAGE_UPLOAD", "LINKEDIN_REGISTER_IMAGE_UPLOAD",
    }
    media_sources = {"$media", "$media[]", "$thumbnail"}
    for adapter in SEEDED_CHANNELS.values():
        for steps in adapter.kinds.values():
            for step in steps:
                media = {name for name, source in step.params.items() if isinstance(source, str) and source in media_sources}
                assert media == set(step.files) | set(step.urls), (adapter.toolkit, step.action)
    links = {step.action for a in SEEDED_CHANNELS.values() for s in a.kinds.values() for step in s if step.urls}
    assert links == {"YOUTUBE_UPDATE_THUMBNAIL"}  # the one URL-only step today (D9)


# A channel action slug: Composio writes them upper case, and so does every caller here.
_SLUG = re.compile(r"\b(?:LINKEDIN|TWITTER|INSTAGRAM|TIKTOK|YOUTUBE)_[A-Z0-9_]+")
# What predates the registry, and may shrink but never grow: the executor's global
# upload list (never widened) and its hand-off to the LinkedIn image workaround, and
# the workaround's own name for the action it stands in for.
_PRE_REGISTRY_LITERALS = Counter({
    **{("core/composio/tool_executor.py", slug): 1 for slug in (
        "TWITTER_UPLOAD_MEDIA", "TWITTER_INITIALIZE_MEDIA_UPLOAD", "TWITTER_UPLOAD_LARGE_MEDIA",
        "TWITTER_APPEND_MEDIA_UPLOAD", "LINKEDIN_CREATE_IMAGE_POST", "LINKEDIN_CREATE_SHARE",
        "LINKEDIN_INITIALIZE_IMAGE_UPLOAD", "LINKEDIN_REGISTER_IMAGE_UPLOAD",
    )},
    ("core/composio/tool_executor.py", "LINKEDIN_CREATE_LINKED_IN_POST"): 2,
    ("core/composio/linkedin_image_workaround.py", "LINKEDIN_CREATE_LINKED_IN_POST"): 1,
})


def _code_strings(path: Path):
    """Every string constant in the file but its docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value) for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
    }
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


def test_no_channel_slug_literal_outside_the_adapter_data():
    data_file = _ORCH / "modules" / "socials" / "channel_adapters.py"
    files = [
        *(_ORCH / "modules" / "socials").rglob("*.py"),
        *(_ORCH / "core" / "composio").rglob("*.py"),
        *(_ORCH / "api").glob("socials*.py"),
    ]
    found = Counter(
        (path.relative_to(_ORCH).as_posix(), slug)
        for path in files if path != data_file
        for text in _code_strings(path) for slug in _SLUG.findall(text)
    )
    assert found - _PRE_REGISTRY_LITERALS == Counter(), "channel slugs belong in modules/socials/channel_adapters.py"
    # The data file is data: no import, no function, no class — only its docstring and assignments.
    tree = ast.parse(data_file.read_text(encoding="utf-8"))
    assert all(isinstance(node, (ast.Expr, ast.Assign)) for node in tree.body)
    assert {target.id for node in tree.body if isinstance(node, ast.Assign) for target in node.targets} == {
        "CHANNEL_ADAPTERS", "GENERIC_ADAPTER",
    }


# ---------------------------------------------------------------------------
# The route: a plain def, behind the gate, in the committed manifest
# ---------------------------------------------------------------------------


def test_the_route_is_a_plain_def_behind_the_gate_and_in_the_committed_manifest(channels):
    (route,) = [r for r in socials_api.router.routes if isinstance(r, APIRoute) and r.path == "/api/socials/channels"]
    assert route.methods == {"GET"}
    assert not inspect.iscoroutinefunction(route.endpoint)  # F105: FastAPI runs it in the threadpool
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "GET", "path": "/api/socials/channels"} in manifest["routes"]

    channels.ctx = _ctx(WS_OFF)
    assert channels.client.get("/api/socials/channels").status_code == 404
