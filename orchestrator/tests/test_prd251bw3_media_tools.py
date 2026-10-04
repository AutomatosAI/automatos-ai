"""PRD-251B Wave 3, US-B304 (and US-B305's references) — the AI tools a workspace uses.

Pinned:

* **The section** (``GET /api/socials/media-tools``): each generation and voice toolkit the
  registry knows with its state, Templates and Kokoro built in, the choices per media type
  the workspace has now, its defaults, both caps and this month's media spend.
* **Defaults are validated against what is offered now**: a toolkit that does not make that
  media type, one not connected, or an unknown media type is a 422 and nothing is stored;
  only owners and admins change them. The caps are written beside the monthly one; a
  per-post cap overrides ``SOCIALS_MEDIA_POST_CAP_USD`` (fail closed when unreadable).
* **A render uses them**: the default toolkit for a slot kind is tried first; the brand
  kit's style follows every prompt without changing the recorded one.
* **Liked references** go only to a generate action the registry flags ``reference_image``
  (fal's stills by the Wave 3 migration), as a link a platform can fetch, and only when the
  workspace sends them; a reference not in storage is left out.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from botocore.exceptions import ClientError

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials_brand as socials_brand  # noqa: E402
import api.socials_media_tools as media_tools_api  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from config import config  # noqa: E402
from core.social_templates import IMAGE_SLOT, VIDEO_SLOT  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.socials import media_caps, media_tools, service  # noqa: E402
from modules.socials.capabilities import (  # noqa: E402
    BALANCE,
    ESTIMATE,
    GENERATE_IMAGE,
    GENERATE_VIDEO,
    REFERENCE_IMAGE,
    STATUS,
    MediaCapabilities,
    OfferedAction,
    parse_media_actions,
)
from modules.socials.recipes import footage as footage_recipes  # noqa: E402
from modules.socials.recipes import footage_routes  # noqa: E402
from modules.socials.recipes.footage_toolkits import RECIPES, SUBMIT, Shot, prompt_for  # noqa: E402
from modules.socials.settings import media_post_cap_usd  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402

api = api_harness.api
VERSIONS = _ORCH / "alembic" / "versions"
STYLE = "Brand style (from the brand kit's references): mood: calm."


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WAVE1 = _load(VERSIONS / "prd251_wave1.py", "prd251_wave1_migration_media_tools")
WAVE3 = _load(VERSIONS / "prd251b_wave3.py", "prd251b_wave3_migration_media_tools")


def _action(toolkit, slug, *capabilities):
    return OfferedAction(toolkit=toolkit, slug=slug, capabilities=frozenset(capabilities), display_name=slug, parameters={})


def _offered(*actions):
    out = {}
    for action in actions:
        out.setdefault(action.toolkit, {})[action.slug] = action
    return out


FAL = (
    _action("fal_ai", "FAL_AI_SUBMIT_ASYNC_JOB", GENERATE_VIDEO, GENERATE_IMAGE, REFERENCE_IMAGE),
    _action("fal_ai", "FAL_AI_QUEUE_GET_STATUS", STATUS),
    _action("fal_ai", "FAL_AI_GET_QUEUE_REQUEST_RESULT", STATUS),
    _action("fal_ai", "FAL_AI_ESTIMATE_PRICING", ESTIMATE),
)
KIE_STILLS = (
    _action("kieai", "KIEAI_GENERATE_FLUX_KONTEXT_IMAGE", GENERATE_IMAGE),
    _action("kieai", "KIEAI_GET_FLUX_KONTEXT_IMAGE_DETAILS", STATUS),
    _action("kieai", "KIEAI_GET_ACCOUNT_CREDITS", BALANCE),
)
ALLOWLISTED = {name: frozenset({GENERATE_VIDEO, GENERATE_IMAGE}) for name in ("fal_ai", "kieai", "higgsfield_mcp")}


def _caps(*actions, connected=None):
    offered = _offered(*actions)
    return MediaCapabilities(offered=offered, allowlisted=ALLOWLISTED, connected=frozenset(connected or offered), withheld={})


BOTH = _caps(*FAL, *KIE_STILLS)


@pytest.fixture(autouse=True)
def toolkit_order(monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_FOOTAGE_TOOLKITS", "fal_ai,kieai,higgsfield_mcp")


# ── what is offered, and the defaults ──────────────────────────────────────


def _values(choices):
    return [choice["value"] for choice in choices]


def test_the_choices_per_media_type_are_what_the_workspace_has_now():
    offered = media_tools.offered(BOTH)
    assert _values(offered["images"]) == ["templates", "fal_ai", "kieai"]
    assert _values(offered["ai_images"]) == ["fal_ai", "kieai", "ask"]
    assert _values(offered["footage"]) == ["fal_ai", "off"]  # Kie's stills action makes no footage
    assert _values(offered["voice"]) == ["kokoro"]
    nothing = media_tools.offered(_caps())
    assert (_values(nothing["ai_images"]), _values(nothing["footage"])) == (["ask"], ["off"])


def test_the_rows_say_each_toolkits_state_and_the_built_ins():
    rows = media_tools.toolkit_rows(_caps(*FAL, connected={"fal_ai"}))
    by_toolkit = {row["toolkit"]: row for row in rows}
    assert by_toolkit["fal_ai"]["status"] == "available" and by_toolkit["fal_ai"]["kind"] == "Images and footage"
    assert by_toolkit["kieai"]["status"] == "connect"  # allowlisted, not connected: connect it in Composio
    assert (by_toolkit["templates"]["status"], by_toolkit["kokoro"]["status"]) == ("builtin", "builtin")


def test_defaults_are_checked_against_the_choices_offered():
    choices = media_tools.offered(BOTH)
    current = media_tools.defaults_of(None)
    assert current == {"images": "templates", "ai_images": "ask", "footage": "off", "voice": "kokoro"}
    assert media_tools.validate_defaults({"ai_images": "kieai", "footage": "fal_ai"}, choices, current) == {
        **current, "ai_images": "kieai", "footage": "fal_ai",
    }
    for bad in ({"footage": "kieai"}, {"voice": "elevenlabs"}, {"ai_images": "higgsfield_mcp"}, {"teleport": "fal_ai"}):
        with pytest.raises(service.InvalidPost):
            media_tools.validate_defaults(bad, choices, current)


def test_a_render_tries_the_default_toolkit_first():
    assert media_tools.prefer_for({"media_tools": {"footage": "fal_ai", "ai_images": "kieai"}}) == {
        VIDEO_SLOT: "fal_ai", IMAGE_SLOT: "kieai",
    }
    assert media_tools.prefer_for({"media_tools": {"footage": "off", "ai_images": "ask"}}) == {}
    assert footage_routes.route_for(IMAGE_SLOT, BOTH).recipe.toolkit == "fal_ai"
    assert footage_routes.route_for(IMAGE_SLOT, BOTH, "kieai").recipe.toolkit == "kieai"
    assert footage_routes.route_for(VIDEO_SLOT, BOTH, "kieai").recipe.toolkit == "fal_ai"  # Kie makes no footage here
    assert footage_routes.route_for(IMAGE_SLOT, BOTH, "nonsense").recipe.toolkit == "fal_ai"


SLOTS = {
    "hook": {"kind": "video", "label": "Hook footage", "path": "assets/slots/hook.mp4"},
    "still_1": {"kind": "image", "label": "Still", "path": "assets/slots/still_1.png"},
}


def test_the_plan_carries_the_style_and_the_default_without_changing_the_prompt():
    footage = {"hook": {"prompt": "A tide at dawn"}, "still_1": {"prompt": "A harbour"}}
    plan = footage_recipes.plan_for(footage, SLOTS, BOTH, width=1080, height=1920, style=STYLE,
                                    prefer={IMAGE_SLOT: "kieai"}, references=("https://ref",))
    shots = {shot.slot: (shot, route) for shot, route in plan.shots}
    assert shots["still_1"][1].recipe.toolkit == "kieai" and shots["hook"][1].recipe.toolkit == "fal_ai"
    hook = shots["hook"][0]
    assert (hook.prompt, hook.style, hook.references) == ("A tide at dawn", STYLE, ("https://ref",))
    assert prompt_for(hook.prompt, hook.style).endswith(f"no readable text, no logos {STYLE}")


def test_a_slot_waiting_for_its_ai_option_to_be_picked_is_not_made():
    footage = {"still_1": {"prompt": "A harbour", "options": [], "options_state": "ready"}}
    plan = footage_recipes.plan_for(footage, SLOTS, BOTH, width=1080, height=1080)
    assert plan.shots == () and plan.fallback == {"still_1": "Still: pick one of its AI options first"}


# ── the caps ───────────────────────────────────────────────────────────────


def test_the_per_post_cap_is_the_workspaces_own_or_the_configured_one(monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_MEDIA_POST_CAP_USD", 10.0)
    assert media_post_cap_usd(None) == 10.0
    assert media_post_cap_usd({"socials": {"media_post_cap_usd": 2.5}}) == 2.5
    assert media_post_cap_usd({"socials": {"media_post_cap_usd": None}}) == 10.0
    assert media_post_cap_usd({"socials": {"media_post_cap_usd": "lots"}}) == 0.0  # fail closed


# ── the routes ─────────────────────────────────────────────────────────────


@pytest.fixture
def tools(api, monkeypatch):
    monkeypatch.setattr(media_tools_api, "media_capabilities", lambda db, ws: BOTH)
    monkeypatch.setattr(media_caps, "month_spend_usd", lambda db, ws, start, end: 1.25)
    monkeypatch.setattr(config, "SOCIALS_MEDIA_POST_CAP_USD", 10.0)
    monkeypatch.setattr(config, "SOCIALS_MEDIA_MONTHLY_CAP_USD", 30.0)
    return api


def _settings(api):
    api.session.expire_all()
    return api.session.get(Workspace, WS_A).settings


def test_the_section_lists_tools_choices_defaults_caps_and_spend(tools):
    body = tools.client.get("/api/socials/media-tools").json()
    assert {row["toolkit"] for row in body["toolkits"]} >= {"fal_ai", "kieai", "templates", "kokoro"}
    assert _values(body["offered"]["footage"]) == ["fal_ai", "off"]
    assert body["defaults"] == media_tools.DEFAULTS
    assert body["caps"] == {"monthly_usd": 30.0, "per_post_usd": 10.0, "problem": None}
    assert body["spend"]["month_usd"] == 1.25 and body["spend"]["period_end"]


def test_an_owner_sets_defaults_and_both_caps(tools):
    changed = tools.client.put("/api/socials/media-tools", json={
        "defaults": {"ai_images": "kieai", "footage": "fal_ai"}, "monthly_cap_usd": 50, "per_post_cap_usd": 4.5,
    })
    assert changed.status_code == 200, changed.text
    body = changed.json()
    assert (body["defaults"]["ai_images"], body["defaults"]["footage"]) == ("kieai", "fal_ai")
    assert body["caps"] == {"monthly_usd": 50.0, "per_post_usd": 4.5, "problem": None}
    settings = _settings(tools)
    assert settings["media_tools"]["ai_images"] == "kieai"
    assert settings["socials"] == {"enabled": True, "media_monthly_cap_usd": 50.0, "media_post_cap_usd": 4.5}
    reset = tools.client.put("/api/socials/media-tools", json={"per_post_cap_usd": None}).json()
    assert reset["caps"]["per_post_usd"] == 10.0 and reset["defaults"]["ai_images"] == "kieai"


@pytest.mark.parametrize("body", [
    {"defaults": {"footage": "kieai"}},
    {"defaults": {"voice": "elevenlabs"}},
    {"defaults": {"teleport": "fal_ai"}},
    {"monthly_cap_usd": -1},
    {"per_post_cap_usd": "lots"},
    {"surprise": True},
])
def test_a_choice_not_offered_or_a_bad_cap_is_422_and_nothing_is_stored(tools, body):
    resp = tools.client.put("/api/socials/media-tools", json=body)
    assert resp.status_code == 422, resp.text
    settings = _settings(tools)
    assert "media_tools" not in settings and settings["socials"] == {"enabled": True}


def test_only_owners_and_admins_change_the_tools(tools):
    tools.role = "editor"  # edits posts, never the workspace's tools
    assert tools.client.put("/api/socials/media-tools", json={"defaults": {"ai_images": "kieai"}}).status_code == 403
    assert tools.client.get("/api/socials/media-tools").status_code == 200


# ── liked references go where the registry says they may ───────────────────


def _shot(kind=IMAGE_SLOT, references=("https://signed/ref.png",)):
    return Shot(slot="still_1", kind=kind, path="assets/slots/still_1.png", label="Still", prompt="A harbour",
                aspect_ratio="1:1", style=STYLE, references=references)


def test_a_reference_goes_only_to_a_flagged_action_that_makes_stills():
    fal = footage_routes.route_for(IMAGE_SLOT, BOTH)
    assert fal.recipe.with_reference({"prompt": "p"}, _shot(), fal) == {"prompt": "p", "image_url": "https://signed/ref.png"}
    assert fal.recipe.with_reference({"prompt": "p"}, _shot(references=()), fal) == {"prompt": "p"}
    video = footage_routes.route_for(VIDEO_SLOT, BOTH)
    assert video.recipe.with_reference({"prompt": "p"}, _shot(kind=VIDEO_SLOT), video) == {"prompt": "p"}
    kie = footage_routes.route_for(IMAGE_SLOT, BOTH, "kieai")  # not flagged: Flux Kontext would edit the image
    assert kie.recipe.with_reference({"prompt": "p"}, _shot(), kie) == {"prompt": "p"}
    flagged_kie = footage_routes.route_for(IMAGE_SLOT, _caps(*FAL, *KIE_STILLS[1:], _action(
        "kieai", "KIEAI_GENERATE_FLUX_KONTEXT_IMAGE", GENERATE_IMAGE, REFERENCE_IMAGE)), "kieai")
    assert flagged_kie.recipe.with_reference({"prompt": "p"}, _shot(), flagged_kie)["input_image"] == "https://signed/ref.png"


def test_fal_submits_the_styled_prompt_and_the_reference():
    route = footage_routes.route_for(IMAGE_SLOT, BOTH)
    asked = []

    async def ask(role, params, required=()):
        asked.append((role, params))
        return {"request_id": "req-1"}

    assert asyncio.run(RECIPES["fal_ai"].submit(ask, _shot(), route)) == "req-1"
    ((role, params),) = asked
    assert role == SUBMIT and params["input"]["image_url"] == "https://signed/ref.png"
    assert params["input"]["prompt"].startswith("A harbour, no readable text, no logos") and params["input"]["prompt"].endswith(STYLE)


def test_the_wave_3_migration_flags_fals_submit_once_and_back():
    seed = json.dumps(WAVE1.SOCIALS_MEDIA_ACTIONS_SEED)
    flagged = WAVE3.flagged(seed)
    parsed = parse_media_actions(flagged)
    assert REFERENCE_IMAGE in parsed["fal_ai"]["FAL_AI_SUBMIT_ASYNC_JOB"]
    assert REFERENCE_IMAGE not in parsed["kieai"]["KIEAI_GENERATE_FLUX_KONTEXT_IMAGE"]
    assert WAVE3.flagged(flagged) == flagged
    assert parse_media_actions(WAVE3.unflagged(flagged)) == parse_media_actions(seed)
    assert WAVE3.flagged("not json") == "not json" and WAVE3.flagged(None) is None
    reshaped = json.dumps({"kieai": {"generate_image": ["KIEAI_GENERATE_FLUX_KONTEXT_IMAGE"]}})
    assert json.loads(WAVE3.flagged(reshaped)) == json.loads(reshaped)  # no fal_ai there: left alone


def test_the_wave_3_migration_rewrites_the_seeded_row_and_its_description():
    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE system_settings (category TEXT, key TEXT, value TEXT, default_value TEXT, description TEXT)"))
        WAVE3._rewrite(conn, WAVE3.flagged, (WAVE3.OLD_WORDS, WAVE3.NEW_WORDS))  # no row yet: nothing to do
        row = next(r for r in WAVE1.settings_seed() if r["key"] == "media_actions")
        conn.execute(sa.text("INSERT INTO system_settings VALUES ('socials', 'media_actions', :v, :d, :t)"),
                     {"v": row["value"], "d": row["default_value"], "t": row["description"]})
        WAVE3._rewrite(conn, WAVE3.flagged, (WAVE3.OLD_WORDS, WAVE3.NEW_WORDS))
        stored = conn.execute(sa.text("SELECT value, default_value, description FROM system_settings")).one()
    for value in (stored.value, stored.default_value):
        assert REFERENCE_IMAGE in parse_media_actions(value)["fal_ai"]["FAL_AI_SUBMIT_ASYNC_JOB"]
    assert WAVE3.NEW_WORDS in stored.description


class _S3:
    def __init__(self, missing=()):
        self.missing, self.copied = set(missing), []

    def head_object(self, Bucket, Key):
        if Key in self.missing:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {}

    def copy_object(self, Bucket, Key, CopySource):
        self.copied.append((CopySource["Key"], Bucket, Key))

    def generate_presigned_url(self, method, Params, ExpiresIn, HttpMethod):
        return f"https://signed/{Params['Bucket']}/{Params['Key']}"


def _style(*refs, send_liked=True):
    return {"brand_style": {"references": list(refs), "send_liked": send_liked}}


def _reference(ref_id, stance="like"):
    return {"id": ref_id, "stance": stance, "path": f"{WS_A}/brand/references/{ref_id}.png", "content_type": "image/png"}


@pytest.fixture
def s3(monkeypatch):
    client = _S3()
    monkeypatch.setattr(socials_brand, "media_public_url_available", lambda: True)
    monkeypatch.setattr(socials_brand, "get_s3_client", lambda: client)
    monkeypatch.setattr(socials_brand, "get_public_s3_client", lambda: client)
    monkeypatch.setattr(socials_brand, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(config, "S3_DOCUMENTS_BUCKET", "documents")
    monkeypatch.setattr(config, "SOCIALS_PUBLIC_MEDIA_BUCKET", "")
    return client


def test_the_newest_liked_reference_is_linked_when_the_workspace_sends_them(s3, monkeypatch):
    settings = _style(_reference("old"), _reference("bad", "avoid"), _reference("new"))
    assert socials_brand.liked_reference_links(settings) == (f"https://signed/documents/workspaces/{WS_A}/brand/references/new.png",)
    assert socials_brand.liked_reference_links({**settings, "brand_style": {**settings["brand_style"], "send_liked": False}}) == ()
    assert socials_brand.liked_reference_links(_style(_reference("bad", "avoid"))) == ()
    s3.missing.add(f"workspaces/{WS_A}/brand/references/new.png")  # the mirror never landed: the next one
    assert socials_brand.liked_reference_links(settings)[0].endswith("/old.png")
    monkeypatch.setattr(socials_brand, "media_public_url_available", lambda: False)
    assert socials_brand.liked_reference_links(settings) == ()


def test_with_a_public_media_bucket_the_reference_is_copied_there_first(s3, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_PUBLIC_MEDIA_BUCKET", "public-media")
    (link,) = socials_brand.liked_reference_links(_style(_reference("new")))
    assert link == f"https://signed/public-media/brand-references/{WS_A}/brand/references/new.png"
    assert s3.copied == [(f"workspaces/{WS_A}/brand/references/new.png", "public-media", f"brand-references/{WS_A}/brand/references/new.png")]


def test_a_renders_inputs_are_the_style_the_defaults_and_the_links(s3):
    db = SimpleNamespace(get=lambda model, ident: workspace)
    workspace = SimpleNamespace(id=WS_A, settings={
        **_style(_reference("new")), "media_tools": {"ai_images": "kieai"},
    })
    workspace.settings["brand_style"]["profile"] = {"mood": ["calm"]}
    inputs = socials_brand.generation_inputs(db, workspace)
    assert inputs["style"] == STYLE and inputs["prefer"] == {IMAGE_SLOT: "kieai"}
    assert inputs["references"][0].endswith("/new.png")
