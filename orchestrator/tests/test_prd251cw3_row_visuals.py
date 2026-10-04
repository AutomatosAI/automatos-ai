"""PRD-251C Wave 3, US-C302 — a row's visual.

Pinned:

* **The row.** ``cadence[].visual`` is one of the mix's sources, with a toolkit only for AI
  images or footage; a text row takes none, and AI footage needs a video row. Its slots
  carry it.
* **The save.** A toolkit the workspace cannot use for that media now (not connected, not one
  Socials makes media with, or one that does not make that kind here) is a 422 saying why,
  for a new plan and a changed one, and nothing is saved; one it can use is kept.
* **The override.** The maker takes the row's visual whatever the plan's mix says, and each
  slot's ask names the row's toolkit (``via``).
* **Caps unchanged.** The render makes a ``via`` slot with that toolkit only, never another,
  and prices it as every shot is priced (a toolkit that prices nothing ahead is booked at its
  ceiling); the money checks after that are the render's own, unchanged. The Plan page's
  estimate prices a shot the same way (``media_tools.shot_usd``).
"""
from __future__ import annotations

import asyncio
import functools
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials_targets as socials_targets  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from config import config  # noqa: E402
from core.models.socials import SocialCampaign  # noqa: E402
from modules.socials import media_tools, plan_row_visuals, plans  # noqa: E402
from modules.socials.recipes import footage as footage_recipes  # noqa: E402
from services import socials_plan_maker as maker_mod  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402
from tests.test_prd251bw2_plans import _body  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251bw2_plans import bank  # noqa: E402,F401  (the fixture)
from tests.test_prd251bw3_media_tools import FAL, KIE_STILLS, _caps  # noqa: E402
from tests.test_prd251bw3_plan_visuals import BLOCKS, TEMPLATE_ID, TOPIC, _db, _plan, _PostsApi, _slot  # noqa: E402

api = api_harness.api
NOW = datetime(2026, 10, 14, 7, 30, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def toolkit_order(monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_FOOTAGE_TOOLKITS", "fal_ai,kieai,higgsfield_mcp")


def _row(source=None, toolkit=None, fmt="image", time="09:00"):
    row = {"channels": ["linkedin"], "format": fmt, "days": ["mon"], "time": time}
    return {**row, "visual": {"source": source, "toolkit": toolkit}} if source else row


# ── the row ────────────────────────────────────────────────────────────────


def test_a_rows_visual_is_kept_and_checked():
    rows = plans.validate_cadence(
        [_row("ai_images", "fal_ai"), _row("ai_footage", fmt="video", time="10:00"), _row("library", time="11:00"), _row(time="12:00")], {},
    )
    assert [row["visual"] for row in rows] == [
        {"source": "ai_images", "toolkit": "fal_ai"}, {"source": "ai_footage", "toolkit": None}, {"source": "library", "toolkit": None}, None,
    ]
    refused = [
        ({**_row(fmt="text"), "visual": {"source": "templates"}}, "templates is not for text rows"),
        (_row("ai_footage"), "ai_footage is not for image rows"),
        (_row("library", "fal_ai"), "names the Composio toolkit"),
        (_row("ai_images", "Fal AI!"), "names the Composio toolkit"),
        (_row("stock_photos"), "its source one of"),
        ({**_row(), "visual": {"source": "templates", "colour": "red"}}, "its source one of"),
    ]
    for row, why in refused:
        with pytest.raises(plans.InvalidPlan, match=why):
            plans.validate_cadence([row], {})


def test_a_rows_slots_carry_its_visual(bank):  # noqa: F811
    plan = _create_plan(bank, timezone="UTC", cadence=[_row("library"), _row(time="12:00")])
    stored = bank.session.get(SocialCampaign, uuid.UUID(plan["id"]))
    slots = plans.expand_slots(stored, datetime(2026, 10, 12, tzinfo=timezone.utc), datetime(2026, 10, 13, tzinfo=timezone.utc))
    assert [(slot.visual_source, slot.visual_toolkit) for slot in slots] == [("library", None), (None, None)]
    assert slots[0].to_dict()["visual"] == {"source": "library", "toolkit": None} and slots[1].to_dict()["visual"] is None


# ── the save ───────────────────────────────────────────────────────────────


def test_a_toolkit_the_workspace_cannot_use_is_refused_and_nothing_is_saved(bank, monkeypatch):  # noqa: F811
    monkeypatch.setattr(plan_row_visuals, "media_capabilities", lambda db, ws: _caps(*FAL, *KIE_STILLS))
    refused = [
        (_row("ai_images", "higgsfield_mcp"), "Higgsfield is not connected: connect it in Composio"),
        (_row("ai_images", "dalle"), "dalle is not a toolkit Socials makes images or footage with"),
        (_row("ai_footage", "kieai", fmt="video"), "Kie.ai does not offer"),
    ]
    for row, why in refused:
        resp = bank.client.post("/api/socials/plans", json=_body(cadence=[row]))
        assert resp.status_code == 422 and why in resp.text, resp.text
        assert "cadence[0].visual.toolkit" in resp.text
    assert bank.client.get("/api/socials/plans").json()["plans"] == []
    kept = _create_plan(bank, cadence=[_row("ai_images", "kieai")])
    assert kept["cadence"][0]["visual"] == {"source": "ai_images", "toolkit": "kieai"}
    changed = bank.client.put(f"/api/socials/plans/{kept['id']}", json={"cadence": [_row("ai_images", "higgsfield_mcp")]})
    assert changed.status_code == 422, changed.text
    assert bank.client.get(f"/api/socials/plans/{kept['id']}").json()["cadence"][0]["visual"]["toolkit"] == "kieai"


def test_a_save_without_a_named_toolkit_never_reads_the_registry(bank, monkeypatch):  # noqa: F811
    def unreachable(db, ws):
        raise AssertionError("the registry is read only for a row that names a toolkit")

    monkeypatch.setattr(plan_row_visuals, "media_capabilities", unreachable)
    assert _create_plan(bank, cadence=[_row("ai_images"), _row(time="12:00")])["cadence"][0]["visual"]["toolkit"] is None


# ── the override ───────────────────────────────────────────────────────────


def _write(monkeypatch, mix, slot):
    posts_api = _PostsApi()

    def propose(db, plan, slot, topic, now, visual_slots=()):
        return {"title": "Lisbon", "copy": {"base": "See you there."}, "variables": {}, "sources": {},
                "template_id": str(TEMPLATE_ID), "visual_prompts": {}}

    monkeypatch.setattr(maker_mod, "_posts_api", lambda: posts_api)
    monkeypatch.setattr(maker_mod, "_propose", propose)
    monkeypatch.setattr(maker_mod, "slot_targets", lambda db, ws, slot: [])
    monkeypatch.setattr(socials_targets, "set_post_targets", lambda db, post, actor, targets, agent=None: None)
    post = SimpleNamespace(id=uuid.uuid4(), template_id=TEMPLATE_ID)
    run = functools.partial(maker_mod.write, _db(), SimpleNamespace(id=WS_A), _plan(mix), slot, TOPIC, post, NOW)
    anyio.run(functools.partial(anyio.to_thread.run_sync, run))
    (changes,) = posts_api.edits
    return changes


def test_the_rows_visual_overrides_the_mix_and_its_asks_name_the_toolkit(monkeypatch):
    changes = _write(monkeypatch, {"templates": 100}, _slot(visual_source="ai_images", visual_toolkit="kieai"))
    assert set(changes["footage"]) == {"still_1", "still_2", "still_3"}
    assert {ask["via"] for ask in changes["footage"].values()} == {"kieai"}
    default = _write(monkeypatch, {"templates": 100}, _slot(visual_source="ai_images"))
    assert all("via" not in ask for ask in default["footage"].values())  # the workspace's default toolkit
    assert "footage" not in _write(monkeypatch, {"ai_images": 100}, _slot(visual_source="templates"))


# ── caps unchanged ─────────────────────────────────────────────────────────


def _plan_for(footage, caps):
    return footage_recipes.plan_for(footage, BLOCKS["slots"], caps, width=1080, height=1920)


def test_a_via_slot_is_made_by_that_toolkit_only_and_priced_as_every_shot():
    both = _caps(*FAL, *KIE_STILLS)
    assert _plan_for({"still_1": {"prompt": "A harbour"}}, both).shots[0][1].recipe.toolkit == "fal_ai"  # first in the order
    named = _plan_for({"still_1": {"prompt": "A harbour", "via": "kieai"}}, both)
    ((shot, route),) = named.shots
    assert route.recipe.toolkit == "kieai"
    ((_, _, price),) = asyncio.run(footage_recipes._price(None, WS_A, named.shots))
    assert price == pytest.approx(config.SOCIALS_FOOTAGE_CEILING_IMAGE_USD)  # booked as any Kie.ai still
    gone = _plan_for({"still_1": {"prompt": "A harbour", "via": "kieai"}}, _caps(*FAL))
    assert gone.shots == () and gone.fallback == {"still_1": "Kie.ai is not connected: connect it in Composio"}  # never fal.ai
    assert media_tools.shot_usd() == {"image": price, "video": pytest.approx(config.SOCIALS_FOOTAGE_CEILING_VIDEO_USD)}


def test_a_post_may_keep_a_via_and_a_bad_one_is_refused():
    from modules.socials import service

    assert service.validate_footage({"still_1": {"prompt": "A harbour", "via": "kieai"}}) == {"still_1": {"prompt": "A harbour", "via": "kieai"}}
    with pytest.raises(service.InvalidPost, match="via names a Composio toolkit"):
        service.validate_footage({"still_1": {"prompt": "A harbour", "via": "Kie AI"}})
