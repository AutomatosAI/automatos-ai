"""PRD-251C Wave 1, US-C105 — research and the composer read the history.

Pinned:

* the seeded research prompt reads the history first and adds only what neither the history
  nor any bank covers;
* a marketplace research row still holding PRD-251B's prompt takes the new one at boot; a
  curated one stays as it is, and another Playbook's row is never touched;
* research asks the Deliverables tool to leave out a Socials post's own files
  (``exclude_source_types``); the tool's default for every other caller is unchanged, and a
  value it cannot use is refused with why;
* the composer gets the opening lines of the workspace's last posts (their count is config),
  never another workspace's, and is told to open differently.
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
import tests.test_prd251w2_compose as compose_tests  # noqa: E402
from config import config  # noqa: E402
from core.models.socials import SocialPost  # noqa: E402
from core.seeds import seed_socials_package as seed  # noqa: E402
from modules.socials import compose, render  # noqa: E402
from modules.tools.discovery import handlers_deliverables  # noqa: E402
from tests.test_prd251_api import WS_A, WS_B  # noqa: E402

api = api_harness.api
composer = compose_tests.composer
UTC = timezone.utc
(RESEARCH,) = [spec for spec in seed.SOCIALS_PLAYBOOKS if spec["template_id"] == seed.RESEARCH_PLAYBOOK_TEMPLATE_ID]


def test_the_seeded_research_prompt_reads_the_history_first_and_adds_only_what_is_new():
    (step,) = RESEARCH["steps"]
    prompt = step["prompt_template"]
    assert prompt.index("history") < prompt.index("Research only the sources")
    assert "platform_get_social_history" in prompt and "neither the history nor any bank" in prompt
    assert f'platform_list_deliverables with exclude_source_types ["{render.DELIVERABLE_SOURCE_TYPE}"]' in prompt


class _OneRow:
    """db.query(...).filter(...).first() answering one row."""

    def __init__(self, row):
        self.row = row

    def query(self, *entities):
        return self

    def filter(self, *clauses):
        return self

    def first(self):
        return self.row


def _row(prompt):
    step = {"step_id": "research", "order": 1, "agent_id": 11, "agent_name": "Social Media Director",
            "error_handling": "stop", "output_key": "topics", "prompt_template": prompt}
    return SimpleNamespace(owner_type="marketplace", steps=[step])


def test_a_marketplace_row_with_the_old_prompt_takes_the_new_one_and_a_curated_one_stays():
    (step,) = RESEARCH["steps"]
    old = _row(seed._RESEARCH_PROMPT_251B)
    assert seed._ensure_playbook(_OneRow(old), RESEARCH, {}) == seed.UPDATED
    assert old.steps[0]["prompt_template"] == step["prompt_template"]
    assert old.steps[0]["agent_id"] == 11  # the rest of the step stays
    assert seed._ensure_playbook(_OneRow(old), RESEARCH, {}) == seed.PRESENT  # up to date now

    curated = _row("Our own research prompt: read the history, then the roadmap.")
    assert seed._ensure_playbook(_OneRow(curated), RESEARCH, {}) == seed.PRESENT
    assert curated.steps[0]["prompt_template"] == "Our own research prompt: read the history, then the roadmap."

    other = next(spec for spec in seed.SOCIALS_PLAYBOOKS if spec["template_id"] != seed.RESEARCH_PLAYBOOK_TEMPLATE_ID)
    assert seed.refreshed_steps(other, [{"step_id": "research", "prompt_template": seed._RESEARCH_PROMPT_251B}]) is None


class _Deliverables:
    """DeliverableService: records what the tool asked for."""

    asked = []

    def __init__(self, db, workspace_id):
        pass

    def list_deliverables(self, **kwargs):
        _Deliverables.asked.append(kwargs)
        return {"success": True, "deliverables": [], "total": 0}


def test_research_leaves_a_socials_posts_own_files_out_and_every_other_caller_is_unchanged(monkeypatch):
    monkeypatch.setattr("services.deliverable_service.DeliverableService", _Deliverables)
    _Deliverables.asked = []
    for params in ({}, {"exclude_source_types": ["social_post"]}, {"exclude_source_types": "social_post"},
                   {"exclude_source_types": ["social_post", "heartbeat"]}):
        assert asyncio.run(handlers_deliverables.list_deliverables(None, WS_A, params))["success"] is True
    assert [ask["source_type_exclude"] for ask in _Deliverables.asked] == [None, "social_post", "social_post", "social_post,heartbeat"]
    for unusable in ([1], ["social_post,chat"], [""], {"kind": "social_post"}):
        refused = asyncio.run(handlers_deliverables.list_deliverables(None, WS_A, {"exclude_source_types": unusable}))
        assert refused == {"success": False, "error": handlers_deliverables.EXCLUDED_ORIGINS_REFUSED}
    assert len(_Deliverables.asked) == 4  # a refused call reads nothing


def _posted(api, workspace_id, opening, days_ago):
    api.session.add(SocialPost(
        id=uuid.uuid4(), workspace_id=workspace_id, created_by="member-1", title=opening, status="published",
        content_hash="0" * 64, copy={"base": f"{opening}\nMore below."}, created_at=datetime.now(UTC) - timedelta(days=days_ago),
    ))
    api.session.commit()


def test_the_composer_gets_how_the_workspaces_last_posts_began(composer, monkeypatch):
    _posted(composer.api, WS_A, "Three weeks to Lisbon.", days_ago=3)
    _posted(composer.api, WS_A, "Missions, in one minute.", days_ago=1)
    _posted(composer.api, WS_B, "Not ours.", days_ago=1)
    assert compose_tests._compose(composer, [compose_tests._answer(composer)]).status_code == 200
    system, material = composer.model.asked[0][0]["content"], json.loads(composer.model.asked[0][1]["content"])
    assert material["recent_openings"] == ["Missions, in one minute.", "Three weeks to Lisbon."]
    assert compose.RECENT_OPENINGS_NOTE in system

    monkeypatch.setattr(config, "SOCIALS_COMPOSE_RECENT_OPENINGS", 1)
    assert compose_tests._compose(composer, [compose_tests._answer(composer)]).status_code == 200
    assert json.loads(composer.model.asked[0][1]["content"])["recent_openings"] == ["Missions, in one minute."]
