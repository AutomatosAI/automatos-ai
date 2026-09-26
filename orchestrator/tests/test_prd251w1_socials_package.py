"""PRD-251 US-120: the Socials package: two agents, four Playbooks, guided setup.

PURE (no database):
  * the seed data: each agent's persona and built-in skills (none a publisher
    skill or html-to-png), the four Playbooks through the create route's own
    validators, every input and ``{{ step_N.key }}`` a step reads, every tool a
    persona or a step names, the templates the Playbooks render;
  * the package: its members, questions, connects and guide steps, and
    ``platform_search_packages`` finding it;
  * the installer: a clone keeps its persona, a Playbook's install reuses the
    workspace's clone of its agent, a fixed render step is never given an agent,
    a re-install leaves the workspace's copy alone, template ids are unique across
    workspaces;
  * the executor: a scheduled run gets the declared defaults, every step stamps
    the run's progress, and a render step keeps stamping while it waits.

@integration (the orchestrator-tests job's Postgres, one rolled-back transaction
each): ``POST /api/marketplace/packages/socials/install`` clones both agents with
persona and skills and all four Playbooks, a re-install adds nothing, a second
workspace gets its own copies, and a skill synced after the first boot attaches
on the next.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import core.models  # noqa: E402,F401  (registers every mapper)
import api.marketplace as marketplace_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.tools.discovery.handlers_packages as hp  # noqa: E402
import services.package_installer as pi  # noqa: E402
from api import recipe_executor as rex  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.builtin_skills import load_manifest  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.core import PLAYBOOK_DOCUMENT_STEP, Agent, Skill, WorkflowTemplate  # noqa: E402
from core.models.marketplace_packages import MarketplacePackage  # noqa: E402
from core.seeds.seed_builtin_skills import seed_builtin_skills  # noqa: E402
from core.seeds.seed_packages import PACKAGES, seed_packages  # noqa: E402
from core.seeds.seed_socials_package import (  # noqa: E402
    BRAND_DESIGNER,
    CAROUSEL_TEMPLATE,
    DIRECTOR,
    LAUNCH_VIDEO_TEMPLATE,
    SOCIALS_AGENTS,
    SOCIALS_PACKAGE,
    SOCIALS_PLAYBOOKS,
    STEP_TIMEOUT_SECONDS,
    TOTAL_TIMEOUT_SECONDS,
    WEEKLY_POST_COUNT,
    playbook_agents,
    playbook_columns,
    seed_socials_marketplace,
)
from core.services.playbook_inputs import with_input_defaults  # noqa: E402
from core.services.playbook_step_refs import STEP_REFERENCE  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.socials.settings import KEY_MEDIA_MONTHLY_CAP, WORKSPACE_SOCIALS_SETTINGS_KEY  # noqa: E402
from modules.tools.discovery import cascade_installer as ci  # noqa: E402
from modules.tools.discovery.action_registry import get_action_registry  # noqa: E402
from tests.helpers_playbook_run import WS as RUN_WS, _Session, done, patch_edges, run_playbook  # noqa: E402

INSTALL_ROUTE = "/api/marketplace/packages/socials/install"
# D14: these post straight to a channel, or render outside media-render. No Socials
# agent carries one, and the seeds never name them (the acceptance script greps).
PUBLISHER_SKILLS = frozenset({"instagram-curator", "twitter-engager", "linkedin-content-creator", "html-to-png"})
PLAYBOOK_IDS = [spec["template_id"] for spec in SOCIALS_PLAYBOOKS]
SKILL_NAMES = sorted({name for spec in SOCIALS_AGENTS for name in spec["skills"]})
AGENTS = {spec["slug"]: spec for spec in SOCIALS_AGENTS}
PLAYBOOKS = {spec["name"]: spec for spec in SOCIALS_PLAYBOOKS}
VERSIONS = _ORCH / "alembic" / "versions"
_INPUT = re.compile(r"\{input\.(\w+)\}")
_TOOL_NAME = re.compile(r"\b(platform_[a-z_]+|search_knowledge|scratchpad_write)\b")
# The ids an agent row stands for in the pure tests: slug → a row with id and name.
FAKE_AGENTS = {spec["slug"]: SimpleNamespace(id=900 + i, name=spec["name"]) for i, spec in enumerate(SOCIALS_AGENTS)}


def _texts():
    """Every persona and step prompt the package seeds."""
    for spec in SOCIALS_AGENTS:
        yield f"persona of {spec['slug']}", spec["custom_persona_prompt"]
    for spec in SOCIALS_PLAYBOOKS:
        for step in spec["steps"]:
            if step.get("prompt_template"):
                yield f"{spec['name']} step {step['order']}", step["prompt_template"]


def _row(spec, agents=FAKE_AGENTS) -> WorkflowTemplate:
    return WorkflowTemplate(**playbook_columns(spec, agents))


# ---------------------------------------------------------------------------
# 1. The agents: persona, built-in skills, never a publisher skill
# ---------------------------------------------------------------------------


def test_the_director_and_the_brand_designer_carry_the_skills_the_story_names():
    assert [spec["slug"] for spec in SOCIALS_AGENTS] == [DIRECTOR, BRAND_DESIGNER]
    assert AGENTS[DIRECTOR]["name"] == "Social Media Director"
    assert AGENTS[BRAND_DESIGNER]["name"] == "Brand Designer"
    assert AGENTS[DIRECTOR]["skills"] == [
        "social-video-director", "social-ops", "social-production-workflow", "social-template-payloads",
        "social-media-strategist", "social-brand-voice", "carousel-design-system", "image-prompt-engineer",
        "short-video-editing-coach",
    ]
    assert AGENTS[BRAND_DESIGNER]["skills"] == [
        "brand-kit-builder", "visual-storyteller", "image-prompt-engineer", "social-brand-voice",
    ]
    # brand-guardian audits the platform's own copy, not a customer's brand.
    assert "brand-guardian" not in SKILL_NAMES


def test_every_skill_an_agent_links_is_a_builtin_skill_synced_from_automatos_skills():
    manifest = load_manifest()
    assert set(SKILL_NAMES) <= set(manifest), sorted(set(SKILL_NAMES) - set(manifest))
    for name in SKILL_NAMES:
        assert manifest[name].skill_source == f"builtin:{name}"


def test_no_agent_names_a_publisher_skill_or_html_to_png():
    assert not PUBLISHER_SKILLS & set(SKILL_NAMES)
    seed_source = (_ORCH / "core" / "seeds" / "seed_socials_package.py").read_text(encoding="utf-8")
    assert not [name for name in PUBLISHER_SKILLS if name in seed_source]


def test_the_personas_say_what_each_agent_does_and_that_neither_publishes():
    director = AGENTS[DIRECTOR]["custom_persona_prompt"]
    designer = AGENTS[BRAND_DESIGNER]["custom_persona_prompt"]
    for persona in (director, designer):
        assert "never publish, schedule or approve" in persona
    for tool in ("platform_create_social_post", "platform_submit_social_post", "platform_get_social_post",
                 "platform_list_templates", "platform_get_template_schema"):
        assert tool in director
    for tool in ("platform_web_fetch", "platform_get_brand_kit", "platform_update_brand_kit"):
        assert tool in designer
    assert "social_handles replaces the whole map" in designer


def test_every_tool_a_persona_or_a_step_names_is_one_the_platform_has():
    registered = {action.name for action in get_action_registry().get_all()}
    from modules.tools.registry.tool_registry import ToolRegistry

    for where, text in _texts():
        for name in set(_TOOL_NAME.findall(text)):
            if name.startswith("platform_"):
                assert name in registered, f"{where} names {name}, which is not a platform action"
            elif name == "search_knowledge":
                assert ToolRegistry().get_tool(name) is not None, where


# ---------------------------------------------------------------------------
# 2. The Playbooks: valid rows, every input and reference answered
# ---------------------------------------------------------------------------


def test_the_four_playbooks_pass_the_create_routes_validators():
    assert [spec["name"] for spec in SOCIALS_PLAYBOOKS] == [
        "Brand kit from your website", "Launch video", "Weekly social posts", "Image carousel",
    ]
    for spec in SOCIALS_PLAYBOOKS:
        row = _row(spec)
        for check in (row.validate_steps, row.validate_execution_config, row.validate_schedule_config):
            assert check() == (True, None), spec["name"]
        assert row.owner_type == "marketplace" and row.workspace_id is None and row.is_approved is True
        assert row.is_system is False  # a workspace's copy stays deletable
        assert row.template_id.startswith("marketplace-")
        assert row.recommended_agents == [FAKE_AGENTS[slug].name for slug in playbook_agents(spec)]


def test_agent_steps_name_the_agent_and_a_fixed_step_names_none():
    for spec in SOCIALS_PLAYBOOKS:
        for stored, seeded in zip(_row(spec).steps, spec["steps"]):
            assert "agent_slug" not in stored
            if seeded.get("type") == PLAYBOOK_DOCUMENT_STEP:
                assert "agent_id" not in stored and "agent_name" not in stored
            else:
                agent = FAKE_AGENTS[seeded["agent_slug"]]
                assert (stored["agent_id"], stored["agent_name"]) == (agent.id, agent.name)
    assert playbook_agents(PLAYBOOKS["Brand kit from your website"]) == [BRAND_DESIGNER]
    for name in ("Launch video", "Weekly social posts", "Image carousel"):
        assert playbook_agents(PLAYBOOKS[name]) == [DIRECTOR]


def test_every_input_a_step_reads_is_declared():
    for spec in SOCIALS_PLAYBOOKS:
        for step in spec["steps"]:
            for name in _INPUT.findall(step.get("prompt_template") or ""):
                assert name in spec["inputs"], f"{spec['name']} step {step['order']} reads undeclared {name}"
        for name, declared in spec["inputs"].items():
            assert declared.get("required") or "default" in declared, f"{spec['name']}: {name}"


def test_the_launch_videos_render_step_reads_only_what_step_one_saves():
    spec = PLAYBOOKS["Launch video"]
    script, render, draft = spec["steps"]
    assert render["type"] == PLAYBOOK_DOCUMENT_STEP and "prompt_template" not in render
    config_ = render["config"]
    assert (config_["format"], config_["template_name"]) == ("social_video", LAUNCH_VIDEO_TEMPLATE)
    assert config_["data"] == "{{ step_1.variables }}"
    references = STEP_REFERENCE.findall(json.dumps(config_))
    assert {(int(step), key) for step, key in references} == {(1, "variables"), (1, "title")}
    for _, key in references:
        assert f'"{key}"' in script["prompt_template"] and "scratchpad_write" in script["prompt_template"]
    # The draft step attaches the rendered file and does not render it again.
    assert "render false" in draft["prompt_template"] and '"9:16"' in draft["prompt_template"]
    assert "platform_submit_social_post" in draft["prompt_template"]


def test_the_templates_the_playbooks_render_are_seeded_starters():
    formats = {starter["name"]: starter for starter in social_starters()}
    assert formats[LAUNCH_VIDEO_TEMPLATE]["format"] == "social_video"
    assert "1080x1920" in formats[LAUNCH_VIDEO_TEMPLATE]["blocks"]["sizes"]  # the 9:16 the post attaches
    carousel = formats[CAROUSEL_TEMPLATE]
    assert carousel["format"] == "social_image" and len(carousel["blocks"]["stills"]) > 1
    # generate_document refuses a template that renders several images, so the
    # carousel renders as its post, which keeps every slide.
    _slides, draft = PLAYBOOKS["Image carousel"]["steps"]
    assert not any(step.get("type") == PLAYBOOK_DOCUMENT_STEP for step in PLAYBOOKS["Image carousel"]["steps"])
    assert f'template "{CAROUSEL_TEMPLATE}"' in draft["prompt_template"] and "render true" in draft["prompt_template"]


def test_the_weekly_playbook_proposes_five_posts_by_default_and_is_seeded_unscheduled():
    spec = PLAYBOOKS["Weekly social posts"]
    row = _row(spec)
    assert row.schedule_config == {"type": "manual"}
    assert spec["inputs"]["count"]["default"] == WEEKLY_POST_COUNT == 5
    assert "platform_schedule_playbook" in spec["description"]
    plan, draft = spec["steps"]
    for source in ("platform_list_deliverables", "platform_list_blog_posts", "platform_browse_reports"):
        assert source in plan["prompt_template"]
    assert "Proposed for:" in draft["prompt_template"] and "render true" in draft["prompt_template"]


def test_a_run_has_the_time_a_render_takes():
    assert TOTAL_TIMEOUT_SECONDS >= config.SOCIALS_RENDER_MAX_WAIT_SECONDS + 2 * STEP_TIMEOUT_SECONDS
    assert config.PLAYBOOK_PROGRESS_STAMP_SECONDS < config.TASK_STALL_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# 3. The package: members, guided setup, search
# ---------------------------------------------------------------------------


def _package(slug="socials") -> dict:
    return next(pkg for pkg in PACKAGES if pkg["slug"] == slug)


def _load_migration(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"{name}_for_socials_package", VERSIONS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_package_is_the_two_agents_then_the_four_playbooks():
    assert _package() is SOCIALS_PACKAGE
    assert [(m["type"], m["ref"]) for m in SOCIALS_PACKAGE["members"]] == (
        [("agent", DIRECTOR), ("agent", BRAND_DESIGNER)] + [("playbook", pid) for pid in PLAYBOOK_IDS]
    )
    assert SOCIALS_PACKAGE["showcase"] is True


def test_the_setup_manifest_carries_the_questions_connects_guide_steps_and_report():
    manifest = SOCIALS_PACKAGE["setup_manifest"]
    questions = {q["id"]: q for q in manifest["questions"]}
    assert list(questions) == ["website_url", "channels", "posting_cadence", "media_monthly_cap_usd"]
    assert questions["media_monthly_cap_usd"]["setting"] == f"{WORKSPACE_SOCIALS_SETTINGS_KEY}.{KEY_MEDIA_MONTHLY_CAP}"

    connects = manifest["required_connects"]
    assert all(c["optional"] is True for c in connects)
    groups = {}
    for c in connects:
        groups.setdefault(c["group"], []).append(c["app_name"])
    assert groups["publish"] == ["LINKEDIN", "TWITTER", "INSTAGRAM"]
    media = _load_migration("prd251_wave1").SOCIALS_MEDIA_ACTIONS_SEED
    assert set(groups["generation"]) == {t.upper() for t, caps in media.items() if "generate_video" in caps}
    assert set(groups["voice"]) == {t.upper() for t, caps in media.items() if "tts" in caps}

    steps = manifest["guide_steps"]
    assert [s["step"] for s in steps] == [1, 2, 3, 4]
    assert [s["title"] for s in steps] == [
        "Turn Socials on", "Build your brand kit", "Make your launch video", "Approve it in the Socials tab",
    ]
    assert "Brand kit from your website" in steps[1]["description"] and "Launch video" in steps[2]["description"]
    assert [r["name"] for r in manifest["report_templates"]] == ["weekly-social-report"]


def _search(monkeypatch, signals: dict) -> dict:
    instances = [MarketplacePackage(**dict(pkg)) for pkg in PACKAGES]
    monkeypatch.setattr("services.marketplace_packages.list_packages", lambda db: instances)
    monkeypatch.setattr(hp, "_load_workspace", lambda db, workspace_id: None)
    return asyncio.run(hp.search_packages(SimpleNamespace(), "ws-1", signals))


@pytest.mark.parametrize("signals", [
    {"text": "we post short videos and carousels on LinkedIn and Instagram"},
    {"urls": ["https://www.linkedin.com/company/harbourline"]},
    {"platforms": ["tiktok"]},
])
def test_platform_search_packages_finds_socials_with_its_guided_setup(monkeypatch, signals):
    out = _search(monkeypatch, signals)
    socials = next(m for m in out["matches"] if m["slug"] == "socials")
    manifest = SOCIALS_PACKAGE["setup_manifest"]
    assert socials["contents"] == {"agent": 2, "playbook": 4}
    assert socials["questions"] == manifest["questions"]
    assert socials["required_connects"] == manifest["required_connects"]
    assert socials["guide_steps"] == manifest["guide_steps"]


def test_a_shopify_store_is_still_offered_a_shopify_package_first(monkeypatch):
    out = _search(monkeypatch, {"platforms": ["shopify"], "text": "run my store orders inventory customers revenue"})
    assert out["matches"][0]["slug"] == "shopify-management"
    assert "socials" not in [m["slug"] for m in out["matches"]]


# ---------------------------------------------------------------------------
# 4. The installer
# ---------------------------------------------------------------------------


def _marketplace_agent(**overrides):
    values = dict(
        id=11, name="Social Media Director", description="d", agent_type="specialized", configuration={},
        model_config=None, tags=["socials"], status="active", original_creator_id=None, version="1.0.0",
        persona_id=None, custom_persona_prompt=AGENTS[DIRECTOR]["custom_persona_prompt"],
        use_custom_persona=True, skills=[], install_count=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_a_clone_keeps_the_marketplace_agents_persona():
    persona_id = uuid.uuid4()
    cloned, _name = ci.clone_agent_to_workspace(MagicMock(), uuid.uuid4(), _marketplace_agent(persona_id=persona_id))
    assert cloned.custom_persona_prompt == AGENTS[DIRECTOR]["custom_persona_prompt"]
    assert cloned.use_custom_persona is True and cloned.persona_id == persona_id
    assert cloned.owner_type == "workspace" and cloned.cloned_from_id == 11


def test_the_remap_never_gives_a_fixed_step_an_agent_and_never_changes_the_marketplace_steps():
    marketplace_steps = playbook_columns(PLAYBOOKS["Launch video"], {DIRECTOR: SimpleNamespace(id=11, name="Social Media Director")})["steps"]
    recipe = WorkflowTemplate(steps=marketplace_steps)
    before = json.dumps(marketplace_steps, sort_keys=True)

    ci._remap_recipe_steps(MagicMock(), recipe, {"Social Media Director": 501}, {11: 501})

    script, render, draft = recipe.steps
    assert script["agent_id"] == draft["agent_id"] == 501
    assert "agent_id" not in render
    assert json.dumps(marketplace_steps, sort_keys=True) == before  # new dicts, the source untouched


class _TemplateIds:
    """db.query(WorkflowTemplate.id).filter(template_id == x).first(): taken ids answer a row."""

    def __init__(self, taken):
        self.taken = set(taken)
        self.asked = []

    def query(self, *entities):
        return self

    def filter(self, clause):
        self._value = clause.right.value
        self.asked.append(self._value)
        return self

    def first(self):
        return (1,) if self._value in self.taken else None


def test_a_template_id_is_unique_across_workspaces_not_only_within_one():
    ws = uuid.UUID("12345678-9abc-def0-1234-56789abcdef0")
    assert pi.free_template_id(_TemplateIds([]), "socials-launch-video", ws) == "socials-launch-video"
    # Another workspace installed it first: this one takes its own suffix.
    assert pi.free_template_id(_TemplateIds(["socials-launch-video"]), "socials-launch-video", ws) == (
        "socials-launch-video-12345678"
    )
    db = _TemplateIds(["socials-launch-video", "socials-launch-video-12345678"])
    assert pi.free_template_id(db, "socials-launch-video", ws) == "socials-launch-video-12345678-2"
    assert len(db.asked) == 3


def test_the_standalone_playbook_install_takes_its_template_id_the_same_way():
    # The marketplace Playbooks tab installs one Playbook through its own route;
    # its per-workspace counter collided on the second workspace (unique index).
    source = (_ORCH / "api" / "workflow_recipes.py").read_text(encoding="utf-8")
    assert "free_template_id(db, base_template_id, ctx.workspace_id)" in source


class _Rows:
    """db.query(...).filter(...).first() answering the given rows in call order."""

    def __init__(self, *rows):
        self._rows = list(rows)

    def query(self, *entities):
        return self

    def filter(self, *clauses):
        return self

    def first(self):
        return self._rows.pop(0)

    def flush(self):
        pass


@pytest.mark.parametrize("existing, remap", [(None, True), (SimpleNamespace(id=77, name="Launch video"), False)])
def test_a_reinstall_leaves_the_workspaces_copy_of_a_playbook_as_it_is(monkeypatch, existing, remap):
    marketplace = SimpleNamespace(id=31, name="Launch video", template_id="marketplace-socials-launch-video")
    seen = {}

    async def cascade(**kwargs):
        seen.update(kwargs)
        return ci.CascadeResult()

    monkeypatch.setattr(ci, "cascade_recipe_dependencies", cascade)
    monkeypatch.setattr(pi, "_clone_recipe_to_workspace", lambda db, ws, recipe, uid: (SimpleNamespace(id=78), recipe.name))
    manifest = asyncio.run(pi._install_playbook(_Rows(marketplace, existing), uuid.uuid4(), marketplace.template_id))

    assert seen["remap_steps"] is remap
    assert manifest.by_type("playbook")[0].status == ("cloned" if remap else "already_installed")


def test_a_playbooks_install_reuses_the_workspaces_clone_of_its_agent(monkeypatch):
    director = _marketplace_agent()
    installed = SimpleNamespace(id=501, name="Social Media Director")
    recipe = playbook_columns(PLAYBOOKS["Launch video"], {DIRECTOR: director})
    cloned_recipe = WorkflowTemplate(steps=recipe["steps"])

    async def agent_cascade(db, ws, marketplace_agent, cloned_agent):
        assert cloned_agent is installed
        return ci.CascadeResult()

    monkeypatch.setattr(ci, "workspace_clone_of", lambda db, ws, agent: installed)
    monkeypatch.setattr(ci, "clone_agent_to_workspace", lambda *a, **k: pytest.fail("the agent was installed already"))
    monkeypatch.setattr(ci, "cascade_agent_dependencies", agent_cascade)

    result = asyncio.run(ci.cascade_recipe_dependencies(
        _Rows(director), uuid.uuid4(), SimpleNamespace(**recipe), cloned_recipe,
    ))

    assert result.cloned_items == [] and director.install_count == 0
    assert [step.get("agent_id") for step in cloned_recipe.steps] == [501, None, 501]


# ---------------------------------------------------------------------------
# 5. The executor: declared defaults, progress stamps
# ---------------------------------------------------------------------------


def test_the_declared_defaults_fill_only_what_a_run_was_not_given():
    declared = {"count": {"type": "integer", "default": 5}, "topic": {"type": "string", "required": True}}
    given = {"channels": "linkedin"}
    filled = with_input_defaults(given, declared)
    assert filled == {"channels": "linkedin", "count": 5}
    assert given == {"channels": "linkedin"}  # a new dict
    assert with_input_defaults({"count": 3}, declared) == {"count": 3}
    assert with_input_defaults({"a": 1}, None) == {"a": 1}


def _one_step_run(monkeypatch, *, inputs, prompt):
    from core.models.core import BoardTask, RecipeExecution
    from core.models.workspaces import Workspace

    prompts = []

    async def step(**kwargs):
        prompts.append(kwargs["clean_prompt"])
        return done("Five posts drafted.")

    execution = SimpleNamespace(
        execution_id="exec-120-defaults", recipe_id=120, workspace_id=RUN_WS, status="pending", current_step=0,
        step_results=None, error_message=None, completed_at=None, started_at=None, output_data=None,
        execution_metadata={},
    )
    steps = [{"step_id": "plan", "order": 1, "agent_id": 7, "prompt_template": prompt,
              "error_handling": "stop", "max_retries": 0}]
    session = _Session({
        WorkflowTemplate: [SimpleNamespace(id=120, name="Weekly social posts", steps=steps, execution_config={}, inputs=inputs)],
        RecipeExecution: [execution],
        Workspace: [SimpleNamespace(deleted_at=None, paused_at=None, paused_reason=None)],
        Agent: [SimpleNamespace(id=7, name="Social Media Director", configuration={}, status="active")],
        BoardTask: [SimpleNamespace(id=1200, status="in_progress", result=None, error_message=None,
                                    review_feedback=None, completed_at=None)],
    })
    patch_edges(monkeypatch, session=session, step=step)
    # A scheduled run: nothing given (services/playbook_scheduler.py launches with {}).
    asyncio.run(rex._execute_recipe_inner("exec-120-defaults", 120, RUN_WS, {}, None))
    return execution, prompts


def test_a_scheduled_run_gets_the_playbooks_declared_defaults(monkeypatch):
    spec = PLAYBOOKS["Weekly social posts"]
    execution, prompts = _one_step_run(monkeypatch, inputs=spec["inputs"], prompt=spec["steps"][0]["prompt_template"])
    assert execution.status == "completed", execution.error_message
    (prompt,) = prompts
    assert f"{WEEKLY_POST_COUNT} posts across these channels: linkedin, twitter, instagram" in prompt
    assert "{input." not in prompt


def test_an_input_without_a_default_is_still_named_not_invented(monkeypatch):
    spec = PLAYBOOKS["Launch video"]
    execution, prompts = _one_step_run(monkeypatch, inputs=spec["inputs"], prompt=spec["steps"][0]["prompt_template"])
    assert execution.status == "failed" and prompts == []
    # F182: the run asks the owner for it by name before step 1, and invents nothing.
    assert "- launch:" in execution.error_message


def test_every_step_stamps_the_runs_progress(monkeypatch):
    execution, _card = run_playbook(monkeypatch, outcomes=[done("one"), done("two")], step_seconds=1, exec_config={})
    assert execution.status == "completed", execution.error_message
    assert "last_progress_at" in execution.execution_metadata


def test_a_render_step_keeps_stamping_while_it_waits_and_stops_after():
    stamps = []

    async def run():
        result = await rex._stamping_progress(asyncio.sleep(0.2, result="rendered"), lambda: stamps.append(1), 0.02)
        after = len(stamps)
        await asyncio.sleep(0.1)
        return result, after

    result, after = asyncio.run(run())
    assert result == "rendered"
    assert after >= 3 and len(stamps) == after


def test_the_render_step_runs_under_the_stamp():
    source = (_ORCH / "api" / "recipe_executor.py").read_text(encoding="utf-8")
    branch = source[source.index("if step_type == PLAYBOOK_DOCUMENT_STEP:"):]
    render = branch[: branch.index("register_as_deliverable")]
    assert "_stamping_progress(" in render and "gen_service.generate(" in render
    assert "app_config.PLAYBOOK_PROGRESS_STAMP_SECONDS" in render


def test_boot_seeds_the_socials_marketplace_after_the_builtin_skills_and_before_the_packages():
    source = (_ORCH / "main.py").read_text(encoding="utf-8")
    skills = source.index("seed_builtin_skills(db)")
    socials = source.index("seed_socials_marketplace(db)")
    packages = source.index("seed_packages(db, create_only=True)")
    assert skills < socials < packages
    block = source[source.rindex("try:", 0, socials): source.index("except Exception", socials)]
    assert "from core.seeds.seed_socials_package import seed_socials_marketplace" in block


# ---------------------------------------------------------------------------
# 6. On Postgres: install, re-install, a second workspace, a later sync
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_engine():
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            for table in ("agents", "skills", "agent_skills", "workflow_recipes", "marketplace_packages",
                          "workspace_enabled_skills", "agent_tool_assignments", "workspaces"):
                conn.execute(sa.text(f"SELECT 1 FROM {table} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Socials package install needs the test database: {exc}")
    yield engine
    engine.dispose()


@contextmanager
def _rolled_back(engine):
    with engine.connect() as conn:
        trans = conn.begin()
        session = Session(bind=conn, join_transaction_mode="create_savepoint")
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            # The cascade reads a legacy catalogue table (marketplace_items) that
            # only alembic creates; the test schema has none. Rolled back below.
            conn.execute(sa.text(
                "CREATE TABLE IF NOT EXISTS marketplace_items (id serial PRIMARY KEY, type text, name text, metadata jsonb)"
            ))
            yield conn, session
        finally:
            session.close()
            trans.rollback()


def _skills_manifest(tmp_path: Path, *, synced) -> Path:
    """The built-in skills of the two agents, as a manifest; ``synced`` have seed files."""
    manifest = tmp_path / "core" / "seeds" / "skills" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"skills": {
        name: {"seed": f"{name}.md", "source": f"test/{name}/SKILL.md"} for name in SKILL_NAMES
    }}), encoding="utf-8")
    for name in synced:
        _sync(manifest, name)
    return manifest


def _sync(manifest: Path, name: str) -> None:
    """What the owner's scripts/sync-skills.py run leaves: the skill's seed file."""
    (manifest.parent / f"{name}.md").write_text(
        f'---\nname: {name}\ndescription: Fixture {name} for the Socials package tests\nversion: "1.0.0"\n---\n\n'
        f"Fixture body for {name}.\n",
        encoding="utf-8",
    )


def _publisher_rows(session) -> None:
    """The old publisher skills and html-to-png exist as global skills: never attached."""
    for name in sorted(PUBLISHER_SKILLS):
        if session.query(Skill).filter(Skill.name == name, Skill.workspace_id.is_(None)).first() is None:
            session.add(Skill(name=name, description=f"Fixture {name}", skill_type="technical",
                              is_active=True, workspace_id=None, skill_source="fixture"))
    session.flush()


def _boot(session, manifest: Path) -> dict:
    """The leader worker's boot seeds, in main.py's order."""
    seed_builtin_skills(session, manifest_path=manifest)
    outcome = seed_socials_marketplace(session)
    seed_packages(session, create_only=True)
    session.commit()
    return outcome


def _workspace(conn) -> uuid.UUID:
    ws = uuid.uuid4()
    conn.execute(
        sa.text("INSERT INTO workspaces (id, name, onboarding) VALUES (CAST(:id AS uuid), :name, CAST(:ob AS jsonb))"),
        {"id": str(ws), "name": f"prd251w1-socials-{str(ws)[:8]}", "ob": json.dumps({"stage": "completed"})},
    )
    return ws


def _client(session, ws: uuid.UUID) -> TestClient:
    app = FastAPI()
    app.include_router(marketplace_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
        workspace_id=ws,
        user=UserContext(id="owner-1", clerk_user_id="clerk-owner-1", system_role="user"),
        auth_type="clerk",
    )
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app)


def _marketplace_rows(session):
    agents = {
        a.slug: a for a in session.query(Agent).filter(Agent.owner_type == "marketplace", Agent.slug.in_(list(AGENTS)))
    }
    playbooks = {
        p.template_id: p
        for p in session.query(WorkflowTemplate).filter(
            WorkflowTemplate.owner_type == "marketplace", WorkflowTemplate.template_id.in_(PLAYBOOK_IDS)
        )
    }
    return agents, playbooks


def _installed(session, ws):
    agents = session.query(Agent).filter(Agent.workspace_id == ws, Agent.owner_type == "workspace").all()
    playbooks = session.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == ws, WorkflowTemplate.owner_type == "workspace"
    ).all()
    return agents, playbooks


def _assert_installed(session, ws):
    """Both agents with persona and skills, all four Playbooks on this workspace's agents."""
    market_agents, market_playbooks = _marketplace_rows(session)
    agents, playbooks = _installed(session, ws)
    assert len(agents) == 2 and len(playbooks) == 4
    clones = {a.cloned_from_id: a for a in agents}
    clone_of = {}
    for spec in SOCIALS_AGENTS:
        clone = clones[market_agents[spec["slug"]].id]
        clone_of[spec["slug"]] = clone.id
        assert clone.name == spec["name"] and clone.owner_id == str(ws)
        assert clone.custom_persona_prompt == spec["custom_persona_prompt"] and clone.use_custom_persona is True
        names = {skill.name for skill in clone.skills}
        assert names == set(spec["skills"]) and not names & PUBLISHER_SKILLS
    enabled = session.execute(
        sa.text("SELECT count(*) FROM workspace_enabled_skills WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": str(ws)}
    ).scalar()
    assert enabled == len(SKILL_NAMES)

    copies = {p.cloned_from_id: p for p in playbooks}
    for spec in SOCIALS_PLAYBOOKS:
        market = market_playbooks[spec["template_id"]]
        copy = copies[market.id]
        assert copy.name == spec["name"] and copy.template_id.startswith(spec["template_id"].replace("marketplace-", ""))
        assert copy.validate_steps() == (True, None)
        for stored, market_step, seeded in zip(copy.steps, market.steps, spec["steps"]):
            if seeded.get("type") == PLAYBOOK_DOCUMENT_STEP:
                assert "agent_id" not in stored and "agent_id" not in market_step
            else:
                assert stored["agent_id"] == clone_of[seeded["agent_slug"]]
                assert market_step["agent_id"] == market_agents[seeded["agent_slug"]].id
    return {a.id for a in agents}, {p.template_id: json.dumps(p.steps, sort_keys=True) for p in playbooks}


@pytest.mark.integration
def test_installing_the_package_clones_both_agents_and_all_four_playbooks_and_a_reinstall_adds_nothing(
    pg_engine, monkeypatch, tmp_path,
):
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    manifest = _skills_manifest(tmp_path, synced=SKILL_NAMES)
    with _rolled_back(pg_engine) as (conn, session):
        _publisher_rows(session)
        _boot(session, manifest)
        ws = _workspace(conn)
        client = _client(session, ws)

        first = client.post(INSTALL_ROUTE)
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["success"] is True, body
        kinds = [r["type"] for r in body["registrations"]]
        assert kinds.count("agent") == 2 and kinds.count("playbook") == 4
        assert all(r["workspace_owned"] for r in body["registrations"])
        agent_ids, steps = _assert_installed(session, ws)

        again = client.post(INSTALL_ROUTE)
        assert again.status_code == 200, again.text
        assert again.json()["success"] is True and again.json()["added_count"] == 0
        assert _assert_installed(session, ws) == (agent_ids, steps)

        # A second workspace gets copies of its own: template ids are unique across workspaces.
        other = _workspace(conn)
        second = _client(session, other).post(INSTALL_ROUTE)
        assert second.status_code == 200, second.text
        assert second.json()["success"] is True, second.json()
        _other_agents, other_steps = _assert_installed(session, other)
        assert not set(other_steps) & set(steps)
        market_agents, _ = _marketplace_rows(session)
        assert {a.install_count for a in market_agents.values()} == {2}


@pytest.mark.integration
def test_a_skill_synced_after_the_first_boot_attaches_on_the_next_and_nothing_is_seeded_twice(pg_engine, tmp_path):
    late = "short-video-editing-coach"
    manifest = _skills_manifest(tmp_path, synced=[name for name in SKILL_NAMES if name != late])
    with _rolled_back(pg_engine) as (conn, session):
        assert session.query(Skill).filter(Skill.name == late, Skill.workspace_id.is_(None)).first() is None, (
            f"a global '{late}' skill already exists in the test database"
        )
        _publisher_rows(session)
        first = _boot(session, manifest)
        assert first["skills"][DIRECTOR]["missing"] == [late]
        agents, _ = _marketplace_rows(session)
        director_skills = [s.name for s in agents[DIRECTOR].skills]
        assert late not in director_skills and director_skills[0] == "social-video-director"

        _sync(manifest, late)  # the owner syncs it after the first boot
        second = _boot(session, manifest)

        assert second["skills"][DIRECTOR] == {"attached": [late], "missing": []}
        assert second["skills"][BRAND_DESIGNER] == {"attached": [], "missing": []}
        assert set(second["agents"].values()) == {"present"} and set(second["playbooks"].values()) == {"present"}
        agents, playbooks = _marketplace_rows(session)
        for spec in SOCIALS_AGENTS:
            names = [s.name for s in agents[spec["slug"]].skills]
            assert sorted(names) == sorted(spec["skills"]) and not set(names) & PUBLISHER_SKILLS
        assert session.query(Agent).filter(Agent.owner_type == "marketplace", Agent.slug.in_(list(AGENTS))).count() == 2
        assert len(playbooks) == 4


@pytest.mark.integration
def test_platform_search_packages_finds_the_seeded_socials_package(pg_engine, tmp_path):
    manifest = _skills_manifest(tmp_path, synced=SKILL_NAMES)
    with _rolled_back(pg_engine) as (conn, session):
        _boot(session, manifest)
        ws = _workspace(conn)
        out = asyncio.run(hp.search_packages(session, ws, {"text": "social media videos for LinkedIn and Instagram"}))
        socials = next(m for m in out["matches"] if m["slug"] == "socials")
        manifest_ = SOCIALS_PACKAGE["setup_manifest"]
        assert socials["questions"] == manifest_["questions"]
        assert socials["required_connects"] == manifest_["required_connects"]
        assert socials["guide_steps"] == manifest_["guide_steps"]
