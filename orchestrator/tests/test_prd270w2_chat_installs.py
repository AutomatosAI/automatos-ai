"""PRD-270 W2 US-270-14g (§5.6, §5.8, K17, K28-1) — Auto's chat install of a Cloud item asks first.

Over Postgres, each test in ONE transaction rolled back at its end, with ``test_prd270w2_local_reads.py``'s Cloud
rows (a Cloud plugin as plugin_import.py commits one, then a Cloud agent, a persona agent, a Playbook and a package
through the real item pass). Auto's calls come from the owner's own chat turn, through the unified executor as
the chat's tool loop makes them, with the workspace at full autonomy: neither skips this gate. The cards are
decided through the approval routes (``grant_approval``, ``deny_approval``) or the click's resume. Proven:

* platform_install_marketplace_agent, platform_install_package and platform_install_plugin on a Cloud item each
  return requires_confirmation with a pending tool_call grant whose question_md is the item's disclosure ('Reviewed
  on Automatos Cloud', never Official), and install nothing; a denied card installs nothing and the same call asks
  again; the approved card re-runs the stored call, which installs it once; the same call again asks again
  (single-use);
* a call naming the Cloud agent by name asks with the card bound to its id, the prompt, the model, the apps and
  the 'Needs …' lines on it, and its click installs that agent;
* a seeded agent, a seeded package and a plugin without a mapping install as today, with no grant;
* the card's text is inert: a prompt holding a link, an image, HTML and a fence of its own sits in one fenced
  block, a name holding them is escaped, and an agent with no prompt says so;
* a yes whose card no longer reads as the item does (a Sync changed it since) is spent, and a new card asks;
* an error inside the check returns the ask, with no card, and installs nothing;
* install_item and the package route on a Cloud item never meet this gate.
"""
from __future__ import annotations

import asyncio
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa
from fastapi import HTTPException

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.approval_grants as approval_api  # noqa: E402
import api.marketplace as marketplace_api  # noqa: E402
import core.security.rate_limiter as rate_limiter  # noqa: E402
import modules.tools.execution.tool_grants as tool_grants  # noqa: E402
import services.click_memory as click_memory  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.models.approval_grants import ApprovalGrant, GrantStatus  # noqa: E402
from core.models.core import Agent, Chat, WorkflowTemplate  # noqa: E402
from core.models.marketplace_packages import MarketplacePackage  # noqa: E402
from core.models.marketplace_plugins import MarketplacePlugin, WorkspaceEnabledPlugin  # noqa: E402
from core.services.approval_grants import grant_grant  # noqa: E402
from core.models.marketplace_cloud import MarketplaceCloudItem  # noqa: E402
from core.services.auto_autonomy import FULL, STANDARD, set_autonomy_level  # noqa: E402
from modules.tools.discovery.cloud_install_asks import (  # noqa: E402
    ACT, AGENT_INSTALL, CHANGED, COULD_NOT_CHECK, PACKAGE_INSTALL, PLUGIN_INSTALL,
)
from modules.tools.execution.stale_cards import SUPERSEDED_BY  # noqa: E402
from services.marketplace_bundles import chat_disclosure  # noqa: E402
from services.marketplace_bundles.chat_disclosure import (  # noqa: E402
    NO_PROMPT, NOT_SHOWN, PROMPT_SHOWN, REVIEWED, THE_REST, disclosure_markdown,
)
from services.marketplace_bundles.import_items import import_items  # noqa: E402
from services.marketplace_bundles.validate import validate_bundle  # noqa: E402
from services.marketplace_cloud import fetch  # noqa: E402
from tests.test_prd270w2_local_reads import (  # noqa: E402
    PUBLISHER, _cloud_rows, _fetched, _made, _rolled_back, _workspace,
)

RIGHT_TO_LEFT = chr(0x202E)  # a bidi override, shown on the card by name
EVIL = "[x](https://evil.example) ![i](https://evil.example/p.png) <img src=x>"
EVIL_PROMPT = f"Before.\n```\n{EVIL}\n```\n{RIGHT_TO_LEFT}After."
LATER_PROMPT = "A prompt a later Sync wrote."
ZERO_WIDTH_SPACE = chr(0x200B)


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            for probe in (
                "SELECT agent_id, recipe_id, package_id, plugin_id, unresolved, synced_at FROM marketplace_cloud_items LIMIT 1",
                "SELECT question_md, details, revoked_by FROM approval_grants LIMIT 1",
                "SELECT workspace_id, user_id, role, is_active FROM workspace_members LIMIT 1",
                "SELECT enabled_at FROM workspace_enabled_plugins LIMIT 1",
                "SELECT cloned_from_id FROM agents LIMIT 1",
            ):
                conn.execute(sa.text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Cloud chat-install tests need Postgres with the PRD-270 W2 schema: {exc}")
    yield eng
    eng.dispose()


def _never(*_args, **_kwargs):
    raise AssertionError("a chat install never asks Automatos Cloud")


def _owner(db, ws) -> SimpleNamespace:
    tag = uuid.uuid4().hex[:8]
    clerk = f"user_owner_{tag}"
    user_id = db.execute(sa.text("INSERT INTO users (email, username, clerk_user_id) VALUES (:e, :u, :c) RETURNING id"),
                         {"e": f"owner-{tag}@harbourline.test", "u": f"owner-{tag}", "c": clerk}).scalar()
    db.execute(sa.text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                       "VALUES (CAST(:ws AS uuid), :user, 'owner', TRUE)"), {"ws": str(ws), "user": user_id})
    return SimpleNamespace(id=user_id, clerk=clerk)


@pytest.fixture
def cafe(engine, monkeypatch):
    """The owner's workspace at full autonomy, Auto, the owner's chat and the Cloud rows, in one rolled-back
    transaction; no notification, no network, no rate limit, no remembered click."""
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(fetch, "fetch_bytes", _never)
    monkeypatch.setattr(fetch, "_client", _never)
    monkeypatch.setattr(tool_grants, "_notify_approval_pending", lambda grant, workspace_id: None)
    monkeypatch.setattr(rate_limiter, "check_rate_limit", AsyncMock(return_value=None))
    monkeypatch.setattr(click_memory, "remember_the_click", AsyncMock(return_value=None))
    monkeypatch.setattr(approval_api, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(approval_api, "said_in_the_chat", lambda db, grant: None)
    with _rolled_back(engine) as db:
        ws = _workspace(db)
        owner = _owner(db, ws)
        auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                     workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws),
                     is_system_agent=True)
        chat = Chat(id=uuid.uuid4(), user_id=owner.id, workspace_id=ws, title="Installs", visibility="private")
        db.add_all([auto, chat])
        db.flush()
        set_autonomy_level(db, ws, FULL)
        db.flush()
        yield SimpleNamespace(db=db, ws=ws, owner=owner, auto=auto, chat=chat,
                              cloud=_cloud_rows(db, uuid.uuid4().hex[:10]))


# --------------------------------------------------------------------------- the owner's turn, the cards


def _owners_chat(cafe) -> dict:
    return {"user_id": cafe.owner.clerk, "driving_user_id": str(cafe.owner.id),
            "conversation_id": str(cafe.chat.id), "turn_id": "t-1"}


def _call(cafe, action: str, params: dict, *, owners_turn: bool = True) -> dict:
    """Auto's call in the owner's chat turn, as the chat's tool loop makes it (or on Auto's own run, with no
    person driving it)."""
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    return asyncio.run(UnifiedToolExecutor(db_session=cafe.db).execute_tool(
        tool_name=action, parameters=dict(params), agent_id=cafe.auto.id, workspace_id=cafe.ws,
        caller_context=_owners_chat(cafe) if owners_turn else None))


def _clicker(cafe) -> SimpleNamespace:
    return SimpleNamespace(workspace_id=cafe.ws,
                           user=SimpleNamespace(id=cafe.owner.clerk, clerk_user_id=cafe.owner.clerk, email=None))


def _approve(cafe, grant_id: int) -> None:
    asyncio.run(approval_api.grant_approval(grant_id, ctx=_clicker(cafe), db=cafe.db))


def _deny(cafe, grant_id: int) -> None:
    asyncio.run(approval_api.deny_approval(grant_id, ctx=_clicker(cafe), db=cafe.db))


def _grant(cafe, grant_id: int) -> ApprovalGrant:
    grant = cafe.db.get(ApprovalGrant, grant_id)
    cafe.db.refresh(grant)
    return grant


def _grants(cafe) -> list:
    return cafe.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == cafe.ws).all()


def _copies(cafe, agent_id: int) -> int:
    return cafe.db.query(Agent).filter(Agent.workspace_id == cafe.ws, Agent.owner_type == "workspace",
                                       Agent.cloned_from_id == agent_id).count()


def _enabled(cafe, plugin_id) -> int:
    return cafe.db.query(WorkspaceEnabledPlugin).filter(WorkspaceEnabledPlugin.workspace_id == cafe.ws,
                                                        WorkspaceEnabledPlugin.plugin_id == plugin_id).count()


def _asked(cafe, answer: dict, action: str) -> ApprovalGrant:
    """The answer is the ask with its pending tool_call card, the card the item's disclosure."""
    assert answer["success"] is False and answer.get("requires_confirmation") is True, answer
    grant = _grant(cafe, answer["grant_id"])
    assert (grant.status, grant.subject_type, grant.tool_name) == (GrantStatus.PENDING.value, "tool_call", action)
    assert grant.question_md == answer["question_md"]
    assert REVIEWED in grant.question_md and "Official" not in grant.question_md
    return grant


# --------------------------------------------------------------------------- each Cloud kind


def _kinds(cafe) -> dict:
    """Each kind: its call, what its card shows, and how many of it are installed here."""
    cloud = cafe.cloud
    return {
        "agent": (AGENT_INSTALL, {"agent_id": cloud.helper.id}, "You summarise the sources you are given.",
                  lambda: _copies(cafe, cloud.helper.id)),
        "package": (PACKAGE_INSTALL, {"slug": f"t270lr-pack-{cloud.tag}"}, "You summarise the sources you are given.",
                    lambda: _copies(cafe, cloud.helper.id)),
        "plugin": (PLUGIN_INSTALL, {"plugin_id": str(cloud.plugin.id)}, "No security issues found.",
                   lambda: _enabled(cafe, cloud.plugin.id)),
    }


@pytest.mark.parametrize("kind", ["agent", "package", "plugin"])
def test_a_cloud_items_install_waits_for_its_card_and_runs_once_on_approve(cafe, kind):
    action, params, shown, installed = _kinds(cafe)[kind]

    first = _asked(cafe, _call(cafe, action, params), action)
    assert shown in first.question_md
    assert installed() == 0

    _deny(cafe, first.id)
    assert _grant(cafe, first.id).status == GrantStatus.DENIED.value
    second = _asked(cafe, _call(cafe, action, params), action)
    assert second.id != first.id and installed() == 0

    _approve(cafe, second.id)
    approved = _grant(cafe, second.id)
    assert approved.details["executed_result"]["success"] is True, approved.details
    assert (approved.status, approved.revoked_by) == (GrantStatus.REVOKED.value, tool_grants.GRANT_CONSUMED_BY)
    assert installed() == 1

    third = _asked(cafe, _call(cafe, action, params), action)
    assert third.id not in (first.id, second.id)
    assert installed() == 1


def test_a_call_naming_the_cloud_agent_asks_with_its_card_bound_to_its_id(cafe):
    cloud = cafe.cloud
    key_app, oauth_app, new_app = cloud.held.apps
    grant = _asked(cafe, _call(cafe, AGENT_INSTALL, {"agent_name": cloud.helper.name}), AGENT_INSTALL)
    assert grant.details["params"] == {"agent_id": cloud.helper.id}

    question = grant.question_md.splitlines()
    for line in (f"- {REVIEWED}", f"- Publisher: {PUBLISHER}", "- Version: 1.0.0",
                 f"- {key_app}", f"- {oauth_app}: needs a connection", f"- {new_app}: needs a connection",
                 f"Model: {cloud.held.model} (openrouter): on this install", "You summarise the sources you are given.",
                 f"- {cloud.skill}", f"- Needs the skill t270lr-missing-{cloud.tag}: not on this install",
                 f"- Needs the plugin t270lr-gone-{cloud.tag}: not on this install"):
        assert line in question, line
    assert _copies(cafe, cloud.helper.id) == 0

    # The same agent by its id as text, or with a key the action doesn't take: the same card, never a run.
    for said in ({"agent_id": str(cloud.helper.id)}, {"agent_id": cloud.helper.id, "agent_name": "anything"}):
        again = _call(cafe, AGENT_INSTALL, said)
        assert again.get("requires_confirmation") is True and again["grant_id"] == grant.id, (said, again)

    _approve(cafe, grant.id)
    assert _copies(cafe, cloud.helper.id) == 1


def test_auto_s_own_run_with_the_dial_down_asks_too(cafe):
    set_autonomy_level(cafe.db, cafe.ws, STANDARD)
    cafe.db.flush()
    _asked(cafe, _call(cafe, AGENT_INSTALL, {"agent_id": cafe.cloud.helper.id}, owners_turn=False), AGENT_INSTALL)
    assert _copies(cafe, cafe.cloud.helper.id) == 0


@pytest.mark.parametrize("claims", ["a_seeded_agents_slug", "a_seeded_packages_member", "a_seeded_playbooks_helper"])
def test_a_cloud_agent_named_as_a_local_item_finds_its_agent_is_refused_by_the_sync(cafe, claims):
    """The security reviews of US-270-14g: an install by reference reads a slug or a name (``.first()`` of either),
    and a seeded package or Playbook installs the agent it names, so a Cloud agent named after a seeded agent's
    slug, or after an agent a seeded item names that isn't here, would be installed without its card."""
    seeded, tag = cafe.cloud.held.seeded, cafe.cloud.tag
    absent = f"t270ci-absent-{tag}"   # no agent here has it
    if claims == "a_seeded_packages_member":
        cafe.db.add(MarketplacePackage(slug=f"t270ci-refs-{tag}", name="A seeded pack",
                                       members=[{"type": "agent", "ref": absent, "name": "The absent one"}]))
    if claims == "a_seeded_playbooks_helper":
        cafe.db.add(WorkflowTemplate(template_id=f"t270ci-refs-{tag}", name="A seeded Playbook", description="Seeded.",
                                     owner_type="marketplace", owner_id="marketplace", is_approved=True,
                                     is_public=True, template_definition={"suggested_agents": [absent]}, steps=[],
                                     created_by="seed"))
    cafe.db.flush()
    claimed = seeded.slug if claims == "a_seeded_agents_slug" else absent
    squatter = {**_made("agent", tag, "Test Squatter", {}), "name": claimed.upper()}
    free = {**_made("agent", tag, "Test Free Agent", {}), "name": f"Test Free Agent {tag}"}
    for bundle in (squatter, free):
        validate_bundle(bundle)

    written = import_items(cafe.db, [_fetched(squatter), _fetched(free)])

    assert written.ok and written.view()["agent"]["refused"] == 1 and written.view()["agent"]["new"] == 1, written
    mapped = {row.external_id for row in cafe.db.query(MarketplaceCloudItem.external_id)}
    assert squatter["externalId"] not in mapped and free["externalId"] in mapped


def test_a_seeded_agent_a_seeded_package_and_an_unmapped_plugin_install_as_today(cafe):
    seeded, tag = cafe.cloud.held.seeded, cafe.cloud.tag
    loose = MarketplacePlugin(slug=f"t270ci-local-{tag}", name="A local kit", version="1.0.0",
                              approval_status="approved", is_active=True)
    pack = MarketplacePackage(slug=f"t270ci-seeded-{tag}", name="A seeded pack",
                              members=[{"type": "agent", "ref": seeded.slug, "name": "The writer"}])
    cafe.db.add_all([loose, pack])
    cafe.db.flush()

    for action, params in ((AGENT_INSTALL, {"agent_id": seeded.id}), (PACKAGE_INSTALL, {"slug": pack.slug}),
                           (PLUGIN_INSTALL, {"plugin_id": str(loose.id)})):
        answer = _call(cafe, action, params)
        assert answer.get("success") is True and not answer.get("requires_confirmation"), (action, answer)
    assert _copies(cafe, seeded.id) == 1 and _enabled(cafe, loose.id) == 1
    assert _grants(cafe) == []


# --------------------------------------------------------------------------- the card's text


def _fence_lines(question: str) -> list:
    return [line for line in question.splitlines() if line.lstrip().startswith("``")]


def _outside_the_fences(question: str) -> str:
    return re.sub(r"^```text\n.*?\n```$", "", question, flags=re.DOTALL | re.MULTILINE)


def test_the_cards_text_is_inert_and_an_agent_without_a_prompt_says_so(cafe):
    helper = cafe.cloud.helper
    helper.custom_persona_prompt = EVIL_PROMPT
    helper.name = f"Helper {EVIL} www.evil.example owner@evil.example{RIGHT_TO_LEFT}"
    cafe.db.flush()

    question = _asked(cafe, _call(cafe, AGENT_INSTALL, {"agent_id": helper.id}), AGENT_INSTALL).question_md
    assert _fence_lines(question) == ["```text", "```"]   # the prompt's own fence is broken, so one block holds it
    block = re.search(r"^```text\n(.*?)\n```$", question, flags=re.DOTALL | re.MULTILINE).group(1)
    assert block.replace(ZERO_WIDTH_SPACE, "") == EVIL_PROMPT.replace(RIGHT_TO_LEFT, "(U+202E)") and EVIL in block
    assert RIGHT_TO_LEFT not in question   # named, never hidden
    outside = _outside_the_fences(question)
    assert re.search(r"(?<!\\)[\[<]", outside) is None, outside   # no link, image or HTML opens
    assert "https://" not in outside and "www." not in outside.lower() and "owner@" not in outside
    assert r"Helper \[x\](https\://evil.example)" in outside and "owner\\@evil.example(U+202E)" in outside

    helper.use_custom_persona, helper.custom_persona_prompt, helper.persona_id = False, None, None
    cafe.db.flush()
    assert f"Its prompt: {NO_PROMPT}." in disclosure_markdown(cafe.db, "agent", str(helper.id)).splitlines()

    planner = disclosure_markdown(cafe.db, "agent", str(cafe.cloud.planner.id))
    assert f"Its prompt (from its persona Test Planner {cafe.cloud.tag}):" in planner
    assert "You plan the week in plain sentences." in planner
    assert f"Model: test-vendor/t270lr-absent-{cafe.cloud.tag}: not on this install" in planner


def test_a_long_prompt_is_cut_and_says_how_much_is_left_out(cafe):
    helper = cafe.cloud.helper
    helper.custom_persona_prompt = "word " * PROMPT_SHOWN
    cafe.db.flush()
    question = disclosure_markdown(cafe.db, "agent", str(helper.id))
    block = re.search(r"^```text\n(.*?)\n```$", question, flags=re.DOTALL | re.MULTILINE).group(1)
    assert len(block) == PROMPT_SHOWN
    left_out = len(helper.custom_persona_prompt) - PROMPT_SHOWN
    assert f"```\n{NOT_SHOWN.format(count=left_out)} {THE_REST}" in question


# --------------------------------------------------------------------------- a changed item, an error, the web


def test_a_card_still_waiting_when_the_item_changes_is_replaced_by_one_that_shows_it(cafe):
    helper = cafe.cloud.helper
    first = _asked(cafe, _call(cafe, AGENT_INSTALL, {"agent_id": helper.id}), AGENT_INSTALL)
    helper.custom_persona_prompt = LATER_PROMPT   # what a Sync wrote while the card waited
    cafe.db.flush()

    second = _asked(cafe, _call(cafe, AGENT_INSTALL, {"agent_id": helper.id}), AGENT_INSTALL)

    assert second.id != first.id and LATER_PROMPT in second.question_md
    replaced = _grant(cafe, first.id)
    assert (replaced.status, replaced.revoked_by) == (GrantStatus.EXPIRED.value, SUPERSEDED_BY)
    with pytest.raises(HTTPException) as refused:   # its click runs nothing and says it was replaced
        _approve(cafe, first.id)
    assert refused.value.status_code == 422 and _copies(cafe, helper.id) == 0


def test_a_yes_whose_card_no_longer_reads_as_the_item_is_spent_and_a_new_card_asks(cafe):
    helper = cafe.cloud.helper
    grant = _asked(cafe, _call(cafe, AGENT_INSTALL, {"agent_id": helper.id}), AGENT_INSTALL)
    grant_grant(grant, granted_by=f"user:{cafe.owner.clerk}")
    helper.custom_persona_prompt = LATER_PROMPT   # what a Sync between the card and the call wrote
    cafe.db.flush()

    answer = _call(cafe, AGENT_INSTALL, {"agent_id": helper.id})

    assert answer["message"] == CHANGED.format(act=ACT.format(label=f"agent #{helper.id}")), answer
    assert answer.get("requires_confirmation") is True and answer["grant_id"] != grant.id
    spent = _grant(cafe, grant.id)
    assert (spent.status, spent.revoked_by) == (GrantStatus.REVOKED.value, tool_grants.GRANT_CONSUMED_BY)
    assert _copies(cafe, helper.id) == 0
    waiting = [g for g in _grants(cafe) if g.status == GrantStatus.PENDING.value]
    assert [g.id for g in waiting] == [answer["grant_id"]] and LATER_PROMPT in waiting[0].question_md


def test_an_error_inside_the_check_returns_the_ask_with_no_card(cafe, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("the disclosure could not be read")

    monkeypatch.setattr(chat_disclosure, "disclosure_markdown", broken)
    answer = _call(cafe, AGENT_INSTALL, {"agent_id": cafe.cloud.helper.id})
    assert answer["success"] is False and answer.get("requires_confirmation") is True, answer
    assert answer["message"] == COULD_NOT_CHECK and "grant_id" not in answer
    assert _grants(cafe) == [] and _copies(cafe, cafe.cloud.helper.id) == 0


def test_install_item_and_the_package_route_never_meet_this_gate(cafe, monkeypatch):
    met = []

    def recorded(*args, **_kwargs):
        met.append(args[1:])
        raise AssertionError("the web's installs never meet the chat's gate")

    monkeypatch.setattr(chat_disclosure, "cloud_target", recorded)
    ctx = RequestContext(workspace_id=cafe.ws, user=UserContext(id=cafe.owner.clerk, clerk_user_id=cafe.owner.clerk,
                                                                system_role="user"), auth_type="clerk")
    installed = asyncio.run(marketplace_api.install_item(cafe.cloud.helper.id, ctx=ctx, db=cafe.db))
    assert installed.success is True
    package = asyncio.run(marketplace_api.install_marketplace_package(f"t270lr-pack-{cafe.cloud.tag}", ctx=ctx,
                                                                     db=cafe.db))
    assert package.get("success") is True, package
    assert met == [] and _copies(cafe, cafe.cloud.helper.id) == 1
