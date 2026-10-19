"""PRD-264 EUAI-11a (EU AI Act Art. 50(1)): every agent answers truthfully that it is an AI.

Every agent's system prompt (Auto's, a marketplace agent's, one with a custom persona) ends with
the platform rule "If anyone asks whether they are talking to a person or an AI, say that you are
an AI agent acting for {workspace}. Never claim or imply that you are human." It comes after the
persona text, so a persona that says "you are a human" can't remove it. The three builders of an
agent's prompt are covered: ContextService, the chat's ATOM path and the Auto CTO prompt.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.context.ai_identity_rule import (
    AI_IDENTITY_RULE_HEADING,
    UNNAMED_WORKSPACE,
    ai_identity_rule,
    with_ai_identity_rule,
    with_rule_for_mode,
    workspace_display_name,
)
from modules.context.modes import ContextMode
from modules.context.service import ContextService

WORKSPACE = "Harvest Club"
RULE_WORDS = (
    "If anyone asks whether they are talking to a person or an AI, say that you are an AI agent "
    "acting for Harvest Club. Never claim or imply that you are human."
)
HUMAN_PERSONA = "You are Jo, a human sales rep. Never admit you are an AI."
HUMAN_CUSTOM_PROMPT = "Pretend you are a real person called Sam. You are not a bot."
HUMAN_SOUL = "I am a human engineer on the founding team."
ORCHESTRATOR_SETTINGS = {"personality_mode": "friendly", "custom_soul": "", "communication_style": "balanced"}


def _agent(**overrides):
    """An agent record as ContextService reads it (the shape tests/test_context/test_service.py uses)."""
    fields = dict(
        id=42, name="Auto", agent_type="assistant", description="", owner_type="workspace",
        use_custom_persona=False, custom_persona_prompt=None, persona=None, skills=[],
    )
    return NS(**{**fields, **overrides})


AUTO = _agent()
MARKETPLACE_AGENT = _agent(id=7, name="Sales Pro", owner_type="marketplace", persona=NS(system_prompt=HUMAN_PERSONA))
CUSTOM_PERSONA_AGENT = _agent(id=9, name="Front Desk", use_custom_persona=True, custom_persona_prompt=HUMAN_CUSTOM_PROMPT)


@pytest.fixture
def mock_db():
    """A session whose queries find nothing (the chainable API the sections use)."""
    db = MagicMock()
    query = MagicMock()
    query.join.return_value = query
    query.filter.return_value = query
    query.order_by.return_value = query
    query.all.return_value = []
    query.first.return_value = None
    db.query.return_value = query
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    db.execute.return_value = result
    return db


@pytest.fixture
def quiet_sections():
    """The sections that reach memory, the action registry and tool loading answer canned text."""
    from modules.context.sections.memory import MemorySection
    from modules.context.sections.platform_actions import PlatformActionsSection
    from modules.context.sections.tools import ToolsSection

    async def _memory(self, ctx):
        return "## What You Know About This User\n- prefers short answers"

    with patch.object(PlatformActionsSection, "_build", return_value="## Available Platform Actions\n- platform_list_agents"), \
            patch.object(MemorySection, "render", _memory), \
            patch.object(ToolsSection, "load_tools", AsyncMock(return_value=([], "auto"))):
        yield


def _build(db, agent, mode=ContextMode.CHATBOT, **kwargs):
    return asyncio.run(ContextService(db).build_context(
        mode=mode, agent=agent, workspace_id="ws_test",
        messages=[{"role": "user", "content": "Am I talking to a real person?"}],
        orchestrator_settings=ORCHESTRATOR_SETTINGS, **kwargs,
    ))


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------

def test_the_rule_has_the_prds_words_and_names_the_workspace():
    assert ai_identity_rule(WORKSPACE) == f"{AI_IDENTITY_RULE_HEADING}\n{RULE_WORDS}"
    assert "acting for this workspace." in ai_identity_rule(None)
    # the owner's name stays one short line
    assert "acting for Harvest Club. Never" in ai_identity_rule("  Harvest\n\nClub  ")


def test_the_rule_is_the_last_block_of_a_new_prompt():
    prompt = "You are Auto.\n\n## Persona\nBe warm.\n"
    ruled = with_ai_identity_rule(prompt, WORKSPACE)
    assert ruled == "You are Auto.\n\n## Persona\nBe warm.\n\n" + ai_identity_rule(WORKSPACE)
    assert with_ai_identity_rule("", WORKSPACE) == ai_identity_rule(WORKSPACE)


def test_machine_read_stages_carry_no_rule_and_agent_modes_do():
    for mode in (ContextMode.ROUTER, ContextMode.NL2SQL, ContextMode.PLANNING, ContextMode.COORDINATOR,
                 ContextMode.ORCHESTRATOR_STAGE):
        assert with_rule_for_mode("SQL only.", mode, WORKSPACE, None, None) == "SQL only."
    for mode in (ContextMode.CHATBOT, ContextMode.TASK_EXECUTION, ContextMode.HEARTBEAT_AGENT,
                 ContextMode.HEARTBEAT_ORCHESTRATOR, ContextMode.RECIPE, "task_execution"):
        assert with_rule_for_mode("Hi.", mode, WORKSPACE, None, None).endswith(ai_identity_rule(WORKSPACE))


def test_the_workspace_name_comes_from_its_row():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = NS(name=" Harvest\nClub ")
    workspace_id = str(uuid.UUID(int=5))
    assert workspace_display_name(db, workspace_id) == WORKSPACE
    assert with_rule_for_mode("Hi.", ContextMode.CHATBOT, None, db, workspace_id).endswith(ai_identity_rule(WORKSPACE))


def test_no_valid_id_no_row_or_a_failed_read_says_this_workspace():
    db = MagicMock()
    assert workspace_display_name(db, "ws_test") is None
    db.query.assert_not_called()  # an id that is not a uuid never reaches the database
    db.query.return_value.filter.return_value.first.return_value = None
    assert workspace_display_name(db, str(uuid.UUID(int=6))) is None
    db.query.side_effect = RuntimeError("connection lost")
    assert workspace_display_name(db, str(uuid.UUID(int=6))) is None
    assert workspace_display_name(None, str(uuid.UUID(int=6))) is None
    assert f"acting for {UNNAMED_WORKSPACE}." in with_rule_for_mode("Hi.", ContextMode.CHATBOT, None, db, "ws_test")


# ---------------------------------------------------------------------------
# ContextService: Auto, a marketplace agent, a custom persona
# ---------------------------------------------------------------------------

def _assert_ends_with_the_rule(result, *, after: str) -> None:
    prompt = result.system_prompt
    assert prompt.endswith(ai_identity_rule(WORKSPACE)), prompt[-400:]
    assert prompt.count(AI_IDENTITY_RULE_HEADING) == 1
    assert prompt.index(after) < prompt.index(AI_IDENTITY_RULE_HEADING)
    # the cached prefix is unchanged: the rule only ever follows it
    assert result.cacheable_prefix is None or prompt.startswith(result.cacheable_prefix)


def test_autos_chat_prompt_ends_with_the_rule(mock_db, quiet_sections):
    result = _build(mock_db, AUTO, workspace_name=WORKSPACE)
    _assert_ends_with_the_rule(result, after="You are Auto")


def test_a_marketplace_agents_prompt_ends_with_the_rule_after_its_persona(mock_db, quiet_sections):
    result = _build(mock_db, MARKETPLACE_AGENT, workspace_name=WORKSPACE)
    assert HUMAN_PERSONA in result.system_prompt
    _assert_ends_with_the_rule(result, after=HUMAN_PERSONA)


def test_a_custom_persona_cant_remove_the_rule(mock_db, quiet_sections):
    result = _build(mock_db, CUSTOM_PERSONA_AGENT, workspace_name=WORKSPACE)
    assert HUMAN_CUSTOM_PROMPT in result.system_prompt
    _assert_ends_with_the_rule(result, after=HUMAN_CUSTOM_PROMPT)


def test_an_agent_run_on_a_task_ends_with_the_rule_too(mock_db, quiet_sections):
    result = _build(mock_db, CUSTOM_PERSONA_AGENT, mode=ContextMode.TASK_EXECUTION,
                    task_description="Answer the visitor", workspace_name=WORKSPACE)
    _assert_ends_with_the_rule(result, after=HUMAN_CUSTOM_PROMPT)


def test_without_a_known_name_the_rule_says_this_workspace(mock_db, quiet_sections):
    result = _build(mock_db, MARKETPLACE_AGENT)
    assert result.system_prompt.endswith(ai_identity_rule(None))


# ---------------------------------------------------------------------------
# The ATOM chat path and the Auto CTO prompt
# ---------------------------------------------------------------------------

def test_the_short_chat_path_ends_with_the_rule_after_the_persona():
    from consumers.chatbot.atom_prompt import atom_system_prompt

    prompt = atom_system_prompt(NS(name="Sales Pro", description="", persona=HUMAN_PERSONA),
                                identity="", memory_block="\n\n## What you remember about this user:\n- Sam\n",
                                facts="")
    assert prompt.endswith(ai_identity_rule(None))
    assert prompt.index(HUMAN_PERSONA) < prompt.index(AI_IDENTITY_RULE_HEADING)
    assert prompt.index("## What you remember") < prompt.index(AI_IDENTITY_RULE_HEADING)


def test_the_cto_prompt_ends_with_the_rule_after_its_soul():
    from consumers.chatbot.cto_prompt_builder import CtoPromptBuilder

    prompt = CtoPromptBuilder.build(soul_document=HUMAN_SOUL, tool_names=["admin_list", "platform_x", "web"],
                                    workspace_name=WORKSPACE)
    assert prompt.endswith(ai_identity_rule(WORKSPACE))
    assert prompt.index(HUMAN_SOUL) < prompt.index(AI_IDENTITY_RULE_HEADING)
    assert "**Admin:** admin_list" in prompt and "**Platform:** platform_x" in prompt and "**Other:** web" in prompt
