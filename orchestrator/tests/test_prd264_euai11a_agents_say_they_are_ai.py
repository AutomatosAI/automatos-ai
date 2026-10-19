"""PRD-264 EUAI-11a (EU AI Act Art. 50(1)): every agent answers truthfully that it is an AI.

Every agent's system prompt (Auto's, a marketplace agent's, one with a custom persona) ends with
the platform rule "If anyone asks whether they are talking to a person or an AI, say that you are
an AI agent acting for {principal}. Never claim or imply that you are human." It comes after the
persona text, so a persona that says "you are a human" can't remove it. The three builders of an
agent's prompt are covered: ContextService, the chat's ATOM path and the Auto CTO prompt.

FR-15: {principal} is the business profile's company name, else the workspace name only when it is
not the auto-created "{name or email}'s Workspace" form and has no "@"; otherwise the clause goes
and the rule reads "say that you are an AI agent." The owner's name or email never reaches a prompt.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.context.ai_identity_rule import (
    AI_IDENTITY_RULE_HEADING,
    MAX_PRINCIPAL_NAME_CHARS,
    ai_identity_rule,
    principal_for_workspace,
    public_principal_name,
    public_workspace_name,
    with_ai_identity_rule,
    with_rule_for_mode,
)
from modules.context.modes import ContextMode
from modules.context.service import ContextService

WORKSPACE = "Harvest Club"
COMPANY = "Harvest Club Coffee Ltd"
RULE_WORDS = (
    "If anyone asks whether they are talking to a person or an AI, say that you are an AI agent "
    "acting for Harvest Club. Never claim or imply that you are human."
)
UNNAMED_RULE_WORDS = (
    "If anyone asks whether they are talking to a person or an AI, say that you are an AI agent. "
    "Never claim or imply that you are human."
)
# Names core/auth/hybrid.py gives a new personal workspace ("{name or email or 'My'}'s Workspace").
OWNER_NAMED_WORKSPACE = "John Smith's Workspace"
OWNER_EMAIL_WORKSPACE = "john@acme.example's Workspace"
PRIVATE_WORKSPACE_NAMES = (
    OWNER_NAMED_WORKSPACE,
    OWNER_EMAIL_WORKSPACE,
    "My's Workspace",
    "Jo’s workspace",           # a typographic apostrophe, any case
    "sales@acme.example",            # an email, whatever the form
    "Front desk @ Harvest Club",
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

def test_the_rule_has_the_prds_words_and_names_the_principal():
    assert ai_identity_rule(WORKSPACE) == f"{AI_IDENTITY_RULE_HEADING}\n{RULE_WORDS}"
    # the principal stays one short line
    assert "acting for Harvest Club. Never" in ai_identity_rule("  Harvest\n\nClub  ")


def test_with_no_public_name_the_clause_is_dropped():
    for nothing in (None, "", "   \n "):
        assert ai_identity_rule(nothing) == f"{AI_IDENTITY_RULE_HEADING}\n{UNNAMED_RULE_WORDS}"
        assert "acting for" not in ai_identity_rule(nothing)


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


# ---------------------------------------------------------------------------
# FR-15: the principal is a public name, never the owner's own name or email
# ---------------------------------------------------------------------------

def test_a_workspace_name_that_carries_the_owner_is_never_public():
    for name in PRIVATE_WORKSPACE_NAMES:
        assert public_workspace_name(name) is None, name
    for nothing in (None, "", "  ", 42):
        assert public_workspace_name(nothing) is None
    assert public_workspace_name(" Harvest\nClub ") == WORKSPACE
    # a name the owner chose that only mentions a workspace is still public
    assert public_workspace_name("Workspace Coffee") == "Workspace Coffee"


def test_a_long_name_is_capped_but_the_owner_form_is_caught_before_the_cap():
    assert public_workspace_name("A" * 500) == "A" * MAX_PRINCIPAL_NAME_CHARS
    assert public_workspace_name("J" * 500 + "'s Workspace") is None


def test_the_business_profiles_company_name_comes_first():
    assert public_principal_name(COMPANY, WORKSPACE) == COMPANY
    assert public_principal_name(COMPANY, OWNER_NAMED_WORKSPACE) == COMPANY
    assert public_principal_name(None, WORKSPACE) == WORKSPACE
    assert public_principal_name("  ", WORKSPACE) == WORKSPACE
    assert public_principal_name(None, OWNER_EMAIL_WORKSPACE) is None


def _db(company=None, workspace=None):
    """A session whose business_profiles query finds ``company`` and whose workspaces query finds ``workspace``."""
    rows = {
        "BusinessProfile": NS(company_name=company) if company is not None else None,
        "Workspace": NS(name=workspace) if workspace is not None else None,
    }

    def _query(model):
        query = MagicMock()
        query.filter.return_value = query
        query.order_by.return_value = query
        query.first.return_value = rows[model.__name__]
        return query

    db = MagicMock()
    db.query.side_effect = _query
    return db


WORKSPACE_ID = str(uuid.UUID(int=5))


def test_the_principal_comes_from_the_business_profile_then_the_workspace_row():
    assert principal_for_workspace(_db(company=COMPANY, workspace=OWNER_NAMED_WORKSPACE), WORKSPACE_ID) == COMPANY
    assert principal_for_workspace(_db(workspace=" Harvest\nClub "), WORKSPACE_ID) == WORKSPACE
    ruled = with_rule_for_mode("Hi.", ContextMode.CHATBOT, None, _db(workspace=WORKSPACE), WORKSPACE_ID)
    assert ruled.endswith(ai_identity_rule(WORKSPACE))


def test_an_auto_created_workspace_name_never_reaches_the_prompt():
    for name in PRIVATE_WORKSPACE_NAMES:
        assert principal_for_workspace(_db(workspace=name), WORKSPACE_ID) is None, name
        ruled = with_rule_for_mode("Hi.", ContextMode.CHATBOT, None, _db(workspace=name), WORKSPACE_ID)
        assert ruled == "Hi.\n\n" + ai_identity_rule(None)
        assert name not in ruled
    # a name the caller passes is filtered the same way
    ruled = with_rule_for_mode("Hi.", ContextMode.CHATBOT, OWNER_EMAIL_WORKSPACE, _db(), WORKSPACE_ID)
    assert "john@acme.example" not in ruled and ruled.endswith(ai_identity_rule(None))


def test_no_valid_id_no_row_or_a_failed_read_names_no_one():
    db = MagicMock()
    assert principal_for_workspace(db, "ws_test") is None
    db.query.assert_not_called()  # an id that is not a uuid never reaches the database
    assert principal_for_workspace(_db(), WORKSPACE_ID) is None
    db.query.side_effect = RuntimeError("connection lost")
    assert principal_for_workspace(db, WORKSPACE_ID) is None
    assert principal_for_workspace(None, WORKSPACE_ID) is None
    assert with_rule_for_mode("Hi.", ContextMode.CHATBOT, None, db, WORKSPACE_ID) == "Hi.\n\n" + ai_identity_rule(None)


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


def test_without_a_known_name_the_rule_names_no_one(mock_db, quiet_sections):
    result = _build(mock_db, MARKETPLACE_AGENT)
    assert result.system_prompt.endswith(f"{AI_IDENTITY_RULE_HEADING}\n{UNNAMED_RULE_WORDS}")


def test_an_owner_named_workspace_is_not_named_in_the_assembled_prompt(mock_db, quiet_sections):
    result = _build(mock_db, CUSTOM_PERSONA_AGENT, workspace_name=OWNER_NAMED_WORKSPACE)
    assert result.system_prompt.endswith(ai_identity_rule(None))
    assert "John Smith" not in result.system_prompt


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


def test_the_cto_prompt_never_names_an_owner_named_workspace():
    from consumers.chatbot.cto_prompt_builder import CtoPromptBuilder

    prompt = CtoPromptBuilder.build(soul_document=HUMAN_SOUL, workspace_name=OWNER_EMAIL_WORKSPACE)
    assert prompt.endswith(ai_identity_rule(None))
    assert "john@acme.example" not in prompt
