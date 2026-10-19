"""PRD-264 EUAI-11a (EU AI Act Art. 50(1)): every agent answers truthfully that it is an AI.

Every agent's system prompt (Auto's, a marketplace agent's, one with a custom persona) ends with
the platform rule "If anyone asks whether they are talking to a person or an AI, say that you are
an AI agent acting for {principal}. Never claim or imply that you are human." It comes after the
persona text, so a persona that says "you are a human" can't remove it. The three builders of an
agent's prompt are covered: ContextService, the chat's ATOM path and the Auto CTO prompt.

FR-15: {principal} is decided once, in core/public_principal.py. On a widget turn it is the Site's
public display_name and never the workspace's name; otherwise the business profile's company
name, else the workspace name only when it is not the auto-created "{name or email}'s Workspace"
form; a name with an "@" is never used. Otherwise the clause goes and the rule reads "say that you
are an AI agent." The owner's name or email never reaches a prompt.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.public_principal import (
    COMPANY_NAME_SQL,
    SITE_NAME_SQL,
    WORKSPACE_NAME_SQL,
    public_principal_name,
    public_site_name,
    site_principal_name,
    workspace_principal_name,
)
from modules.context.ai_identity_rule import (
    AI_IDENTITY_RULE_HEADING,
    MAX_PRINCIPAL_NAME_CHARS,
    ai_identity_rule,
    principal_for_turn,
    with_ai_identity_rule,
    with_rule_for_mode,
)
from modules.context.modes import ContextMode
from modules.context.service import ContextService

WORKSPACE = "Harvest Club"
COMPANY = "Harvest Club Coffee Ltd"
SITE = "Harvest Club Online"
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
    "jo's WORKSPACE",                # any case
    "sales@acme.example",            # an email, whatever the form
    "Front desk @ Harvest Club",
)
HUMAN_PERSONA = "You are Jo, a human sales rep. Never admit you are an AI."
HUMAN_CUSTOM_PROMPT = "Pretend you are a real person called Sam. You are not a bot."
HUMAN_SOUL = "I am a human engineer on the founding team."
ORCHESTRATOR_SETTINGS = {"personality_mode": "friendly", "custom_soul": "", "communication_style": "balanced"}
WORKSPACE_ID = str(uuid.UUID(int=5))


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


def _sql(statement) -> str:
    """A statement's SQL on one line, so two surfaces' reads of the same rows compare equal."""
    return " ".join(str(getattr(statement, "text", "")).split())


def _result(row):
    """A result whose ``fetchone`` is ``row``, and whose other reads find nothing."""
    result = MagicMock()
    result.fetchone.return_value = row
    result.first.return_value = row
    result.scalar.return_value = None
    result.scalars.return_value.all.return_value = []
    return result


def _db(company=None, workspace=None, site=None):
    """A session whose FR-15 reads find these names, and whose other queries find nothing.

    The business profile, workspace and Site reads are matched by their SQL, so any surface that
    reads those rows with the same statements gets the same answers.
    """
    answers = {_sql(COMPANY_NAME_SQL): company, _sql(WORKSPACE_NAME_SQL): workspace, _sql(SITE_NAME_SQL): site}

    def _execute(statement, *args, **kwargs):
        value = answers.get(_sql(statement))
        return _result((value,) if value is not None else None)

    db = MagicMock()
    query = MagicMock()
    for chained in ("join", "filter", "order_by", "limit"):
        getattr(query, chained).return_value = query
    query.all.return_value = []
    query.first.return_value = None
    db.query.return_value = query
    db.execute.side_effect = _execute
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
        mode=mode, agent=agent, workspace_id=WORKSPACE_ID,
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
    assert f"acting for {'A' * MAX_PRINCIPAL_NAME_CHARS}. Never" in ai_identity_rule("A" * 500)


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
    db = _db(workspace=WORKSPACE)
    for mode in (ContextMode.ROUTER, ContextMode.NL2SQL, ContextMode.PLANNING, ContextMode.COORDINATOR,
                 ContextMode.ORCHESTRATOR_STAGE):
        assert with_rule_for_mode("SQL only.", mode, db, WORKSPACE_ID, False) == "SQL only."
    for mode in (ContextMode.CHATBOT, ContextMode.TASK_EXECUTION, ContextMode.HEARTBEAT_AGENT,
                 ContextMode.HEARTBEAT_ORCHESTRATOR, ContextMode.RECIPE, "task_execution"):
        assert with_rule_for_mode("Hi.", mode, db, WORKSPACE_ID, False).endswith(ai_identity_rule(WORKSPACE))


# ---------------------------------------------------------------------------
# FR-15: one decision, in core/public_principal.py
# ---------------------------------------------------------------------------

def test_a_workspace_name_that_carries_the_owner_is_never_public():
    for name in PRIVATE_WORKSPACE_NAMES:
        assert public_principal_name(None, name) is None, name
    for nothing in (None, "", "  ", 42):
        assert public_principal_name(None, nothing) is None
    assert public_principal_name(None, " Harvest Club ") == WORKSPACE
    # a name the owner chose that only mentions a workspace is still public
    assert public_principal_name(None, "Workspace Coffee") == "Workspace Coffee"


def test_the_business_profiles_company_name_comes_first_and_never_carries_an_email():
    assert public_principal_name(COMPANY, WORKSPACE) == COMPANY
    assert public_principal_name(COMPANY, OWNER_NAMED_WORKSPACE) == COMPANY
    assert public_principal_name(None, WORKSPACE) == WORKSPACE
    assert public_principal_name("  ", WORKSPACE) == WORKSPACE
    assert public_principal_name(None, OWNER_EMAIL_WORKSPACE) is None
    # a company name holding an email is refused like a workspace name holding one
    assert public_principal_name("owner@acme.example", WORKSPACE) == WORKSPACE
    assert public_principal_name("owner@acme.example", OWNER_NAMED_WORKSPACE) is None


def test_a_sites_name_is_public_unless_it_copied_the_owner_form():
    assert public_site_name(f" {SITE} ") == SITE
    for name in PRIVATE_WORKSPACE_NAMES:
        assert public_site_name(name) is None, name


def test_the_reads_pick_the_newest_named_profile_and_the_default_site():
    company_sql = _sql(COMPANY_NAME_SQL)
    assert "company_name <> ''" in company_sql and "ORDER BY updated_at DESC LIMIT 1" in company_sql
    assert "ORDER BY created_at ASC LIMIT 1" in _sql(SITE_NAME_SQL)
    for statement in (COMPANY_NAME_SQL, WORKSPACE_NAME_SQL, SITE_NAME_SQL):
        assert "CAST(:ws AS uuid)" in _sql(statement)


def test_the_principal_comes_from_the_business_profile_then_the_workspace_row():
    assert workspace_principal_name(_db(company=COMPANY, workspace=OWNER_NAMED_WORKSPACE), WORKSPACE_ID) == COMPANY
    assert workspace_principal_name(_db(workspace=f" {WORKSPACE} "), WORKSPACE_ID) == WORKSPACE
    assert workspace_principal_name(_db(company="owner@acme.example", workspace=WORKSPACE), WORKSPACE_ID) == WORKSPACE
    ruled = with_rule_for_mode("Hi.", ContextMode.CHATBOT, _db(workspace=WORKSPACE), WORKSPACE_ID, False)
    assert ruled.endswith(ai_identity_rule(WORKSPACE))


def test_an_auto_created_workspace_name_never_reaches_the_prompt():
    for name in PRIVATE_WORKSPACE_NAMES:
        assert workspace_principal_name(_db(workspace=name), WORKSPACE_ID) is None, name
        ruled = with_rule_for_mode("Hi.", ContextMode.CHATBOT, _db(workspace=name), WORKSPACE_ID, False)
        assert ruled == "Hi.\n\n" + ai_identity_rule(None)
        assert name not in ruled


def test_no_valid_id_no_row_or_a_failed_read_names_no_one():
    db = MagicMock()
    assert workspace_principal_name(db, "ws_test") is None
    assert site_principal_name(db, "ws_test") is None
    db.execute.assert_not_called()  # an id that is not a uuid never reaches the database
    assert workspace_principal_name(_db(), WORKSPACE_ID) is None
    assert site_principal_name(_db(), WORKSPACE_ID) is None
    db.execute.side_effect = RuntimeError("connection lost")
    assert workspace_principal_name(db, WORKSPACE_ID) is None
    assert site_principal_name(db, WORKSPACE_ID) is None
    assert workspace_principal_name(None, WORKSPACE_ID) is None
    assert with_rule_for_mode("Hi.", ContextMode.CHATBOT, db, WORKSPACE_ID, False) == "Hi.\n\n" + ai_identity_rule(None)


# ---------------------------------------------------------------------------
# FR-15 on a widget turn: the Site's public name, never the workspace's
# ---------------------------------------------------------------------------

def test_a_widget_turn_names_the_site_never_the_workspace_or_company():
    named = _db(company=COMPANY, workspace=WORKSPACE, site=SITE)
    assert principal_for_turn(named, WORKSPACE_ID, widget_mode=True) == SITE
    assert principal_for_turn(named, WORKSPACE_ID, widget_mode=False) == COMPANY
    for db in (_db(company=COMPANY, workspace=WORKSPACE), _db(workspace="Jane Doe", site=OWNER_NAMED_WORKSPACE)):
        ruled = with_rule_for_mode("Hi.", ContextMode.CHATBOT, db, WORKSPACE_ID, True)
        assert ruled == "Hi.\n\n" + ai_identity_rule(None)
        assert WORKSPACE not in ruled and "Jane Doe" not in ruled and "John Smith" not in ruled


def test_the_assembled_widget_prompt_never_names_the_workspace(quiet_sections):
    # a renamed personal workspace (its owner's full name) passes the name filter: a widget turn still never names it
    db = _db(workspace="Jane Doe")
    result = _build(db, MARKETPLACE_AGENT, widget_mode=True)
    assert result.system_prompt.endswith(f"{AI_IDENTITY_RULE_HEADING}\n{UNNAMED_RULE_WORDS}")
    assert "Jane Doe" not in result.system_prompt
    named = _build(_db(company=COMPANY, workspace="Jane Doe", site=SITE), MARKETPLACE_AGENT, widget_mode=True)
    assert named.system_prompt.endswith(ai_identity_rule(SITE))
    assert "Jane Doe" not in named.system_prompt and COMPANY not in named.system_prompt


# ---------------------------------------------------------------------------
# One decision for every surface: the channel notices read the same rows the same way
# ---------------------------------------------------------------------------

SAME_ROWS = (
    dict(company=COMPANY, workspace=OWNER_NAMED_WORKSPACE),
    dict(company="owner@acme.example", workspace=WORKSPACE),
    dict(company="  ", workspace=f" {WORKSPACE} "),
    dict(workspace=OWNER_EMAIL_WORKSPACE),
    dict(workspace="Jo’s workspace"),
    dict(workspace="A" * 500),
    dict(),
)


def test_the_prompt_rule_and_the_channel_notice_name_the_same_principal():
    """Runs once EUAI-03's channel notice (channels/ai_notice.py) is on the branch."""
    ai_notice = pytest.importorskip("channels.ai_notice")
    for rows in SAME_ROWS:
        expected = workspace_principal_name(_db(**rows), WORKSPACE_ID)
        assert ai_notice.principal_name(_db(**rows), WORKSPACE_ID) == expected, rows
        assert principal_for_turn(_db(**rows), WORKSPACE_ID, widget_mode=False) == expected, rows


def test_the_notice_rules_and_the_kernel_decide_names_alike():
    """Runs once EUAI-02a's notice rules (modules/policy/ai_disclosure.py) are on the branch."""
    ai_disclosure = pytest.importorskip("modules.policy.ai_disclosure")
    names = (None, "", "  ", 42, COMPANY, WORKSPACE, SITE, "owner@acme.example", "Jo’s workspace", *PRIVATE_WORKSPACE_NAMES)
    for company in names:
        for workspace in names:
            assert ai_disclosure.public_principal_name(company, workspace) == public_principal_name(company, workspace)
        assert ai_disclosure.public_site_name(company) == public_site_name(company)


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


def test_autos_chat_prompt_ends_with_the_rule(quiet_sections):
    result = _build(_db(workspace=WORKSPACE), AUTO)
    _assert_ends_with_the_rule(result, after="You are Auto")


def test_a_marketplace_agents_prompt_ends_with_the_rule_after_its_persona(quiet_sections):
    result = _build(_db(workspace=WORKSPACE), MARKETPLACE_AGENT)
    assert HUMAN_PERSONA in result.system_prompt
    _assert_ends_with_the_rule(result, after=HUMAN_PERSONA)


def test_a_custom_persona_cant_remove_the_rule(quiet_sections):
    result = _build(_db(workspace=WORKSPACE), CUSTOM_PERSONA_AGENT)
    assert HUMAN_CUSTOM_PROMPT in result.system_prompt
    _assert_ends_with_the_rule(result, after=HUMAN_CUSTOM_PROMPT)


def test_an_agent_run_on_a_task_ends_with_the_rule_too(quiet_sections):
    result = _build(_db(workspace=WORKSPACE), CUSTOM_PERSONA_AGENT, mode=ContextMode.TASK_EXECUTION,
                    task_description="Answer the visitor")
    _assert_ends_with_the_rule(result, after=HUMAN_CUSTOM_PROMPT)


def test_without_a_known_name_the_rule_names_no_one(quiet_sections):
    result = _build(_db(), MARKETPLACE_AGENT)
    assert result.system_prompt.endswith(f"{AI_IDENTITY_RULE_HEADING}\n{UNNAMED_RULE_WORDS}")


def test_an_owner_named_workspace_is_not_named_in_the_assembled_prompt(quiet_sections):
    result = _build(_db(workspace=OWNER_NAMED_WORKSPACE), CUSTOM_PERSONA_AGENT)
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
                                    db_session=_db(workspace=WORKSPACE), workspace_id=WORKSPACE_ID)
    assert prompt.endswith(ai_identity_rule(WORKSPACE))
    assert prompt.index(HUMAN_SOUL) < prompt.index(AI_IDENTITY_RULE_HEADING)
    assert "**Admin:** admin_list" in prompt and "**Platform:** platform_x" in prompt and "**Other:** web" in prompt


def test_the_cto_prompt_never_names_an_owner_named_workspace():
    from consumers.chatbot.cto_prompt_builder import CtoPromptBuilder

    prompt = CtoPromptBuilder.build(soul_document=HUMAN_SOUL, db_session=_db(workspace=OWNER_EMAIL_WORKSPACE),
                                    workspace_id=WORKSPACE_ID)
    assert prompt.endswith(ai_identity_rule(None))
    assert "john@acme.example" not in prompt


def _cto_turn_prompt(db, workspace_id, widget_mode=False):
    """The system prompt Auto's CTO turn sends, built by the service's real CTO override."""
    from consumers.chatbot.cto_prompt_builder import CtoPromptBuilder
    from consumers.chatbot.service import StreamingChatService

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db, svc.workspace_id, svc.widget_mode = db, workspace_id, widget_mode
    llm_messages = [{"role": "system", "content": "the chat's prompt"}]
    with patch.object(CtoPromptBuilder, "get_platform_state_snapshot", return_value={}):
        svc._apply_cto_override(llm_messages, NS(orchestrator=NS(_last_memory_result=None)),
                                NS(custom_persona_prompt=HUMAN_SOUL, configuration={}), [], None, 7)
    return llm_messages[0]["content"]


def test_autos_cto_turn_names_the_workspaces_public_name_like_context_service():
    prompt = _cto_turn_prompt(_db(company=COMPANY, workspace=OWNER_NAMED_WORKSPACE), WORKSPACE_ID)
    assert prompt.endswith(ai_identity_rule(COMPANY))
    assert prompt.index(HUMAN_SOUL) < prompt.index(AI_IDENTITY_RULE_HEADING)
    assert _cto_turn_prompt(_db(workspace=WORKSPACE), WORKSPACE_ID).endswith(ai_identity_rule(WORKSPACE))
    owner_named = _cto_turn_prompt(_db(workspace=OWNER_NAMED_WORKSPACE), WORKSPACE_ID)
    assert owner_named.endswith(ai_identity_rule(None)) and "John Smith" not in owner_named


def test_a_cto_turn_on_the_widget_names_only_the_site():
    named = _db(company=COMPANY, workspace=WORKSPACE, site=SITE)
    assert _cto_turn_prompt(named, WORKSPACE_ID, widget_mode=True).endswith(ai_identity_rule(SITE))
    unnamed = _cto_turn_prompt(_db(company=COMPANY, workspace=WORKSPACE), WORKSPACE_ID, widget_mode=True)
    assert unnamed.endswith(ai_identity_rule(None)) and WORKSPACE not in unnamed
