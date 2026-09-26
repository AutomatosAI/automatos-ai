"""F167 (b) — a session never claims an approval nobody gave.

Ticket #999's session wrote that a Chrome print command "needed your approval,
and it went through". No question card existed: the host ran the command on its
own rule. To a session, a command that runs looks the same whether the host
allowed it outright or the operator approved a hold. So the ticket prompt says:
claim an approval only when a question card was answered.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from services.cli_session_prompt import SKILLS_HEADER, TOOLS_HEADER, session_system_prompt

RULE = "never write that anything needed, got or went through the operator's approval unless a question card was answered"


def _agent():
    return NS(id=58, name="Printer", description="Prints reports to PDF.", use_custom_persona=False,
              custom_persona_prompt=None, persona=None, configuration={},
              skills=[NS(id=1, name="pdf", description="Makes PDFs.", prompt_template="Use Chrome to print.",
                         is_active=True, content_hash=None, tools_schema=None)])


def test_a_ticket_session_is_told_to_claim_an_approval_only_when_a_card_was_answered():
    prompt = session_system_prompt(_agent())
    tools = prompt[prompt.index(TOOLS_HEADER):prompt.index(SKILLS_HEADER)]
    approvals = next(line for line in tools.splitlines() if line.startswith("- Approvals:"))
    assert RULE in approvals and "`ask_human`" in approvals and "never that it was approved" in approvals
    assert session_system_prompt(_agent()) == prompt  # stable per agent (the prompt-cache invariant)


def test_the_operators_own_terminal_gets_no_such_line():
    """The Canvas terminal is the operator typing: no gate, no tools block."""
    assert RULE not in session_system_prompt(_agent(), ticket_session=False)
