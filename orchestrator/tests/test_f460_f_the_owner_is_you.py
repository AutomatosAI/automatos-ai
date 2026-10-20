"""F460 (night 18): Auto spoke of the owner in the third person, to the owner: "ask Gerard whether…",
"He hasn't confirmed", "needs Gerard's decision". Both lanes' identity clause now say that the person a
document, card or tool note names (or calls 'the owner') is the one Auto is talking to; with no name
known, neither prompt changes.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from consumers.chatbot.atom_prompt import atom_system_prompt
from consumers.chatbot.owner_is_you import owner_is_you
from consumers.chatbot.personality import AutomatosPersonality
from consumers.chatbot.service import atom_identity_clause

SETTINGS = {"personality_mode": "friendly", "custom_soul": "", "communication_style": "balanced"}
NAMELESS_OPENING = "You are Auto, a capable AI assistant. You're ready to help.\n\n## Who You Are\n"
CLAUSE = ("When a document, card or tool note names Gerard or 'the owner', that is the person you are talking to: "
          "say 'you' and 'your', never 'Gerard', 'he' or 'the owner', except inside a draft addressed to someone else.")


def _full(user_name=None):
    return AutomatosPersonality.get_base_system_prompt(user_name=user_name, agent_name="Auto",
                                                       orchestrator_settings=SETTINGS)


def test_the_full_lane_says_the_owner_named_is_the_person_it_talks_to():
    assert owner_is_you("Gerard") == CLAUSE
    assert _full("Gerard").startswith(f"You are Auto, a capable AI assistant. You're talking to Gerard. {CLAUSE}\n\n")


def test_the_atom_lane_says_it_too():
    identity = atom_identity_clause("Gerard")
    prompt = atom_system_prompt(NS(name="Auto", description="", persona=None), identity=identity, memory_block="",
                                facts="")

    assert identity == f" You're talking to Gerard. {CLAUSE}" and identity in prompt


def test_with_no_name_neither_prompt_changes():
    assert _full().startswith(NAMELESS_OPENING) and "When a document, card or tool note names" not in _full()
    assert atom_identity_clause(None) == atom_identity_clause("  ") == ""
