"""Brand kit at generation (F): an answer that repeats its prompt keeps the rules block as it is.

Main CI on 15452da7f (#953 merged): a playbook step whose answer repeated its prompt
came back with the rules block's own example signed ("Never leave a placeholder such
as Gerard, Harbourline Coffee Roasters.") and a banned-words note naming "exquisite",
which only the rules' list used. The answer check now leaves a repeated rules block
alone and checks the text around it.
"""
from __future__ import annotations

from services.brand_rules import BANNED_NOTE_LEAD, on_brand_text, rules_for_kit

SIGNED = "Gerard, Harbourline Coffee Roasters"
KIT = {"name": "Harbourline Coffee Roasters",
       "voice": {"tone": ["warm", "plain", "local"], "banned_phrases": ["exquisite"], "sign_off": SIGNED}}


def _echoed(answer: str) -> str:
    return f"Write the club note.\n\n{rules_for_kit(KIT)}\n\n{answer}"


def test_the_rules_block_in_an_answer_is_left_as_it_is():
    out = on_brand_text(_echoed("Best,\n[Your name]"), KIT)
    assert rules_for_kit(KIT) in out
    assert out.endswith(f"Best,\n{SIGNED}")
    assert BANNED_NOTE_LEAD not in out


def test_a_banned_word_outside_the_rules_block_is_still_said():
    out = on_brand_text(_echoed("Our exquisite Guji.\n\nBest,\n[Your name]"), KIT)
    assert rules_for_kit(KIT) in out
    assert out.endswith(f'{BANNED_NOTE_LEAD}: "exquisite". Change them before it goes out.')


def test_an_answer_without_the_rules_block_is_checked_as_before():
    assert on_brand_text("Our exquisite Guji.\n\nBest,\n[Your name]", KIT) == (
        f"Our exquisite Guji.\n\nBest,\n{SIGNED}\n\n"
        f'{BANNED_NOTE_LEAD}: "exquisite". Change them before it goes out.')
