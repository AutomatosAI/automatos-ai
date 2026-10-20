"""PRD-263 US-066 — the persona doctrine loses the DELEGATE lane, and every shipped Auto
persona still carrying the previous block is lifted at boot.

PRD-256 removed the DELEGATE lane (a specialist answers the chat) from the router:
consumers/chatbot/auto.py routes to respond, assign or mission. Point 2 of
MANAGER_DOCTRINE_BLOCK still named DELEGATE, and every shipped persona (the seeded
friendly default and the three Settings presets) carries that block. So the backfill
(sync_auto_personas, run at boot) must recognise each shipped text with the previous
block and lift it to the same voice with the current one, while a customized persona
is never touched.

Pure and LLM-free: the real _backfill_auto_persona and sync_auto_personas over
SimpleNamespace rows, with a MagicMock db (as in test_prd226_prof_tech_backfill).
"""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.seeds.seed_auto_agent import (
    MANAGER_DOCTRINE_BLOCK,
    _FRIENDLY_FALLBACK,
    _PERSONALITY_BASE_VOICES,
    _PREVIOUS_FRIENDLY_FALLBACK,
    _PREVIOUS_MANAGER_DOCTRINE_BLOCK,
    _backfill_auto_persona,
    _default_persona,
    compose_persona_with_doctrine,
    sync_auto_personas,
)

# sha256 of MANAGER_DOCTRINE_BLOCK as it shipped before US-066 (main 5c4832155): the lift
# keys must be the exact text rows hold, not a reconstruction that only looks right.
SHIPPED_BLOCK_SHA256 = "2e770b83c159677c5f64eba3e98661df965665bbfffaac5cc9b47313315fb3ea"
PRESET_MODES = ("friendly", "professional", "technical")


def _row(persona: str, configuration=None) -> SimpleNamespace:
    return SimpleNamespace(
        custom_persona_prompt=persona,
        use_custom_persona=True,
        configuration=configuration,
        workspace_id="ws-us066",
        slug="auto-x",
    )


def _with_previous_block(voice: str) -> str:
    return compose_persona_with_doctrine(voice, _PREVIOUS_MANAGER_DOCTRINE_BLOCK)


def test_the_previous_block_is_the_block_that_shipped():
    digest = hashlib.sha256(_PREVIOUS_MANAGER_DOCTRINE_BLOCK.encode("utf-8")).hexdigest()
    assert digest == SHIPPED_BLOCK_SHA256


def test_the_current_block_names_respond_and_not_delegate():
    assert "RESPOND" in MANAGER_DOCTRINE_BLOCK
    assert "DELEGATE" not in MANAGER_DOCTRINE_BLOCK
    assert "DELEGATE" in _PREVIOUS_MANAGER_DOCTRINE_BLOCK


def test_the_seeded_friendly_default_with_the_previous_block_is_lifted():
    for base in (_FRIENDLY_FALLBACK, _PREVIOUS_FRIENDLY_FALLBACK):
        row = _row(_with_previous_block(base), {"personality_mode": "friendly"})

        assert _backfill_auto_persona(row) == "updated"
        assert row.custom_persona_prompt == _default_persona()
        assert "DELEGATE" not in row.custom_persona_prompt


def test_each_preset_with_the_previous_block_is_lifted_to_its_own_voice():
    for mode in PRESET_MODES:
        voice = _PERSONALITY_BASE_VOICES[mode]
        row = _row(_with_previous_block(voice), {"personality_mode": mode})

        assert _backfill_auto_persona(row) == "updated", mode
        assert row.custom_persona_prompt == compose_persona_with_doctrine(voice), mode
        assert row.configuration == {"personality_mode": mode}, mode


def test_professional_and_technical_rows_are_lifted_and_neither_is_reported_skipped():
    rows = [
        _row(_with_previous_block(_PERSONALITY_BASE_VOICES[mode]), {"personality_mode": mode})
        for mode in ("professional", "technical")
    ]
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = rows

    assert sync_auto_personas(db) == {"updated": 2, "skipped": 0, "current": 0}
    assert rows[0].custom_persona_prompt == compose_persona_with_doctrine(_PERSONALITY_BASE_VOICES["professional"])
    assert rows[1].custom_persona_prompt == compose_persona_with_doctrine(_PERSONALITY_BASE_VOICES["technical"])


def test_a_customized_persona_carrying_the_previous_block_is_untouched():
    custom = _with_previous_block("**My personality:**\n- I am the Auto this workspace wrote for itself")
    row = _row(custom, {"personality_mode": "custom"})

    assert _backfill_auto_persona(row) == "skipped"
    assert row.custom_persona_prompt == custom


def test_a_lifted_row_is_current_on_the_next_boot():
    row = _row(_with_previous_block(_FRIENDLY_FALLBACK))
    _backfill_auto_persona(row)
    lifted = row.custom_persona_prompt

    assert _backfill_auto_persona(row) == "current"
    assert row.custom_persona_prompt == lifted
