"""Voice quality: Auto sounds like a person, not a read-aloud chat log.

The fault these cover: nothing in the voice path ever told Auto she was being
HEARD, and her raw text went straight to TTS — so a normal chat answer
(bullets, ``**bold**``, headers, links) was read out with the punctuation
pronounced. Plus the slow-turn filler fired per turn, so an utterance storm
played "One moment." over itself.

Pure unit tests — no DB, no socket, no model.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import patch

import pytest

from modules.voice.spoken_style import (
    SPOKEN_OUTPUT_CONTRACT,
    speechify,
    split_speech_unit,
)


# ── speechify ──────────────────────────────────────────────────────────────


def test_bullets_and_bold_become_plain_speech() -> None:
    said = speechify("Here's the state:\n\n- **Auth** is broken\n- Billing is fine")
    assert "*" not in said and "-" not in said
    assert "Auth is broken" in said and "Billing is fine" in said
    assert ":." not in said  # the lead-in colon isn't doubled up


def test_numbered_list_loses_its_numbering() -> None:
    said = speechify("1. Fix it\n2. Ship it")
    assert said.startswith("Fix it")
    assert "1." not in said and "2." not in said


def test_headers_and_rules_are_dropped() -> None:
    said = speechify("## Next steps\n---\nWe ship on Friday.")
    assert "#" not in said and "---" not in said
    assert "Next steps" in said and "We ship on Friday." in said


def test_link_speaks_its_label_not_its_url() -> None:
    said = speechify("Check [the dashboard](https://app.automatos.app/x?y=1) now")
    assert "the dashboard" in said
    assert "http" not in said and "automatos.app" not in said


def test_code_fence_says_nothing() -> None:
    assert speechify("```python\nprint('hi')\n```") == ""


def test_inline_code_keeps_its_contents() -> None:
    assert "db.py" in speechify("It's in `db.py` line 42")


def test_emoji_are_stripped() -> None:
    said = speechify("All green 🎉✅")
    assert "🎉" not in said and "✅" not in said
    assert "All green" in said


def test_plain_prose_is_untouched() -> None:
    text = "Evening, Gerard. All good here."
    assert speechify(text) == text


def test_lone_asterisk_in_prose_is_not_emphasis() -> None:
    # "2 * 3" must not be eaten as an unclosed emphasis marker.
    assert speechify("The cost is 2 * 3 dollars") == "The cost is 2 * 3 dollars"


def test_punctuation_only_unit_is_silent() -> None:
    assert speechify("---") == ""
    assert speechify("   ") == ""


# ── speech units ───────────────────────────────────────────────────────────


def test_unit_splits_on_sentence_end() -> None:
    unit, rest = split_speech_unit("Hello there. How are", 180)
    assert unit == "Hello there."
    assert rest == " How are"


def test_incomplete_sentence_waits() -> None:
    unit, rest = split_speech_unit("no terminator yet", 180)
    assert unit == ""
    assert rest == "no terminator yet"


def test_runaway_sentence_is_cut_at_a_word_boundary() -> None:
    buf = "word " * 60  # no terminator at all
    unit, rest = split_speech_unit(buf, 50)
    assert unit and len(unit) <= 50
    assert not unit.endswith("wor")  # cut on a space, never mid-word
    assert rest


def test_newline_is_a_boundary() -> None:
    unit, _ = split_speech_unit("First line\nsecond", 180)
    assert unit.strip() == "First line"


# ── frame streaming ────────────────────────────────────────────────────────


async def _chunks(*texts: str) -> AsyncIterator[str]:
    for t in texts:
        yield "0:" + json.dumps(t)


async def _collect(gen: AsyncIterator[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [f async for f in gen]


class _Cfg:
    VOICE_LIVE_SPEECH_UNITS = True
    VOICE_LIVE_SPEECH_UNIT_MAX_CHARS = 180


@pytest.mark.asyncio
async def test_frames_are_sanitised_across_chunk_boundaries() -> None:
    """The reason units exist: '**' arrives split over two chunks, so raw
    per-token sanitising could never catch it."""
    from modules.voice.providers import retell as mod

    with patch("config.config", _Cfg):
        frames = await _collect(
            mod.retell_response_frames(7, _chunks("Auth is *", "*broken*", "* now."))
        )
    spoken = "".join(f["content"] for f in frames)
    assert "*" not in spoken
    assert "Auth is broken now." in spoken
    assert frames[-1]["content_complete"] is True
    assert all(f["response_id"] == 7 for f in frames)


@pytest.mark.asyncio
async def test_terminal_frame_always_closes_the_turn() -> None:
    from modules.voice.providers import retell as mod

    with patch("config.config", _Cfg):
        frames = await _collect(mod.retell_response_frames(3, _chunks("```\ncode\n```")))
    # Nothing speakable, but Retell still needs the turn closed or it hangs.
    assert frames[-1] == {"response_id": 3, "content": "", "content_complete": True}


@pytest.mark.asyncio
async def test_multi_sentence_reply_streams_progressively() -> None:
    """First audio must not wait for the whole generation."""
    from modules.voice.providers import retell as mod

    with patch("config.config", _Cfg):
        frames = await _collect(
            mod.retell_response_frames(1, _chunks("One. ", "Two. ", "Three."))
        )
    speech = [f for f in frames if f["content"].strip()]
    assert len(speech) >= 3, "sentences should ship as they complete, not in one lump"


@pytest.mark.asyncio
async def test_legacy_passthrough_when_units_disabled() -> None:
    from modules.voice.providers import retell as mod

    class _Off(_Cfg):
        VOICE_LIVE_SPEECH_UNITS = False

    with patch("config.config", _Off):
        frames = await _collect(mod.retell_response_frames(2, _chunks("**raw**")))
    assert frames[0]["content"] == "**raw**"


# ── the spoken-output contract ─────────────────────────────────────────────


def test_contract_names_the_things_that_break_tts() -> None:
    lowered = SPOKEN_OUTPUT_CONTRACT.lower()
    for token in ("markdown", "heard", "sentences", "url"):
        assert token in lowered


def test_spoken_style_appends_to_the_system_prompt() -> None:
    from consumers.chatbot.service import StreamingChatService as S

    msgs = [
        {"role": "system", "content": "You are Auto."},
        {"role": "user", "content": "hi"},
    ]
    S._apply_spoken_style(msgs)
    assert msgs[0]["content"].startswith("You are Auto.")
    assert "YOU ARE SPEAKING OUT LOUD" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "hi"}  # user turn untouched


def test_spoken_style_is_dial_gated() -> None:
    from consumers.chatbot.service import StreamingChatService as S

    class _Off:
        VOICE_LIVE_SPOKEN_STYLE = False

    msgs = [{"role": "system", "content": "You are Auto."}]
    with patch("config.config", _Off):
        S._apply_spoken_style(msgs)
    assert msgs[0]["content"] == "You are Auto."


def test_spoken_style_survives_a_missing_system_message() -> None:
    from consumers.chatbot.service import StreamingChatService as S

    msgs = [{"role": "user", "content": "hi"}]
    S._apply_spoken_style(msgs)
    assert msgs[0]["role"] == "system"


# ── agent tuning ───────────────────────────────────────────────────────────


def test_tuning_no_longer_cancels_the_speaker() -> None:
    """The aggressive mode cancelled Gerard's own voice — a whole live call
    logged zero turns until he shouted."""
    from modules.voice import retell_api

    tuning = retell_api.build_agent_tuning()
    assert tuning["denoising_mode"] == "noise-cancellation"
    assert tuning["interruption_sensitivity"] >= 0.4, "0.2 blocked normal barge-in"


def test_tuning_carries_voice_and_speech_normalisation() -> None:
    from modules.voice import retell_api

    tuning = retell_api.build_agent_tuning()
    assert tuning["voice_id"]
    assert tuning["normalize_for_speech"] is True
    assert tuning["reminder_max_count"] == 1  # stop the "still there?" nagging
    assert "voice_speed" in tuning and "voice_temperature" in tuning


def test_tuning_can_leave_a_human_picked_voice_alone() -> None:
    from modules.voice import retell_api

    tuning = retell_api.build_agent_tuning(include_voice=False)
    assert "voice_id" not in tuning
    assert tuning["denoising_mode"] == "noise-cancellation"  # hearing still tuned
