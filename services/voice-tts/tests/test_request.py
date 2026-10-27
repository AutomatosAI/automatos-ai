"""POST /speak's body: what is spoken, what is refused and with which code, and the sentences."""

from __future__ import annotations

import pytest

from helpers import catalogue
from voice_tts.request import RequestError, clamp_speed, parse_speech, sentences


@pytest.fixture
def voices(tmp_path):
    return catalogue(tmp_path)


def _refusal(payload, settings, voices) -> RequestError:
    with pytest.raises(RequestError) as caught:
        parse_speech(payload, settings, voices)
    return caught.value


def test_a_text_with_no_voice_or_speed_is_spoken_by_the_default_at_normal_speed(settings, voices):
    spec = parse_speech({"text": "  Hello there.  "}, settings, voices)

    assert spec.text == "Hello there."
    assert spec.voice.id == "af_heart" and spec.voice.espeak == "en-us"
    assert spec.speed == 1.0


def test_null_voice_and_speed_are_the_defaults(settings, voices):
    spec = parse_speech({"text": "Hi", "voice": None, "speed": None}, settings, voices)

    assert (spec.voice.id, spec.speed) == ("af_heart", 1.0)


def test_a_british_voice_is_phonemized_as_british_english(settings, voices):
    spec = parse_speech({"text": "Tomato", "voice": "bm_george"}, settings, voices)

    assert spec.voice.espeak == "en-gb" and spec.voice.accent == "en-GB"


@pytest.mark.parametrize("speed, spoken", [(0.5, 0.8), (0.8, 0.8), (1, 1.0), (1.1, 1.1), (1.2, 1.2), (3, 1.2)])
def test_speed_is_clamped_to_point_eight_to_one_point_two(speed, spoken):
    assert clamp_speed(speed) == spoken


@pytest.mark.parametrize("speed", ["fast", True, float("nan"), float("inf"), [1.0], {"v": 1}])
def test_a_speed_that_is_not_a_finite_number_is_a_422(settings, voices, speed):
    refused = _refusal({"text": "Hi", "speed": speed}, settings, voices)

    assert (refused.status, refused.code) == (422, "tts_bad_request")
    assert "speed" in refused.detail


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_is_a_422(settings, voices, text):
    refused = _refusal({"text": text}, settings, voices)

    assert (refused.status, refused.code) == (422, "tts_bad_request")


@pytest.mark.parametrize("text", [None, 42, ["Hi"]])
def test_text_that_is_not_a_string_is_a_422(settings, voices, text):
    assert _refusal({"text": text}, settings, voices).status == 422


def test_text_over_the_limit_after_trimming_is_a_413(settings, voices):
    at_limit = "a" * settings.max_chars
    assert parse_speech({"text": f"  {at_limit}  "}, settings, voices).text == at_limit

    refused = _refusal({"text": at_limit + "b"}, settings, voices)

    assert (refused.status, refused.code) == (413, "tts_too_long")


@pytest.mark.parametrize("voice", ["af_nobody", "AF_HEART", "", 7])
def test_a_voice_the_image_does_not_hold_is_a_422(settings, voices, voice):
    refused = _refusal({"text": "Hi", "voice": voice}, settings, voices)

    assert (refused.status, refused.code) == (422, "tts_bad_request")
    assert "voice" in refused.detail


@pytest.mark.parametrize("payload", [[], "text", None, {"voice": "af_heart"}, {"text": "Hi", "lang": "en-us"}])
def test_a_body_that_is_not_the_contracts_object_is_a_422(settings, voices, payload):
    assert _refusal(payload, settings, voices).status == 422


def test_a_long_text_is_spoken_a_sentence_at_a_time():
    text = "Your board has three tasks waiting. Two are due today! Shall I start on the first one?"

    assert sentences(text) == (
        "Your board has three tasks waiting.",
        "Two are due today!",
        "Shall I start on the first one?",
    )


def test_short_sentences_and_abbreviations_join_the_next_one():
    text = 'Sure. I spoke with Dr. Lee about the U.S. launch. He said "Go." Then it shipped on time.'

    assert sentences(text) == (
        "Sure. I spoke with Dr. Lee about the U.S. launch.",
        'He said "Go." Then it shipped on time.',
    )


def test_a_single_sentence_or_a_trailing_fragment_is_one_piece():
    assert sentences("Done") == ("Done",)
    assert sentences("All three tasks are done. Ok") == ("All three tasks are done.", "Ok")
