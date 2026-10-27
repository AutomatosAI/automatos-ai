"""GET /voices's listing: read from the voices file, described by Kokoro's voice ids, English first."""

from __future__ import annotations

import pytest

from helpers import catalogue, voices_file
from voice_tts.voices import LANGUAGES, VoicesError, describe, load_catalogue, voice_ids


@pytest.mark.parametrize("voice_id, name, accent, gender, espeak", [
    ("af_heart", "Heart", "en-US", "female", "en-us"),
    ("am_michael", "Michael", "en-US", "male", "en-us"),
    ("bf_isabella", "Isabella", "en-GB", "female", "en-gb"),
    ("bm_george", "George", "en-GB", "male", "en-gb"),
    ("ef_dora", "Dora", "es-ES", "female", "es"),
    ("ff_siwis", "Siwis", "fr-FR", "female", "fr-fr"),
    ("hm_omega", "Omega", "hi-IN", "male", "hi"),
    ("if_sara", "Sara", "it-IT", "female", "it"),
    ("jf_alpha", "Alpha", "ja-JP", "female", "ja"),
    ("pm_alex", "Alex", "pt-BR", "male", "pt-br"),
    ("zf_xiaoxiao", "Xiaoxiao", "zh-CN", "female", "cmn"),
])
def test_a_voice_id_names_its_voice_accent_gender_and_language(voice_id, name, accent, gender, espeak):
    voice = describe(voice_id)

    assert (voice.name, voice.accent, voice.gender, voice.espeak) == (name, accent, gender, espeak)
    assert voice.listing() == {"id": voice_id, "name": name, "accent": accent, "gender": gender}


@pytest.mark.parametrize("voice_id", ["heart", "af-heart", "xf_heart", "ax_heart", "af_", "Af_heart"])
def test_an_id_outside_kokoros_convention_is_refused(voice_id):
    with pytest.raises(VoicesError):
        describe(voice_id)


def test_the_listing_is_american_then_british_then_the_other_languages_each_by_id(tmp_path):
    body = catalogue(tmp_path).body()

    assert body["default"] == "af_heart"
    assert [voice["id"] for voice in body["voices"]] == [
        "af_heart", "am_adam", "bf_emma", "bm_george", "ef_dora", "jf_alpha", "zf_xiaobei",
    ]
    assert set(body["voices"][0]) == {"id", "name", "accent", "gender"}


def test_an_id_the_service_could_not_place_is_left_out(tmp_path):
    voices = catalogue(tmp_path, ids=("af_heart", "santa_claus", "qf_unknown"))

    assert [voice.id for voice in voices.voices] == ["af_heart"]


def test_a_default_the_file_lacks_is_refused(tmp_path):
    with pytest.raises(VoicesError, match="default voice"):
        catalogue(tmp_path, default="af_nobody")


def test_a_missing_or_empty_voices_file_is_refused(tmp_path):
    with pytest.raises(VoicesError):
        voice_ids(str(tmp_path / "missing.bin"))
    with pytest.raises(VoicesError):
        load_catalogue(str(voices_file(tmp_path / "empty.bin", ids=())), "af_heart")


def test_the_image_holds_upstreams_54_voices_english_first(settings):
    voices = load_catalogue(settings.kokoro_voices_path, settings.default_voice)
    ids = [voice.id for voice in voices.voices]

    assert len(ids) == 54
    assert voices.default == "af_heart" and ids[0] == "af_alloy"
    english = [voice_id for voice_id in ids if voice_id[0] in "ab"]
    assert ids[: len(english)] == english and len(english) == 28
    assert {voice_id[0] for voice_id in ids} == set(LANGUAGES)
