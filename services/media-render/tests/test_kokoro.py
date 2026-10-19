"""Kokoro speaks inside the image: one line, one WAV, longer than 0.5 s (US-101),
with the voiced segments on-screen text lands on (US-102). Synthesis runs no exiftool
and writes no mark (PRD-264 EUAI-06a): a render's voice step is timed in CI, and the
WAVs that leave the service are marked by POST /tts (test_provenance*.py).

This loads the real 310 MB model through the image's espeak-ng data, so it
also proves the data moved to the short path works end to end.

PRD-262 US-052: every voice of the catalogue speaks a line from the rewritten
voices file, and the image's cut espeak-ng data gives Kokoro the same phonemes for
a fixed English sentence as the package's full data gave the build.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import espeakng_loader
import pytest
import soundfile
from kokoro_onnx.config import EspeakConfig
from kokoro_onnx.tokenizer import Tokenizer

from media_render import provenance
from media_render.config import load_settings
from media_render.espeak_build import LANGUAGES
from media_render.kokoro_tts import synthesize_line
from media_render.voices_build import load_catalogue

# Written by the py stage (media_render/espeak_build.py); the test stage copies it in.
PHONEMES = Path("/opt/media-render/espeak-phonemes.json")


def test_kokoro_synthesises_one_line(tmp_path):
    settings = load_settings()
    spoken = synthesize_line("Rendered on brand, in three seconds.", tmp_path / "l01.wav", settings)
    assert spoken.path.is_file()
    assert spoken.sample_rate == 24000
    assert spoken.seconds > 0.5
    samples, rate = soundfile.read(str(spoken.path))
    assert rate == spoken.sample_rate
    assert len(samples) / rate > 0.5
    assert float(abs(samples).max()) > 0.01, "the line is silent"


def test_speaking_a_line_runs_no_exiftool_and_writes_no_mark(tmp_path, monkeypatch):
    """EUAI-06a: the mark is not paid for inside the timed voice step; the WAV comes out unmarked."""
    settings = load_settings()
    real_run = subprocess.run
    ran = []

    def watched(argv, *args, **kwargs):
        ran.append(list(argv) if isinstance(argv, (list, tuple)) else [argv])
        return real_run(argv, *args, **kwargs)

    def no_mark(*args, **kwargs):
        raise AssertionError("synthesize_line marked its WAV")

    monkeypatch.setattr(subprocess, "run", watched)
    monkeypatch.setattr(provenance, "mark", no_mark)
    spoken = synthesize_line("No mark inside the timed step.", tmp_path / "l01.wav", settings)
    monkeypatch.undo()
    exiftool = Path(settings.exiftool_bin).name
    assert [argv for argv in ran if argv and Path(str(argv[0])).name == exiftool] == []
    assert provenance.read_source_type(spoken.path, exiftool_bin=settings.exiftool_bin, timeout_seconds=60) is None
    assert provenance.XMP_CHUNK not in spoken.path.read_bytes(), "the WAV carries an XMP chunk"


def test_a_line_with_pauses_has_a_segment_per_phrase(tmp_path):
    spoken = synthesize_line("Orders. Stock. Customers. The books.", tmp_path / "l02.wav", load_settings())
    starts = [start for start, _ in spoken.segments]
    assert len(spoken.segments) >= 3, spoken.segments
    assert starts == sorted(starts)
    assert all(0 <= start < end <= spoken.seconds for start, end in spoken.segments)


def test_an_empty_line_is_refused(tmp_path):
    try:
        synthesize_line("   ", tmp_path / "x.wav", load_settings())
    except ValueError:
        return
    raise AssertionError("an empty line was spoken")


@pytest.mark.parametrize("voice", load_catalogue())
def test_every_catalogue_voice_speaks_a_line(voice, tmp_path):
    # In the language the orchestrator sends with it: a British voice (b…) en-gb.
    lang = "en-gb" if voice.startswith("b") else "en-us"
    spoken = synthesize_line("Your brand, in every voice.", tmp_path / f"{voice}.wav", load_settings(), voice=voice, lang=lang)
    samples, rate = soundfile.read(str(spoken.path))
    assert spoken.seconds > 0.5 and len(samples) / rate > 0.5
    assert float(abs(samples).max()) > 0.01, f"{voice} is silent"


def test_the_english_phonemes_match_the_snapshot_the_build_took_with_the_full_data():
    snapshot = json.loads(PHONEMES.read_text(encoding="utf-8"))
    assert tuple(snapshot["phonemes"]) == LANGUAGES and snapshot["dictionaries"] == ["en_dict"]
    data = load_settings().espeak_data_path
    tokenizer = Tokenizer(EspeakConfig(lib_path=espeakng_loader.get_library_path(), data_path=data))
    for language, expected in snapshot["phonemes"].items():
        assert expected, language
        assert tokenizer.phonemize(snapshot["sentence"], language) == expected, language
