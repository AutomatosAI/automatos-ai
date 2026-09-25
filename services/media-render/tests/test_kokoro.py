"""Kokoro speaks inside the image: one line, one WAV, longer than 0.5 s (US-101),
with the voiced segments on-screen text lands on (US-102).

This loads the real 310 MB model through the image's espeak-ng data, so it
also proves the short-path copy works end to end.
"""

from __future__ import annotations

import soundfile

from media_render.config import load_settings
from media_render.kokoro_tts import synthesize_line


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
