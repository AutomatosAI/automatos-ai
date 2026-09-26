"""The mix (US-102): the reference graph, generalised, and -14 LUFS on real audio.

The graph tests pin the reference's parameters (docs/PRDS/prd251-reference/
mix-reference.py). The loudness tests run the real two-pass ffmpeg mix inside
the image on a real Kokoro line and on a music-voice-SFX mix, and measure the
result independently with ebur128.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile

from media_render import audio
from media_render.config import load_settings
from media_render.kokoro_tts import synthesize_line
from media_render.timing import voiced_segments

FMT = "aformat=sample_rates=48000:channel_layouts=stereo"
TOLERANCE_LU = 1.0

LOUDNORM_LOG = """[Parsed_loudnorm_12 @ 0x55d5c5f8e4c0]
{
\t"input_i" : "-23.19",
\t"input_tp" : "-6.05",
\t"input_lra" : "3.90",
\t"input_thresh" : "-33.40",
\t"output_i" : "-14.26",
\t"output_tp" : "-1.50",
\t"output_lra" : "2.60",
\t"output_thresh" : "-24.37",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.26"
}
"""

EBUR128_LOG = """[Parsed_ebur128_0 @ 0x5581] Summary:

  Integrated loudness:
    I:         -14.2 LUFS
    Threshold: -24.4 LUFS

  Loudness range:
    LRA:         2.1 LU
    Threshold: -34.3 LUFS
    LRA low:   -15.3 LUFS
    LRA high:  -13.2 LUFS

  True peak:
    Peak:       -1.4 dBFS
"""


def full_plan(root: Path) -> audio.MixPlan:
    return audio.MixPlan(
        duration=40.0,
        voice=(audio.VoiceTrack(root / "l01.wav", 0.30), audio.VoiceTrack(root / "l02.wav", 2.40)),
        music=audio.MusicTrack(root / "bed.mp3", start=32.0),
        sfx=(audio.SfxTrack(root / "switch.ogg", 3.28, 0.7),),
    )


def test_the_full_mix_is_the_reference_graph(tmp_path):
    inputs, graph = audio.filter_graph(full_plan(tmp_path), audio.LOUDNORM)
    assert inputs[:6] == ["-ss", "32", "-t", "40", "-i", str(tmp_path / "bed.mp3")]
    assert inputs[6:] == ["-i", str(tmp_path / "l01.wav"), "-i", str(tmp_path / "l02.wav"), "-i", str(tmp_path / "switch.ogg")]
    parts = graph.split(";")
    assert parts == [
        f"[0:a]{FMT},afade=t=in:st=0:d=0.02,afade=t=out:st=38.3:d=1.7,atrim=0:40[m]",
        f"[1:a]{FMT},adelay=300|300[v0]",
        f"[2:a]{FMT},adelay=2400|2400[v1]",
        "[v0][v1]amix=inputs=2:normalize=0:duration=longest,apad=whole_dur=40,atrim=0:40[vo]",
        "[vo]asplit=2[vomix][vokey]",
        "[m][vokey]sidechaincompress=threshold=0.015:ratio=10:attack=10:release=420:makeup=1[mduck]",
        "[mduck]volume=0.6[mq]",
        f"[3:a]{FMT},volume=0.7,adelay=3280|3280[s0]",
        "[s0]amix=inputs=1:normalize=0:duration=longest,apad=whole_dur=40,atrim=0:40[sx]",
        "[mq][vomix][sx]amix=inputs=3:normalize=0:duration=longest,atrim=0:40,apad=pad_dur=1,"
        f"loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,atrim=0:40,{FMT}[out]",
    ]


def test_voice_alone_is_normalised_without_a_ducking_key(tmp_path):
    plan = audio.MixPlan(duration=3.0, voice=(audio.VoiceTrack(tmp_path / "l01.wav", 0.3),))
    _, graph = audio.filter_graph(plan, audio.LOUDNORM)
    assert "sidechaincompress" not in graph
    assert graph.endswith(f"[vo]atrim=0:3,apad=pad_dur=1,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,atrim=0:3,{FMT}[out]")


def test_music_alone_sits_at_the_bed_level(tmp_path):
    plan = audio.MixPlan(duration=10.0, music=audio.MusicTrack(tmp_path / "bed.mp3", start=0, fade_in=0, fade_out=0))
    _, graph = audio.filter_graph(plan, audio.LOUDNORM)
    assert f"[0:a]{FMT},atrim=0:10[m]" in graph and "[m]volume=0.6[mq]" in graph and "[mq]atrim=0:10," in graph


def test_a_silent_plan_has_no_graph_but_still_gets_a_track(tmp_path):
    with pytest.raises(ValueError):
        audio.filter_graph(audio.MixPlan(duration=3.0), audio.LOUDNORM)
    argv = audio.silence_argv("ffmpeg", 3.0, tmp_path / "mix.wav")
    assert "anullsrc=r=48000:cl=stereo" in argv and argv[argv.index("-t") + 1] == "3"


def test_the_two_passes_measure_then_normalise(tmp_path):
    plan = full_plan(tmp_path)
    measure = audio.measure_argv("ffmpeg", plan)
    assert "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json" in measure[measure.index("-filter_complex") + 1]
    assert measure[-3:] == ["-f", "null", "-"]
    stats = audio.loudnorm_stats("noise before\n" + LOUDNORM_LOG)
    second = audio.second_pass_loudnorm(stats)
    assert second == (
        "loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=-23.19:measured_TP=-6.05:measured_LRA=3.90"
        ":measured_thresh=-33.40:offset=0.26:linear=true:print_format=json"
    )
    mix = audio.mix_argv("ffmpeg", plan, second, tmp_path / "mix.wav")
    assert second in mix[mix.index("-filter_complex") + 1]
    assert mix[-3:] == ["-c:a", "pcm_s16le", str(tmp_path / "mix.wav")]


def test_a_silent_measurement_falls_back_to_one_pass():
    silent = {"input_i": "-inf", "input_tp": "-inf", "input_lra": "0.00", "input_thresh": "-70.00", "target_offset": "inf"}
    assert audio.second_pass_loudnorm(silent) is None


def test_measured_values_are_clamped_to_what_loudnorm_accepts():
    stats = {"input_i": "-120.5", "input_tp": "3.2", "input_lra": "140", "input_thresh": "-130", "target_offset": "-150"}
    second = audio.second_pass_loudnorm(stats)
    assert "measured_I=-99.00" in second and "measured_LRA=99.00" in second and "offset=-99.00" in second


def test_the_ebur128_summary_is_read():
    assert audio.ebur128_summary("t: 0.1 M: -20 S: -20 I: -30.0 LUFS\n" + EBUR128_LOG) == {
        "integrated_lufs": -14.2,
        "lra": 2.1,
        "true_peak_dbfs": -1.4,
    }


# ── real ffmpeg, real Kokoro: the loudness target ────────────────────────────


def _mixed(plan: audio.MixPlan, output: Path) -> dict:
    settings = load_settings()
    report = audio.run_mix(plan, output, ffmpeg_bin=settings.ffmpeg_bin, timeout_seconds=settings.mix_timeout_seconds)
    measured = audio.measure_loudness(output, ffmpeg_bin=settings.ffmpeg_bin, timeout_seconds=settings.mix_timeout_seconds)
    print(f"loudnorm report {report}; ebur128 {measured}")
    return {"report": report, "measured": measured, "info": soundfile.info(str(output))}


def test_a_spoken_line_mixes_to_minus_14_lufs(tmp_path):
    # The fixture's own audio: one Kokoro line at 0.3 s in a 3.0 s composition,
    # shorter than loudnorm's 3 s window without the padding.
    line = synthesize_line("Rendered on brand, in three seconds.", tmp_path / "l01.wav", load_settings())
    mixed = _mixed(audio.MixPlan(duration=3.0, voice=(audio.VoiceTrack(line.path, 0.3),)), tmp_path / "mix.wav")
    assert abs(mixed["measured"]["integrated_lufs"] - audio.LOUDNESS_TARGET_LUFS) <= TOLERANCE_LU
    assert mixed["measured"]["true_peak_dbfs"] <= -1.0
    assert (mixed["info"].samplerate, mixed["info"].channels) == (48000, 2)
    assert abs(mixed["info"].duration - 3.0) < 0.01


def _music_bed(path: Path, seconds: float) -> Path:
    rate = 48000
    t = np.arange(int(seconds * rate)) / rate
    chord = sum(0.12 * np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63))
    beat = t % 0.5
    kick = 0.5 * np.sin(2 * np.pi * 55.0 * t) * np.exp(-beat * 18.0)
    bed = (chord + kick).astype(np.float32)
    soundfile.write(str(path), np.stack([bed, bed], axis=1), rate)
    return path


def _click(path: Path) -> Path:
    rate = 48000
    t = np.arange(int(0.08 * rate)) / rate
    burst = (0.8 * np.sin(2 * np.pi * 1800.0 * t) * np.exp(-t * 60.0)).astype(np.float32)
    soundfile.write(str(path), burst, rate)
    return path


def test_music_voice_and_sfx_mix_to_minus_14_lufs(tmp_path):
    settings = load_settings()
    first = synthesize_line("Orders. Stock. Customers. The books.", tmp_path / "l01.wav", settings)
    second = synthesize_line("The right agent takes it.", tmp_path / "l02.wav", settings)
    plan = audio.MixPlan(
        duration=10.0,
        voice=(audio.VoiceTrack(first.path, 0.5), audio.VoiceTrack(second.path, 5.0)),
        music=audio.MusicTrack(_music_bed(tmp_path / "bed.wav", 14.0), start=2.0),
        sfx=(audio.SfxTrack(_click(tmp_path / "click.wav"), 4.2, 0.6),),
    )
    mixed = _mixed(plan, tmp_path / "mix.wav")
    assert abs(mixed["measured"]["integrated_lufs"] - audio.LOUDNESS_TARGET_LUFS) <= TOLERANCE_LU
    assert mixed["measured"]["true_peak_dbfs"] <= -1.0
    assert abs(mixed["info"].duration - 10.0) < 0.01


def test_a_silent_plan_writes_a_silent_track(tmp_path):
    settings = load_settings()
    report = audio.run_mix(
        audio.MixPlan(duration=2.0), tmp_path / "mix.wav", ffmpeg_bin=settings.ffmpeg_bin, timeout_seconds=60
    )
    samples, rate = soundfile.read(str(tmp_path / "mix.wav"))
    assert report == {"silent": True}
    assert rate == 48000 and samples.shape == (96000, 2) and not samples.any()


# ── word timing ──────────────────────────────────────────────────────────────


def _bursts(spans, rate=24000, total=1.5):
    samples = np.zeros(int(total * rate), dtype=np.float32)
    for start, end in spans:
        t = np.arange(int(start * rate), int(end * rate))
        samples[t] = 0.5 * np.sin(2 * np.pi * 200.0 * t / rate)
    return samples, rate


def test_segments_split_at_gaps_of_70_ms_or_more():
    samples, rate = _bursts([(0.10, 0.40), (0.45, 0.60), (0.80, 1.20)])
    segments = voiced_segments(samples, rate)
    # The 50 ms pause stays inside a segment; the 200 ms one splits.
    assert len(segments) == 2
    (a_start, a_end), (b_start, b_end) = segments
    assert abs(a_start - 0.10) <= 0.02 and abs(a_end - 0.60) <= 0.02
    assert abs(b_start - 0.80) <= 0.02 and abs(b_end - 1.20) <= 0.02


def test_silence_has_no_segments():
    assert voiced_segments(np.zeros(24000, dtype=np.float32), 24000) == []
    assert voiced_segments(np.zeros(0, dtype=np.float32), 24000) == []
