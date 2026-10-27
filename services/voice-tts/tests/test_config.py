"""The settings: the image's defaults, the container's CPU quota, and refusals that name the variable."""

from __future__ import annotations

import pytest

from voice_tts.config import ConfigError, available_cpus, load_settings


def _cpu_max(tmp_path, text: str):
    path = tmp_path / "cpu.max"
    path.write_text(text, encoding="ascii")
    return path


def test_the_defaults_describe_the_image(tmp_path):
    settings = load_settings({}, cpus=4)

    assert (settings.port, settings.bind_host, settings.default_voice) == (8100, "0.0.0.0", "af_heart")
    assert settings.kokoro_model_path == "/opt/kokoro/kokoro-v1.0.onnx"
    assert settings.kokoro_voices_path == "/opt/kokoro/voices-v1.0.bin"
    assert settings.espeak_data_path == "/opt/espeak"
    assert (settings.max_chars, settings.mp3_kbps, settings.lame_bin) == (500, 64, "lame")
    assert (settings.max_concurrent, settings.threads) == (4, 4)
    assert settings.token == "" and not settings.is_production


def test_the_token_never_reaches_a_repr():
    settings = load_settings({"VOICE_TTS_TOKEN": "  s3cret-voice-token  "}, cpus=1)

    assert settings.token == "s3cret-voice-token"
    assert "s3cret-voice-token" not in repr(settings)


def test_the_concurrency_and_threads_can_be_set():
    settings = load_settings({"VOICE_TTS_MAX_CONCURRENT": "3", "VOICE_TTS_THREADS": "2"}, cpus=8)

    assert (settings.max_concurrent, settings.threads) == (3, 2)


@pytest.mark.parametrize("name, value", [
    ("VOICE_TTS_MAX_CONCURRENT", "0"),
    ("VOICE_TTS_MAX_CHARS", "-5"),
    ("VOICE_TTS_PORT", "70000"),
    ("VOICE_TTS_MP3_KBPS", "65"),
    ("VOICE_TTS_SENTENCE_PAUSE_SECONDS", "lots"),
])
def test_a_bad_value_is_refused_naming_its_variable(name, value):
    with pytest.raises(ConfigError, match=name):
        load_settings({name: value}, cpus=2)


def test_the_cpu_count_is_the_containers_quota_rounded_up(tmp_path):
    assert available_cpus(_cpu_max(tmp_path, "200000 100000\n"), cpu_count=32) == 2
    assert available_cpus(_cpu_max(tmp_path, "150000 100000\n"), cpu_count=32) == 2
    assert available_cpus(_cpu_max(tmp_path, "50000 100000\n"), cpu_count=32) == 1


def test_no_quota_or_no_cgroup_file_is_the_machines_cpus(tmp_path):
    assert available_cpus(_cpu_max(tmp_path, "max 100000\n"), cpu_count=6) == 6
    assert available_cpus(tmp_path / "absent", cpu_count=6) == 6


def test_a_quota_above_the_machine_is_held_to_the_machine(tmp_path):
    assert available_cpus(_cpu_max(tmp_path, "800000 100000\n"), cpu_count=4) == 4


def test_production_is_named_by_environment():
    assert load_settings({"ENVIRONMENT": "Production"}, cpus=1).is_production
