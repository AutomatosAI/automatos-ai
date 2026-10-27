"""Boot refuses a container that could not speak, and the image itself boots."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import pytest

from voice_tts.boot import BOOT_FAILURE_EXIT_CODE, boot_problems


def test_the_image_passes_its_boot_checks(settings):
    assert boot_problems(settings) == []


@pytest.mark.parametrize("environment", ["production", "development", "staging", ""])
def test_no_token_is_refused_in_every_environment(settings, environment):
    problems = boot_problems(replace(settings, environment=environment, token=""))

    assert any("VOICE_TTS_TOKEN is not set" in problem for problem in problems)


def test_no_token_boots_only_when_a_developer_allowed_it(settings):
    assert boot_problems(replace(settings, token="", allow_no_token=True)) == []


def test_a_missing_model_voices_file_default_voice_or_encoder_is_refused(settings, tmp_path):
    problems = boot_problems(replace(
        settings,
        kokoro_model_path=str(tmp_path / "missing.onnx"),
        default_voice="af_nobody",
        lame_bin="/nonexistent/lame",
    ))

    assert any("Kokoro model" in problem for problem in problems)
    assert any("default voice" in problem for problem in problems)
    assert any("LAME" in problem for problem in problems)


def test_an_espeak_path_espeak_ng_would_truncate_is_refused(settings, tmp_path):
    long = tmp_path / ("e" * 170)
    long.mkdir()
    (long / "phontab").write_bytes(b"")

    problems = boot_problems(replace(settings, espeak_data_path=str(long)))

    assert any("truncates" in problem for problem in problems)


def test_the_entrypoint_exits_2_on_a_boot_problem():
    env = {**os.environ, "ENVIRONMENT": "development", "VOICE_TTS_TOKEN": "", "LOG_RELAY_ENABLED": "false"}
    env.pop("VOICE_TTS_ALLOW_NO_TOKEN", None)

    result = subprocess.run(
        [sys.executable, "-m", "voice_tts", "boot-check"], capture_output=True, text=True, env=env, timeout=60,
        cwd="/app",
    )

    assert result.returncode == BOOT_FAILURE_EXIT_CODE
    assert "VOICE_TTS_TOKEN is not set" in result.stderr
