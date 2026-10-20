"""The boot assertions (US-101): the espeak path trap, the Hyperframes switches,
and production's token rule. Pure functions plus one real `boot-check` process.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from media_render import boot
from media_render.config import ESPEAK_DATA_PATH_LIMIT, HYPERFRAMES_OFF_SWITCHES, TOKEN_ENV, ConfigError, load_settings

ROOT = Path(__file__).resolve().parents[1]
SWITCHES_ON = {name: "1" for name in HYPERFRAMES_OFF_SWITCHES}


def _espeak_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "phontab").write_bytes(b"\0")
    return path


def _path_of_length(base: Path, length: int) -> Path:
    head = str(base.resolve()) + "/"
    assert len(head) < length
    return Path(head + "e" * (length - len(head)))


def test_a_short_real_espeak_path_passes(tmp_path):
    assert boot.espeak_data_path_problems(str(_espeak_dir(tmp_path / "espeak"))) == []


def test_a_path_of_160_characters_is_refused(tmp_path):
    path = _path_of_length(tmp_path, ESPEAK_DATA_PATH_LIMIT)
    assert len(str(path)) == 160
    problems = boot.espeak_data_path_problems(str(path))
    assert len(problems) == 1 and "160 characters" in problems[0]


def test_159_characters_is_short_enough(tmp_path):
    path = _espeak_dir(_path_of_length(tmp_path, ESPEAK_DATA_PATH_LIMIT - 1))
    assert boot.espeak_data_path_problems(str(path)) == []


def test_a_short_symlink_to_a_long_real_path_is_refused(tmp_path):
    # phonemizer resolves symlinks, so the resolved length is the one espeak sees.
    real = _espeak_dir(_path_of_length(tmp_path / "real", ESPEAK_DATA_PATH_LIMIT + 5))
    link = tmp_path / "l"
    link.symlink_to(real)
    assert len(str(link)) < ESPEAK_DATA_PATH_LIMIT
    assert boot.espeak_data_path_problems(str(link))


def test_a_directory_without_espeak_data_is_refused(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    problems = boot.espeak_data_path_problems(str(empty))
    assert len(problems) == 1 and "phontab" in problems[0]


def test_every_hyperframes_switch_is_required():
    assert boot.hyperframes_env_problems(SWITCHES_ON) == []
    for name in HYPERFRAMES_OFF_SWITCHES:
        missing = {k: v for k, v in SWITCHES_ON.items() if k != name}
        assert any(name in problem for problem in boot.hyperframes_env_problems(missing))
        assert any(name in problem for problem in boot.hyperframes_env_problems({**SWITCHES_ON, name: "0"}))


def test_production_refuses_to_boot_without_a_token():
    assert boot.token_problems(load_settings({"ENVIRONMENT": "production"}))
    assert boot.token_problems(load_settings({"ENVIRONMENT": "production", TOKEN_ENV: "s3cret"})) == []
    assert boot.token_problems(load_settings({})) == []


def test_boot_problems_collects_every_failure(tmp_path):
    settings = load_settings(
        {"ENVIRONMENT": "production", "MEDIA_RENDER_ESPEAK_DATA_PATH": str(_path_of_length(tmp_path, 170))}
    )
    problems = boot.boot_problems(settings, {})
    assert len(problems) == 1 + len(HYPERFRAMES_OFF_SWITCHES) + 1
    try:
        boot.assert_boot_environment(settings, {})
    except boot.BootError as exc:
        assert exc.problems == problems
    else:
        raise AssertionError("assert_boot_environment did not raise")


def test_a_missing_exiftool_stops_the_container(tmp_path):
    """PRD-264 EUAI-06a: the AI mark's writer is part of a working image."""
    assert boot.marker_problems(load_settings({})) == []
    missing = load_settings({"MEDIA_RENDER_EXIFTOOL_BIN": str(tmp_path / "no-exiftool")})
    problems = boot.marker_problems(missing)
    assert len(problems) == 1 and "exiftool" in problems[0]
    assert problems[0] in boot.boot_problems(missing, {})


def test_a_broken_music_manifest_stops_the_container(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    settings = load_settings({"MEDIA_RENDER_MUSIC_DIR": str(music)})
    assert boot.music_library_problems(settings) == [], "no manifest is an empty library"
    (music / "manifest.json").write_text('{"tracks": [{"id": "gone", "file": "gone.mp3"}]}')
    problems = boot.music_library_problems(settings)
    assert len(problems) == 1 and "gone.mp3" in problems[0]
    (music / "manifest.json").write_text('{"tracks": [{"id": "up", "file": "../outside.mp3"}]}')
    assert "outside the library" in boot.music_library_problems(settings)[0]


def test_the_container_command_exits_2_on_a_long_espeak_path():
    env = {**os.environ, "MEDIA_RENDER_ESPEAK_DATA_PATH": "/opt/" + "e" * 160, "LOG_RELAY_ENABLED": "false"}
    proc = subprocess.run(
        [sys.executable, "-m", "media_render", "boot-check"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == boot.BOOT_FAILURE_EXIT_CODE, proc.stderr
    assert "espeak-ng data path is 165 characters" in proc.stderr


# ── PRD-262 US-052: the voices and the language follow the image's data ──────


def _boot_check(**env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "media_render", "boot-check"],
        cwd=ROOT,
        env={**os.environ, "LOG_RELAY_ENABLED": "false", **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_the_voices_file_and_the_default_voice_are_checked_at_boot(tmp_path):
    assert boot.kokoro_voice_problems(load_settings({})) == [], "the image's voices file holds af_heart"
    (problem,) = boot.kokoro_voice_problems(load_settings({"MEDIA_RENDER_KOKORO_VOICES": str(tmp_path / "none.bin")}))
    assert problem.startswith("MEDIA_RENDER_KOKORO_VOICES: the Kokoro voices file") and "cannot be read" in problem
    empty = tmp_path / "empty.bin"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("README.txt", "no voices here")
    (problem,) = boot.kokoro_voice_problems(load_settings({"MEDIA_RENDER_KOKORO_VOICES": str(empty)}))
    assert problem.endswith("holds no voices"), problem
    # am_santa is in upstream's file, not in the image's: a default every request would be refused with.
    (problem,) = boot.kokoro_voice_problems(load_settings({"MEDIA_RENDER_KOKORO_VOICE": "am_santa"}))
    assert problem.startswith("MEDIA_RENDER_KOKORO_VOICE='am_santa' is not in the voices file /opt/kokoro/"), problem
    proc = _boot_check(MEDIA_RENDER_KOKORO_VOICE="am_santa")
    assert proc.returncode == boot.BOOT_FAILURE_EXIT_CODE and "MEDIA_RENDER_KOKORO_VOICE='am_santa'" in proc.stderr, proc.stderr


def test_a_language_the_image_does_not_speak_stops_the_boot():
    for lang in ("fr-fr", "en", "EN-US"):
        expected = f"MEDIA_RENDER_KOKORO_LANG={lang!r} is not valid: expected one of en-us, en-gb"
        with pytest.raises(ConfigError, match=re.escape(expected)):
            load_settings({"MEDIA_RENDER_KOKORO_LANG": lang})
    assert load_settings({"MEDIA_RENDER_KOKORO_LANG": "en-gb"}).kokoro_lang == "en-gb"
    proc = _boot_check(MEDIA_RENDER_KOKORO_LANG="fr-fr")
    assert proc.returncode == boot.BOOT_FAILURE_EXIT_CODE, proc.stderr
    assert "MEDIA_RENDER_KOKORO_LANG='fr-fr' is not valid" in proc.stderr
