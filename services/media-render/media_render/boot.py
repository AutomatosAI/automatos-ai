"""Boot assertions: the container refuses to start in a state that renders wrong.

``python -m media_render`` runs these before the server or any job. Each check
returns plain-language problems; any problem fails the boot with exit code 2,
so the container stops instead of serving broken renders.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Mapping, Sequence

from .config import ESPEAK_DATA_PATH_LIMIT, HYPERFRAMES_OFF_SWITCHES, TOKEN_ENV, Settings
from .music import MusicLibraryError, load_library
from .signing_keys import configured_problems
from .voices_build import VoicesError, voice_names

BOOT_FAILURE_EXIT_CODE = 2


class BootError(RuntimeError):
    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("media-render refused to boot: " + "; ".join(self.problems))


def espeak_data_path_problems(path: str) -> List[str]:
    """espeak-ng truncates its data path at about 160 characters (PRD-251 Traps).

    phonemizer resolves symlinks before handing the path over, so the REAL path
    is what counts: a short symlink to a long directory fails the same way.
    """
    real = os.path.realpath(path)
    longest = max(len(path), len(real))
    if longest >= ESPEAK_DATA_PATH_LIMIT:
        return [
            f"the espeak-ng data path is {longest} characters (resolved: {real!r}); espeak-ng "
            f"truncates it at about {ESPEAK_DATA_PATH_LIMIT}, so copy the data to a short real "
            "path such as /opt/espeak"
        ]
    if not Path(real, "phontab").is_file():
        return [f"no espeak-ng data at {real!r} (phontab is missing)"]
    return []


def hyperframes_env_problems(env: Mapping[str, str]) -> List[str]:
    return [
        f"{name} must be 1 (Hyperframes telemetry, skill installs and update checks stay off)"
        for name in HYPERFRAMES_OFF_SWITCHES
        if (env.get(name) or "").strip() != "1"
    ]


def token_problems(settings: Settings) -> List[str]:
    """Production never runs an unauthenticated renderer; development may."""
    if settings.is_production and not settings.internal_token:
        return [f"{TOKEN_ENV} is not set, and production refuses an unauthenticated renderer"]
    return []


def music_library_problems(settings: Settings) -> List[str]:
    """A broken music manifest stops the container; an absent one is an empty library."""
    try:
        load_library(settings.music_dir)
    except MusicLibraryError as exc:
        return [f"the music library at {settings.music_dir} cannot be used: {exc}"]
    return []


def kokoro_voice_problems(settings: Settings) -> List[str]:
    """The voices file lists its voices, and the default voice is one of them (PRD-262 US-052).

    A request names only a voice the file holds (validate.voice_name), so a default
    the file lacks would refuse every request that names none.
    """
    try:
        voices = voice_names(settings.kokoro_voices_path)
    except VoicesError as exc:
        return [f"MEDIA_RENDER_KOKORO_VOICES: {exc}"]
    if settings.kokoro_voice not in voices:
        return [
            f"MEDIA_RENDER_KOKORO_VOICE={settings.kokoro_voice!r} is not in the voices file "
            f"{settings.kokoro_voices_path}, which holds {', '.join(sorted(voices))}"
        ]
    return []


def marker_problems(settings: Settings) -> List[str]:
    """PRD-264 EUAI-06a: without exiftool no AI-made file can be marked, and an unmarked one never ships."""
    if shutil.which(settings.exiftool_bin) is None:
        return [f"exiftool ({settings.exiftool_bin!r}) is not installed, so no AI-made file could be marked"]
    return []


def signing_problems(settings: Settings) -> List[str]:
    """PRD-264 EUAI-06b: a configured C2PA chain and key that could sign nothing stop the container.

    None configured is no problem: this install makes its own on first use (signing_keys.py)."""
    return configured_problems(settings)


def boot_problems(settings: Settings, env: Mapping[str, str]) -> List[str]:
    return [
        *espeak_data_path_problems(settings.espeak_data_path),
        *hyperframes_env_problems(env),
        *token_problems(settings),
        *music_library_problems(settings),
        *kokoro_voice_problems(settings),
        *marker_problems(settings),
        *signing_problems(settings),
    ]


def assert_boot_environment(settings: Settings, env: Mapping[str, str]) -> None:
    problems = boot_problems(settings, env)
    if problems:
        raise BootError(problems)
