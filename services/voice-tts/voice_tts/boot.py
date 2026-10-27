"""Boot assertions: the container refuses to start in a state that could not speak.

``python -m voice_tts`` runs these before the server. Each check returns
plain-language problems; any problem fails the boot with exit code 2, so the
container stops instead of answering every request with an error.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Sequence

from .config import TOKEN_ENV, Settings
from .voices import VoicesError, load_catalogue

BOOT_FAILURE_EXIT_CODE = 2
# espeak-ng truncates its data path at about 160 characters, and phonemizer resolves
# symlinks first, so the resolved path must stay under this (media-render's trap, PRD-251).
ESPEAK_DATA_PATH_LIMIT = 160


class BootError(RuntimeError):
    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("voice-tts refused to boot: " + "; ".join(self.problems))


def token_problems(settings: Settings) -> List[str]:
    """Production never speaks for anyone who asks; development may."""
    if settings.is_production and not settings.token:
        return [f"{TOKEN_ENV} is not set, and production refuses an unauthenticated voice"]
    return []


def model_problems(settings: Settings) -> List[str]:
    if not Path(settings.kokoro_model_path).is_file():
        return [f"the Kokoro model {settings.kokoro_model_path!r} is missing (VOICE_TTS_KOKORO_MODEL)"]
    return []


def voice_problems(settings: Settings) -> List[str]:
    """The voices file lists its voices, and the default voice is one of them."""
    try:
        load_catalogue(settings.kokoro_voices_path, settings.default_voice)
    except VoicesError as exc:
        return [f"VOICE_TTS_KOKORO_VOICES / VOICE_TTS_DEFAULT_VOICE: {exc}"]
    return []


def espeak_problems(path: str) -> List[str]:
    real = os.path.realpath(path)
    longest = max(len(path), len(real))
    if longest >= ESPEAK_DATA_PATH_LIMIT:
        return [
            f"the espeak-ng data path is {longest} characters (resolved: {real!r}); espeak-ng truncates it at "
            f"about {ESPEAK_DATA_PATH_LIMIT}, so use a short real path such as /opt/espeak"
        ]
    if not Path(real, "phontab").is_file():
        return [f"no espeak-ng data at {real!r} (phontab is missing)"]
    return []


def encoder_problems(settings: Settings) -> List[str]:
    if shutil.which(settings.lame_bin) is None:
        return [f"LAME ({settings.lame_bin!r}) is not installed, so no MP3 could be made"]
    return []


def boot_problems(settings: Settings) -> List[str]:
    return [
        *token_problems(settings),
        *model_problems(settings),
        *voice_problems(settings),
        *espeak_problems(settings.espeak_data_path),
        *encoder_problems(settings),
    ]


def assert_boot_environment(settings: Settings) -> None:
    problems = boot_problems(settings)
    if problems:
        raise BootError(problems)
