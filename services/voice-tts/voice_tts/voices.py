"""The voices this image holds, and what each one is, read from Kokoro's voice ids.

``voices-v1.0.bin`` is an npz: a zip with one ``<voice>.npy`` per voice. Kokoro names
each voice ``<language><gender>_<name>``: ``af_heart`` is American English, female,
"Heart"; ``bm_george`` British English, male, "George". The first letter is also the
language the text is phonemized in (espeak-ng's code, ``LANGUAGES``).

``GET /voices`` lists them English first (American, then British), then the others by
language, each group by id. The listing is read once, at boot.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, NamedTuple, Tuple

NPY = ".npy"
VOICE_ID = re.compile(r"^(?P<language>[a-z])(?P<gender>[fm])_(?P<name>[a-z0-9]{1,32})$")
GENDERS = {"f": "female", "m": "male"}


class Language(NamedTuple):
    accent: str
    espeak: str


# Kokoro's language letters, in listing order: English first. ``accent`` is the BCP 47 tag
# the listing shows; ``espeak`` the language espeak-ng phonemizes the text in.
LANGUAGES: Mapping[str, Language] = {
    "a": Language("en-US", "en-us"),
    "b": Language("en-GB", "en-gb"),
    "e": Language("es-ES", "es"),
    "f": Language("fr-FR", "fr-fr"),
    "h": Language("hi-IN", "hi"),
    "i": Language("it-IT", "it"),
    "j": Language("ja-JP", "ja"),
    "p": Language("pt-BR", "pt-br"),
    "z": Language("zh-CN", "cmn"),
}
_ORDER = {letter: index for index, letter in enumerate(LANGUAGES)}


class VoicesError(ValueError):
    """A voices file cannot be used."""


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    accent: str
    gender: str
    espeak: str

    def listing(self) -> Dict[str, str]:
        return {"id": self.id, "name": self.name, "accent": self.accent, "gender": self.gender}


def describe(voice_id: str) -> Voice:
    """What a Kokoro voice id says about its voice. Raises VoicesError for an id that breaks the convention."""
    match = VOICE_ID.match(voice_id)
    language = LANGUAGES.get(match.group("language")) if match else None
    if match is None or language is None:
        raise VoicesError(f"{voice_id!r} is not a Kokoro voice id (<language><gender>_<name>)")
    return Voice(
        id=voice_id,
        name=match.group("name").capitalize(),
        accent=language.accent,
        gender=GENDERS[match.group("gender")],
        espeak=language.espeak,
    )


def voice_ids(path: str) -> FrozenSet[str]:
    """The voice ids a Kokoro voices file holds: the names of its ``<voice>.npy`` members."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise VoicesError(f"the Kokoro voices file {path} cannot be read: {exc}") from None
    held = frozenset(name[: -len(NPY)] for name in names if name.endswith(NPY))
    if not held:
        raise VoicesError(f"the Kokoro voices file {path} holds no voices")
    return held


def _sort_key(voice: Voice) -> Tuple[int, str]:
    return _ORDER[voice.id[0]], voice.id


@dataclass(frozen=True)
class Catalogue:
    """The voices the service speaks with, and the one it uses when a request names none."""

    default: str
    voices: Tuple[Voice, ...]

    def get(self, voice_id: str) -> Voice:
        """The voice, or KeyError when this image does not hold it."""
        for voice in self.voices:
            if voice.id == voice_id:
                return voice
        raise KeyError(voice_id)

    def body(self) -> Dict[str, Any]:
        """``GET /voices``'s answer."""
        return {"default": self.default, "voices": [voice.listing() for voice in self.voices]}


def load_catalogue(path: str, default: str) -> Catalogue:
    """The voices file's voices, listed English first, and the default, which must be one of them.

    A voice whose id breaks Kokoro's convention is left out: the service could not tell which
    language to speak it in. Raises VoicesError when none is left or the default is missing.
    """
    known: List[Voice] = []
    for voice_id in voice_ids(path):
        try:
            known.append(describe(voice_id))
        except VoicesError:
            continue
    if not known:
        raise VoicesError(f"the Kokoro voices file {path} holds no voice the service can speak with")
    if default not in {voice.id for voice in known}:
        raise VoicesError(f"the default voice {default!r} is not in the voices file {path}")
    return Catalogue(default=default, voices=tuple(sorted(known, key=_sort_key)))
