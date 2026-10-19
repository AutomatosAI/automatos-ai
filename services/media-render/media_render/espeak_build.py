"""The espeak-ng data the image keeps (PRD-262 US-052): the languages the product speaks.

    python media_render/espeak_build.py /opt/espeak /opt/media-render/espeak-phonemes.json

The Dockerfile's py stage runs it in the RUN that installs the lock and moves the
espeak-ng data to ``/opt/espeak``, so nothing it deletes is left in a lower layer.

Kokoro speaks the ``LANGUAGES`` below and no other. The orchestrator sends en-us for
an American voice and en-gb for a British one
(``orchestrator/modules/socials/kokoro_voices.py``), and the service refuses any
other language (``validate.language``, and ``MEDIA_RENDER_KOKORO_LANG`` at boot).
espeak-ng reads the words of both from one dictionary, ``en_dict``. The package
ships 114 dictionaries (17.63 MB); the other 113 (17.46 MB) are deleted. The
phoneme tables, the intonations and the language and voice definitions stay.

The cut is proved in the same build:

1. Kokoro's phonemes for ``SENTENCE`` are taken in each language with the data as
   the package ships it, the way Kokoro takes them (kokoro-onnx's ``Tokenizer``,
   given the data path), each time in a new interpreter, because espeak-ng keeps
   process-wide state;
2. every dictionary but those the languages read is deleted;
3. the phonemes are taken again. The build fails unless they are the same as
   before, and none is empty;
4. they are written to the snapshot file. The image's own suite takes them once more
   through the service's settings and must get the same (``tests/test_kokoro.py``).

English text reads the same. What changes is a word in another alphabet inside a
line (Cyrillic, Greek, Chinese): espeak-ng no longer has that language's dictionary.

This file runs on its own, before the package is copied into the image, so it
imports nothing from media_render; kokoro-onnx is imported only by the phonemizing
child. The service reads ``LANGUAGES`` from here.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import AbstractSet, Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

LANGUAGES = ("en-us", "en-gb")
# Numbers, a currency, a percentage, a time, an abbreviation, a possessive, an
# accented loanword and words the two languages say differently.
SENTENCE = (
    "On 3 March, Dr Lee's team shipped 1,250 orders at $4.99 each, up 12% on last year; "
    "the new schedule starts at 9:30, with a tomato tasting at the café."
)
DICTIONARY_SUFFIX = "_dict"
PHONEMIZE_TIMEOUT_SECONDS = 180

logger = logging.getLogger("media_render.espeak_build")


class EspeakBuildError(RuntimeError):
    """The espeak-ng data cannot be cut down without changing what Kokoro says."""


def dictionaries_for(languages: Sequence[str] = LANGUAGES) -> FrozenSet[str]:
    """The dictionaries espeak-ng reads the languages' words from: a language's base, so en_dict for en-us and en-gb."""
    return frozenset(language.split("-", 1)[0] + DICTIONARY_SUFFIX for language in languages)


def prune(data: Path, keep: AbstractSet[str]) -> Tuple[Tuple[str, ...], int]:
    """Delete every dictionary in ``data`` but ``keep``: the names deleted, and their bytes."""
    missing = sorted(name for name in keep if not (data / name).is_file())
    if missing:
        raise EspeakBuildError(f"{data} has no {', '.join(missing)}")
    dropped = sorted(
        path for path in data.iterdir()
        if path.is_file() and path.name.endswith(DICTIONARY_SUFFIX) and path.name not in keep
    )
    size = sum(path.stat().st_size for path in dropped)
    for path in dropped:
        path.unlink()
    return tuple(path.name for path in dropped), size


def _phonemize_here(data: Path) -> Dict[str, str]:
    """Kokoro's phonemes for SENTENCE in each language, in this interpreter (the child's side)."""
    import espeakng_loader
    from kokoro_onnx.config import EspeakConfig
    from kokoro_onnx.tokenizer import Tokenizer

    tokenizer = Tokenizer(EspeakConfig(lib_path=espeakng_loader.get_library_path(), data_path=str(data)))
    return {language: tokenizer.phonemize(SENTENCE, language) for language in LANGUAGES}


def phonemes(data: Path) -> Dict[str, str]:
    """Kokoro's phonemes for SENTENCE in each language with the espeak-ng data at ``data``, from a new interpreter."""
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--phonemize", str(data)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=PHONEMIZE_TIMEOUT_SECONDS,
        check=False,
    )
    if child.returncode != 0:
        raise EspeakBuildError(f"phonemizing with {data} failed (exit {child.returncode}): {child.stderr[-2000:]}")
    lines = [line for line in child.stdout.splitlines() if line.strip()]
    try:
        taken = json.loads(lines[-1]) if lines else {}
    except ValueError:
        raise EspeakBuildError(f"phonemizing with {data} printed no phonemes: {child.stdout[-2000:]}") from None
    empty = [language for language in LANGUAGES if not taken.get(language)]
    if empty:
        raise EspeakBuildError(f"phonemizing with {data} gave no phonemes in {', '.join(empty)}")
    return {language: taken[language] for language in LANGUAGES}


def build(data: Path, snapshot: Path) -> Dict[str, Any]:
    """Cut the data at ``data`` to the languages' dictionaries, proving Kokoro's phonemes unchanged: the snapshot."""
    keep = dictionaries_for()
    before = phonemes(data)
    dropped, size = prune(data, keep)
    after = phonemes(data)
    changed = [language for language in LANGUAGES if after[language] != before[language]]
    if changed:
        detail = "; ".join(f"{language}: {before[language]!r} -> {after[language]!r}" for language in changed)
        raise EspeakBuildError(f"deleting the other dictionaries changed Kokoro's phonemes ({detail})")
    record = {
        "about": "Kokoro's phonemes for the sentence, taken with the espeak-ng data as its package ships it and "
        "again after the other dictionaries were deleted (PRD-262 US-052, media_render/espeak_build.py).",
        "sentence": SENTENCE,
        "dictionaries": sorted(keep),
        "phonemes": after,
    }
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("espeak-ng: kept %s; deleted %d other dictionaries, %d bytes", ", ".join(sorted(keep)), len(dropped), size)
    for language in LANGUAGES:
        logger.info("espeak-ng: %s phonemes unchanged: %s", language, after[language])
    return record


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Keep the espeak-ng data Kokoro's languages need (PRD-262 US-052).")
    parser.add_argument("data", type=Path, help="the espeak-ng data directory, /opt/espeak")
    parser.add_argument("snapshot", type=Path, nargs="?", help="where the phonemes are written")
    parser.add_argument("--phonemize", action="store_true", help="print Kokoro's phonemes for the sentence as JSON, and stop")
    args = parser.parse_args(argv)
    if args.phonemize:
        # ASCII escapes: the parent reads the IPA back whatever the locale's encoding.
        sys.stdout.write(json.dumps(_phonemize_here(args.data)) + "\n")
        return 0
    if args.snapshot is None:
        parser.error("the snapshot path is required")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        build(args.data, args.snapshot)
    except (EspeakBuildError, OSError, subprocess.TimeoutExpired) as exc:
        logger.error("espeak-ng: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
