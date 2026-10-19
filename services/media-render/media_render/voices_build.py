"""The Kokoro voices the image keeps (PRD-262 US-052): the catalogue the platform offers.

    python media_render/voices_build.py media_render/voices.json /opt/kokoro/voices-v1.0.bin

The Dockerfile's kokoro stage runs it right after the sha256 check of the upstream
files. ``voices-v1.0.bin`` is an npz: a zip holding one ``<voice>.npy`` per voice,
stored, not compressed. Upstream's holds 54 voices in nine languages. The platform
offers the 27 English ones its catalogue lists
(``orchestrator/modules/socials/kokoro_voices.py``), whose ids ``voices.json`` copies
because the build context cannot read the orchestrator. The file is rewritten in
place to those voices alone, about 14 MB less:

- each member is copied byte for byte, with its name, date and attributes, in the
  upstream file's order, so the same input always gives the same file; the file
  keeps the upstream file's mode and times;
- the build fails when the catalogue names a voice the upstream file lacks, and
  when the rewritten file does not hold exactly the catalogue or fails its CRC check.

The service reads the names back at boot (``voice_names``) and refuses any other
voice (``validate.voice_name``). This file runs on its own, before the package is
copied into the image, so it imports nothing from media_render.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import AbstractSet, FrozenSet, List, NamedTuple, Optional, Sequence, Tuple, Union

CATALOGUE = Path(__file__).with_name("voices.json")
NPY = ".npy"
# Kokoro's voice ids: the language and gender, then the name (af_heart, bm_george).
VOICE_ID = re.compile(r"^[a-z]{2}_[a-z0-9]{1,32}$")

logger = logging.getLogger("media_render.voices_build")


class VoicesError(ValueError):
    """The catalogue or a voices file cannot be used."""


class Rewritten(NamedTuple):
    held: int
    kept: int
    bytes_before: int
    bytes_after: int


def load_catalogue(path: Path = CATALOGUE) -> Tuple[str, ...]:
    """The voice ids ``voices.json`` lists, in its order: each a Kokoro voice id, each once."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VoicesError(f"{path} cannot be read: {exc}") from None
    voices = data.get("voices") if isinstance(data, dict) else None
    if not isinstance(voices, list) or not voices:
        raise VoicesError(f"{path} must list the voice ids under 'voices'")
    wrong = [voice for voice in voices if not isinstance(voice, str) or not VOICE_ID.match(voice)]
    if wrong:
        raise VoicesError(f"{path} lists ids that are not Kokoro voice ids: {wrong}")
    if len(set(voices)) != len(voices):
        raise VoicesError(f"{path} lists a voice more than once")
    return tuple(voices)


def voice_names(path: Union[str, Path]) -> FrozenSet[str]:
    """The voices a Kokoro voices file holds: the names of its ``<voice>.npy`` members."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise VoicesError(f"the Kokoro voices file {path} cannot be read: {exc}") from None
    voices = frozenset(name[: -len(NPY)] for name in names if name.endswith(NPY))
    if not voices:
        raise VoicesError(f"the Kokoro voices file {path} holds no voices")
    return voices


def _copy_of(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """A new entry with the member's name, date and attributes, stored as upstream stores every voice."""
    entry = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    entry.compress_type = zipfile.ZIP_STORED
    entry.external_attr = info.external_attr
    return entry


def _check(path: Path, wanted: AbstractSet[str]) -> None:
    held = voice_names(path)
    if held != wanted:
        raise VoicesError(f"the rewritten {path.name} holds {sorted(held)}, not the catalogue's {sorted(wanted)}")
    with zipfile.ZipFile(path) as archive:
        broken = archive.testzip()
    if broken is not None:
        raise VoicesError(f"the rewritten {path.name} fails its CRC check at {broken}")


def rewrite(path: Path, keep: Sequence[str]) -> Rewritten:
    """Rewrite the voices file at ``path`` in place to hold the voices ``keep`` names, and nothing else."""
    wanted = frozenset(keep)
    held = voice_names(path)
    missing = sorted(wanted - held)
    if missing:
        raise VoicesError(f"{path} has no {', '.join(missing)}: the catalogue names voices the upstream file lacks")
    bytes_before = path.stat().st_size
    staged = path.with_name(path.name + ".rewrite")
    try:
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(staged, "w", zipfile.ZIP_STORED) as out:
            for info in source.infolist():
                if info.filename.endswith(NPY) and info.filename[: -len(NPY)] in wanted:
                    out.writestr(_copy_of(info), source.read(info))
        _check(staged, wanted)
        shutil.copystat(path, staged)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
    return Rewritten(held=len(held), kept=len(wanted), bytes_before=bytes_before, bytes_after=path.stat().st_size)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite Kokoro's voices file to the catalogue's voices (PRD-262 US-052).")
    parser.add_argument("catalogue", type=Path, help="media_render/voices.json")
    parser.add_argument("voices", type=Path, help="the voices file, already checked against kokoro/SHA256SUMS")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        done = rewrite(args.voices, load_catalogue(args.catalogue))
    except VoicesError as exc:
        logger.error("voices: %s", exc)
        return 1
    logger.info(
        "voices: %d of the %d voices kept in %s, %d -> %d bytes",
        done.kept,
        done.held,
        args.voices,
        done.bytes_before,
        done.bytes_after,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
