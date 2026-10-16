"""PRD-259 S-003 — builders for .autoskin packages in the tests (zips made in memory).

The base skin is the shared corpus's first case (``fixtures/skins/cases.json``): a valid
skin with shipped fonts, an 'auto' mascot and ``assets/preview.png``.
"""
from __future__ import annotations

import copy
import io
import json
import struct
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

CASES = Path(__file__).resolve().parent / "skins" / "cases.json"

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 56
WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 60
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 52
MASCOT_BYTES = {"webm": WEBM, "mp4": MP4, "png": PNG}

Member = Tuple[Union[str, zipfile.ZipInfo], bytes]


def base_skin(**overrides) -> dict:
    """A deep copy of the corpus's first (valid) skin, with ``overrides`` applied."""
    skin = copy.deepcopy(json.loads(CASES.read_text(encoding="utf-8"))["cases"][0]["skin"])
    return {**skin, **overrides}


def woff2(size: int = 1024) -> bytes:
    """A woff2 header that passes brand_fonts.check_woff2: signature, TrueType flavor, length, one table."""
    header = b"wOF2" + struct.pack(">IIH", 0x00010000, size, 1)
    return header + b"\x00" * (size - len(header))


def build_zip(members: Iterable[Member], compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    """A zip of ``members`` in order (a ZipInfo keeps its own attributes)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buffer.getvalue()


def package(
    skin: Optional[dict] = None,
    files: Optional[Dict[str, bytes]] = None,
    extra: Iterable[Member] = (),
) -> bytes:
    """skin.json, then ``files`` (default: the preview), then ``extra`` members."""
    chosen = base_skin() if skin is None else skin
    assets = {"assets/preview.png": PNG} if files is None else files
    members = [("skin.json", json.dumps(chosen).encode("utf-8")), *assets.items(), *extra]
    return build_zip(members)


def mascot_files(states: Iterable[str], fill: int = 0) -> Dict[str, bytes]:
    """Every format of each state's clip; ``fill`` pads each file to grow it."""
    return {
        f"assets/mascot/{state}.{fmt}": data + b"\x00" * fill
        for state in states for fmt, data in MASCOT_BYTES.items()
    }


def full_package() -> bytes:
    """A valid package using every asset kind: a font file, the preview and a custom mascot."""
    from services.skins.schema import MASCOT_STATES

    skin = base_skin(
        fonts={"sans": "assets/fonts/brand.woff2", "serif": "newsreader", "mono": "geist-mono"},
        mascot={"mode": "custom", "shell": "gold", "light": "ice"},
    )
    files = {"assets/preview.png": PNG, "assets/fonts/brand.woff2": woff2(), **mascot_files(MASCOT_STATES)}
    return package(skin, files)
