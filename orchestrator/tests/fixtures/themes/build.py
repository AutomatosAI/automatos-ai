"""PRD-261 A-06a — builds .autotheme test packages, in memory for pytest and as a command.

The ``hello`` package is the folder ``hello/`` next to this file. Two variants change it:

* ``hello-clip`` adds a mascot clip and its still: ``variants/idle.mp4`` and ``variants/idle.png``
  are copies of PRD-260's ``frontend/public/auto/idle.{mp4,png}`` (a clip a browser is known to
  play), packed as ``assets/mascot/idle.{mp4,png}`` and declared in ``theme.json``'s ``mascot``;
* ``hello-throws`` swaps ``dist/entry.js`` for ``variants/entry.throws.js``, which throws
  while it loads.

Zips are reproducible: members in path order, one fixed timestamp, deflated. The package
digest is computed by ``services/themes/digest.py`` itself, loaded by file path so the command
needs no orchestrator dependencies (the browser lane runs it with a bare ``python3``)::

    python3 orchestrator/tests/fixtures/themes/build.py --variant hello --out hello.autotheme
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

HERE = Path(__file__).resolve().parent
HELLO_DIR = HERE / "hello"
VARIANTS_DIR = HERE / "variants"
DIGEST_SOURCE = HERE.parents[2] / "services" / "themes" / "digest.py"

VARIANTS = ("hello", "hello-clip", "hello-throws")
MANIFEST = "theme.json"
ENTRY = "dist/entry.js"
CLIP = "assets/mascot/idle.mp4"
STILL = "assets/mascot/idle.png"
CLIP_STATE = "idle"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
FILE_MODE = 0o644 << 16

Member = Tuple[Union[str, zipfile.ZipInfo], bytes]

# The reader refuses a look-schema knob the prepaint table never paints (services/themes/first_paint.py:
# the browser builds a theme's looks from that table alone), so a test's table for the hello package paints
# hello's three knobs: accent and ground by choice tables, radius by a numeric rule.
HELLO_ACCENT_TABLE = {"blue": {"--hello-accent": "#2f6fed"}, "green": {"--hello-accent": "#1e7f3c"}}
HELLO_RADIUS_RULE = {
    "var": "--hello-glow", "knob": "radius", "min": 0, "max": 16, "integer": True, "scale": 100, "capTable": None,
    "color": {"knob": "accent", "rgb": {"blue": "47,111,237", "green": "30,127,60"}}, "template": "rgba-alpha",
}
HELLO_DEFAULTS = {"accent": "blue", "ground": "paper", "radius": 6}


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """theme.json as the package holds it."""
    return (json.dumps(manifest, indent=2) + "\n").encode("utf-8")


def hello_manifest() -> Dict[str, Any]:
    """A fresh copy of the hello package's theme.json."""
    return json.loads((HELLO_DIR / MANIFEST).read_text(encoding="utf-8"))


def changed_manifest(set_fields: Optional[Mapping[str, Any]] = None, unset: Sequence[str] = ()) -> bytes:
    """The hello theme.json with top-level fields set and others removed, as package bytes."""
    merged = {**hello_manifest(), **(set_fields or {})}
    return manifest_bytes({key: value for key, value in merged.items() if key not in unset})


def hello_prepaint(
    knobs: Mapping[str, Any], defaults: Optional[Mapping[str, Any]] = None, pairs: Sequence[Any] = (), **extra: Any,
) -> Dict[str, Any]:
    """A prepaint table that paints every knob of hello's look schema: ``knobs`` over an accent table, radius
    by ``HELLO_RADIUS_RULE``, ``defaults`` over ``HELLO_DEFAULTS``; ``extra`` adds a part (a tone rule)."""
    return {
        "knobs": {"accent": HELLO_ACCENT_TABLE, **knobs}, "pairs": list(pairs), "numeric": [HELLO_RADIUS_RULE],
        "defaults": {**HELLO_DEFAULTS, **(defaults or {})}, **extra,
    }


def _folder_members(folder: Path) -> Dict[str, bytes]:
    """Every file under ``folder`` by its relative path; dotfiles (a Finder .DS_Store) are skipped."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(folder).parts)
    }


def hello_members(variant: str = "hello") -> Dict[str, bytes]:
    """``{path: bytes}`` of a variant's package."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}: one of {', '.join(VARIANTS)}")
    members = _folder_members(HELLO_DIR)
    if variant == "hello-throws":
        return {**members, ENTRY: (VARIANTS_DIR / "entry.throws.js").read_bytes()}
    if variant == "hello-clip":
        clip, still = (VARIANTS_DIR / "idle.mp4").read_bytes(), (VARIANTS_DIR / "idle.png").read_bytes()
        manifest = changed_manifest({"mascot": {"states": [CLIP_STATE]}})
        return {**members, MANIFEST: manifest, CLIP: clip, STILL: still}
    return members


def build_zip(members: Iterable[Member], compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    """A zip of ``members`` in the order given; a str name gets the fixed time and mode."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as zf:
        for name, data in members:
            info = name if isinstance(name, zipfile.ZipInfo) else zipfile.ZipInfo(name, date_time=FIXED_TIME)
            if not isinstance(name, zipfile.ZipInfo):
                info.compress_type, info.external_attr = compression, FILE_MODE
            zf.writestr(info, data)
    return buffer.getvalue()


def build_package(
    variant: str = "hello",
    replace: Optional[Mapping[str, Optional[bytes]]] = None,
    extra: Sequence[Member] = (),
) -> bytes:
    """A variant's package; ``replace`` sets (or, with None, drops) files; ``extra`` is appended as given."""
    members = {**hello_members(variant), **(replace or {})}
    kept: List[Member] = [(path, data) for path, data in sorted(members.items()) if data is not None]
    return build_zip([*kept, *extra])


def digest_module() -> ModuleType:
    """``services/themes/digest.py``, loaded from its file (no package imports run)."""
    spec = importlib.util.spec_from_file_location("theme_package_digest", DIGEST_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {DIGEST_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def members_digest(members: Mapping[str, bytes]) -> str:
    """The package digest of ``{path: bytes}``, by digest.py's own code."""
    digest = digest_module()
    return digest.package_digest({path: digest.sha256_hex(data) for path, data in members.items()})


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write one variant's package to ``--out`` and print its digest."""
    parser = argparse.ArgumentParser(description="Build a PRD-261 test theme package (.autotheme).")
    parser.add_argument("--variant", choices=VARIANTS, default="hello")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    data = build_package(args.variant)
    args.out.write_bytes(data)
    print(f"{args.out} {len(data)} bytes, digest {members_digest(hello_members(args.variant))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
