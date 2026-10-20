"""PRD-261 A-08c — the package ``specs/installed-first-paint.spec.ts`` installs in the local backend.

It is A-06a's ``hello`` package (``orchestrator/tests/fixtures/themes/build.py``) under its own
theme id, so it never meets another spec's upload: the same id and version with other files is
refused with a 409. Its files are the hello theme's own, so the import map's integrity pins the
real ``dist/entry.js``. The builder is loaded from its file, as A-08b's fixture loads it, so a bare
``python3`` runs this::

    python3 e2e/browser/fixtures/hello-a08c/build.py --out hello-a08c.autotheme
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
BUILDER = REPO_ROOT / "orchestrator" / "tests" / "fixtures" / "themes" / "build.py"
VARIANT = "hello"
THEME_ID = "automatos-e2e/hello-a08c"
THEME_NAME = "Hello first paint (A-08c)"


def load_builder() -> ModuleType:
    """A-06a's package builder, from its file (no orchestrator imports run)."""
    spec = importlib.util.spec_from_file_location("theme_fixture_build", BUILDER)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_first_paint_package() -> bytes:
    """The hello package with this fixture's theme id and name."""
    builder = load_builder()
    manifest = json.loads(builder.hello_members(VARIANT)[builder.MANIFEST])
    renamed = {**manifest, "id": THEME_ID, "name": THEME_NAME}
    return builder.build_package(VARIANT, replace={builder.MANIFEST: builder.manifest_bytes(renamed)})


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write the package to ``--out``."""
    parser = argparse.ArgumentParser(description="Build the PRD-261 A-08c first-paint package (.autotheme).")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    data = build_first_paint_package()
    args.out.write_bytes(data)
    sys.stdout.write(f"{args.out} {len(data)} bytes\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
