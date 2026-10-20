"""PRD-261 C-04b — the package ``specs/marketplace-themes-tab.spec.ts`` submits to the Marketplace.

It is A-06a's ``hello`` package (``orchestrator/tests/fixtures/themes/build.py``) under its own
theme id, so its Marketplace listing and its install never meet another spec's: A-07d's spec
uninstalls every ``automatos-e2e/hello`` install it finds, and the same id and version with other
files is refused with a 409. The files are the hello theme's own, so once chosen the page draws
the hello frame. The builder is loaded from its file, as A-08c's fixture loads it, so a bare
``python3`` runs this::

    python3 e2e/browser/fixtures/hello-c04b/build.py --out hello-c04b.autotheme
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
THEME_ID = "automatos-e2e/hello-c04b"
THEME_NAME = "Hello Marketplace (C-04b)"
# The Marketplace refuses a submission with no source/ (C-06p4, 422 SOURCE_REQUIRED); A-06a's
# hello package has none, so this fixture adds one minimal source file.
SOURCE_FILE = "source/src/index.tsx"
SOURCE_BYTES = b"export default {}\n"


def load_builder() -> ModuleType:
    """A-06a's package builder, from its file (no orchestrator imports run)."""
    spec = importlib.util.spec_from_file_location("theme_fixture_build", BUILDER)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_marketplace_package() -> bytes:
    """The hello package with this fixture's theme id and name, and one ``source/`` file."""
    builder = load_builder()
    manifest = json.loads(builder.hello_members(VARIANT)[builder.MANIFEST])
    renamed = {**manifest, "id": THEME_ID, "name": THEME_NAME}
    replace = {builder.MANIFEST: builder.manifest_bytes(renamed), SOURCE_FILE: SOURCE_BYTES}
    return builder.build_package(VARIANT, replace=replace)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write the package to ``--out``."""
    parser = argparse.ArgumentParser(description="Build the PRD-261 C-04b Marketplace package (.autotheme).")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    data = build_marketplace_package()
    args.out.write_bytes(data)
    sys.stdout.write(f"{args.out} {len(data)} bytes\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
