"""PRD-261 C-03b — the package ``specs/admin-theme-review.spec.ts`` submits to the Marketplace.

It is A-06a's ``hello`` package (``orchestrator/tests/fixtures/themes/build.py``) under its own
theme id, so it never meets another spec's upload or submission: the Marketplace refuses the same
id and version with other files, and a version not higher than one it holds (409). The builder is
loaded from its file, as A-08c's fixture loads it, so a bare ``python3`` runs this::

    python3 e2e/browser/fixtures/hello-c03b/build.py --out hello-c03b.autotheme
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
THEME_ID = "automatos-e2e/hello-c03b"
THEME_NAME = "Hello review queue (C-03b)"


def load_builder() -> ModuleType:
    """A-06a's package builder, from its file (no orchestrator imports run)."""
    spec = importlib.util.spec_from_file_location("theme_fixture_build", BUILDER)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_review_package() -> bytes:
    """The hello package with this fixture's theme id and name."""
    builder = load_builder()
    manifest = json.loads(builder.hello_members(VARIANT)[builder.MANIFEST])
    renamed = {**manifest, "id": THEME_ID, "name": THEME_NAME}
    return builder.build_package(VARIANT, replace={builder.MANIFEST: builder.manifest_bytes(renamed)})


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write the package to ``--out``."""
    parser = argparse.ArgumentParser(description="Build the PRD-261 C-03b review-queue package (.autotheme).")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    data = build_review_package()
    args.out.write_bytes(data)
    sys.stdout.write(f"{args.out} {len(data)} bytes\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
