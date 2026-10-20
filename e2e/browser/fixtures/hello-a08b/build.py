"""PRD-261 A-08b — the clip package ``specs/files-proxy.spec.ts`` installs in the local backend.

It is A-06a's ``hello-clip`` variant (``orchestrator/tests/fixtures/themes/build.py``: the hello
theme plus PRD-260's idle clip and still as ``assets/mascot/idle.{mp4,png}``) under its own theme
id, so it never meets another spec's ``automatos-e2e/hello`` upload: the same id and version with
other files is refused with a 409. The builder is loaded from its file, as it loads digest.py, so a
bare ``python3`` runs this::

    python3 e2e/browser/fixtures/hello-a08b/build.py --out hello-a08b.autotheme
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
VARIANT = "hello-clip"
THEME_ID = "automatos-e2e/hello-a08b"
THEME_NAME = "Hello clip (A-08b)"


def load_builder() -> ModuleType:
    """A-06a's package builder, from its file (no orchestrator imports run)."""
    spec = importlib.util.spec_from_file_location("theme_fixture_build", BUILDER)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_clip_package() -> bytes:
    """The hello-clip package with this fixture's theme id and name."""
    builder = load_builder()
    manifest = json.loads(builder.hello_members(VARIANT)[builder.MANIFEST])
    renamed = {**manifest, "id": THEME_ID, "name": THEME_NAME}
    return builder.build_package(VARIANT, replace={builder.MANIFEST: builder.manifest_bytes(renamed)})


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write the package to ``--out``."""
    parser = argparse.ArgumentParser(description="Build the PRD-261 A-08b clip package (.autotheme).")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    data = build_clip_package()
    args.out.write_bytes(data)
    sys.stdout.write(f"{args.out} {len(data)} bytes\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
