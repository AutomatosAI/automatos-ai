"""PRD-261 E3c — the two packages ``specs/theme-storage.spec.ts`` installs in the local backend.

Both are A-06a's ``hello`` package (``orchestrator/tests/fixtures/themes/build.py``) with this folder's
``entry.js`` as ``dist/entry.js``: a frame that keeps one value through the SDK's ``storage`` area. They
differ only in their theme id and name, so the spec proves a theme's state is its own: ``hello-e3``
writes it and reads it back after a reload, ``hello-e3-other`` reads nothing of it. Each id is this
fixture's own, so neither meets another spec's upload (the same id and version with other files is a
409). ``uses`` declares ``storage``, and ``engines`` asks for the 1.1.0 host and SDK, the first that
have it. The builder is loaded from its file, as A-08c's fixture loads it, so a bare ``python3`` runs
this::

    python3 e2e/browser/fixtures/hello-e3/build.py --variant hello-e3 --out hello-e3.autotheme
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
BUILDER = REPO_ROOT / "orchestrator" / "tests" / "fixtures" / "themes" / "build.py"
ENTRY_SOURCE = HERE / "entry.js"
BASE_VARIANT = "hello"
ENGINES = {"automatos": "^1.1.0", "sdk": "^1.1.0"}
USES = ["storage"]
# variant -> (theme id, name)
PACKAGES: Dict[str, Tuple[str, str]] = {
    "hello-e3": ("automatos-e2e/hello-e3", "Hello storage (E3c)"),
    "hello-e3-other": ("automatos-e2e/hello-e3-other", "Hello storage, another theme (E3c)"),
}


def load_builder() -> ModuleType:
    """A-06a's package builder, from its file (no orchestrator imports run)."""
    spec = importlib.util.spec_from_file_location("theme_fixture_build", BUILDER)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_storage_package(variant: str) -> bytes:
    """The hello package with this fixture's entry, the variant's theme id and name, and ``uses`` storage."""
    if variant not in PACKAGES:
        raise ValueError(f"unknown variant {variant!r}: one of {', '.join(sorted(PACKAGES))}")
    theme_id, name = PACKAGES[variant]
    builder = load_builder()
    manifest = json.loads(builder.hello_members(BASE_VARIANT)[builder.MANIFEST])
    changed = {**manifest, "id": theme_id, "name": name, "engines": ENGINES, "uses": USES}
    replace = {builder.MANIFEST: builder.manifest_bytes(changed), builder.ENTRY: ENTRY_SOURCE.read_bytes()}
    return builder.build_package(BASE_VARIANT, replace=replace)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write the variant's package to ``--out``."""
    parser = argparse.ArgumentParser(description="Build a PRD-261 E3c storage test package (.autotheme).")
    parser.add_argument("--variant", choices=sorted(PACKAGES), required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    data = build_storage_package(args.variant)
    args.out.write_bytes(data)
    sys.stdout.write(f"{args.out} {len(data)} bytes\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
