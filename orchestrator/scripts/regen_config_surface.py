#!/usr/bin/env python3
"""Regenerate reports/config-surface.json — the Config deletion guard manifest.

Run from ``orchestrator/``::

    python3 scripts/regen_config_surface.py

Writes the sorted list of Config setting NAMES. **Never values** — this repo is
public and several settings hold credentials. Attributes are read with
``inspect.getattr_static`` so a property that needs a database (or any other
lazy attribute) is recorded without being evaluated.

Adding a setting: run this and commit the updated manifest. Removing one is
deliberate — the guard test will fail until you regenerate, which is the whole
point (see tests/test_config_surface_guard.py).
"""
from __future__ import annotations

import inspect
import json
import pathlib

MANIFEST = pathlib.Path(__file__).resolve().parents[1] / "reports" / "config-surface.json"

_COMMENT = (
    "Config deletion guard (tests/test_config_surface_guard.py). NAMES ONLY — "
    "never values. Regenerate: python3 scripts/regen_config_surface.py"
)


def setting_names(config_obj) -> list[str]:
    """Every public UPPERCASE setting name on the config object.

    ``inspect.getattr_static`` avoids triggering descriptors — a plain
    ``getattr`` evaluates properties, and at least one Config property raises
    without a seeded database, which would make the manifest un-generatable
    outside a fully provisioned environment.
    """
    names: list[str] = []
    for name in dir(config_obj):
        if not name.isupper() or name.startswith("_"):
            continue
        try:
            value = inspect.getattr_static(config_obj, name)
        except AttributeError:
            continue
        if inspect.isfunction(value) or inspect.ismethod(value):
            continue
        if isinstance(value, (staticmethod, classmethod)):
            continue
        names.append(name)
    return sorted(set(names))


def main() -> None:
    from config import config

    names = setting_names(config)
    MANIFEST.write_text(
        json.dumps({"_comment": _COMMENT, "settings": names}, indent=2) + "\n"
    )
    print(f"wrote {len(names)} setting names -> {MANIFEST}")


if __name__ == "__main__":
    main()
