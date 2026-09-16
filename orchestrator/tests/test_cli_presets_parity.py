"""CLI adapter design §8.1 — one registry, two renderings, no drift.

The host's preset table (``services/cli-host/automatos_cli_host/presets.py``)
owns everything operational; the backend's ``core/cli_presets.py`` owns only
what it validates and books. The two are kept in step here: every CLI id the
backend knows is a preset row on the host, and vice versa — read with ``ast``
so neither package has to import the other.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core import cli_presets, cli_runtime  # noqa: E402

HOST_PRESETS = _ORCH.parent / "services" / "cli-host" / "automatos_cli_host" / "presets.py"


def _host_preset_ids() -> set:
    tree = ast.parse(HOST_PRESETS.read_text(encoding="utf-8"))
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "CliPreset":
            for kw in node.keywords:
                if kw.arg == "id" and isinstance(kw.value, ast.Constant):
                    ids.add(kw.value.value)
    return ids


def test_backend_registry_matches_the_hosts_preset_rows():
    host_ids = _host_preset_ids()
    assert host_ids, f"no CliPreset rows found in {HOST_PRESETS}"
    assert set(cli_presets.CLI_PRESETS) == host_ids, (
        f"backend {sorted(cli_presets.CLI_PRESETS)} vs host {sorted(host_ids)} — add the id to both "
        "(design §10 step 8)"
    )


def test_runtime_module_still_exports_the_registry_names():
    """Every importer of core.cli_runtime keeps its names; the rules are the registry's."""
    assert cli_runtime.CLI_PROVIDERS == cli_presets.CLI_PROVIDERS == ("claude", "codex")
    assert cli_runtime.usage_provider_slug("claude") == "claude_code" and cli_runtime.usage_provider_slug("codex") == "codex"
    assert cli_runtime.USAGE_PROVIDER_LABELS == {"claude_code": "Claude Code", "codex": "Codex"}
    assert cli_runtime.is_valid_cli_model("claude", "opus") and cli_runtime.is_valid_cli_model("claude", "claude-opus-5[1m]")
    assert not cli_runtime.is_valid_cli_model("claude", "gpt-5.5") and cli_runtime.is_valid_cli_model("codex", "gpt-5.5")
    assert cli_runtime.is_valid_cli_model("codex", None) and not cli_runtime.is_valid_cli_model("grok", "x")
    assert cli_runtime.validate_runtime_configuration({"runtime": "cli", "provider": "codex", "model": "gpt-5.5"}, cli_enabled=True) == []
