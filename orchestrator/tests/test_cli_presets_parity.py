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


# ── PRD-245 S0.6: the Bash allowlist the session prompt renders is the host's ──

HOST_POLICY = _ORCH.parent / "services" / "cli-host" / "automatos_cli_host" / "policy.py"


def _host_default_bash_allow() -> set:
    tree = ast.parse(HOST_POLICY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        targets = [getattr(t, "id", None) for t in getattr(node, "targets", [])]
        if isinstance(node, ast.Assign) and "DEFAULT_BASH_ALLOW" in targets:
            return {elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)}
    return set()


def test_session_bash_verbs_mirror_the_hosts_default_allowlist():
    """The host's policy is the rule; the backend renders a copy into the session
    prompt so the agent knows what runs without asking. Drift fails here."""
    host = _host_default_bash_allow()
    assert host, f"no DEFAULT_BASH_ALLOW tuple found in {HOST_POLICY}"
    ours = set(cli_presets.SESSION_BASH_VERBS)
    assert ours == host, (
        f"backend-only {sorted(ours - host)} / host-only {sorted(host - ours)} — "
        "mirror the host's DEFAULT_BASH_ALLOW in core/cli_presets.SESSION_BASH_VERBS"
    )
    assert len(cli_presets.SESSION_BASH_VERBS) == len(ours)   # no duplicates in the rendered list


# ── PRD-245 W1: the three claim keys the session bridge rides on ──────────────

HOST_SESSION = _ORCH.parent / "services" / "cli-host" / "automatos_cli_host" / "session.py"

# What the backend puts in the claim payload for the bridge, and what the host's
# ``Session._session_tools`` reads back out. The spelling has already moved once
# (the PRD said ``session_tools_url``; the build ships ``session_tools_path``),
# and a drift here is SILENT: the host reads nothing, writes no MCP config, and
# every session runs with no platform tools and no error anywhere.
SESSION_BRIDGE_CLAIM_KEYS = {"session_tools", "session_tools_path", "session_token"}


def _host_session_tools_keys() -> set:
    """The ``self.ticket.get("…")`` keys inside the host's ``_session_tools``."""
    tree = ast.parse(HOST_SESSION.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_session_tools":
            return {
                c.args[0].value
                for c in ast.walk(node)
                if isinstance(c, ast.Call)
                and getattr(c.func, "attr", None) == "get"
                and c.args and isinstance(c.args[0], ast.Constant)
                and isinstance(c.args[0].value, str)
            }
    return set()


def test_the_host_reads_exactly_the_claim_keys_the_backend_writes():
    host = _host_session_tools_keys()
    assert host, f"no _session_tools function found in {HOST_SESSION}"
    assert host == SESSION_BRIDGE_CLAIM_KEYS, (
        f"host-only {sorted(host - SESSION_BRIDGE_CLAIM_KEYS)} / "
        f"backend-only {sorted(SESSION_BRIDGE_CLAIM_KEYS - host)} — rename on BOTH sides, "
        "and move EXPECTED_CLI_HOST_VERSION with it"
    )


def test_the_claim_payload_carries_those_keys_and_the_version_moved_with_them():
    """The tripwire with teeth.

    ``test_host_contract_version_moved_with_the_claim_shape`` asserts a literal,
    so it stayed green through the very change it is named for. Binding the
    version to the claim's own key set means a new field cannot land without
    editing this line — which is the moment to decide whether the host version
    moves too.
    """
    from services import cli_host_service as svc

    claim_keys = _claim_payload_keys()
    assert SESSION_BRIDGE_CLAIM_KEYS <= claim_keys, (
        f"the claim no longer carries {sorted(SESSION_BRIDGE_CLAIM_KEYS - claim_keys)}"
    )
    assert (svc.EXPECTED_CLI_HOST_VERSION, sorted(claim_keys)) == (
        "0.8.0",
        sorted(_EXPECTED_CLAIM_KEYS),
    ), (
        "the claim payload's shape changed — bump EXPECTED_CLI_HOST_VERSION and the host's "
        "__version__ together, then update this expectation"
    )


# The claim payload as of host contract 0.8.0.
_EXPECTED_CLAIM_KEYS = {
    "task_id", "workspace_id", "title", "prompt", "attachment_ids", "review_mode",
    "agent_id", "agent_name", "provider", "model", "allowed_tools",
    "cwd", "worktree", "attempt", "session_id", "lease_seconds",
    "system_prompt", "resume_session_id",
    # the bridge (0.8.0)
    "session_tools", "session_tools_path", "session_token",
}

BACKEND_SERVICE = _ORCH / "services" / "cli_host_service.py"


def _claim_payload_keys() -> set:
    """The literal keys of the dict ``claim_for_host`` appends to its result."""
    tree = ast.parse(BACKEND_SERVICE.read_text(encoding="utf-8"))
    best: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "claim_for_host":
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "append":
                    for arg in call.args:
                        if isinstance(arg, ast.Dict):
                            keys = {
                                k.value for k in arg.keys
                                if isinstance(k, ast.Constant) and isinstance(k.value, str)
                            }
                            if len(keys) > len(best):
                                best = keys
    return best
