"""PRD-143 S9 — tool scaffolder: router endpoint → 3-file skeleton.

scripts/scaffold_platform_tool.py turns a live FastAPI route into an
ActionDefinition skeleton + workspace-scoped handler skeleton for HUMAN
curation (scaffold-then-curate, PRD-143 FR-7). The contract proven here:

- introspects real routes (params, body model, docstring) and emits
  conventional skeletons: platform_* naming, workspace-scoped handler
  signature, permission_level guessed from the HTTP verb, # CURATE checklist;
- endpoints on su-locked routers (S6/S7) emit super_admin_only=True with a
  review flag — detected from the require_super_admin dependency IDENTITY on
  the router, not a hardcoded module list;
- NEVER writes into modules/tools/discovery/ and never touches
  platform_actions.py — registration stays manual.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# Dummy POSTGRES_* satisfies the config chain at import (blessed pattern,
# see test_prd143_su_executor_gate.py) — the port points at nothing so any
# fail-soft connect refuses instantly. CI exports real vars (setdefault no-ops).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH_ROOT = Path(__file__).resolve().parent.parent
_SCAFFOLDER = _ORCH_ROOT / "scripts" / "scaffold_platform_tool.py"
_DISCOVERY_DIR = _ORCH_ROOT / "modules" / "tools" / "discovery"

_spec = importlib.util.spec_from_file_location("scaffold_platform_tool", _SCAFFOLDER)
spt = importlib.util.module_from_spec(_spec)
# sys.modules registration BEFORE exec — py3.10 dataclasses resolve string
# annotations via sys.modules[cls.__module__] at class-creation time.
sys.modules["scaffold_platform_tool"] = spt
_spec.loader.exec_module(spt)


def _curate_lines(source: str) -> list[str]:
    return [ln for ln in source.splitlines() if "# CURATE" in ln]


# (router module, 'METHOD /path', expected action name, expected permission,
#  body/path properties that must appear in the emitted parameters schema)
SAMPLES = [
    pytest.param(
        "api/agents.py", "POST /api/agents",
        "platform_create_agents", "write", ("name", "agent_type"),
        id="agents-create",
    ),
    pytest.param(
        "api/agents.py", "GET /api/agents",
        "platform_list_agents", "read", (),
        id="agents-list",
    ),
    pytest.param(
        "api/agents.py", "DELETE /api/agents/{agent_id}",
        "platform_delete_agents", "destructive", ("agent_id",),
        id="agents-delete",
    ),
    pytest.param(
        "api/workspaces.py", "GET /api/workspaces/current",
        "platform_get_workspaces_current", "read", (),
        id="workspaces-get",
    ),
]


@pytest.mark.parametrize("module_path,endpoint,name,permission,props", SAMPLES)
def test_emits_action_and_handler_skeleton_for_sample_endpoint(
    module_path, endpoint, name, permission, props
):
    result = spt.scaffold(module_path, endpoint)

    # --- action skeleton ---
    action = result.action_source
    assert "ActionDefinition(" in action
    assert f'name="{name}"' in action
    assert f'permission_level="{permission}"' in action
    assert "workspace_scoped=True" in action
    # Operator tier by default — the Rev 2 inversion (su only for obs routers).
    assert "super_admin_only=False" in action
    if permission == "destructive":
        assert "requires_confirmation=True" in action

    # The CURATE checklist names every review item (PRD-143 FR-7).
    curate = _curate_lines(action)
    assert curate, "emitted action skeleton must carry a # CURATE checklist"
    for item in ("name", "description", "params", "tier", "permission_level"):
        assert any(item in ln for ln in curate), f"missing CURATE item: {item}"

    # Introspected params from the real route (body model / path params).
    for prop in props:
        assert f"'{prop}'" in action, f"parameters schema missing {prop}"

    # --- handler skeleton ---
    handler = result.handler_source
    handler_fn = name.removeprefix("platform_")
    assert f"async def {handler_fn}(" in handler
    assert "db: Session, workspace_id: UUID, params: Dict[str, Any]" in handler
    assert "NotImplementedError" in handler  # skeleton, not an implementation

    # Both skeletons are valid Python and clean of env reads.
    compile(action, result.action_filename, "exec")
    compile(handler, result.handler_filename, "exec")
    assert "os.getenv" not in action and "os.getenv" not in handler


def test_docstring_seeds_description():
    result = spt.scaffold("api/agents.py", "POST /api/agents")
    # create_agent's docstring: "Create a new agent with enhanced fields"
    assert "Create a new agent" in result.action_source


def test_obs_router_endpoint_flagged_su():
    # api/heartbeat.py is su-locked router-wide (S6) — detected by dependency
    # identity, so the skeleton must flip the tier and flag it for review.
    result = spt.scaffold("api/heartbeat.py", "GET /api/heartbeat/status")
    assert "super_admin_only=True" in result.action_source
    assert "super_admin_only=False" not in result.action_source
    su_lines = [
        ln for ln in _curate_lines(result.action_source)
        if "REVIEW" in ln or "su-locked" in ln or "super_admin" in ln
    ]
    assert su_lines, "su tier must carry an explicit review flag"


def test_never_writes_into_discovery_dir(tmp_path):
    before = sorted(p.name for p in _DISCOVERY_DIR.iterdir())
    platform_actions = (_DISCOVERY_DIR / "platform_actions.py").read_text()

    # Direct refusal — discovery dir and any path under it.
    for target in (_DISCOVERY_DIR, _DISCOVERY_DIR / "gen"):
        with pytest.raises(ValueError):
            spt.scaffold("api/agents.py", "POST /api/agents", out_dir=target)

    # CLI path refuses too (exit code 2, nothing written).
    assert spt.main(
        ["api/agents.py", "POST /api/agents", "--out-dir", str(_DISCOVERY_DIR)]
    ) == 2

    assert sorted(p.name for p in _DISCOVERY_DIR.iterdir()) == before
    assert (_DISCOVERY_DIR / "platform_actions.py").read_text() == platform_actions

    # Positive path: a staging dir gets exactly the two skeleton files.
    out = tmp_path / "gen"
    result = spt.scaffold("api/agents.py", "POST /api/agents", out_dir=out)
    written = sorted(p.name for p in out.iterdir())
    assert written == sorted([result.action_filename, result.handler_filename])
    assert result.action_path.read_text() == result.action_source
    assert result.handler_path.read_text() == result.handler_source


def test_verb_to_permission_level_mapping():
    for verb in ("GET", "HEAD", "OPTIONS", "get"):
        assert spt.verb_to_permission_level(verb) == "read"
    for verb in ("POST", "PUT", "PATCH", "post"):
        assert spt.verb_to_permission_level(verb) == "write"
    assert spt.verb_to_permission_level("DELETE") == "destructive"
    # Fail-closed: an unknown verb gets maximum review scrutiny.
    assert spt.verb_to_permission_level("TRACE") == "destructive"


def test_unknown_endpoint_raises_with_candidates():
    with pytest.raises(ValueError) as exc:
        spt.scaffold("api/agents.py", "POST /api/no-such-route")
    assert "POST /api/no-such-route" in str(exc.value)
