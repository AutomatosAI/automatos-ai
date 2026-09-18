"""Workspace lifecycle (PRD-247 S0.1): provision, model dials, seed, purge.

Provision and purge run two orchestrator scripts *inside* the backend container
over stdin (``docker exec -i … python -``), so they work the same whether or
not ``/app`` is a bind mount of this checkout, and the runner never needs the
database URL. Purge goes through the product's own two steps — soft-delete then
``services.workspace_purge.purge_workspace_sync`` — synchronously, so the run
gets the ``PurgeResult`` back (the admin route queues the same purge in a
background task and returns before it happens).

No key is minted for the run: the harness is the anonymous local operator and
``X-Workspace-ID`` is the boundary (see ``config``). ``seed_agents`` therefore
checks that what it creates echoes the sim workspace id and stops the run if
it does not — before a single ticket is filed.

Two model dials, both recorded in the run:

* per agent — ``PUT /api/agents/{id}/model-config`` on the agents the pack
  seeded (only those);
* global — ``system_settings`` has no workspace column, so the chatbot and
  ``*_llm`` tiers are shared by every workspace on the stack. The runner
  snapshots them to disk before touching them and restores them in ``finally``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .api import Api, ApiError
from .config import (CREATE_WORKSPACE_SCRIPT, DEFAULT_WORKSPACE_ID, LLM_SETTING_CATEGORY_SUFFIX,
                     LLM_SETTING_EXTRA_CATEGORIES, LLM_SETTING_KEYS, PURGE_WORKSPACE_SCRIPT, Settings)
from .packs import AgentSpec

SIM_SLUG_PREFIX = "sim-"
SCRIPT_TIMEOUT_S = 240
MODEL_KEYS = ("llm_model", "model")
PROVIDER_KEYS = ("llm_provider", "provider")


class WorkspaceError(RuntimeError):
    """Provisioning or purging did not happen; the message carries the script's stderr."""


@dataclass(frozen=True)
class SimWorkspace:
    id: str
    slug: str
    name: str
    api_key: str = ""  # only when provision(mint_key=True); unused by the runner


def run_script_in_backend(settings: Settings, script: Path, args: Sequence[str],
                          timeout_s: int = SCRIPT_TIMEOUT_S) -> str:
    """Pipe a local script into ``python -`` in the backend container; return its stdout."""
    if not script.exists():
        raise WorkspaceError(f"{script} is missing from this checkout")
    cmd = [settings.docker, "exec", "-i", "-w", "/app", settings.backend_container, "python", "-", *args]
    try:
        proc = subprocess.run(cmd, input=script.read_text(encoding="utf-8"), capture_output=True,
                              text=True, timeout=timeout_s, check=False)
    except FileNotFoundError as exc:
        raise WorkspaceError(f"{settings.docker!r} is not on PATH: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceError(f"{script.name} did not finish in {timeout_s}s") from exc
    if proc.returncode != 0:  # the purge script reports refusals as JSON on stdout
        detail = (proc.stderr.strip() or proc.stdout.strip())[-2000:]
        raise WorkspaceError(f"{script.name} exited {proc.returncode}: {detail}")
    return proc.stdout


def parse_key_values(stdout: str) -> dict[str, str]:
    """``KEY=VALUE`` lines on stdout (the create script's contract); comments are stderr."""
    pairs: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            pairs[key.strip()] = value.strip()
    return pairs


def provision(settings: Settings, slug: str, name: str, purpose: str, mint_key: bool = False) -> SimWorkspace:
    """Create (or find, by slug) a sim workspace. No credential unless asked for one."""
    if not slug.startswith(SIM_SLUG_PREFIX):
        raise WorkspaceError(f"sim workspaces are named {SIM_SLUG_PREFIX}*, got {slug!r}")
    args = ["--slug", slug, "--name", name, "--key-name", f"sim key {slug}", "--purpose", purpose]
    stdout = run_script_in_backend(settings, CREATE_WORKSPACE_SCRIPT, args if mint_key else [*args, "--no-key"])
    pairs = parse_key_values(stdout)
    workspace_id = pairs.get("WORKSPACE_ID", "")
    if not workspace_id:
        raise WorkspaceError(f"create script printed no WORKSPACE_ID; stdout was: {stdout[:400]!r}")
    if workspace_id == DEFAULT_WORKSPACE_ID:
        raise WorkspaceError("refusing to run against the operator's default workspace")
    return SimWorkspace(id=workspace_id, slug=slug, name=name, api_key=pairs.get("API_KEY", ""))


def purge(settings: Settings, workspace_id: str, expect_purpose: str = "sim") -> dict[str, Any]:
    """Soft-delete then purge; the script refuses anything not created for the sim."""
    if workspace_id == DEFAULT_WORKSPACE_ID:
        raise WorkspaceError("refusing to purge the operator's default workspace")
    stdout = run_script_in_backend(settings, PURGE_WORKSPACE_SCRIPT, [
        "--workspace-id", workspace_id, "--expect-purpose", expect_purpose,
    ])
    last = [line for line in stdout.splitlines() if line.strip().startswith("{")]
    if not last:
        raise WorkspaceError(f"purge script printed no result; stdout was: {stdout[:400]!r}")
    return json.loads(last[-1])


# -- agents ---------------------------------------------------------------------

def assert_scoped(api: Api, created: Mapping[str, Any], what: str) -> None:
    """The boundary check: a row the run created must carry the sim workspace id."""
    echoed = created.get("workspace_id")
    if echoed is not None and str(echoed) != str(api.workspace_id):
        raise WorkspaceError(
            f"{what} landed in workspace {echoed}, not the sim workspace {api.workspace_id}: "
            "X-Workspace-ID is not being honoured — stopping before anything else is created"
        )


def seed_agents(api: Api, specs: Sequence[AgentSpec]) -> dict[str, dict[str, Any]]:
    """POST /api/agents/ for every pack agent; returns key -> created agent. Stops on a scoping miss."""
    created: dict[str, dict[str, Any]] = {}
    for spec in specs:
        body = {
            "name": spec.name, "description": spec.description or None, "job_title": spec.job_title or None,
            "agent_type": spec.agent_type, "configuration": {}, "tags": list(spec.tags) + ["sim"],
        }
        agent = api.post("/api/agents/", body, label=f"seed:{spec.key}")
        assert_scoped(api, agent if isinstance(agent, Mapping) else {}, f"agent '{spec.key}'")
        created[spec.key] = agent
    return created


def set_agent_models(api: Api, agents: Mapping[str, Mapping[str, Any]], provider: str,
                     model_id: str) -> tuple[dict[str, Any], ...]:
    """Pin every seeded agent to the cheap model; a refusal is recorded, not fatal."""
    outcomes = []
    for key, agent in agents.items():
        try:
            api.put(f"/api/agents/{agent['id']}/model-config", {"provider": provider, "model_id": model_id},
                    label=f"model:{key}")
            outcomes.append({"agent": key, "id": agent["id"], "ok": True})
        except ApiError as exc:
            outcomes.append({"agent": key, "id": agent["id"], "ok": False, "status": exc.status, "body": exc.body[:300]})
    return tuple(outcomes)


# -- global LLM tiers -------------------------------------------------------------

def _is_llm_row(row: Mapping[str, Any]) -> bool:
    category = str(row.get("category") or "")
    tier = category.endswith(LLM_SETTING_CATEGORY_SUFFIX) or category in LLM_SETTING_EXTRA_CATEGORIES
    return tier and str(row.get("key") or "") in LLM_SETTING_KEYS


def snapshot_llm_settings(api: Api) -> tuple[dict[str, Any], ...]:
    """The current model/provider rows of every LLM tier — what restore() puts back."""
    rows = api.get("/api/system-settings/", label="settings:snapshot")
    listed = rows if isinstance(rows, list) else (rows or {}).get("items") or []
    return tuple({"id": r["id"], "category": r.get("category"), "key": r.get("key"), "value": r.get("value")}
                 for r in listed if isinstance(r, dict) and "id" in r and _is_llm_row(r))


def _bulk_update(api: Api, updates: Sequence[Mapping[str, Any]], label: str) -> Any:
    """``POST /api/system-settings/bulk-update`` takes the list itself; fall back to a wrapped body."""
    try:
        return api.post("/api/system-settings/bulk-update", list(updates), label=label)
    except ApiError as exc:
        if exc.status != 422:
            raise
        return api.post("/api/system-settings/bulk-update", {"updates": list(updates)}, label=label + ":wrapped")


def apply_llm_settings(api: Api, snapshot: Sequence[Mapping[str, Any]], provider: str,
                       model_id: str) -> tuple[dict[str, Any], ...]:
    """Point every tier in the snapshot at the cheap model. Returns what was sent."""
    # Rows whose value is NULL are left alone: bulk-update cannot write NULL back,
    # so touching them would be a change restore() could not undo.
    updates = tuple(
        {"id": row["id"], "value": model_id if row["key"] in MODEL_KEYS else provider}
        for row in snapshot if row["key"] in MODEL_KEYS + PROVIDER_KEYS and row.get("value") is not None
    )
    if updates:
        _bulk_update(api, updates, "settings:apply")
    return updates


def restore_llm_settings(api: Api, snapshot: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    updates = tuple({"id": row["id"], "value": row["value"]} for row in snapshot if row.get("value") is not None)
    if updates:
        _bulk_update(api, updates, "settings:restore")
    return updates
