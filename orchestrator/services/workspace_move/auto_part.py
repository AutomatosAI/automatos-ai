"""PRD-273 §5.1 / §5.3 — ``workspace.json`` (the allow-listed settings) and ``auto.json`` (Auto's soul).

``workspaces`` has no ``workspace_id`` column, so no table rule reaches its settings. They are
allow-listed key by key instead: the brand kit, the Orchestrator block without its model
(``llm`` and ``model``: the target keeps its own provider and model, §8-Q7), the voice and push
settings. ``byok_overrides`` and ``integrations`` carry keys and connections and never leave,
and neither does any key added later until it is named here.

Auto's soul is its agent row (slug ``auto-{workspace_id}``, a system agent), the row the
Orchestrator settings page reads: ``settings['orchestrator']['custom_soul']`` can be stale on
Automatos Cloud, so ``auto.json`` reads the row. ``source_agent_id`` names the row in
``sql/agents.jsonl``, so an import maps it (and every row naming it) to the target's own Auto.
The GDPR export (``services/gdpr_service.py``) carries the same ``workspace.json``.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models import Agent
from core.models.workspaces import Workspace

# The settings keys a move carries, and the Orchestrator keys it leaves behind (§8-Q7).
ALLOWED_SETTINGS_KEYS = ("brand_kit", "orchestrator", "voice_live", "push")
ORCHESTRATOR_DROPPED_KEYS = ("llm", "model")
# Named so the guard test can prove they never travel; any unnamed key stays behind as well.
NEVER_SETTINGS_KEYS = ("byok_overrides", "integrations")
AUTO_SLUG = "auto-{workspace_id}"
AUTO_FIELDS = ("custom_persona_prompt", "use_custom_persona", "configuration")


def allowed_settings(settings: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The allow-listed keys of a workspace's settings, copied (the stored dict is untouched)."""
    stored = settings if isinstance(settings, Mapping) else {}
    kept = {key: deepcopy(stored[key]) for key in ALLOWED_SETTINGS_KEYS if key in stored}
    orchestrator = kept.get("orchestrator")
    if isinstance(orchestrator, Mapping):
        kept["orchestrator"] = {
            key: value for key, value in orchestrator.items() if key not in ORCHESTRATOR_DROPPED_KEYS
        }
    return kept


def workspace_document(db: Session, workspace_id: UUID | str) -> Dict[str, Any]:
    """``workspace.json``: the workspace's name and its allow-listed settings. Raises
    LookupError when the workspace does not exist."""
    workspace = db.query(Workspace).filter(Workspace.id == UUID(str(workspace_id))).first()
    if workspace is None:
        raise LookupError("The workspace does not exist.")
    return {"name": workspace.name, "settings": allowed_settings(workspace.settings)}


def auto_document(db: Session, workspace_id: UUID | str, external_id: Optional[str] = None) -> Dict[str, Any]:
    """``auto.json``: the Auto agent row's soul and configuration, and its id in this file.
    ``agent`` is None when the workspace has no Auto row yet (it is seeded on first use)."""
    ws = UUID(str(workspace_id))
    slug = AUTO_SLUG.format(workspace_id=ws)
    row = (
        db.query(Agent.id, Agent.slug, *(getattr(Agent, name) for name in AUTO_FIELDS))
        .filter(Agent.slug == slug, Agent.is_system_agent.is_(True), Agent.workspace_id == ws)
        .first()
    )
    if row is None:
        return {"agent": None}
    agent = {name: deepcopy(getattr(row, name)) for name in AUTO_FIELDS}
    return {"agent": {"source_agent_id": row.id, "slug": row.slug, "externalId": external_id, **agent}}


__all__ = [
    "ALLOWED_SETTINGS_KEYS", "AUTO_FIELDS", "AUTO_SLUG", "NEVER_SETTINGS_KEYS", "ORCHESTRATOR_DROPPED_KEYS",
    "allowed_settings", "auto_document", "workspace_document",
]
