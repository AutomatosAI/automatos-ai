"""PRD-273 §4.3 / §5.1 — ``bundles/agents.jsonl`` and ``bundles/playbooks.jsonl``.

Each of the workspace's agents and Playbooks as a PRD-270 item bundle (``marketplace_bundles``'
schema 1, checked by its ``validate_bundle``), so the import resolves what each one references
the way a Sync with Automatos Cloud does: global skills by name, plugins and shared personas
by slug, the model by its id, Composio apps by name, and lists what the target lacks. The
bodies come from the Cloud export's own builders (``export_agents.agent_bodies``,
``export_playbooks.playbook_body``); a Playbook step names its agent by the externalId of
that agent's bundle in this file (Auto's too, which ``auto.json`` repeats).

One line per row: ``{"source_id", "revision", "bundle"}``. ``source_id`` is the row's id in
``sql/agents.jsonl`` or ``sql/workflow_recipes.jsonl``, which keep everything the format has
no field for (team, job title, schedule). A row the format refuses (a prompt over its cap,
say) has no line; the manifest lists it under ``unbundled`` with the refusal's code.
"""
from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.core import Agent, WorkflowTemplate as WorkflowRecipe
from services.marketplace_bundles.canonical import revision_of
from services.marketplace_bundles.errors import BundleError
from services.marketplace_bundles.export_agents import AGENT_BUNDLE_COLUMNS, agent_bodies
from services.marketplace_bundles.export_playbooks import PLAYBOOK_BUNDLE_COLUMNS, playbook_body
from services.marketplace_bundles.exported import Lineage, envelope, fitted_publisher
from services.marketplace_bundles.schema import DESCRIPTION_MAX, EXTERNAL_ID_LENGTH, VERSION_PATTERN
from services.marketplace_bundles.validate import validate_bundle
from services.themes.digest import sha256_hex

logger = logging.getLogger(__name__)

AGENTS_PART = "bundles/agents.jsonl"
PLAYBOOKS_PART = "bundles/playbooks.jsonl"
AGENT_TYPE = "agent"
PLAYBOOK_TYPE = "playbook"
EXTERNAL_ID_NAMESPACE = "automatos-workspace"
# A row whose version the format does not allow travels as this one (the sql row keeps its own).
FALLBACK_VERSION = "1.0.0"
_VERSION = re.compile(VERSION_PATTERN)


def external_id(item_type: str, workspace_id: UUID | str, row_id: Any) -> str:
    """A workspace row's externalId in this file: stable for the same row of the same workspace."""
    return sha256_hex(f"{EXTERNAL_ID_NAMESPACE}:{item_type}:{workspace_id}:{row_id}".encode("utf-8"))[
        :EXTERNAL_ID_LENGTH
    ]


def _fitted(row: Any) -> SimpleNamespace:
    """The row's display fields fitted to the format (a version it allows, a description in its cap)."""
    fields = dict(row._mapping)
    version = fields.get("version")
    fields["version"] = version if isinstance(version, str) and _VERSION.fullmatch(version) else FALLBACK_VERSION
    fields["description"] = (fields.get("description") or "")[:DESCRIPTION_MAX]
    return SimpleNamespace(**fields)


def _line(item_type: str, row: SimpleNamespace, ext_id: str, publisher: str, body: Dict[str, Any],
          unbundled: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """One bundle line, or None (recorded in ``unbundled``) when the format refuses the bundle."""
    lineage = Lineage(external_id=ext_id, newest=row, row_ids=frozenset({row.id}))
    bundle = envelope(item_type, lineage, publisher, body)
    try:
        validate_bundle(bundle)
    except BundleError as refused:
        logger.warning("[workspace-move] %s %s has no bundle: %s (%s)", item_type, row.id, refused.code, refused.message)
        unbundled.append({"type": item_type, "source_id": row.id, "code": refused.code})
        return None
    return {"source_id": row.id, "revision": revision_of(bundle), "bundle": bundle}


def _agent_rows(db: Session, workspace_id: UUID | str) -> List[SimpleNamespace]:
    rows = db.query(*AGENT_BUNDLE_COLUMNS).filter(Agent.workspace_id == workspace_id).order_by(Agent.id).all()
    return [_fitted(row) for row in rows]


def _playbook_rows(db: Session, workspace_id: UUID | str) -> List[SimpleNamespace]:
    rows = (
        db.query(*PLAYBOOK_BUNDLE_COLUMNS).filter(WorkflowRecipe.workspace_id == workspace_id)
        .order_by(WorkflowRecipe.id).all()
    )
    return [_fitted(row) for row in rows]


def _lines(lines: Iterable[Optional[Dict[str, Any]]]) -> Iterable[Dict[str, Any]]:
    return (line for line in lines if line is not None)


def write_bundles(db: Session, writer: Any, workspace_id: UUID | str,
                  workspace_name: str) -> Tuple[Mapping[int, str], List[Dict[str, Any]]]:
    """Write both bundle parts; (agent id -> externalId, the rows that have no bundle)."""
    publisher = fitted_publisher(workspace_name or "")
    unbundled: List[Dict[str, Any]] = []
    agents = _agent_rows(db, workspace_id)
    agent_ids = {row.id: external_id(AGENT_TYPE, workspace_id, row.id) for row in agents}
    bodies = agent_bodies(db, agents)
    writer.write_rows(AGENTS_PART, _lines(
        _line(AGENT_TYPE, row, agent_ids[row.id], publisher, bodies[row.id], unbundled) for row in agents
    ))
    writer.write_rows(PLAYBOOKS_PART, _lines(
        _line(PLAYBOOK_TYPE, row, external_id(PLAYBOOK_TYPE, workspace_id, row.id), publisher,
              playbook_body(row, agent_ids), unbundled)
        for row in _playbook_rows(db, workspace_id)
    ))
    return agent_ids, unbundled


__all__ = ["AGENTS_PART", "PLAYBOOKS_PART", "external_id", "write_bundles"]
