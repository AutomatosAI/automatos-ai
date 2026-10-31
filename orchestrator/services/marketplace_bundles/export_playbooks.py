"""PRD-270 W2 US-270-12a (K9, K10) — Automatos Cloud's approved community Playbooks, as bundles.

The rows: ``owner_type='marketplace'``, approved, public, not a system Playbook, and a template id no
local edition seeds itself (seeded.py). Each lineage (lineage.py) exports its newest row. The query names
its columns, so a Playbook's schedule config (a trigger's webhook id is a secret: the webhook route's URL
is its only key), template definition, learning data, owner, workspace and install count are never read.

A community Playbook's steps carry the agent ids of the SUBMITTER'S workspace (the submit route copies
them), and those ids never leave: each step names its agent by a local id of the bundle's ``agents``
(a1, a2… in the order the steps first name them), whose reference is ``{slug}`` for a seeded
marketplace agent, ``{item}`` for an agent of an exported lineage (one of ``exported_agents``' rows, or
the submitter's own agent the lineage was cloned from), and otherwise ``{name}``, the step's own agent
name, shown to the user and never resolved. Only marketplace rows are read for this: an agent of any
workspace, the submitter's or another tenant's, is never loaded, so nothing of it but what the step
itself says can travel. A step keeps only the format's keys (schema.STEP_KEYS; its config only the
document step's keys); inputs, outputs and execution keep theirs; the apps are the required tools,
upper-cased.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Tuple

from sqlalchemy.orm import Session

from core.models.core import Agent, WorkflowTemplate as WorkflowRecipe
from services.marketplace_bundles.exported import (
    MARKETPLACE_OWNER, ExportedItem, Lineage, envelope, exported_item, items_by_row, newest_per_lineage,
    publisher_name,
)
from services.marketplace_bundles.schema import (
    DOCUMENT_STEP_TYPE, EXECUTION_KEYS, IO_FIELD_KEYS, STEP_CONFIG_KEYS, STEP_KEYS, STEP_SPEC,
)
from services.marketplace_bundles.seeded import is_seeded_agent, seeded_template_ids
from services.marketplace_public.publishers import publisher_names

ITEM_TYPE = "playbook"
AGENT_ITEM_TYPE = "agent"
LOCAL_ID_PREFIX = "a"
# The step keys whose null travels as null (the format allows it); any other null key is left out.
NULLABLE_STEP_KEYS = tuple(key for key in STEP_KEYS if STEP_SPEC["fields"][key].get("nullable"))

_COLUMNS = (
    WorkflowRecipe.id, WorkflowRecipe.template_id, WorkflowRecipe.cloned_from_id, WorkflowRecipe.original_creator_id,
    WorkflowRecipe.name, WorkflowRecipe.description, WorkflowRecipe.marketplace_category,
    WorkflowRecipe.marketplace_icon, WorkflowRecipe.tags, WorkflowRecipe.version, WorkflowRecipe.steps,
    WorkflowRecipe.inputs, WorkflowRecipe.outputs, WorkflowRecipe.execution_config, WorkflowRecipe.required_tools,
)


@dataclass(frozen=True)
class _AgentRefs:
    """What a step's agent can be named by: the marketplace rows the steps name, and the row ids that
    name an exported agent (exported.items_by_row)."""

    rows: Mapping[int, Any]
    items: Mapping[Any, str]

    def ref_of(self, step: Mapping[str, Any]) -> Optional[Dict[str, str]]:
        """The step's agent as a reference: {slug}, {item} or the step's own {name}; None when nothing
        names it (the Playbook's bundle is then refused)."""
        agent_id = _agent_id(step)
        row = self.rows.get(agent_id) if agent_id is not None else None
        if row is not None and is_seeded_agent(row.slug):
            return {"slug": row.slug}
        if agent_id is not None and agent_id in self.items:
            return {"item": self.items[agent_id]}
        name = _name(step.get("agent_name"))
        return {"name": name} if name else None


def exported_playbooks(db: Session, agent_items: Iterable[ExportedItem]) -> List[ExportedItem]:
    """One ExportedItem per lineage of Automatos Cloud's exportable community Playbooks, newest row
    first; ``agent_items`` are ``exported_agents``' items. A Playbook whose bundle validate_bundle
    refuses is logged and left out."""
    seeded = seeded_template_ids()
    lineages = newest_per_lineage(ITEM_TYPE, (row for row in _approved_rows(db) if row.template_id not in seeded))
    if not lineages:
        return []
    refs = _AgentRefs(_step_agent_rows(db, lineages), items_by_row(agent_items, AGENT_ITEM_TYPE))
    names = publisher_names(db, (lineage.newest.original_creator_id for lineage in lineages))
    items = (
        exported_item(ITEM_TYPE, lineage, envelope(
            ITEM_TYPE, lineage, publisher_name(names, lineage.newest.original_creator_id),
            _body(lineage.newest, refs),
        ))
        for lineage in lineages
    )
    return [item for item in items if item is not None]


def _approved_rows(db: Session) -> List[Any]:
    """K9's Playbooks, seeded ones included, newest first (the id breaks a tie)."""
    return (
        db.query(*_COLUMNS)
        .filter(
            WorkflowRecipe.owner_type == MARKETPLACE_OWNER, WorkflowRecipe.is_approved.is_(True),
            WorkflowRecipe.is_public.is_(True), WorkflowRecipe.is_system.isnot(True),
        )
        .order_by(WorkflowRecipe.created_at.desc(), WorkflowRecipe.id.desc())
        .all()
    )


def _step_agent_rows(db: Session, lineages: Iterable[Lineage]) -> Dict[int, Any]:
    """The marketplace rows the exported Playbooks' steps name, by id, with their slug only: a row of any
    workspace is never read."""
    ids = {agent_id for lineage in lineages for step in _steps(lineage.newest.steps)
           for agent_id in (_agent_id(step),) if agent_id is not None}
    if not ids:
        return {}
    rows = db.query(Agent.id, Agent.slug).filter(Agent.id.in_(ids), Agent.owner_type == MARKETPLACE_OWNER).all()
    return {row.id: row for row in rows}


def _body(row: Any, refs: _AgentRefs) -> Dict[str, Any]:
    """The playbook object of ``row``'s bundle."""
    steps = _steps(row.steps)
    agents, local_ids = _playbook_agents(steps, refs)
    return {
        "agents": agents,
        "steps": [_step(step, local_ids.get(_agent_key(step))) for step in steps],
        "inputs": _io(row.inputs),
        "outputs": _io(row.outputs),
        "execution": _execution(row.execution_config),
        "apps": _apps(row.required_tools),
    }


def _playbook_agents(steps: List[Any], refs: _AgentRefs) -> Tuple[Dict[str, Any], Dict[Hashable, str]]:
    """(local id -> reference, step agent key -> local id), local ids in the order steps first name agents."""
    firsts: Dict[Hashable, Any] = {}
    for step in steps:
        key = _agent_key(step)
        if key is not None and key not in firsts:
            firsts[key] = step
    local_ids = {key: f"{LOCAL_ID_PREFIX}{number}" for number, key in enumerate(firsts, start=1)}
    return {local_ids[key]: refs.ref_of(step) for key, step in firsts.items()}, local_ids


def _agent_key(step: Any) -> Optional[Hashable]:
    """What one agent of the steps is told apart by: its id, else its name; None for a document step
    and for a step naming no agent."""
    if not isinstance(step, dict) or step.get("type") == DOCUMENT_STEP_TYPE:
        return None
    agent_id = _agent_id(step)
    if agent_id is not None:
        return agent_id
    name = _name(step.get("agent_name"))
    return ("name", name) if name else None


def _step(step: Any, agent: Optional[str]) -> Any:
    """``step`` with only the format's keys (a null kept only where the format allows one), its agent
    the local id ``agent``; anything but an object is left for the validator to refuse."""
    if not isinstance(step, dict):
        return step
    kept = {key: deepcopy(step[key]) for key in STEP_KEYS
            if key in step and (step[key] is not None or key in NULLABLE_STEP_KEYS)}
    config = kept.get("config")
    if isinstance(config, dict):
        kept = {**kept, "config": {key: config[key] for key in STEP_CONFIG_KEYS if config.get(key) is not None}}
    return {**kept, "agent": agent}


def _io(value: Any) -> Any:
    """Inputs or outputs with only the format's keys on each field; none reads as no fields."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        return value
    return {
        name: {key: deepcopy(field[key]) for key in IO_FIELD_KEYS if field.get(key) is not None}
        if isinstance(field, dict) else field
        for name, field in value.items()
    }


def _execution(value: Any) -> Any:
    """The execution config's keys the format allows; none reads as no keys."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        return value
    return {key: value[key] for key in EXECUTION_KEYS if value.get(key) is not None}


def _apps(required_tools: Any) -> List[str]:
    """The required tools as distinct Composio app names, upper-cased and sorted."""
    if not isinstance(required_tools, (list, tuple)):
        return []
    return sorted({tool.strip().upper() for tool in required_tools if isinstance(tool, str) and tool.strip()})


def _steps(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _agent_id(step: Any) -> Optional[int]:
    """A step's agent id as an integer (a digit string read as one), else None."""
    value = step.get("agent_id") if isinstance(step, dict) else None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    return None


def _name(value: Any) -> Optional[str]:
    return (value.strip() or None) if isinstance(value, str) else None


__all__ = ["exported_playbooks"]
