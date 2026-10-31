"""PRD-270 W2 US-270-13c (§5.6, K14) — locally: an Automatos Cloud Playbook, written as a marketplace row.

The item pass (``import_items``) hands ``upsert_playbook`` each Playbook bundle after the pass's agents, so
a step naming an agent of the same pass finds its row. The Playbook is upserted by its
``marketplace_cloud_items`` mapping (type 'playbook', its externalId), as ``upsert_agent`` does an agent:

* no mapping: a NEW marketplace row (owner 'marketplace', no workspace, public, approved, not featured,
  not a system Playbook, template id ``cloud-<externalId>``, the seed's empty template definition) and its
  mapping;
* a mapping at the bundle's revision whose ``unresolved`` the bundle resolves to again: UNCHANGED, nothing
  written;
* a mapping at another revision, or at the same one while its helpers resolve otherwise now (a Cloud agent
  a later Sync brought, P270W2-RVW-1: the Sync fetches its bundle again for that, ``items_feed``'s
  ``rechecked_entries``): the SAME row UPDATED in place (its name, description, steps, inputs, outputs,
  execution, schedule, agents, apps, tags, category, icon, version and its creator's name, the publisher's,
  as ``upsert_agent`` moves an agent's) and the mapping moved. A local operator's approval, featuring and
  visibility of the row stand, as do its template id and counts.

Its steps are stored as the Socials seed stores them (``seed_socials_package._stored_steps``): each agent
step names its agent by ``agent_id`` and ``agent_name``, which a Playbook's install remaps to the
workspace's clone (``cascade_installer._remap_recipe_steps``), and its recommended agents are those agents'
names, which the install clones (by name). An agent reference resolves (resolve.agent_for_ref) to an
approved marketplace agent: ``{slug}`` a seeded one, ``{item}`` the Cloud agent mapped to it. One that
doesn't (``{name}`` never does) leaves its step with no agent, ``needs_helper`` and ``agent_name`` its name
(F223's shape, ``cascade_installer._unpin_foreign_helpers``), and is recorded in the mapping's
``unresolved`` as a helper (an ``{item}`` one with its agent's externalId); the Playbook is still
written. The row is checked with the model's own three validators (the seed's ``_validate``) before
anything is written.

Never touched: a row without a mapping (seeded, operator-made or submitted here) and a workspace's copy of a
Cloud row. A REFUSED Playbook writes nothing: its mapping naming a row that is no longer a marketplace
Playbook, its template id held by another row, or a validator refusing it; at the revision its mapping
holds (a re-resolve, P270W2-RVW-1) the row stands as it was: UNCHANGED. Refusals are this module's own
sentences (a validator's is the model's own), logged with the externalId, never a value of the bundle.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple

from sqlalchemy.orm import Session

from core.models.core import WorkflowTemplate
from core.models.marketplace_cloud import MarketplaceCloudItem
from services.marketplace_bundles.canonical import local_version
from services.marketplace_bundles.errors import BundleError
from services.marketplace_bundles.resolve import (
    HELPER, MARKETPLACE_OWNER, PLAYBOOK_ITEM, LocalAgent, agent_for_ref, at_revision, held_as_resolved,
    unresolved,
)
from services.marketplace_cloud.pull import NEW, REFUSED, UNCHANGED, UPDATED

logger = logging.getLogger(__name__)

ITEM_TYPE = PLAYBOOK_ITEM
TEMPLATE_ID_PREFIX = "cloud-"
STEP_AGENT = "agent"  # a bundle step's local id of its agent, never stored
# What every Socials Playbook row holds (seed_socials_package.playbook_columns).
TEMPLATE_DEFINITION: Mapping[str, Any] = MappingProxyType({"steps": [], "agents": [], "config": {}, "variables": []})
MANUAL_SCHEDULE: Mapping[str, str] = MappingProxyType({"type": "manual"})
UNNAMED_HELPER = "a helper"  # an {item} agent of no bundle this pass read, as the cascade calls one

NOT_A_MARKETPLACE_ROW = "The Playbook this install held for it is gone or is no longer a marketplace Playbook."
TEMPLATE_ID_HELD = "A Playbook this install holds already has its template id."
NOT_VALID = "It isn't a Playbook this install can run: {problem}"
BAD_VERSION = "Its version can't be recorded here."

NO_NAMES: Mapping[str, str] = MappingProxyType({})


class PlaybookParts(NamedTuple):
    """A Playbook bundle's steps as stored here, its recommended agents' names (in the order the steps first
    name them) and its helpers that didn't resolve."""

    steps: Tuple[Dict[str, Any], ...]
    recommended: Tuple[str, ...]
    unresolved: Tuple[Dict[str, str], ...]


def template_id_of(external_id: str) -> str:
    """The template id of the Cloud Playbook ``external_id``: 'cloud-' and the externalId (38 characters)."""
    return f"{TEMPLATE_ID_PREFIX}{external_id}"


def upsert_playbook(
    db: Session, bundle: Mapping[str, Any], revision: str, publisher: str, agent_names: Mapping[str, str] = NO_NAMES,
) -> Tuple[str, Optional[WorkflowTemplate]]:
    """Write the Playbook ``bundle`` (validated, its canonical bytes' sha256 ``revision``) as the Cloud
    marketplace row its mapping holds, with ``publisher`` as its creator's name. ``agent_names`` are the
    names of the pass's agent bundles by externalId, the name of a helper that didn't resolve.

    Answers (outcome, row): NEW or UPDATED with the row written and flushed, UNCHANGED with the row held
    (nothing written: its mapping is at ``revision`` and its helpers resolve as they did, or a rewrite at that
    revision was refused, so it stands as it was), or REFUSED with None (nothing written). The caller commits;
    a database error is the caller's to roll back.
    """
    external_id = bundle["externalId"]
    mapping = _mapping(db, external_id)
    row = db.get(WorkflowTemplate, mapping.recipe_id) if mapping is not None else None
    parts = resolved_parts(db, bundle[ITEM_TYPE], agent_names)
    if held_as_resolved(mapping, revision, parts.unresolved):
        return UNCHANGED, row
    fields, refusal = _checked_fields(bundle, revision, publisher, parts)
    refusal = _refusal(db, external_id, mapping, row) or refusal
    if refusal is not None:
        logger.warning("Automatos Cloud sync: Playbook %s is refused: %s", external_id, refusal)
        return (UNCHANGED, row) if at_revision(mapping, revision) else (REFUSED, None)
    outcome, row = _written_row(db, row, fields, external_id)
    _map(db, mapping, bundle, revision, row.id, parts.unresolved)
    db.flush()
    return outcome, row


def _mapping(db: Session, external_id: str) -> Optional[MarketplaceCloudItem]:
    return (
        db.query(MarketplaceCloudItem)
        .filter(MarketplaceCloudItem.item_type == ITEM_TYPE, MarketplaceCloudItem.external_id == external_id)
        .one_or_none()
    )


def _refusal(
    db: Session, external_id: str, mapping: Optional[MarketplaceCloudItem], row: Optional[WorkflowTemplate],
) -> Optional[str]:
    """Why the bundle can't be written, or None: its mapped row isn't a marketplace Playbook any more, or (a
    new one) another row already holds its template id."""
    if mapping is not None:
        is_marketplace = row is not None and row.owner_type == MARKETPLACE_OWNER and row.workspace_id is None
        return None if is_marketplace else NOT_A_MARKETPLACE_ROW
    held = db.query(WorkflowTemplate.id).filter(WorkflowTemplate.template_id == template_id_of(external_id)).first()
    return TEMPLATE_ID_HELD if held is not None else None


def resolved_parts(db: Session, body: Mapping[str, Any], agent_names: Mapping[str, str] = NO_NAMES) -> PlaybookParts:
    """The playbook object's steps with their agents found here, its recommended agents and the helpers
    that didn't resolve (each agent the steps name, in the order they first name it)."""
    refs = body["agents"]
    named = _distinct(step[STEP_AGENT] for step in body["steps"] if step.get(STEP_AGENT) is not None)
    found = {local_id: agent_for_ref(db, refs[local_id]) for local_id in named}
    return PlaybookParts(
        steps=tuple(_stored_step(step, refs, found, agent_names) for step in body["steps"]),
        recommended=_distinct(found[local_id].name for local_id in named if found[local_id] is not None),
        unresolved=tuple(unresolved(
            (HELPER, helper_name(refs[local_id], agent_names), refs[local_id].get("item"))
            for local_id in named if found[local_id] is None
        )),
    )


def _stored_step(
    step: Mapping[str, Any], refs: Mapping[str, Any], found: Mapping[str, Optional[LocalAgent]],
    agent_names: Mapping[str, str],
) -> Dict[str, Any]:
    """A bundle step as the seed stores one: its keys but its local agent id, and its agent's ``agent_id``
    and ``agent_name``; with no agent found, no ``agent_id`` and the helper it needs named."""
    rest = {key: deepcopy(value) for key, value in step.items() if key != STEP_AGENT}
    local_id = step.get(STEP_AGENT)
    if local_id is None:
        return rest
    agent = found[local_id]
    if agent is not None:
        return {**rest, "agent_id": agent.id, "agent_name": agent.name}
    helper = helper_name(refs[local_id], agent_names)
    return {**rest, "agent_id": None, "agent_name": helper, "needs_helper": helper}


def helper_name(ref: Mapping[str, str], agent_names: Mapping[str, str] = NO_NAMES) -> str:
    """What a helper that didn't resolve is called: a ``{name}``'s name, the name of the pass's agent bundle
    an ``{item}`` names (else 'a helper'), a ``{slug}``'s slug."""
    if "name" in ref:
        return ref["name"]
    if "item" in ref:
        return agent_names.get(ref["item"], UNNAMED_HELPER)
    return ref.get("slug", UNNAMED_HELPER)


def _distinct(values: Iterable[Any]) -> Tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def _checked_fields(
    bundle: Mapping[str, Any], revision: str, publisher: str, parts: PlaybookParts,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """The row's columns, and why they can't be written (its version, or a validator's refusal) or None."""
    try:
        version = local_version(bundle["version"], revision)
    except BundleError:
        return {}, BAD_VERSION
    fields = playbook_fields(bundle, version, publisher, parts)
    problem = validation_problem(fields)
    return fields, (NOT_VALID.format(problem=problem) if problem is not None else None)


def playbook_fields(bundle: Mapping[str, Any], version: str, publisher: str, parts: PlaybookParts) -> Dict[str, Any]:
    """The columns a Sync writes on a Cloud Playbook row, new or updated."""
    body = bundle[ITEM_TYPE]
    return {
        "name": bundle["name"],
        "description": bundle["description"],
        "steps": [deepcopy(step) for step in parts.steps],
        "inputs": deepcopy(body["inputs"]),
        "outputs": deepcopy(body["outputs"]),
        "execution_config": deepcopy(body["execution"]),
        "schedule_config": dict(MANUAL_SCHEDULE),
        "recommended_agents": list(parts.recommended),
        "required_tools": list(_distinct(body["apps"])),
        "tags": list(bundle["tags"]),
        "marketplace_category": bundle["category"] or None,
        "marketplace_icon": bundle["icon"] or None,
        "version": version,
        "created_by": publisher,
    }


def validation_problem(fields: Mapping[str, Any]) -> Optional[str]:
    """What the model's validators (the create route's, the seed's ``_validate``) say is wrong with the
    steps, execution and schedule of ``fields``, or None. Checked on a row no session holds."""
    probe = WorkflowTemplate(
        steps=fields["steps"], execution_config=fields["execution_config"], schedule_config=fields["schedule_config"],
    )
    for check in (probe.validate_steps, probe.validate_execution_config, probe.validate_schedule_config):
        ok, problem = check()
        if not ok:
            return problem
    return None


def _written_row(
    db: Session, row: Optional[WorkflowTemplate], fields: Dict[str, Any], external_id: str,
) -> Tuple[str, WorkflowTemplate]:
    """A new Cloud marketplace Playbook with ``fields`` (flushed, so it has its id), or ``row`` moved to them."""
    if row is None:
        created = WorkflowTemplate(
            template_id=template_id_of(external_id), owner_type=MARKETPLACE_OWNER, owner_id=MARKETPLACE_OWNER,
            workspace_id=None, template_definition=deepcopy(dict(TEMPLATE_DEFINITION)), is_public=True,
            is_featured=False, is_system=False, is_approved=True, **fields,
        )
        db.add(created)
        db.flush()
        return NEW, created
    for column, value in fields.items():
        setattr(row, column, value)
    return UPDATED, row


def _map(
    db: Session, mapping: Optional[MarketplaceCloudItem], bundle: Mapping[str, Any], revision: str,
    recipe_id: int, missing: Iterable[Dict[str, str]],
) -> None:
    """A new mapping for a new row; for an update, the mapping moved (its synced_at bumps with it)."""
    if mapping is None:
        db.add(MarketplaceCloudItem(
            item_type=ITEM_TYPE, external_id=bundle["externalId"], version=bundle["version"],
            revision=revision, recipe_id=recipe_id, unresolved=list(missing),
        ))
        return
    mapping.version = bundle["version"]
    mapping.revision = revision
    mapping.unresolved = list(missing)


__all__ = [
    "PlaybookParts", "helper_name", "playbook_fields", "resolved_parts", "template_id_of", "upsert_playbook",
    "validation_problem",
]
