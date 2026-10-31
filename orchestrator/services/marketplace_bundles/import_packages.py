"""PRD-270 W2 US-270-13c (§5.6, K14) — locally: an Automatos Cloud package, written as a marketplace package row.

The item pass (``import_items``) hands ``upsert_package`` each package bundle after the pass's agents and
Playbooks, so a member naming an item of the same pass finds its row. The package is upserted by its
``marketplace_cloud_items`` mapping (type 'package', its externalId):

* no mapping: a NEW package row (not a showcase) and its mapping;
* a mapping at the bundle's revision: UNCHANGED, nothing written;
* a mapping at another revision: the SAME row UPDATED in place (its slug, name, description, vertical tags,
  matching, members and setup) and the mapping moved; a local operator's showcase choice stands.

Each member's reference is rewritten to the LOCAL reference ``package_installer.install_package`` takes for
its type: an agent by its integer id (``_find_marketplace_agent`` tries an id first), a Playbook by its
template id, a skill by its id, a plugin by its id, a model by its model id, a tool by its app. A member
resolves (resolve.py) only to a row any workspace may install: an approved marketplace agent or Playbook
(``{slug}`` or ``{templateId}`` a seeded one, ``{item}`` the Cloud row mapped to it), an active global skill
by its name and plugin, an approved, active plugin by its slug, a model an ``llm_models`` row holds. A member
that doesn't resolve REFUSES the package, nothing written: ``install_package`` would stop mid-install at a
missing agent or Playbook and keep what it installed before it.

Never touched: a package row without a mapping (seeded or operator-made). A REFUSED package writes nothing:
its slug held by another package row (one without a mapping, as K14 says, or another Cloud package's, which
the unique slug would refuse anyway), its mapping naming a row that is gone, or a member that doesn't
resolve. Refusals are this module's own sentences, logged with the externalId, never a value of the bundle.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.marketplace_cloud import MarketplaceCloudItem
from core.models.marketplace_packages import MarketplacePackage
from services.marketplace_bundles.resolve import (
    agent_for_ref, global_skills, model_held, playbook_for_ref, plugins_by_slug,
)
from services.marketplace_cloud.pull import NEW, REFUSED, UNCHANGED, UPDATED

logger = logging.getLogger(__name__)

ITEM_TYPE = "package"

SLUG_HELD = "A package this install holds already has its slug."
ROW_GONE = "The package this install held for it is gone."
MEMBER_MISSING = "Its member {number} (a {kind}) isn't one this install holds."

# A member's reference as install_package reads it: an id or a key.
LocalRef = Union[int, str]


def upsert_package(
    db: Session, bundle: Mapping[str, Any], revision: str,
) -> Tuple[str, Optional[MarketplacePackage]]:
    """Write the package ``bundle`` (validated, its canonical bytes' sha256 ``revision``) as the Cloud package
    row its mapping holds, each member's reference rewritten to this install's.

    Answers (outcome, row): NEW or UPDATED with the row written and flushed, UNCHANGED with the row held
    (nothing written), or REFUSED with None (nothing written). The caller commits; a database error is the
    caller's to roll back.
    """
    external_id = bundle["externalId"]
    mapping = _mapping(db, external_id)
    row = db.get(MarketplacePackage, mapping.package_id) if mapping is not None else None
    if mapping is not None and mapping.revision == revision:
        return UNCHANGED, row
    body = bundle[ITEM_TYPE]
    refusal = _refusal(db, body["slug"], mapping, row)
    members: List[Dict[str, Any]] = []
    if refusal is None:
        members, refusal = local_members(db, body["members"])
    if refusal is not None:
        logger.warning("Automatos Cloud sync: package %s is refused: %s", external_id, refusal)
        return REFUSED, None
    outcome, row = _written_row(db, row, package_fields(bundle, members))
    _map(db, mapping, bundle, revision, row.id)
    db.flush()
    return outcome, row


def _mapping(db: Session, external_id: str) -> Optional[MarketplaceCloudItem]:
    return (
        db.query(MarketplaceCloudItem)
        .filter(MarketplaceCloudItem.item_type == ITEM_TYPE, MarketplaceCloudItem.external_id == external_id)
        .one_or_none()
    )


def _refusal(
    db: Session, slug: str, mapping: Optional[MarketplaceCloudItem], row: Optional[MarketplacePackage],
) -> Optional[str]:
    """Why the bundle can't be written, or None: its mapped row is gone, or another row holds its slug."""
    if mapping is not None and row is None:
        return ROW_GONE
    query = db.query(MarketplacePackage.id).filter(MarketplacePackage.slug == slug)
    if row is not None:
        query = query.filter(MarketplacePackage.id != row.id)
    return SLUG_HELD if query.first() is not None else None


def local_members(db: Session, members: List[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """The members as the package row stores them ({type, ref, name, description}, ``ref`` this install's),
    and why one doesn't resolve (``[]`` with the reason) or None."""
    local: List[Dict[str, Any]] = []
    for number, member in enumerate(members, start=1):
        ref = local_ref(db, member["type"], member["ref"])
        if ref is None:
            return [], MEMBER_MISSING.format(number=number, kind=member["type"])
        local.append({
            "type": member["type"], "ref": ref, "name": member["name"], "description": member["description"],
        })
    return local, None


def local_ref(db: Session, member_type: str, ref: Mapping[str, Any]) -> Optional[LocalRef]:
    """The reference ``install_package`` takes for a ``member_type`` member that ``ref`` names, or None when
    nothing here answers it."""
    resolve = _LOCAL_REFS.get(member_type)
    return resolve(db, ref) if resolve is not None else None


def _agent_ref(db: Session, ref: Mapping[str, Any]) -> Optional[int]:
    agent = agent_for_ref(db, ref)
    return agent.id if agent is not None else None


def _playbook_ref(db: Session, ref: Mapping[str, Any]) -> Optional[str]:
    return playbook_for_ref(db, ref)


def _skill_ref(db: Session, ref: Mapping[str, Any]) -> Optional[int]:
    key = (ref["name"], ref.get("package"))
    return global_skills(db, [key]).get(key)


def _plugin_ref(db: Session, ref: Mapping[str, Any]) -> Optional[str]:
    plugin_id: Optional[UUID] = plugins_by_slug(db, [ref["slug"]]).get(ref["slug"])
    return str(plugin_id) if plugin_id is not None else None


def _llm_ref(db: Session, ref: Mapping[str, Any]) -> Optional[str]:
    return ref["modelId"] if model_held(db, ref["modelId"]) else None


def _tool_ref(_db: Session, ref: Mapping[str, Any]) -> str:
    return ref["app"]


_LOCAL_REFS: Mapping[str, Callable[[Session, Mapping[str, Any]], Optional[LocalRef]]] = MappingProxyType({
    "agent": _agent_ref, "playbook": _playbook_ref, "skill": _skill_ref, "plugin": _plugin_ref,
    "llm": _llm_ref, "tool": _tool_ref,
})


def package_fields(bundle: Mapping[str, Any], members: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The columns a Sync writes on a Cloud package row, new or updated."""
    body = bundle[ITEM_TYPE]
    return {
        "slug": body["slug"],
        "name": bundle["name"],
        "description": bundle["description"],
        "vertical_tags": list(body["verticalTags"]),
        "matching": deepcopy(body["matching"]),
        "members": [dict(member) for member in members],
        "setup_manifest": deepcopy(body["setup"]),
    }


def _written_row(
    db: Session, row: Optional[MarketplacePackage], fields: Dict[str, Any],
) -> Tuple[str, MarketplacePackage]:
    """A new Cloud package with ``fields`` (flushed, so it has its id), or ``row`` moved to them."""
    if row is None:
        created = MarketplacePackage(showcase=False, **fields)
        db.add(created)
        db.flush()
        return NEW, created
    for column, value in fields.items():
        setattr(row, column, value)
    return UPDATED, row


def _map(
    db: Session, mapping: Optional[MarketplaceCloudItem], bundle: Mapping[str, Any], revision: str, package_id: UUID,
) -> None:
    """A new mapping for a new row; for an update, the mapping moved. A package written has every member
    resolved, so its ``unresolved`` is empty."""
    if mapping is None:
        db.add(MarketplaceCloudItem(
            item_type=ITEM_TYPE, external_id=bundle["externalId"], version=bundle["version"],
            revision=revision, package_id=package_id, unresolved=[],
        ))
        return
    mapping.version = bundle["version"]
    mapping.revision = revision
    mapping.unresolved = []


__all__ = ["LocalRef", "local_members", "local_ref", "package_fields", "upsert_package"]
