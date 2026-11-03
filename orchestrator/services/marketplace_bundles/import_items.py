"""PRD-270 W2 US-270-13c (§5.6, K14) — locally: the Sync's item pass, all of it or none of it.

The Sync (US-270-13d's ``run_item_passes``) runs this after W1's theme pass and the plugin pass have
committed, handing it the agent, Playbook and package bundles ``items_feed.fetch_bundles`` fetched and
checked. ``import_items`` writes them in ONE transaction: every agent (``upsert_agent``), a flush, every
Playbook (``upsert_playbook``, its steps finding the agents just written), a flush, every package
(``upsert_package``, its members finding both), and ONE commit. A bundle of any other type isn't this
pass's and is left alone.

The pass first takes a transaction advisory lock, so a second process's item pass waits for this one to end
(W1's one-Sync rule holds per process only) and then finds its rows written (unchanged); then it sets a
transaction-local ``lock_timeout``, so a pass that meets a row lock it can't have (an install moving a Cloud
row's install count) fails instead of waiting on. Any database error, or anything unforeseen, rolls EVERY
row of the pass back and records only its plain error in the ``marketplace_cloud_sync`` row (W1's
``pull._failed`` pattern, with W1's ``sync_row``, under the same ``lock_timeout``: the attempt, the success
and the counts the theme pass recorded stand), and the pass counts each bundle it was handed refused. The
theme pass's and the plugin pass's rows, committed before, stand.

The answer is the per-type counts {new, updated, unchanged, refused} (W1's ``SyncCounts``) and the error, if
any. The session the caller hands in carries nothing pending: a failure rolls it back.
"""
from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from services.marketplace_bundles.import_agents import upsert_agent
from services.marketplace_bundles.import_packages import upsert_package
from services.marketplace_bundles.import_playbooks import upsert_playbook
from services.marketplace_bundles.items_feed import FetchedBundle
from services.marketplace_cloud.pull import LAST_ERROR_MAX, REFUSED, SyncCounts, sync_row

logger = logging.getLogger(__name__)

AGENT, PLAYBOOK, PACKAGE = "agent", "playbook", "package"
ITEM_TYPES = (AGENT, PLAYBOOK, PACKAGE)  # the pass's order: what each later type names is written first
PASS_LOCK_KEY = "automatos-cloud:item-pass"
PASS_LOCK_TIMEOUT = "5s"
_PASS_LOCK = text("SELECT pg_advisory_xact_lock(hashtext(:key))")
_LOCK_TIMEOUT = text("SELECT set_config('lock_timeout', :timeout, true)")

NOT_SAVED = "The items from Automatos Cloud couldn't be saved, so none was added; the themes and plugins stand."
UNEXPECTED = "The items from Automatos Cloud failed unexpectedly, so none was added; the server log has the detail."

NO_COUNTS: Mapping[str, SyncCounts] = MappingProxyType({item_type: SyncCounts() for item_type in ITEM_TYPES})


class ItemPass(NamedTuple):
    """The item pass: its counts by type (agent, playbook, package) and, when it failed, its plain error."""

    counts: Mapping[str, SyncCounts] = NO_COUNTS
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """Whether the pass committed."""
        return self.error is None

    def view(self) -> Dict[str, Dict[str, int]]:
        """``{agent, playbook, package: {new, updated, unchanged, refused}}``, as the Sync answers them."""
        return {item_type: self.counts[item_type].view() for item_type in ITEM_TYPES}


def import_items(db: Session, bundles: Iterable[FetchedBundle]) -> ItemPass:
    """Write the agent, Playbook and package ``bundles`` in one transaction and commit (K14); on any
    failure write none of them and record the error."""
    chosen = in_pass_order(bundles)
    try:
        counts = _written(db, chosen)
        db.commit()
    except SQLAlchemyError:
        logger.exception("Automatos Cloud: the item pass couldn't be saved")
        return _failed(db, chosen, NOT_SAVED)
    except Exception:  # anything else still rolls the pass back and is recorded, never a half-written pass
        logger.exception("Automatos Cloud: the item pass failed unexpectedly")
        return _failed(db, chosen, UNEXPECTED)
    logger.info("Automatos Cloud items: %s", ItemPass(counts).view())
    return ItemPass(counts)


def in_pass_order(bundles: Iterable[FetchedBundle]) -> Tuple[FetchedBundle, ...]:
    """The pass's bundles, agents then Playbooks then packages, each type in the order given; any other
    type left out."""
    listed = tuple(bundles)
    return tuple(fetched for item_type in ITEM_TYPES for fetched in listed if fetched.entry.type == item_type)


def _written(db: Session, chosen: Tuple[FetchedBundle, ...]) -> Dict[str, SyncCounts]:
    """Every bundle upserted, a flush after each type; the counts by type. Raises what the database raises."""
    db.execute(_PASS_LOCK, {"key": PASS_LOCK_KEY})
    db.execute(_LOCK_TIMEOUT, {"timeout": PASS_LOCK_TIMEOUT})
    names = agent_names(chosen)
    counts = dict(NO_COUNTS)
    for item_type in ITEM_TYPES:
        for fetched in (one for one in chosen if one.entry.type == item_type):
            outcome, _row = _upsert(db, fetched, names)
            counts = {**counts, item_type: counts[item_type].plus(outcome)}
        db.flush()
    return counts


def agent_names(chosen: Iterable[FetchedBundle]) -> Dict[str, str]:
    """The pass's agent bundles' names by externalId: what a Playbook calls a helper that didn't resolve."""
    return {fetched.bundle["externalId"]: fetched.bundle["name"] for fetched in chosen if fetched.entry.type == AGENT}


def _upsert(db: Session, fetched: FetchedBundle, names: Mapping[str, str]) -> Tuple[str, Any]:
    """One bundle written by its type's writer: (outcome, row)."""
    bundle, revision = fetched.bundle, fetched.entry.revision
    if fetched.entry.type == AGENT:
        return upsert_agent(db, bundle, revision, bundle["publisher"]["name"])
    if fetched.entry.type == PLAYBOOK:
        return upsert_playbook(db, bundle, revision, bundle["publisher"]["name"], names)
    return upsert_package(db, bundle, revision)


def _failed(db: Session, chosen: Tuple[FetchedBundle, ...], message: str) -> ItemPass:
    """Roll every row of the pass back and record its error; each bundle it was handed counts refused."""
    logger.warning("Automatos Cloud: the item pass failed: %s", message)
    try:
        db.rollback()
        db.execute(_LOCK_TIMEOUT, {"timeout": PASS_LOCK_TIMEOUT})  # the rollback ended the pass's setting
        sync = sync_row(db)
        sync.last_error = message[:LAST_ERROR_MAX]
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Automatos Cloud: the item pass's failure couldn't be recorded")
    counts = dict(NO_COUNTS)
    for fetched in chosen:
        counts = {**counts, fetched.entry.type: counts[fetched.entry.type].plus(REFUSED)}
    return ItemPass(counts, message)


__all__ = [
    "ITEM_TYPES", "NOT_SAVED", "PASS_LOCK_KEY", "PASS_LOCK_TIMEOUT", "UNEXPECTED", "ItemPass", "agent_names",
    "import_items", "in_pass_order",
]
