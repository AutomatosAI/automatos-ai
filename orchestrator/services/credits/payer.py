"""The workspace that pays (PRD-274 "Automatos for products", Wave 5d, decisions 24 and 25).

A product's workspace (one carrying ``settings.product_id``) has no credits and no trial of its own: its product's
owner workspace pays for it, under the owner's plan (§8-2). Everything that decides or moves credits asks here
first, so one place answers "who pays":

- the credit gate and every check of whether work may start (``routing.workspace_funding``);
- the meter and an Auto Live call's charge (``debit``): the debit lands on the owner's balance, and records the
  workspace it was for in ``credit_ledger.sub_tenant_id``, which is not ``workspace_id`` and has no foreign key, so
  the owner's usage per workspace survives that workspace's purge;
- the balance alerts (``alerts.check_alerts``) and the summary (``summary.billing_summary``);
- the trial (``trial.grant_trial``) and an admin's grant (``api/admin_credits``), refused for such a workspace.

Every other workspace pays for itself. A product's workspace whose product is gone (it can't be deleted while it
has workspaces) pays for itself too, and so has nothing to spend.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import text

from services.credits.ledger import SPEND_ORDER, Entry, record

_OWNER = text("SELECT owner_workspace_id::text FROM products WHERE id::text = :product")
_OWNERS = text(
    "SELECT w.id::text, p.owner_workspace_id::text FROM workspaces w "
    "JOIN products p ON p.id::text = w.settings->>'product_id' WHERE w.id::text = ANY(:ids)"
)


def payer_of(db: Any, workspace: Any) -> str:
    """The id of the workspace that pays for ``workspace`` (a loaded row): no query unless it is a product's."""
    product_id = (getattr(workspace, "settings", None) or {}).get("product_id")
    if not product_id:
        return str(workspace.id)
    owner = db.execute(_OWNER, {"product": str(product_id)}).scalar()
    return str(owner or workspace.id)


def paying_workspaces(db: Any, workspace_ids: Iterable[Any]) -> Dict[str, str]:
    """Each of ``workspace_ids`` (as strings) to the id of the workspace that pays for it, in one query."""
    ids = sorted({str(w) for w in workspace_ids if w})
    if not ids:
        return {}
    owners = {row[0]: row[1] for row in db.execute(_OWNERS, {"ids": ids}).fetchall()}
    return {ws: owners.get(ws) or ws for ws in ids}


def paying_workspace(db: Any, workspace_id: Any) -> str:
    """The id of the workspace that pays for ``workspace_id``."""
    return paying_workspaces(db, [workspace_id]).get(str(workspace_id), str(workspace_id))


def is_paid_by_its_product(db: Any, workspace_id: Any) -> bool:
    return paying_workspace(db, workspace_id) != str(workspace_id)


def debit(db: Any, workspace_id: Any, entry: Entry, *, spend_buckets: Iterable[str] = SPEND_ORDER,
          payer: Optional[str] = None) -> bool:
    """Records a debit for work done in ``workspace_id`` on the balance of the workspace that pays for it
    (``payer`` when the caller resolved it), with the workspace it was for when that is another. Does not
    commit; False when the source event was already recorded."""
    payer = payer or paying_workspace(db, workspace_id)
    if payer != str(workspace_id):
        entry = replace(entry, sub_tenant_id=str(workspace_id))
    return record(db, payer, entry, spend_buckets=spend_buckets)
