"""
Seed built-in starter templates for document generation (PRD-63).

PRD-251 S1.2: the social video starters (``social_starters.py``) are seeded by
``seed_social_starters`` through the same starter path (``_seed_presets``: a
platform-owned row refreshes in place, a user's copy is never touched). A
workspace gets them when it turns Socials on (``PUT /api/workspaces/current/socials``):
D1 keeps Socials out of sight until then, and the platform switch is off by default.
"""

import hashlib
import logging
import os
from datetime import datetime
from typing import Any, Callable, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.core import DocumentTemplate
from modules.documents.presets import BRAND_BOARD, MEETING_NOTES_BLOCKS, PRESETS
from modules.documents.social_starters import social_starters
from modules.documents.template_summary import STARTER_CREATOR

logger = logging.getLogger(__name__)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# F347: the sha-256 of each source a legacy seed file shipped before its current one.
# A platform-owned row still holding one of them, byte for byte, takes the current
# file; a row anyone edited holds something else and is left alone. invoice.html:
# PRD-63's (the Automatos orange) and PRD-167's, both printing a hardcoded "$".
# F356: every earlier version of the three Jinja starters, which now print with the
# block documents' design system: invoice.html as F347 shipped it and as F345
# (#985) changes it, basic_report.html and executive_summary.html as PRD-63,
# PRD-167 and F350 shipped them. PRD-255 (US-004): each as F356 shipped it (the
# logo at a fixed 14 mm, the executive summary's raw kit colours); they now read the
# kit's logo rules and colour roles.
RETIRED_SEED_SOURCES = {
    "invoice.html": frozenset({
        "46617dfe5eb1ea2a8d1f939a02d2a1813f426374b0f495d492a4cda48a5c8103",
        "5e2edf70209917bead0b13fdfa5b2245bb38989c3f5c4a158673a327992b78d7",
        "84897050f72fd72b891f8a91b31399a4a9fbfbf95299a997333ede86c999783e",
        "b5cb41b9095579bae26c8550b882f748e3313c8872a670a1dabf63a5d6df4362",
        "d65b4be0269c3385aa2855d11efe488f395403405af28ebf8d527088e65a9370",  # F356
    }),
    "basic_report.html": frozenset({
        "d11f47dba1eccdc3b230e9e55c13b16b8b9da6d485f4f5be1c87092143bf38f3",
        "8286f472960ff9f21f947a4610f128a1bee996ed6c24eb46687ba7b95eb43865",
        "9d08c668fcc55a20bad4f645ac23d50fc625cd09a8ec163b6a1c89159fb3122f",  # F356
    }),
    "executive_summary.html": frozenset({
        "1681d85f5eb12a76256d8a5c4e58c266a55012297e6b19cd0d1b5c84a7d92995",
        "a54919317214125ea2100283c72f918fcb5fc9a253651c5f2ac22c493738abe4",
        "0c95d70e370598d16d4d47ac2a73c2726c6fc5f859edd19f8dc22350066ba4cc",
        "6d473496efa6102885c81a78da1de39d8a007b0748a859a4c05eb68245c7b2b1",  # F356
    }),
}

STARTER_TEMPLATES = [
    {
        "name": "Basic Report",
        "description": "General-purpose report with sections, metrics, and professional styling.",
        "format": "pdf",
        "category": "report",
        "template_file": "basic_report.html",
        "data_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "author": {"type": "string"},
                "company_name": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["title", "content"],
                    },
                },
                "metrics": {"type": "object"},
            },
            "required": ["title", "sections"],
        },
        "sample_data": {
            "title": "Monthly Performance Report",
            "date": "2026-02-01",
            "author": "Automatos AI",
            "sections": [
                {"title": "Overview", "content": "This month showed strong growth across all key metrics."},
                {"title": "Highlights", "content": "Revenue up 15%, customer satisfaction at 94%."},
            ],
            "metrics": {"Revenue": "$125K", "Users": "2,340", "Uptime": "99.9%"},
        },
    },
    {
        "name": "Invoice",
        "description": "Professional invoice with line items, tax calculation, and payment terms.",
        "format": "pdf",
        "category": "invoice",
        "template_file": "invoice.html",
        "data_schema": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "address": {"type": "string"}, "email": {"type": "string"}},
                    "required": ["name"],
                },
                "client": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "address": {"type": "string"}, "email": {"type": "string"}},
                    "required": ["name"],
                },
                "invoice_number": {"type": "string"},
                "date": {"type": "string"},
                "due_date": {"type": "string"},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price": {"type": "number"},
                            "total": {"type": "number"},
                        },
                        "required": ["description", "quantity", "unit_price", "total"],
                    },
                },
                "subtotal": {"type": "number"},
                "tax": {"type": "number"},
                "total": {"type": "number"},
                "payment_terms": {"type": "string"},
            },
            # F345: the client, the invoice number, the figures and the payment terms are
            # asked for, never defaulted ("Client Name", "INV-001", "Net 30").
            "required": [
                "company", "client", "invoice_number", "line_items", "subtotal", "tax", "total", "payment_terms",
            ],
        },
        "sample_data": {
            "company": {"name": "Acme Corp", "address": "123 Main St", "email": "billing@acme.com"},
            "client": {"name": "Client LLC", "address": "456 Oak Ave", "email": "client@example.com"},
            "invoice_number": "INV-001",
            "date": "2026-02-01",
            "due_date": "2026-03-01",
            "line_items": [
                {"description": "Consulting Services", "quantity": 10, "unit_price": 150.00, "total": 1500.00},
                {"description": "Software License", "quantity": 1, "unit_price": 500.00, "total": 500.00},
            ],
            "subtotal": 2000.00,
            "tax": 200.00,
            "total": 2200.00,
            "payment_terms": "Net 30",
        },
    },
    {
        "name": "Executive Summary",
        "description": "Executive summary with highlights, metrics dashboard, and recommendations.",
        "format": "pdf",
        "category": "report",
        "template_file": "executive_summary.html",
        "data_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "author": {"type": "string"},
                "highlights": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "object"},
                "recommendations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "highlights"],
        },
        "sample_data": {
            "title": "Q1 Executive Summary",
            "date": "2026-03-31",
            "highlights": ["Revenue grew 22% YoY", "Launched 3 new products", "NPS score reached 72"],
            "metrics": {"Revenue": "$2.4M", "Growth": "22%", "NPS": "72", "Retention": "94%"},
            "recommendations": [
                "Expand sales team by 3 FTEs in Q2",
                "Invest in AI-driven customer support",
                "Launch enterprise tier by Q3",
            ],
        },
    },
    {
        "name": "Meeting Notes",
        "description": (
            "Meeting notes under your letterhead: date, attendees, agenda, discussion, decisions "
            "and an actions table (action, owner, due)."
        ),
        "format": "docx",
        "category": "report",
        "template_file": None,
        # F356: a block layout (presets.MEETING_NOTES_BLOCKS); it had none, so its docx could not render.
        "blocks": MEETING_NOTES_BLOCKS,
        "data_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}},
                "agenda": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
                "decisions": {"type": "array", "items": {"type": "string"}},
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "owner": {"type": "string"},
                            "due_date": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["title", "date", "attendees"],
        },
        "sample_data": {
            "title": "Sprint Planning Meeting",
            "date": "2026-02-18",
            "attendees": ["Alice", "Bob", "Carol"],
            "agenda": ["Review last sprint", "Plan next sprint", "Assign stories"],
            "notes": "Last sprint closed 18 of 21 stories; the three carried over are blocked on the payments API.",
            "decisions": ["Two-week sprints stay", "Payments API work moves to the top of the backlog"],
            "action_items": [
                {"task": "Complete API integration", "owner": "Bob", "due_date": "2026-02-25"},
                {"task": "Draft the release notes", "owner": "Carol", "due_date": "2026-02-27"},
            ],
        },
    },
    {
        "name": "Data Export",
        "description": "Formatted Excel export for tabular data with auto-sizing columns.",
        "format": "xlsx",
        "category": "data",
        "template_file": None,
        "data_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "rows": {"type": "array", "items": {"type": "array"}},
            },
            "required": ["title", "columns", "rows"],
        },
        "sample_data": {
            "title": "Sales Data",
            "columns": ["Date", "Product", "Quantity", "Revenue"],
            "rows": [
                ["2026-01-01", "Widget A", 150, 4500.00],
                ["2026-01-01", "Widget B", 75, 3000.00],
                ["2026-02-01", "Widget A", 200, 6000.00],
            ],
        },
    },
]


def seed_starter_templates(db: Session, workspace_id: UUID) -> int:
    """Insert starter templates into document_templates for a workspace.

    Skips templates that already exist (by name + workspace), except that a
    platform-owned one still holding a retired seed source takes the current one
    (F347, :func:`refresh_retired_source`). Returns the number of templates created.
    """
    created = legacy_refreshed = 0
    for tmpl in STARTER_TEMPLATES:
        exists = (
            db.query(DocumentTemplate)
            .filter(
                DocumentTemplate.workspace_id == workspace_id,
                DocumentTemplate.name == tmpl["name"],
            )
            .first()
        )
        if exists:
            legacy_refreshed += refresh_retired_source(exists, tmpl)
            continue

        template_content = seed_source(tmpl)
        record = DocumentTemplate(
            workspace_id=workspace_id,
            name=tmpl["name"],
            description=tmpl["description"],
            format=tmpl["format"],
            category=tmpl["category"],
            template_content=template_content,
            blocks=tmpl.get("blocks"),
            data_schema=tmpl["data_schema"],
            sample_data=tmpl["sample_data"],
            is_active=True,
            version=1,
            created_by="system",
        )
        db.add(record)
        created += 1

    # PRD-167 S2 → PRD-243: block-native starters, ONE per category, from the
    # presets the Studio itself offers. Copy-on-customise, so a platform-owned
    # starter (created_by="system") is refreshed in place when the preset changes;
    # a row a user made under the same name is never touched. PRD-255 US-009: the Brand
    # Board is a starter of its own, under the same rule.
    added, refreshed = _seed_presets(db, workspace_id, [*PRESETS, BRAND_BOARD])
    created += added
    refreshed += legacy_refreshed

    if created or refreshed:
        db.commit()
        logger.info(
            "Seeded starter templates for workspace %s: %d created, %d refreshed", workspace_id, created, refreshed
        )
    return created


def seed_source(tmpl: dict) -> Optional[str]:
    """The current source of a legacy seed's template file; ``None`` when it has none."""
    if not tmpl.get("template_file"):
        return None
    html_path = os.path.join(TEMPLATES_DIR, tmpl["template_file"])
    if not os.path.exists(html_path):
        return None
    with open(html_path, "r") as f:
        return f.read()


def holds_retired_source(existing, tmpl: dict) -> bool:
    """Whether ``existing`` is a platform-owned, active row still holding a retired seed source. Pure."""
    retired = RETIRED_SEED_SOURCES.get(tmpl.get("template_file") or "", frozenset())
    content = getattr(existing, "template_content", None)
    if not retired or not isinstance(content, str):
        return False
    owned = (getattr(existing, "created_by", None) or "") == STARTER_CREATOR and getattr(existing, "is_active", True) is not False
    return owned and hashlib.sha256(content.encode("utf-8")).hexdigest() in retired


def takes_starter_blocks(existing, tmpl: dict) -> bool:
    """Whether ``existing`` is a platform-owned, active row of a block seed that has no body
    at all (no blocks, no source, no uploaded file): the old Meeting Notes row (F356). Pure."""
    if not tmpl.get("blocks"):
        return False
    owned = (getattr(existing, "created_by", None) or "") == STARTER_CREATOR and getattr(existing, "is_active", True) is not False
    bodies = (getattr(existing, name, None) for name in ("blocks", "template_content", "template_file_path"))
    return owned and not any(bodies)


def refresh_retired_source(existing, tmpl: dict) -> int:
    """Give a row :func:`holds_retired_source` finds the seed's current source (F347), or a
    row :func:`takes_starter_blocks` finds the seed's blocks (F356); 1 when it did."""
    if takes_starter_blocks(existing, tmpl):
        existing.blocks = tmpl["blocks"]
        existing.updated_at = datetime.utcnow()
        return 1
    source = seed_source(tmpl)
    if source is None or not holds_retired_source(existing, tmpl):
        return 0
    existing.template_content = source
    existing.updated_at = datetime.utcnow()
    return 1


def seed_social_starters(db: Session, workspace_id: UUID, *, commit: bool = True) -> dict:
    """Seed the social video starters into a workspace (PRD-251 S1.2); idempotent.

    Called when the workspace turns Socials on. The same starter path as the
    document presets: a missing starter is created, a platform-owned one that
    drifted from its seed is refreshed in place, one a person made or deleted
    under the same name is left alone. Returns the counts. Commits only when
    something changed, and never with ``commit=False``: the Socials switch
    commits the switch and the starters together.
    """
    created, refreshed = _seed_presets(db, workspace_id, social_starters())
    if created or refreshed:
        if commit:
            db.commit()
        logger.info(
            "Seeded social starters for workspace %s: %d created, %d refreshed", workspace_id, created, refreshed
        )
    return {"created": created, "refreshed": refreshed}


def seed_social_starters_where_on(db: Session, socials_on: Callable[[Any], bool]) -> dict:
    """The social starters for every workspace that has Socials on, at boot (PRD-251B).

    ``seed_social_starters`` runs when a workspace turns Socials on, so a starter that
    ships later (the photo cards) would never reach a workspace that turned it on before.
    ``socials_on`` reads a workspace's settings (``modules/socials/settings``, which this
    module may not import). Idempotent: a workspace that has every starter is left as it
    is. Each workspace commits on its own, so one that fails is logged and the others
    still get theirs.
    """
    from core.models.workspaces import Workspace

    on = [row.id for row in db.query(Workspace.id, Workspace.settings).all() if socials_on(row.settings)]
    totals = {"workspaces": len(on), "created": 0, "refreshed": 0}
    for workspace_id in on:
        try:
            counts = seed_social_starters(db, workspace_id)
        except Exception:
            db.rollback()
            logger.exception("Social starters for workspace %s could not be seeded", workspace_id)
            continue
        totals["created"] += counts["created"]
        totals["refreshed"] += counts["refreshed"]
    return totals


def _seed_presets(db: Session, workspace_id: UUID, presets) -> tuple:
    """Create or refresh each starter in ``presets`` (``starter_outcome``); ``(created, refreshed)``. No commit."""
    created = refreshed = 0
    for preset in presets:
        existing = (
            db.query(DocumentTemplate)
            .filter(
                DocumentTemplate.workspace_id == workspace_id,
                DocumentTemplate.name == preset["name"],
            )
            .first()
        )
        outcome = starter_outcome(existing, preset)
        if outcome == "created":
            db.add(
                DocumentTemplate(
                    workspace_id=workspace_id,
                    is_active=True,
                    version=1,
                    created_by=STARTER_CREATOR,
                    **starter_columns(preset),
                )
            )
            created += 1
        elif outcome == "refreshed":
            for column, value in starter_columns(preset).items():
                setattr(existing, column, value)
            existing.updated_at = datetime.utcnow()
            refreshed += 1
    return created, refreshed


def starter_columns(preset: dict) -> dict:
    """The columns a starter row carries from its preset. Pure."""
    return {
        "name": preset["name"],
        "description": preset["description"],
        "format": preset["format"],
        "category": preset["category"],
        "blocks": preset["blocks"],
        "sample_data": preset.get("sample_data", {}),
    }


def starter_outcome(existing, preset: dict) -> str:
    """``created`` (no row), ``refreshed`` (a platform-owned row that drifted from the
    preset), ``unchanged`` (platform-owned and identical), ``user_owned`` (a row a
    person created under the same name — never overwritten), or ``deleted_by_user``
    (a platform starter the person soft-deleted — never resurrected, never
    re-created: their gallery stays the way they left it). Pure."""
    if existing is None:
        return "created"
    if (getattr(existing, "created_by", None) or "") != STARTER_CREATOR:
        return "user_owned"
    if getattr(existing, "is_active", True) is False:
        return "deleted_by_user"
    current = {column: getattr(existing, column, None) for column in starter_columns(preset)}
    return "unchanged" if current == starter_columns(preset) else "refreshed"
