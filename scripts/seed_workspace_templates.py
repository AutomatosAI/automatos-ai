#!/usr/bin/env python3
"""
US-005: Seed Workspace Templates
=================================

Inserts 6 pre-built workspace templates into the ``workspaces`` table with
``is_template=True``.  Each template carries a realistic widget layout on a
12-column grid so users can clone and start working immediately.

Usage:
    python scripts/seed_workspace_templates.py [--dry-run] [--force]

Flags:
    --dry-run   Print what would be inserted without touching the DB.
    --force     Delete existing template rows and re-insert (idempotent refresh).
"""

import argparse
import sys
from pathlib import Path
from uuid import uuid4

# Allow imports from the orchestrator package
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database.database import SessionLocal, init_database  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402

# ── Template definitions ─────────────────────────────────────────────

TEMPLATES = [
    # 1. Data Analysis
    {
        "name": "Data Analysis",
        "description": "Explore datasets, run code, and generate reports with a two-panel data view, code editor, and document output.",
        "template_icon": "\U0001F4CA",
        "template_category": "analytics",
        "layout_mode": "grid",
        "layout": {"columns": 12, "rowHeight": 100},
        "widgets": [
            {
                "id": "w-da-data1",
                "type": "data",
                "title": "Primary Dataset",
                "x": 0, "y": 0, "w": 6, "h": 3,
                "config": {"source": "upload", "format": "csv"},
            },
            {
                "id": "w-da-data2",
                "type": "data",
                "title": "Secondary Dataset",
                "x": 6, "y": 0, "w": 6, "h": 3,
                "config": {"source": "upload", "format": "csv"},
            },
            {
                "id": "w-da-code",
                "type": "code",
                "title": "Analysis Notebook",
                "x": 0, "y": 3, "w": 8, "h": 4,
                "config": {"language": "python", "theme": "dark"},
            },
            {
                "id": "w-da-doc",
                "type": "document",
                "title": "Report Output",
                "x": 8, "y": 3, "w": 4, "h": 4,
                "config": {"format": "markdown"},
            },
        ],
    },
    # 2. CRM Dashboard
    {
        "name": "CRM Dashboard",
        "description": "Manage customer communications, track sales data, and automate follow-up workflows.",
        "template_icon": "\U0001F4BC",
        "template_category": "business",
        "layout_mode": "grid",
        "layout": {"columns": 12, "rowHeight": 100},
        "widgets": [
            {
                "id": "w-crm-email",
                "type": "email",
                "title": "Inbox & Outreach",
                "x": 0, "y": 0, "w": 5, "h": 4,
                "config": {"view": "unified"},
            },
            {
                "id": "w-crm-data",
                "type": "data",
                "title": "Sales Pipeline",
                "x": 5, "y": 0, "w": 7, "h": 4,
                "config": {"source": "crm", "view": "kanban"},
            },
            {
                "id": "w-crm-workflow",
                "type": "workflow",
                "title": "Follow-up Automations",
                "x": 0, "y": 4, "w": 12, "h": 3,
                "config": {"trigger": "email_received"},
            },
        ],
    },
    # 3. DevOps Monitor
    {
        "name": "DevOps Monitor",
        "description": "Watch live terminals, orchestrate deployment workflows, and monitor infrastructure metrics.",
        "template_icon": "\U0001F6E0",
        "template_category": "engineering",
        "layout_mode": "grid",
        "layout": {"columns": 12, "rowHeight": 100},
        "widgets": [
            {
                "id": "w-devops-term",
                "type": "terminal",
                "title": "Live Shell",
                "x": 0, "y": 0, "w": 6, "h": 4,
                "config": {"shell": "bash"},
            },
            {
                "id": "w-devops-workflow",
                "type": "workflow",
                "title": "CI/CD Pipeline",
                "x": 6, "y": 0, "w": 6, "h": 4,
                "config": {"view": "dag"},
            },
            {
                "id": "w-devops-data",
                "type": "data",
                "title": "Infrastructure Metrics",
                "x": 0, "y": 4, "w": 12, "h": 3,
                "config": {"source": "prometheus", "refresh": 30},
            },
        ],
    },
    # 4. Content Creator
    {
        "name": "Content Creator",
        "description": "Draft documents, generate images, and send email campaigns from a single workspace.",
        "template_icon": "\U0000270F",
        "template_category": "creative",
        "layout_mode": "grid",
        "layout": {"columns": 12, "rowHeight": 100},
        "widgets": [
            {
                "id": "w-cc-doc",
                "type": "document",
                "title": "Draft Editor",
                "x": 0, "y": 0, "w": 7, "h": 4,
                "config": {"format": "rich_text"},
            },
            {
                "id": "w-cc-image",
                "type": "image",
                "title": "Image Generator",
                "x": 7, "y": 0, "w": 5, "h": 4,
                "config": {"provider": "dalle", "size": "1024x1024"},
            },
            {
                "id": "w-cc-email",
                "type": "email",
                "title": "Campaign Sender",
                "x": 0, "y": 4, "w": 12, "h": 3,
                "config": {"view": "compose"},
            },
        ],
    },
    # 5. Research Assistant
    {
        "name": "Research Assistant",
        "description": "Read papers, take persistent notes, and prototype ideas with an integrated code editor.",
        "template_icon": "\U0001F50D",
        "template_category": "research",
        "layout_mode": "grid",
        "layout": {"columns": 12, "rowHeight": 100},
        "widgets": [
            {
                "id": "w-ra-doc",
                "type": "document",
                "title": "Reading Pane",
                "x": 0, "y": 0, "w": 6, "h": 4,
                "config": {"format": "markdown", "readonly": True},
            },
            {
                "id": "w-ra-memory",
                "type": "memory",
                "title": "Research Notes",
                "x": 6, "y": 0, "w": 6, "h": 4,
                "config": {"auto_save": True},
            },
            {
                "id": "w-ra-code",
                "type": "code",
                "title": "Prototype Sandbox",
                "x": 0, "y": 4, "w": 12, "h": 3,
                "config": {"language": "python", "theme": "light"},
            },
        ],
    },
    # 6. Project Manager
    {
        "name": "Project Manager",
        "description": "Visualise task workflows, track project metrics, and coordinate team communication via email.",
        "template_icon": "\U0001F4CB",
        "template_category": "business",
        "layout_mode": "grid",
        "layout": {"columns": 12, "rowHeight": 100},
        "widgets": [
            {
                "id": "w-pm-workflow",
                "type": "workflow",
                "title": "Task Board",
                "x": 0, "y": 0, "w": 8, "h": 4,
                "config": {"view": "kanban"},
            },
            {
                "id": "w-pm-data",
                "type": "data",
                "title": "Project Metrics",
                "x": 8, "y": 0, "w": 4, "h": 4,
                "config": {"source": "internal", "chart": "burndown"},
            },
            {
                "id": "w-pm-email",
                "type": "email",
                "title": "Team Updates",
                "x": 0, "y": 4, "w": 12, "h": 3,
                "config": {"view": "thread"},
            },
        ],
    },
]


def seed_templates(dry_run: bool = False, force: bool = False) -> None:
    """Insert (or refresh) the 6 workspace templates."""
    if not dry_run:
        init_database()

    db = SessionLocal()

    try:
        existing = (
            db.query(Workspace)
            .filter(Workspace.is_template == True)
            .all()
        )
        existing_names = {ws.name for ws in existing}

        if force and existing:
            if dry_run:
                print(f"[dry-run] Would delete {len(existing)} existing template(s)")
            else:
                for ws in existing:
                    db.delete(ws)
                db.flush()
                print(f"Deleted {len(existing)} existing template(s)")
            existing_names = set()

        created = 0
        skipped = 0

        for tpl in TEMPLATES:
            if tpl["name"] in existing_names:
                print(f"  skip  {tpl['name']} (already exists)")
                skipped += 1
                continue

            if dry_run:
                print(f"  [dry-run] would insert: {tpl['name']}")
                created += 1
                continue

            ws = Workspace(
                id=uuid4(),
                name=tpl["name"],
                description=tpl["description"],
                template_icon=tpl["template_icon"],
                template_category=tpl["template_category"],
                layout_mode=tpl["layout_mode"],
                layout=tpl["layout"],
                widgets=tpl["widgets"],
                is_template=True,
                is_active=True,
                visibility="public",
            )
            db.add(ws)
            created += 1
            print(f"  insert  {tpl['name']}")

        if not dry_run:
            db.commit()

        print(f"\nDone: {created} created, {skipped} skipped.")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="US-005: Seed workspace templates")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be inserted")
    parser.add_argument("--force", action="store_true", help="Delete existing templates and re-insert")
    args = parser.parse_args()

    seed_templates(dry_run=args.dry_run, force=args.force)
