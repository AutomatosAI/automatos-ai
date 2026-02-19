"""
Seed built-in starter templates for document generation (PRD-63).
"""

import logging
import os
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.core import DocumentTemplate

logger = logging.getLogger(__name__)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

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
                },
                "client": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "address": {"type": "string"}, "email": {"type": "string"}},
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
                    },
                },
                "subtotal": {"type": "number"},
                "tax": {"type": "number"},
                "total": {"type": "number"},
                "payment_terms": {"type": "string"},
            },
            "required": ["company", "client", "line_items", "total"],
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
        "description": "Structured meeting notes with attendees, agenda, and action items.",
        "format": "docx",
        "category": "report",
        "template_file": None,
        "data_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}},
                "agenda": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
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
            "action_items": [
                {"task": "Complete API integration", "owner": "Bob", "due_date": "2026-02-25"},
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

    Skips templates that already exist (by name + workspace).
    Returns the number of templates created.
    """
    created = 0
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
            continue

        template_content = None
        if tmpl["template_file"]:
            html_path = os.path.join(TEMPLATES_DIR, tmpl["template_file"])
            if os.path.exists(html_path):
                with open(html_path, "r") as f:
                    template_content = f.read()

        record = DocumentTemplate(
            workspace_id=workspace_id,
            name=tmpl["name"],
            description=tmpl["description"],
            format=tmpl["format"],
            category=tmpl["category"],
            template_content=template_content,
            data_schema=tmpl["data_schema"],
            sample_data=tmpl["sample_data"],
            is_active=True,
            version=1,
            created_by="system",
        )
        db.add(record)
        created += 1

    if created:
        db.commit()
        logger.info(f"Seeded {created} starter templates for workspace {workspace_id}")
    return created
