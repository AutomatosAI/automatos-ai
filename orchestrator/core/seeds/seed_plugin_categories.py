"""
Seed Plugin Categories
======================

Seeds the plugin_categories table with predefined categories for the marketplace.
Categories are broad — plugins use tags for finer granularity.

Idempotent: upserts on slug (insert or update existing).
"""

import logging
from sqlalchemy.orm import Session

from core.models.marketplace_plugins import PluginCategory

logger = logging.getLogger(__name__)


PREDEFINED_CATEGORIES = [
    # --- Development ---
    {
        "slug": "code-review",
        "name": "Code Review",
        "description": "Plugins for automated code review, linting, and quality analysis.",
        "icon": "🔍",
        "sort_order": 10,
    },
    {
        "slug": "testing",
        "name": "Testing",
        "description": "Plugins for test generation, test automation, and coverage analysis.",
        "icon": "🧪",
        "sort_order": 20,
    },
    {
        "slug": "documentation",
        "name": "Documentation",
        "description": "Plugins for generating, maintaining, and improving documentation.",
        "icon": "📝",
        "sort_order": 30,
    },
    # --- DevOps ---
    {
        "slug": "deployment",
        "name": "Deployment",
        "description": "Plugins for deployment automation, rollback management, and release workflows.",
        "icon": "🚀",
        "sort_order": 40,
    },
    {
        "slug": "monitoring",
        "name": "Monitoring",
        "description": "Plugins for application monitoring, alerting, and observability.",
        "icon": "📊",
        "sort_order": 50,
    },
    {
        "slug": "ci-cd",
        "name": "CI/CD",
        "description": "Plugins for continuous integration and continuous delivery pipeline management.",
        "icon": "⚙️",
        "sort_order": 60,
    },
    # --- Marketing ---
    {
        "slug": "seo",
        "name": "SEO",
        "description": "Plugins for search engine optimization, keyword research, and ranking analysis.",
        "icon": "🔎",
        "sort_order": 70,
    },
    {
        "slug": "content",
        "name": "Content",
        "description": "Plugins for content creation, editing, scheduling, and distribution.",
        "icon": "✍️",
        "sort_order": 80,
    },
    {
        "slug": "analytics",
        "name": "Analytics",
        "description": "Plugins for marketing analytics, attribution, and campaign performance tracking.",
        "icon": "📈",
        "sort_order": 90,
    },
    # --- Sales ---
    {
        "slug": "outreach",
        "name": "Outreach",
        "description": "Plugins for sales outreach, email sequences, and prospecting automation.",
        "icon": "📧",
        "sort_order": 100,
    },
    {
        "slug": "crm",
        "name": "CRM",
        "description": "Plugins for CRM integration, contact management, and deal tracking.",
        "icon": "🤝",
        "sort_order": 110,
    },
    {
        "slug": "prospecting",
        "name": "Prospecting",
        "description": "Plugins for lead generation, enrichment, and qualification.",
        "icon": "🎯",
        "sort_order": 120,
    },
    # --- Support ---
    {
        "slug": "ticketing",
        "name": "Ticketing",
        "description": "Plugins for ticket management, triage, and support workflow automation.",
        "icon": "🎫",
        "sort_order": 130,
    },
    {
        "slug": "knowledge-base",
        "name": "Knowledge Base",
        "description": "Plugins for knowledge base management, FAQ generation, and self-service support.",
        "icon": "📚",
        "sort_order": 140,
    },
    # --- Data ---
    {
        "slug": "analysis",
        "name": "Analysis",
        "description": "Plugins for data analysis, exploration, and insight generation.",
        "icon": "🔬",
        "sort_order": 150,
    },
    {
        "slug": "visualization",
        "name": "Visualization",
        "description": "Plugins for data visualization, charting, and dashboard creation.",
        "icon": "📉",
        "sort_order": 160,
    },
    {
        "slug": "etl",
        "name": "ETL",
        "description": "Plugins for data extraction, transformation, and loading pipelines.",
        "icon": "🔄",
        "sort_order": 170,
    },
    # --- Security ---
    {
        "slug": "scanning",
        "name": "Scanning",
        "description": "Plugins for security scanning, vulnerability detection, and penetration testing.",
        "icon": "🛡️",
        "sort_order": 180,
    },
    {
        "slug": "compliance",
        "name": "Compliance",
        "description": "Plugins for regulatory compliance checking, policy enforcement, and audit trails.",
        "icon": "✅",
        "sort_order": 190,
    },
    {
        "slug": "audit",
        "name": "Audit",
        "description": "Plugins for security auditing, access review, and log analysis.",
        "icon": "🔐",
        "sort_order": 200,
    },
]


def seed_plugin_categories(db: Session):
    """
    Seed predefined plugin categories into the database.

    Upserts on slug: creates new categories or updates existing ones.
    """
    created_count = 0
    updated_count = 0
    skipped_count = 0

    for data in PREDEFINED_CATEGORIES:
        existing = db.query(PluginCategory).filter(PluginCategory.slug == data["slug"]).first()

        if existing:
            changed = False
            for field in ("name", "description", "icon", "sort_order"):
                new_val = data.get(field)
                if getattr(existing, field) != new_val:
                    setattr(existing, field, new_val)
                    changed = True

            if changed:
                updated_count += 1
                logger.debug(f"Updated category: {data['slug']}")
            else:
                skipped_count += 1
                logger.debug(f"Category unchanged: {data['slug']}")
        else:
            category = PluginCategory(**data)
            db.add(category)
            created_count += 1
            logger.debug(f"Created category: {data['slug']}")

    try:
        db.commit()
        logger.info(
            f"Seeded plugin categories: {created_count} created, {updated_count} updated, "
            f"{skipped_count} unchanged"
        )
        return created_count, updated_count
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to seed plugin categories: {e}")
        raise
