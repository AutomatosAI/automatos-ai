#!/usr/bin/env python3
"""
PRD-230 US-008 — Seed the two Shopify marketplace packages.
===========================================================

**Shopify Management** (run the store) + **Shopify Development** (build / theme /
dev) — curated from the EXISTING 12 Shopify marketplace agents seeded by
``core/seeds/seed_shopify_agents.py``. A package is DATA, not code (D4): this is
a plain idempotent upsert into ``marketplace_packages`` keyed by slug, no
migration.

Members reference the real agent **slugs**; the closure resolver (US-004) and
installer (US-005) pull each agent's LLM + skills + plugins + connected-app
requirements at install time (D2). Nothing is duplicated here — the member list
is built from ``SHOPIFY_AGENTS`` so every ref provably resolves to real inventory
(a curated slug that drifts from the roster raises at import, not in prod).

Curation is v1 (Gerard tunes content later). The picks are defensible and cited:

  Management (run-the-store, all SHOPIFY-connected):
    shopify-ops               Shopify Operations Manager   (operations)
    shopify-support           Shopify Support Agent        (customer service)
    shopify-inventory-watchdog Shopify Inventory Watchdog  (inventory)
    shopify-business-analyst  Shopify Business Analyst     (reports / weekly numbers)

  Development (build/theme/dev, all GITHUB-connected):
    shopify-app-dev           Shopify App Architect
    shopify-storefront-dev    Shopify Storefront Developer
    shopify-extension-dev     Shopify Extension Builder
"""
from __future__ import annotations

import logging

from core.database.database import SessionLocal
from core.seeds.seed_shopify_agents import SHOPIFY_AGENTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The real inventory, indexed by slug — the single citation for every member ref.
_AGENTS_BY_SLUG: dict[str, dict] = {a["slug"]: a for a in SHOPIFY_AGENTS}

# Curated rosters (the D5 split). Slugs MUST exist in the roster above.
_MANAGEMENT_AGENT_SLUGS = [
    "shopify-ops",
    "shopify-support",
    "shopify-inventory-watchdog",
    "shopify-business-analyst",
]
_DEVELOPMENT_AGENT_SLUGS = [
    "shopify-app-dev",
    "shopify-storefront-dev",
    "shopify-extension-dev",
]


def _first_sentence(text: str) -> str:
    text = (text or "").strip()
    dot = text.find(". ")
    return text[: dot + 1] if dot != -1 else text


def _agent_members(slugs: list[str]) -> list[dict]:
    """Build agent member refs from the REAL roster. A slug not in inventory
    raises KeyError at import — the seed cannot ship an unresolvable ref."""
    members = []
    for slug in slugs:
        agent = _AGENTS_BY_SLUG[slug]  # fail loud on drift (FR: every ref resolves)
        members.append(
            {
                "type": "agent",
                "ref": slug,
                "name": agent["name"],
                "description": _first_sentence(agent["description"]),
            }
        )
    return members


# The Shopify two-step, stated honestly once (D-connect truth, US-002 doctrine).
_SHOPIFY_TWO_STEP_NOTE = (
    "Connect now for live store data. Then install the Automatos Shopify app — a "
    "Site appears in Settings → Widget SDK, and turning on sync unlocks your "
    "Knowledge Graph + widgets."
)


def _guide_steps(team_noun: str, connect_line: str, first_task: str) -> list[dict]:
    """The D7 three-step flow, carried in the manifest and narrated by US-009."""
    return [
        {
            "step": 1,
            "title": "Your team is installed",
            "description": (
                f"{team_noun} are registered to your workspace — owned by you and "
                "editable. Everything came with its full toolkit (skills, tools, model)."
            ),
        },
        {
            "step": 2,
            "title": "Connect what they need",
            "description": connect_line,
        },
        {
            "step": 3,
            "title": "Put your agents to work",
            "description": (
                f"{first_task} The setup checklist card carries the remaining steps."
            ),
        },
    ]


MANAGEMENT_PACKAGE: dict = {
    "slug": "shopify-management",
    "name": "Shopify Management",
    "description": (
        "Run the store. A team that manages day-to-day operations, answers "
        "customers, watches inventory, and reports your weekly numbers — wired "
        "straight to your Shopify data, no CSVs."
    ),
    "vertical_tags": ["shopify", "ecommerce", "retail"],
    "matching": {
        "platforms": ["shopify"],
        "url_patterns": ["myshopify.com", "shopify"],
        "vocabulary": [
            "store", "orders", "inventory", "customers", "products", "sales",
            "fulfillment", "refunds", "merchant", "revenue",
        ],
    },
    "members": _agent_members(_MANAGEMENT_AGENT_SLUGS),
    "setup_manifest": {
        "questions": [
            {"id": "store_url", "prompt": "What's your Shopify store URL?"},
        ],
        "required_connects": [
            {
                "app_name": "SHOPIFY",
                "app_type": "ECOMMERCE",
                "needs_oauth": True,
                "note": _SHOPIFY_TWO_STEP_NOTE,
            }
        ],
        "guide_steps": _guide_steps(
            team_noun="Four agents",
            connect_line=(
                "Connect Shopify through the chat connect card. " + _SHOPIFY_TWO_STEP_NOTE
            ),
            first_task="Run your first Weekly Numbers report.",
        ),
        "report_templates": [
            {
                "name": "weekly-numbers",
                "title": "Weekly Numbers",
                "description": "Orders, revenue, and top products for the week.",
            },
            {
                "name": "inventory-status",
                "title": "Inventory Status",
                "description": "Low-stock and reorder alerts across your catalog.",
            },
            {
                "name": "customer-service-summary",
                "title": "Customer Service Summary",
                "description": "Open tickets, response times, and recurring issues.",
            },
        ],
    },
    "showcase": True,
}


DEVELOPMENT_PACKAGE: dict = {
    "slug": "shopify-development",
    "name": "Shopify Development",
    "description": (
        "Build the store. A team for theme work, app and extension development — "
        "validated code against official Shopify schemas, straight into your repo."
    ),
    "vertical_tags": ["shopify", "ecommerce", "development"],
    "matching": {
        "platforms": ["shopify"],
        "url_patterns": ["myshopify.com", "shopify"],
        "vocabulary": [
            "theme", "liquid", "app", "extension", "storefront", "checkout",
            "code", "development", "build", "api", "webhook",
        ],
    },
    "members": _agent_members(_DEVELOPMENT_AGENT_SLUGS),
    "setup_manifest": {
        "questions": [
            {"id": "repo_url", "prompt": "Which repo should the team build against?"},
        ],
        "required_connects": [
            {"app_name": "GITHUB", "app_type": "DEVELOPER", "needs_oauth": True},
            {
                "app_name": "SHOPIFY",
                "app_type": "ECOMMERCE",
                "needs_oauth": True,
                "note": "Optional: connect a development store to test against.",
            },
        ],
        "guide_steps": _guide_steps(
            team_noun="Three agents",
            connect_line=(
                "Connect GitHub through the chat connect card (and a dev store if "
                "you have one)."
            ),
            first_task="Kick off a theme or app build.",
        ),
        "report_templates": [
            {
                "name": "theme-audit",
                "title": "Theme / Storefront Audit",
                "description": "Performance, accessibility, and best-practice review.",
            },
        ],
    },
    "showcase": True,
}


PACKAGES: list[dict] = [MANAGEMENT_PACKAGE, DEVELOPMENT_PACKAGE]

# The JSONB fields rebuilt (not mutated) on update — assign fresh copies.
_JSONB_FIELDS = ("vertical_tags", "matching", "members", "setup_manifest")


def seed_packages(db=None) -> tuple[int, int]:
    """Idempotently upsert the Shopify packages into ``marketplace_packages``
    (keyed by slug). Re-running is a no-op create-wise: existing rows are
    refreshed, never duplicated. Returns ``(created, updated)``."""
    from core.models.marketplace_packages import MarketplacePackage

    own_session = db is None
    if own_session:
        db = SessionLocal()

    created, updated = 0, 0
    try:
        for pkg_def in PACKAGES:
            existing = (
                db.query(MarketplacePackage)
                .filter_by(slug=pkg_def["slug"])
                .one_or_none()
            )
            if existing is None:
                db.add(MarketplacePackage(**{
                    "slug": pkg_def["slug"],
                    "name": pkg_def["name"],
                    "description": pkg_def["description"],
                    "vertical_tags": list(pkg_def["vertical_tags"]),
                    "matching": dict(pkg_def["matching"]),
                    "members": list(pkg_def["members"]),
                    "setup_manifest": dict(pkg_def["setup_manifest"]),
                    "showcase": pkg_def["showcase"],
                }))
                created += 1
                logger.info("✅ Created package %s", pkg_def["slug"])
            else:
                # Rebuild JSONB (assign fresh objects — never mutate in place).
                existing.name = pkg_def["name"]
                existing.description = pkg_def["description"]
                existing.vertical_tags = list(pkg_def["vertical_tags"])
                existing.matching = dict(pkg_def["matching"])
                existing.members = list(pkg_def["members"])
                existing.setup_manifest = dict(pkg_def["setup_manifest"])
                existing.showcase = pkg_def["showcase"]
                updated += 1
                logger.info("⏭️  Updated package %s", pkg_def["slug"])
        db.commit()
        logger.info("Done! Packages created: %d, updated: %d", created, updated)
        return created, updated
    except Exception as e:  # pragma: no cover - defensive
        db.rollback()
        logger.error("Error seeding packages: %s", e, exc_info=True)
        raise
    finally:
        if own_session:
            db.close()


if __name__ == "__main__":
    seed_packages()
