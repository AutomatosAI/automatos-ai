#!/usr/bin/env python3
"""Seed marketplace agents from agent_catalog_templates.

Reads from Ralph's agent_catalog_templates table and inserts into the
agents table with owner_type='marketplace' so they appear in the
existing Marketplace Agents tab.

Usage:
    cd orchestrator && python ../scripts/seed_marketplace_agents.py
"""

import json
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'orchestrator', '.env'))

import psycopg2

CATEGORY_MAP = {
    "engineering": "development",
    "design": "design",
    "marketing": "marketing",
    "sales": "sales",
    "product": "business",
    "project-management": "productivity",
    "testing": "development",
    "support": "support",
    "paid-media": "marketing",
    "specialized": "custom",
    "agent-role": "custom",
    "productivity": "productivity",
    "social-media": "marketing",
}


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Get existing marketplace agent names
    cur.execute("SELECT lower(name) FROM agents WHERE owner_type = 'marketplace'")
    existing = {r[0] for r in cur.fetchall()}
    print(f"Existing marketplace agents: {len(existing)}")

    # Read catalog templates
    cur.execute(
        "SELECT slug, name, category, description, persona, "
        "recommended_model, recommended_tools, tags, icon "
        "FROM agent_catalog_templates WHERE is_active = true"
    )
    templates = cur.fetchall()
    print(f"Catalog templates to seed: {len(templates)}")

    inserted = 0
    skipped = 0

    for t in templates:
        slug, name, category, description, persona, model, tools, tags, icon = t

        if name.lower() in existing:
            print(f"  SKIP: {name}")
            skipped += 1
            continue

        marketplace_category = CATEGORY_MAP.get(category, "custom")
        model_id = model or "anthropic/claude-sonnet-4-6"
        provider = "anthropic" if "claude" in model_id else "openrouter"

        model_config = json.dumps({
            "provider": provider,
            "model_id": model_id,
            "temperature": 0.7,
            "max_tokens": 4096,
        })

        tags_json = json.dumps(tags if isinstance(tags, list) else [])

        cur.execute(
            """INSERT INTO agents
            (name, description, agent_type, status, tags,
             owner_type, is_approved, is_featured,
             marketplace_category, marketplace_icon,
             slug, custom_persona_prompt, use_custom_persona,
             model_config, configuration, install_count, version)
            VALUES
            (%s, %s, 'custom', 'active', %s::json,
             'marketplace', true, false,
             %s, %s,
             %s, %s, %s,
             %s::json, '{}'::json, 0, '1.0.0')""",
            (name, description, tags_json,
             marketplace_category, icon,
             slug, persona, bool(persona),
             model_config)
        )
        existing.add(name.lower())
        inserted += 1

    conn.commit()
    print(f"\nDone: {inserted} inserted, {skipped} skipped")

    cur.execute("SELECT count(*) FROM agents WHERE owner_type = 'marketplace'")
    count = cur.fetchone()[0]
    print(f"Total marketplace agents now: {count}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
