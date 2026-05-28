"""
Seed the Blog Pipeline v2: Mission-powered research + playbook orchestration.

The playbook scouts a topic, launches a multi-agent mission for deep research
and writing, generates cover art, then creates an approval task with a publish
gate. Runs Tue/Fri at 09:00 UTC.

Usage:
    python scripts/seed_blog_playbook.py
    WORKSPACE_ID=... python scripts/seed_blog_playbook.py  # seed into a specific workspace
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from core.database.database import get_database_url

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

CONTENT_AGENTS = [
    {
        "name": "QUILL",
        "agent_type": "custom",
        "description": (
            "Blog topic scout. Checks workspace memory and existing posts to find "
            "fresh, trending topics. Launches a research mission via platform_create_mission "
            "for deep, multi-agent content creation. Lightweight — the mission does the "
            "heavy lifting (research, writing, editing, fact-checking)."
        ),
        "category": "Content Creation",
        "tags": ["blog", "content", "topic-scout", "mission-launcher"],
        "tools": [],
        "model_id": "mistralai/mistral-small-3.1-24b-instruct",
        "skills": ["pattern_recognition"],
        "system_prompt": (
            "You are QUILL, the blog topic scout for Automatos AI. Your job is to find "
            "a compelling topic and launch a mission that will research and write it.\n\n"
            "## Workflow\n"
            "1. Use platform_search_memory to check what topics have been covered\n"
            "2. Use platform_list_blog_posts to see existing published content\n"
            "3. Pick a fresh, trending topic in the given category that hasn't been covered\n"
            "4. Launch a mission with platform_create_mission — the goal should be specific:\n"
            "   - What to research (3-5 specific angles/questions)\n"
            "   - Target audience and tone (technical professionals, accessible)\n"
            "   - Required output: a blog post draft created via platform_publish_blog_post\n"
            "   - Quality bar: cite real data/tools/examples, 1000-2000 words\n"
            "   - Must call platform_publish_blog_post(publish_immediately=false) at the end\n\n"
            "## Topic Selection Guidelines\n"
            "- Avoid topics already covered (check memory + existing posts)\n"
            "- Prefer timely topics with recent developments\n"
            "- Focus on practical, actionable content over theory\n"
            "- Tie topics back to AI automation, agents, or orchestration when possible"
        ),
    },
    # CANVAS is a general-purpose image agent — used in chat, social posts,
    # blog covers, and any other on-demand image creation. It is NOT seeded
    # by this blog-specific script. The blog playbook uses the
    # platform_generate_cover_image tool directly (no agent role-binding) so
    # CANVAS stays free for general use without being coupled to the blog
    # pipeline.
]


# ---------------------------------------------------------------------------
# Playbook definition (v2 — mission-powered)
# ---------------------------------------------------------------------------

BLOG_PLAYBOOK = {
    "name": "Blog Pipeline",
    "template_id": "daily-blog-pipeline",
    "description": (
        "Mission-powered blog pipeline. QUILL scouts a trending topic and "
        "calls platform_create_blog_post — the tool fires a mission that "
        "handles research, writing, editing, cover image generation (via "
        "the configured BLOG_COVER_MODEL), and a human-review board task. "
        "Runs Tue/Fri at 09:00 UTC."
    ),
    "category": "Content Creation",
    "tags": ["blog", "content", "mission", "autonomous", "writing", "scheduled"],
    "recipe_type": "workflow",
    "inputs": {
        "category": {
            "type": "string",
            "required": True,
            "default": "AI & Automation",
            "description": "Blog topic category (e.g. AI Agents, Developer Tools, Cloud Infrastructure)",
        },
    },
    "execution_config": {
        "mode": "sequential",
        "max_retries": 1,
        "timeout_per_step": 600,
        "total_timeout": 1800,
    },
    "schedule_config": {
        "type": "cron",
        "cron_expression": "0 9 * * 2,5",
    },
    "steps": [
        {
            "step_id": "quill_launch_mission",
            "order": 1,
            "agent_name": "QUILL",
            "prompt_template": (
                "Pick a fresh, compelling topic in the '{input.category}' "
                "category for the blog and dispatch a full-pipeline mission to "
                "write it.\n\n"
                "## Steps\n"
                "1. Check platform_search_memory and platform_list_blog_posts to "
                "see what topics we've already covered. Avoid duplicates.\n"
                "2. Pick ONE specific, timely topic with a clear angle. Be "
                "concrete (e.g. 'Multi-agent orchestration for Shopify stores' "
                "— not 'AI in e-commerce').\n"
                "3. Call platform_create_blog_post(topic=<your chosen topic>, "
                "category='{input.category}'). That's it. The tool fires a "
                "mission that handles research, writing, publishing, cover "
                "image, and the human-review board task — all server-side.\n\n"
                "## What you do NOT need to do\n"
                "- Do NOT call platform_create_mission directly.\n"
                "- Do NOT build the goal template by hand.\n"
                "- Do NOT call platform_publish_blog_post — the mission does it.\n"
                "- Do NOT pick a sub-topic for cover art — the cover step is "
                "auto-included.\n\n"
                "Your job is topic selection. The platform handles the rest."
            ),
            "max_iterations": 15,
            "error_handling": "stop",
        },
    ],
    "metadata": {
        "required_tools": [],
        "required_skills": ["pattern_recognition"],
        "suggested_agents": ["QUILL"],
        "trigger_type": "cron",
        "schedule": "0 9 * * 2,5",
        "estimated_time": "15-30 minutes",
        "cost_tier": "standard",
    },
}


def seed_blog_playbook():
    """Seed content agents + blog playbook into the marketplace."""
    print("Seeding Blog Pipeline v2 (mission-powered)...")

    engine = create_engine(get_database_url())

    with engine.connect() as db:
        trans = db.begin()

        try:
            # --- Seed agents into marketplace ---
            for agent in CONTENT_AGENTS:
                # Delete existing by name
                db.execute(text(
                    "DELETE FROM marketplace_items WHERE type = 'agent' "
                    "AND creator_name = 'Automatos Team' AND name = :name"
                ), {"name": agent["name"]})

                metadata = {
                    "required_tools": agent["tools"],
                    "required_skills": agent.get("skills", []),
                    "model_id": agent["model_id"],
                    "system_prompt": agent.get("system_prompt", ""),
                    "agent_type": agent["agent_type"],
                }

                result = db.execute(text("""
                    INSERT INTO marketplace_items (
                        type, name, description, creator_name, category, tags,
                        install_count, is_featured, is_approved, version, metadata,
                        created_at, updated_at
                    ) VALUES (
                        'agent', :name, :description, 'Automatos Team', :category, :tags,
                        0, true, true, '1.0.0', :metadata,
                        NOW(), NOW()
                    )
                    RETURNING id
                """), {
                    "name": agent["name"],
                    "description": agent["description"],
                    "category": agent["category"],
                    "tags": json.dumps(agent["tags"]),
                    "metadata": json.dumps(metadata),
                })
                agent_id = result.scalar()
                print(f"  Agent: {agent['name']} (ID: {agent_id})")

            # Also remove EDITOR from marketplace (no longer needed — mission handles editing)
            db.execute(text(
                "DELETE FROM marketplace_items WHERE type = 'agent' "
                "AND creator_name = 'Automatos Team' AND name = 'EDITOR'"
            ))
            print("  Removed: EDITOR (mission handles editing now)")

            # --- Seed playbook into marketplace ---
            db.execute(text(
                "DELETE FROM marketplace_items WHERE type = 'recipe' "
                "AND creator_name = 'Automatos Team' AND name IN (:n1, :n2)"
            ), {"n1": BLOG_PLAYBOOK["name"], "n2": "Daily Blog Pipeline"})

            playbook_metadata = BLOG_PLAYBOOK["metadata"].copy()
            playbook_metadata["steps"] = [
                f"Step {s['order']}: {s['agent_name']} — {s['step_id']}"
                for s in BLOG_PLAYBOOK["steps"]
            ]

            result = db.execute(text("""
                INSERT INTO marketplace_items (
                    type, name, description, creator_name, category, tags,
                    install_count, is_featured, is_approved, version, metadata,
                    created_at, updated_at
                ) VALUES (
                    'recipe', :name, :description, 'Automatos Team', :category, :tags,
                    0, true, true, '2.0.0', :metadata,
                    NOW(), NOW()
                )
                RETURNING id
            """), {
                "name": BLOG_PLAYBOOK["name"],
                "description": BLOG_PLAYBOOK["description"],
                "category": BLOG_PLAYBOOK["category"],
                "tags": json.dumps(BLOG_PLAYBOOK["tags"]),
                "metadata": json.dumps(playbook_metadata),
            })
            playbook_id = result.scalar()
            print(f"  Playbook: {BLOG_PLAYBOOK['name']} (ID: {playbook_id})")

            # --- If WORKSPACE_ID is set, also update the live playbook ---
            workspace_id = os.environ.get("WORKSPACE_ID")
            if workspace_id:
                print(f"\n  Updating live playbook in workspace {workspace_id}...")

                # Look up agent IDs by name in this workspace
                agent_map = {}
                for agent in CONTENT_AGENTS:
                    row = db.execute(text(
                        "SELECT id FROM agents WHERE workspace_id = :ws AND name = :name LIMIT 1"
                    ), {"ws": workspace_id, "name": agent["name"]}).fetchone()
                    if row:
                        agent_map[agent["name"]] = row[0]
                        print(f"    Found agent: {agent['name']} (ID: {row[0]})")
                    else:
                        print(f"    Agent not found: {agent['name']} — install from marketplace first")

                if "QUILL" in agent_map:
                    # Build steps with actual agent IDs
                    steps = []
                    for step in BLOG_PLAYBOOK["steps"]:
                        agent_id = agent_map.get(step["agent_name"])
                        if agent_id:
                            steps.append({
                                "step_id": step["step_id"],
                                "order": step["order"],
                                "agent_id": agent_id,
                                "prompt_template": step["prompt_template"],
                                "max_iterations": step.get("max_iterations", 15),
                                "error_handling": step.get("error_handling", "stop"),
                            })

                    # Update existing playbook or create new one
                    existing = db.execute(text(
                        "SELECT id FROM workflow_recipes WHERE workspace_id = :ws "
                        "AND template_id = :tid LIMIT 1"
                    ), {"ws": workspace_id, "tid": BLOG_PLAYBOOK["template_id"]}).fetchone()

                    if existing:
                        db.execute(text("""
                            UPDATE workflow_recipes SET
                                name = :name,
                                description = :description,
                                steps = :steps,
                                inputs = :inputs,
                                execution_config = :exec_config,
                                schedule_config = :sched_config,
                                tags = :tags,
                                category = :category,
                                updated_at = NOW()
                            WHERE id = :id
                        """), {
                            "id": existing[0],
                            "name": BLOG_PLAYBOOK["name"],
                            "description": BLOG_PLAYBOOK["description"],
                            "steps": json.dumps(steps),
                            "inputs": json.dumps(BLOG_PLAYBOOK["inputs"]),
                            "exec_config": json.dumps(BLOG_PLAYBOOK["execution_config"]),
                            "sched_config": json.dumps(BLOG_PLAYBOOK["schedule_config"]),
                            "tags": json.dumps(BLOG_PLAYBOOK["tags"]),
                            "category": BLOG_PLAYBOOK["category"],
                        })
                        print(f"    Updated existing playbook (ID: {existing[0]})")
                    else:
                        db.execute(text("""
                            INSERT INTO workflow_recipes (
                                workspace_id, owner_type, template_id, name, description,
                                steps, inputs, execution_config, schedule_config,
                                tags, category, is_public, is_featured,
                                created_at, updated_at
                            ) VALUES (
                                :ws, 'workspace', :tid, :name, :description,
                                :steps, :inputs, :exec_config, :sched_config,
                                :tags, :category, true, false,
                                NOW(), NOW()
                            )
                            RETURNING id
                        """), {
                            "ws": workspace_id,
                            "tid": BLOG_PLAYBOOK["template_id"],
                            "name": BLOG_PLAYBOOK["name"],
                            "description": BLOG_PLAYBOOK["description"],
                            "steps": json.dumps(steps),
                            "inputs": json.dumps(BLOG_PLAYBOOK["inputs"]),
                            "exec_config": json.dumps(BLOG_PLAYBOOK["execution_config"]),
                            "sched_config": json.dumps(BLOG_PLAYBOOK["schedule_config"]),
                            "tags": json.dumps(BLOG_PLAYBOOK["tags"]),
                            "category": BLOG_PLAYBOOK["category"],
                        })
                        print("    Created new playbook in workspace")

                    # QUILL stays on a cheap tool-capable text model (topic
                    # scouting + topic selection is text-only). CANVAS is left
                    # alone — operators may have customized it for general
                    # image creation use cases beyond blog covers.
                    for name, model in [
                        ("QUILL", "mistralai/mistral-small-3.1-24b-instruct"),
                    ]:
                        if name in agent_map:
                            db.execute(text(
                                "UPDATE agents SET model_id = :model, updated_at = NOW() "
                                "WHERE id = :id"
                            ), {"id": agent_map[name], "model": model})
                            print(f"    Updated {name} model → {model}")

                else:
                    print("    Skipping workspace playbook — QUILL agent required")

            trans.commit()
            print("\nDone!")

        except Exception as e:
            trans.rollback()
            print(f"Error: {e}")
            raise


if __name__ == "__main__":
    seed_blog_playbook()
