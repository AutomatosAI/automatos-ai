"""
Seed the Daily Blog Pipeline: QUILL, EDITOR, CANVAS agents + playbook.

Creates three content agents in the marketplace and a 4-step daily playbook
that researches, writes, reviews, designs cover art, and queues for publish.

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
            "Blog writer agent. Researches trending topics, checks workspace memory "
            "for previously covered subjects, and drafts engaging long-form blog posts "
            "in markdown. Creates posts as drafts for editorial review before publishing. "
            "Uses web search when available to ground articles in current data."
        ),
        "category": "Content Creation",
        "tags": ["blog", "writing", "content", "research", "seo", "draft"],
        "tools": [],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": ["pattern_recognition", "Data Analysis"],
        "system_prompt": (
            "You are QUILL, an expert blog writer for Automatos AI. Your job is to "
            "research and write high-quality, engaging blog posts in markdown format.\n\n"
            "## Workflow\n"
            "1. Use platform_search_memory to check what topics have been covered before\n"
            "2. Use platform_list_blog_posts to see existing published content\n"
            "3. Research a trending topic relevant to the given category\n"
            "4. Write a comprehensive blog post (800-1500 words) in clean markdown\n"
            "5. Create the draft with platform_publish_blog_post(publish_immediately=false)\n\n"
            "## Writing Guidelines\n"
            "- Strong, SEO-friendly headline\n"
            "- Compelling opening paragraph that hooks the reader\n"
            "- Clear section structure with ## headings\n"
            "- Include data points, examples, or comparisons where possible\n"
            "- End with a conclusion and call-to-action\n"
            "- Write a concise excerpt (under 300 chars) for the preview card\n"
            "- Add relevant tags and a category\n"
            "- ALWAYS create as draft (publish_immediately=false) — never publish directly"
        ),
    },
    {
        "name": "EDITOR",
        "agent_type": "custom",
        "description": (
            "Blog editor agent. Reviews draft blog posts for clarity, engagement, "
            "grammar, SEO quality, and factual accuracy. Improves content without "
            "changing the author's voice. Never publishes — only improves drafts."
        ),
        "category": "Content Creation",
        "tags": ["blog", "editing", "review", "seo", "quality", "draft"],
        "tools": [],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": ["pattern_recognition"],
        "system_prompt": (
            "You are EDITOR, an expert blog editor for Automatos AI. Your job is to "
            "review and improve draft blog posts without changing the writer's voice.\n\n"
            "## Workflow\n"
            "1. Use platform_list_blog_posts(status=draft) to find the latest draft\n"
            "2. Use platform_get_blog_post to read the full content\n"
            "3. Review for: clarity, flow, grammar, engagement, SEO, factual accuracy\n"
            "4. Improve the post with platform_update_blog_post\n\n"
            "## Editorial Guidelines\n"
            "- Fix grammar and awkward phrasing\n"
            "- Strengthen weak openings and conclusions\n"
            "- Improve headline for SEO and click-through\n"
            "- Ensure section headings are clear and descriptive\n"
            "- Add transition sentences between sections if flow is choppy\n"
            "- Tighten verbose paragraphs — cut fluff\n"
            "- Verify the excerpt is compelling and under 300 chars\n"
            "- NEVER publish — only improve the draft"
        ),
    },
    {
        "name": "CANVAS",
        "agent_type": "custom",
        "description": (
            "Cover art designer agent. Generates cover images for blog posts using "
            "AI image generation (DALL-E via Composio when available). Reads the post "
            "title and excerpt to create a relevant, visually appealing cover image. "
            "Falls back to tagging posts that need manual cover art."
        ),
        "category": "Content Creation",
        "tags": ["blog", "design", "cover-image", "dall-e", "image-generation"],
        "tools": [],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": [],
        "system_prompt": (
            "You are CANVAS, a cover art designer for Automatos AI blog posts. Your job "
            "is to generate or source compelling cover images for draft blog posts.\n\n"
            "## Workflow\n"
            "1. Use platform_list_blog_posts(status=draft) to find the latest draft\n"
            "2. Use platform_get_blog_post to read the title and excerpt\n"
            "3. Generate a cover image description based on the post topic\n"
            "4. If you have access to an image generation tool (DALL-E, etc.), generate the image\n"
            "5. Update the post with platform_update_blog_post(cover_image_url=...)\n"
            "6. If no image generation tool is available, update the post tags to include "
            "   'needs-cover-image' so a human can add one later\n\n"
            "## Design Guidelines\n"
            "- Cover images should be wide format (16:9 aspect ratio)\n"
            "- Use clean, modern design aesthetic\n"
            "- Avoid text in the image (the title overlay is handled by CSS)\n"
            "- Make the image relevant to the post topic\n"
            "- Prefer abstract/conceptual imagery over literal illustrations"
        ),
    },
]


# ---------------------------------------------------------------------------
# Playbook definition
# ---------------------------------------------------------------------------

BLOG_PLAYBOOK = {
    "name": "Daily Blog Pipeline",
    "template_id": "daily-blog-pipeline",
    "description": (
        "Autonomous daily blog pipeline. QUILL researches and writes a draft, "
        "EDITOR reviews and improves it, CANVAS generates a cover image, and a "
        "review task is created for human approval. Runs daily at 09:00 UTC."
    ),
    "category": "Content Creation",
    "tags": ["blog", "content", "daily", "autonomous", "writing", "scheduled"],
    "recipe_type": "workflow",
    "inputs": {
        "category": {
            "type": "string",
            "required": True,
            "default": "AI & Automation",
            "description": "Blog topic category (e.g. AI, Engineering, Business)",
        },
    },
    "execution_config": {
        "mode": "sequential",
        "max_retries": 1,
        "timeout_per_step": 300,
        "total_timeout": 900,
    },
    "schedule_config": {
        "type": "cron",
        "cron_expression": "0 9 * * *",
    },
    "steps": [
        {
            "step_id": "quill_write",
            "order": 1,
            "agent_name": "QUILL",
            "prompt_template": (
                "Research a trending topic in the '{input.category}' category and write "
                "an engaging blog post. First check platform_search_memory and "
                "platform_list_blog_posts to avoid repeating topics we've already covered. "
                "Create the post as a draft using platform_publish_blog_post with "
                "publish_immediately=false. Include relevant tags and a compelling excerpt."
            ),
            "max_iterations": 15,
            "error_handling": "stop",
        },
        {
            "step_id": "editor_review",
            "order": 2,
            "agent_name": "EDITOR",
            "prompt_template": (
                "Review and improve the latest draft blog post. Use "
                "platform_list_blog_posts(status=draft) to find it, then "
                "platform_get_blog_post to read the full content. Improve clarity, "
                "engagement, grammar, and SEO using platform_update_blog_post. "
                "Do NOT publish — only improve the draft."
            ),
            "max_iterations": 15,
            "error_handling": "skip",
        },
        {
            "step_id": "canvas_design",
            "order": 3,
            "agent_name": "CANVAS",
            "prompt_template": (
                "Find the latest draft blog post and generate a cover image for it. "
                "Use platform_list_blog_posts(status=draft) to find it, then "
                "platform_get_blog_post to read the title and excerpt. Generate an "
                "appropriate cover image and update the post with the cover_image_url "
                "using platform_update_blog_post. If no image generation tool is "
                "available, add 'needs-cover-image' to the post tags."
            ),
            "max_iterations": 10,
            "error_handling": "skip",
        },
        {
            "step_id": "create_review_task",
            "order": 4,
            "agent_name": "QUILL",
            "prompt_template": (
                "The blog post draft has been written, reviewed, and designed. "
                "First, use platform_list_blog_posts(status=draft) to find the latest draft post. "
                "Then create a board task for human approval using platform_create_task with:\n"
                "- title: 'Review & Publish: [the actual post title]'\n"
                "- description: 'Blog post ready for final review. Approve to publish live.'\n"
                "- approval_action: {\"type\": \"publish_blog\", \"post_id\": \"[the actual post UUID]\"}\n"
                "- priority: 'high'\n"
                "- tags: ['blog', 'approval']\n\n"
                "The approval_action field is CRITICAL — it enables the one-click publish "
                "button on the board. Use the actual post_id from the draft you found."
            ),
            "max_iterations": 5,
            "error_handling": "skip",
        },
    ],
    "metadata": {
        "required_tools": [],
        "required_skills": ["pattern_recognition", "Data Analysis"],
        "suggested_agents": ["QUILL", "EDITOR", "CANVAS"],
        "trigger_type": "cron",
        "schedule": "0 9 * * *",
        "estimated_time": "10-15 minutes",
        "cost_tier": "premium",
    },
}


def seed_blog_playbook():
    """Seed content agents + daily blog playbook into the marketplace."""
    print("Seeding Daily Blog Pipeline...")

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

            # --- Seed playbook into marketplace ---
            db.execute(text(
                "DELETE FROM marketplace_items WHERE type = 'recipe' "
                "AND creator_name = 'Automatos Team' AND name = :name"
            ), {"name": BLOG_PLAYBOOK["name"]})

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
                    0, true, true, '1.0.0', :metadata,
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

            # --- If WORKSPACE_ID is set, also create the actual playbook in workflow_recipes ---
            workspace_id = os.environ.get("WORKSPACE_ID")
            if workspace_id:
                print(f"\n  Creating playbook in workspace {workspace_id}...")

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

                if len(agent_map) >= 2:  # At least QUILL and EDITOR
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

                    # Delete existing playbook by template_id
                    db.execute(text(
                        "DELETE FROM workflow_recipes WHERE workspace_id = :ws "
                        "AND template_id = :tid"
                    ), {"ws": workspace_id, "tid": BLOG_PLAYBOOK["template_id"]})

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
                    print(f"    Playbook created in workspace!")
                else:
                    print("    Skipping workspace playbook — install agents first")

            trans.commit()
            print("\nDone!")

        except Exception as e:
            trans.rollback()
            print(f"Error: {e}")
            raise


if __name__ == "__main__":
    seed_blog_playbook()
