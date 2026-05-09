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
    {
        "name": "CANVAS",
        "agent_type": "custom",
        "description": (
            "Cover art designer agent. Generates cover images for blog posts and "
            "persists them to workspace storage. Image-generation focus only — does "
            "NOT attach images to posts; that is done by a downstream tool-capable "
            "agent that reads the URL from field memory."
        ),
        "category": "Content Creation",
        "tags": ["blog", "design", "cover-image", "image-generation"],
        "tools": [],
        # gemini-2.5-flash supports tool use AND multi-modal output, so CANVAS can
        # both call workspace_write_file to persist and (when wired) generate images.
        "model_id": "google/gemini-2.5-flash",
        "skills": [],
        "system_prompt": (
            "You are CANVAS, a cover art designer for Automatos AI blog posts. Your job "
            "is to generate a cover image and persist it to workspace storage. A separate "
            "downstream step attaches the image URL to the blog post — DO NOT call "
            "platform_update_blog_post yourself.\n\n"
            "## Workflow\n"
            "1. Use platform_list_blog_posts(status=draft) to find the latest draft\n"
            "2. Use platform_get_blog_post to read its title, excerpt, slug, and category\n"
            "3. Generate a cover image (16:9, abstract/conceptual, no embedded text). "
            "Use composio_execute to call an image-generation action (DALL-E, "
            "Stability, Replicate, or Gemini) — pick whichever is installed in this "
            "workspace. If the workflow has an image_prompt input, use that.\n"
            "4. Persist the generated image to the workspace at "
            "`content/blog/images/{slug}.png` using workspace_write_file (binary).\n"
            "5. Get the public URL via workspace_get_public_url for that path.\n"
            "6. Output a single line of JSON to your final response: "
            "`{\"post_id\": \"<uuid>\", \"cover_image_url\": \"<public_url>\"}` so the "
            "downstream attach step can read it from field memory.\n\n"
            "## Design Guidelines\n"
            "- 16:9 aspect ratio, abstract/conceptual imagery\n"
            "- No embedded text (title overlay is handled by CSS at render time)\n"
            "- Modern, clean aesthetic that matches Automatos brand\n"
            "- Prefer geometric shapes, gradients, and tech motifs over literal scenes"
        ),
    },
]


# ---------------------------------------------------------------------------
# Playbook definition (v2 — mission-powered)
# ---------------------------------------------------------------------------

BLOG_PLAYBOOK = {
    "name": "Blog Pipeline",
    "template_id": "daily-blog-pipeline",
    "description": (
        "Mission-powered blog pipeline. QUILL scouts a trending topic and launches "
        "a research mission that handles deep research, writing, and editing via "
        "multiple agents. CANVAS generates cover art. Finally, a review task is "
        "created with a one-click publish approval gate. Runs Tue/Fri at 09:00 UTC."
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
                "Find a fresh, trending topic in the '{input.category}' category for our blog.\n\n"
                "1. Check platform_search_memory and platform_list_blog_posts to see what "
                "we've already covered — avoid duplicates.\n"
                "2. Pick a specific, compelling topic with a clear angle.\n"
                "3. Launch a mission using platform_create_mission with a goal that covers "
                "the ENTIRE pipeline end-to-end:\n\n"
                "   Goal template:\n"
                "   'Research and write a high-quality blog post about [TOPIC]. "
                "   Investigate [2-3 specific angles]. Include real-world examples, "
                "   data points, and expert perspectives. The post should be 1000-2000 words, "
                "   written for technical professionals.\n\n"
                "   The mission MUST complete ALL of these steps:\n"
                "   1. Research the topic thoroughly from multiple angles.\n"
                "   2. Write the FULL blog post draft as polished prose — actual paragraphs, "
                "headings, examples, and conclusions. NOT an outline, summary, or "
                "bracketed placeholder.\n"
                "   3. Edit and SEO-review for accuracy, clarity, and readability.\n"
                "   4. Publish the draft via platform_publish_blog_post — IMPORTANT: pass the "
                "FULL article body (1000-2000 words of actual writing) as the `content` "
                "argument. Do NOT pass placeholder text like \"[blog content here]\" or a "
                "summary — the server validates content and will reject anything that "
                "looks like a placeholder. Required args: title, content (full article), "
                "excerpt (under 300 chars), tags (array), category: {input.category}, "
                "publish_immediately: false. Save the returned post_id — it is needed for "
                "the next steps.\n"
                "   5. Generate a cover image: dispatch a CANVAS task that produces the "
                "image and persists it to workspace storage. CANVAS will write its output "
                "to field memory as JSON: {\"post_id\":\"...\",\"cover_image_url\":\"...\"}.\n"
                "   6. Attach the cover: read CANVAS output from field memory, then call "
                "platform_update_blog_post(post_id=<from step 4>, cover_image_url=<from "
                "CANVAS>). This step needs a tool-capable agent (NOT the image-gen "
                "model) — assign it to QUILL or any role with a text LLM.\n"
                "   7. Create a board task for human review via platform_create_task with "
                "title: Review & Publish: [post title], approval_action: "
                "{type: publish_blog, post_id: [the post UUID]}, priority: high, "
                "auto_approve: true, tags: [blog, approval]'\n\n"
                "The mission handles EVERYTHING — research, writing, images, attaching, "
                "and publishing."
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

                    # Update agent models. QUILL stays on a cheap text model (topic
                    # scouting + writing is text-heavy). CANVAS uses gemini-2.5-flash
                    # which is multi-modal AND tool-capable — so it can call image-gen
                    # actions via composio_execute and persist via workspace_write_file.
                    # Do NOT use gemini-3-pro-image-preview here — it does not support
                    # tool use, so the agent cannot save its output anywhere.
                    for name, model in [
                        ("QUILL", "mistralai/mistral-small-3.1-24b-instruct"),
                        ("CANVAS", "google/gemini-2.5-flash"),
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
