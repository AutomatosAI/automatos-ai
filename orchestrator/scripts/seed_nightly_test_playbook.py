"""
Seed the Nightly Test Pipeline playbook.

5-step playbook that runs the API test suite, reads results, fetches logs,
creates bug tickets on the board for each failure, and submits a QA report.

The key fix: each logical phase is its own step with focused instructions.
Earlier versions jammed all 5 phases into one monolithic prompt, so the LLM
would summarise failures instead of calling platform_create_task per failure.

Usage:
    python scripts/seed_nightly_test_playbook.py
    WORKSPACE_ID=... python scripts/seed_nightly_test_playbook.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine, text
from core.database.database import get_database_url


# ---------------------------------------------------------------------------
# Playbook definition — 5 steps, each with a focused prompt
# ---------------------------------------------------------------------------

NIGHTLY_PLAYBOOK = {
    "name": "Nightly Test Pipeline",
    "template_id": "nightly-test-pipeline",
    "description": (
        "Runs the API test suite nightly, reads results, fetches Loki logs for "
        "each failure, creates a board task per failure assigned to PATCHER, "
        "and submits a QA report."
    ),
    "category": "QA & Testing",
    "tags": ["nightly-test", "qa", "automated", "testing", "ci"],
    "recipe_type": "workflow",
    "inputs": {},
    "execution_config": {
        "mode": "sequential",
        "max_retries": 1,
        "timeout_per_step": 600,
        "total_timeout": 1800,
    },
    "schedule_config": {
        "type": "cron",
        "cron_expression": "0 3 * * *",  # 03:00 UTC daily
    },
    "steps": [
        # ── Step 1: Run tests ──────────────────────────────────────────
        {
            "step_id": "run_tests",
            "order": 1,
            "agent_name": "QA_RUNNER",
            "prompt_template": (
                "You are a QA agent. Run the nightly test suite.\n\n"
                "Execute this command using workspace_exec:\n"
                "  python3 tests/run_nightly.py\n\n"
                "After it finishes, use scratchpad_write with key='test_exit_code' "
                "and value set to the exit code from the output.\n\n"
                "Then read the results file using workspace_read_file with "
                "path: artifacts/results/test-summary.json\n\n"
                "IMPORTANT: The path is artifacts/results/test-summary.json "
                "(workspace root, NOT inside repos/). If workspace_read_file fails, "
                "try: workspace_exec with command: cat artifacts/results/test-summary.json\n\n"
                "Once you have the JSON content, use scratchpad_write with "
                "key='test_results' and the FULL JSON content as the value. "
                "Do NOT summarise — write the complete JSON."
            ),
            "max_iterations": 10,
            "error_handling": "stop",
        },
        # ── Step 2: Fetch logs per failure ─────────────────────────────
        {
            "step_id": "fetch_logs",
            "order": 2,
            "agent_name": "QA_RUNNER",
            "prompt_template": (
                "You are a QA agent. Read the test results from the previous step "
                "using scratchpad_read with key='test_results'.\n\n"
                "Parse the JSON. If the 'failures' array is empty, write "
                "scratchpad_write key='failure_logs' value='[]' and stop.\n\n"
                "For EACH failure in the array:\n"
                "1. Extract the test file name from the nodeid "
                "   (e.g. 'test_heartbeat' from 'tests/api/test_heartbeat.py::test_name')\n"
                "2. Call platform_execute with:\n"
                "   action: platform_query_loki_logs\n"
                "   params:\n"
                "     service: automatos-backend\n"
                "     search: <test_file_name>\n"
                "     minutes: 15\n"
                "     limit: 20\n\n"
                "Collect all log results. Then use scratchpad_write with "
                "key='failure_logs' and value set to a JSON array where each "
                "entry has {nodeid, logs} — the nodeid from the failure and "
                "the log lines you fetched.\n\n"
                "If Loki returns no logs for a failure, set logs to "
                "'No matching logs found'."
            ),
            "max_iterations": 25,
            "error_handling": "continue",
        },
        # ── Step 3: Create bug tickets ─────────────────────────────────
        {
            "step_id": "create_tickets",
            "order": 3,
            "agent_name": "QA_RUNNER",
            "prompt_template": (
                "You are a QA agent. Your ONLY job is to create board tasks for "
                "test failures. Do NOT do anything else.\n\n"
                "1. Read scratchpad_read key='test_results' to get the test JSON.\n"
                "2. Read scratchpad_read key='failure_logs' to get the log data.\n\n"
                "Parse the test results JSON. Look at the 'failures' array.\n\n"
                "## If there are failures\n\n"
                "For EACH failure, call platform_execute with:\n"
                "  action: platform_create_task\n"
                "  params:\n"
                "    title: \"[Nightly Test] {nodeid} — {first 80 chars of assertion_message}\"\n"
                "    description: (use the EXACT format below)\n"
                "    priority: \"high\"\n"
                "    tags: [\"nightly-test\", \"bug\", \"automated\"]\n"
                "    assigned_agent_name: \"PATCHER\"\n\n"
                "The description MUST follow this EXACT format — copy errors VERBATIM:\n\n"
                "```\n"
                "## Failed Test\n"
                "**Test:** {nodeid}\n"
                "**Source Files:** {source_files joined by comma}\n\n"
                "## Error Traceback\n"
                "```\n"
                "{paste the ENTIRE error field from the JSON — do NOT summarise}\n"
                "```\n\n"
                "## Assertion Message\n"
                "{paste assertion_message VERBATIM}\n\n"
                "## Relevant Logs\n"
                "{paste log lines from failure_logs for this nodeid, "
                "or 'No matching logs found'}\n\n"
                "## Suggested Fix\n"
                "{Based on assertion_message and source_files, describe the fix}\n"
                "```\n\n"
                "CRITICAL: You MUST call platform_execute/platform_create_task "
                "once per failure. Do NOT skip any. Do NOT summarise.\n\n"
                "After creating all tasks, use scratchpad_write with "
                "key='tickets_created' and value set to the number of tasks created.\n\n"
                "## If ALL tests passed\n\n"
                "Create ONE task:\n"
                "  platform_execute action=platform_create_task params:\n"
                "    title: \"[Nightly Test] All tests passed\"\n"
                "    description: \"Nightly test suite passed with no failures.\"\n"
                "    priority: \"low\"\n"
                "    tags: [\"nightly-test\", \"passed\"]\n\n"
                "Then scratchpad_write key='tickets_created' value='1'."
            ),
            "max_iterations": 30,
            "error_handling": "continue",
        },
        # ── Step 4: Submit QA report ───────────────────────────────────
        {
            "step_id": "qa_report",
            "order": 4,
            "agent_name": "QA_RUNNER",
            "prompt_template": (
                "You are a QA agent. Submit a QA report summarising the nightly run.\n\n"
                "1. Read scratchpad_read key='test_results' for the numbers.\n"
                "2. Read scratchpad_read key='tickets_created' for ticket count.\n\n"
                "Call platform_execute with:\n"
                "  action: platform_submit_report\n"
                "  params:\n"
                "    title: \"Nightly Test Report\"\n"
                "    content: (a summary with: total tests, passed, failed, pass rate, "
                "duration, and how many bug tickets were created)\n"
                "    report_type: \"qa\"\n"
                "    status: \"pass\" if 0 failures, \"fail\" otherwise"
            ),
            "max_iterations": 5,
            "error_handling": "continue",
        },
    ],
    "metadata": {
        "required_tools": [
            "workspace_exec",
            "workspace_read_file",
            "platform_query_loki_logs",
            "platform_create_task",
            "platform_submit_report",
        ],
        "required_skills": [],
        "suggested_agents": ["QA_RUNNER"],
        "trigger_type": "cron",
        "schedule": "0 3 * * *",
        "estimated_time": "5-15 minutes",
        "cost_tier": "standard",
    },
}

# Agent that runs the tests — reuse an existing QA agent if present
QA_AGENT = {
    "name": "QA_RUNNER",
    "agent_type": "custom",
    "description": (
        "Autonomous QA agent. Runs test suites, reads results, fetches logs, "
        "creates bug tickets on the board, and submits QA reports. "
        "Executes all steps without asking for confirmation."
    ),
    "category": "QA & Testing",
    "tags": ["qa", "testing", "nightly", "automated", "bug-reporter"],
    "tools": [],
    "model_id": "anthropic/claude-sonnet-4",
    "skills": [],
    "system_prompt": (
        "You are QA_RUNNER, an autonomous QA agent for Automatos AI. "
        "You execute test suites and create actionable bug tickets.\n\n"
        "## Rules\n"
        "- Execute ALL steps without asking for confirmation\n"
        "- NEVER summarise errors — copy them VERBATIM into tickets\n"
        "- Always use platform_execute to call platform actions\n"
        "- Use scratchpad_write to pass data between playbook steps\n"
        "- Do NOT use JIRA or any external tool — use platform_create_task ONLY"
    ),
}


def seed_nightly_playbook():
    """Seed QA agent + nightly test playbook."""
    print("Seeding Nightly Test Pipeline...")

    engine = create_engine(get_database_url())

    with engine.connect() as db:
        trans = db.begin()

        try:
            # --- Seed QA agent into marketplace ---
            db.execute(text(
                "DELETE FROM marketplace_items WHERE type = 'agent' "
                "AND creator_name = 'Automatos Team' AND name = :name"
            ), {"name": QA_AGENT["name"]})

            metadata = {
                "model_id": QA_AGENT["model_id"],
                "system_prompt": QA_AGENT["system_prompt"],
                "skills": QA_AGENT["skills"],
                "tools": QA_AGENT["tools"],
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
                "name": QA_AGENT["name"],
                "description": QA_AGENT["description"],
                "category": QA_AGENT["category"],
                "tags": json.dumps(QA_AGENT["tags"]),
                "metadata": json.dumps(metadata),
            })
            agent_id = result.scalar()
            print(f"  Agent: {QA_AGENT['name']} (marketplace ID: {agent_id})")

            # --- Seed playbook into marketplace ---
            db.execute(text(
                "DELETE FROM marketplace_items WHERE type = 'recipe' "
                "AND creator_name = 'Automatos Team' AND name = :name"
            ), {"name": NIGHTLY_PLAYBOOK["name"]})

            playbook_metadata = NIGHTLY_PLAYBOOK["metadata"].copy()
            playbook_metadata["steps"] = [
                f"Step {s['order']}: {s['agent_name']} — {s['step_id']}"
                for s in NIGHTLY_PLAYBOOK["steps"]
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
                "name": NIGHTLY_PLAYBOOK["name"],
                "description": NIGHTLY_PLAYBOOK["description"],
                "category": NIGHTLY_PLAYBOOK["category"],
                "tags": json.dumps(NIGHTLY_PLAYBOOK["tags"]),
                "metadata": json.dumps(playbook_metadata),
            })
            playbook_id = result.scalar()
            print(f"  Playbook: {NIGHTLY_PLAYBOOK['name']} (marketplace ID: {playbook_id})")

            # --- If WORKSPACE_ID is set, update the live playbook ---
            workspace_id = os.environ.get("WORKSPACE_ID")
            if workspace_id:
                print(f"\n  Updating live playbook in workspace {workspace_id}...")

                # Look up QA_RUNNER agent (or fall back to any agent named PATCHER)
                qa_agent_row = db.execute(text(
                    "SELECT id FROM agents "
                    "WHERE workspace_id = :ws AND LOWER(name) IN ('qa_runner', 'patcher') "
                    "ORDER BY CASE WHEN LOWER(name) = 'qa_runner' THEN 0 ELSE 1 END "
                    "LIMIT 1"
                ), {"ws": workspace_id}).fetchone()

                if not qa_agent_row:
                    print("    No QA_RUNNER or PATCHER agent found. Creating QA_RUNNER...")
                    qa_agent_row = db.execute(text("""
                        INSERT INTO agents (
                            workspace_id, name, agent_type, description,
                            system_prompt, model_id, status,
                            created_at, updated_at
                        ) VALUES (
                            :ws, :name, 'custom', :description,
                            :system_prompt, :model_id, 'active',
                            NOW(), NOW()
                        )
                        RETURNING id
                    """), {
                        "ws": workspace_id,
                        "name": QA_AGENT["name"],
                        "description": QA_AGENT["description"],
                        "system_prompt": QA_AGENT["system_prompt"],
                        "model_id": QA_AGENT["model_id"],
                    }).fetchone()
                    print(f"    Created agent: QA_RUNNER (ID: {qa_agent_row[0]})")
                else:
                    print(f"    Found agent (ID: {qa_agent_row[0]})")

                qa_agent_id = qa_agent_row[0]

                # Build steps with the resolved agent ID
                steps = []
                for step in NIGHTLY_PLAYBOOK["steps"]:
                    steps.append({
                        "step_id": step["step_id"],
                        "order": step["order"],
                        "agent_id": qa_agent_id,
                        "prompt_template": step["prompt_template"],
                        "max_iterations": step.get("max_iterations", 25),
                        "error_handling": step.get("error_handling", "stop"),
                    })

                # Update or insert the playbook
                existing = db.execute(text(
                    "SELECT id FROM workflow_recipes WHERE workspace_id = :ws "
                    "AND template_id = :tid LIMIT 1"
                ), {"ws": workspace_id, "tid": NIGHTLY_PLAYBOOK["template_id"]}).fetchone()

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
                        "name": NIGHTLY_PLAYBOOK["name"],
                        "description": NIGHTLY_PLAYBOOK["description"],
                        "steps": json.dumps(steps),
                        "inputs": json.dumps(NIGHTLY_PLAYBOOK["inputs"]),
                        "exec_config": json.dumps(NIGHTLY_PLAYBOOK["execution_config"]),
                        "sched_config": json.dumps(NIGHTLY_PLAYBOOK["schedule_config"]),
                        "tags": json.dumps(NIGHTLY_PLAYBOOK["tags"]),
                        "category": NIGHTLY_PLAYBOOK["category"],
                    })
                    print(f"    Updated existing playbook (ID: {existing[0]})")
                else:
                    result = db.execute(text("""
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
                        "tid": NIGHTLY_PLAYBOOK["template_id"],
                        "name": NIGHTLY_PLAYBOOK["name"],
                        "description": NIGHTLY_PLAYBOOK["description"],
                        "steps": json.dumps(steps),
                        "inputs": json.dumps(NIGHTLY_PLAYBOOK["inputs"]),
                        "exec_config": json.dumps(NIGHTLY_PLAYBOOK["execution_config"]),
                        "sched_config": json.dumps(NIGHTLY_PLAYBOOK["schedule_config"]),
                        "tags": json.dumps(NIGHTLY_PLAYBOOK["tags"]),
                        "category": NIGHTLY_PLAYBOOK["category"],
                    })
                    new_id = result.scalar()
                    print(f"    Created new playbook (ID: {new_id})")

            trans.commit()
            print("\nDone.")

        except Exception as e:
            trans.rollback()
            print(f"\nERROR: {e}")
            raise


if __name__ == "__main__":
    seed_nightly_playbook()
