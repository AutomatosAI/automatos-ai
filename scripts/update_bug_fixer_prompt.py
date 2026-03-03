#!/usr/bin/env python3
"""
Update Bug Fixer Recipe — Step 2 Prompt
=========================================

Replaces the bug fixer recipe's step 2 prompt to use workspace tools
(workspace_git, workspace_grep, workspace_read_file, workspace_write_file,
workspace_exec) instead of raw GitHub API actions that require blob SHAs
and base64 encoding.

Usage:
    python scripts/update_bug_fixer_prompt.py [--dry-run]

Flags:
    --dry-run   Print the update without touching the DB.
"""

import argparse
import sys
from pathlib import Path

# Allow imports from the orchestrator package
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database.database import SessionLocal, init_database  # noqa: E402
from core.models.core import WorkflowTemplate  # noqa: E402
from sqlalchemy.orm.attributes import flag_modified  # noqa: E402

# ── New Step 2 Prompt ────────────────────────────────────────────────

NEW_STEP_2_PROMPT = """\
You are fixing Jira ticket {input.issue_key} in the repo AutomatosAI/automatos-ai.

## Step A: Set up the workspace
1. workspace_git operation="checkout" args="-b fix/{input.issue_key}" to create a fix branch.

## Step B: Find the relevant code
1. workspace_grep to search for keywords from the ticket (error messages, function names).
2. workspace_read_file to read files you need to understand.
3. Identify the root cause before making changes.

## Step C: Apply the fix
1. workspace_write_file with corrected content for each file.
2. Keep changes minimal — fix the bug, don't refactor.

## Step D: Verify
1. workspace_exec command="pytest tests/ -x -q" (or the relevant test file).
2. If tests fail, adjust and re-verify.

## Step E: Commit and push
1. workspace_git operation="add" args="-A"
2. workspace_git operation="commit" args="-m '[{input.issue_key}] <short fix description>'"
3. workspace_git operation="push" args="origin fix/{input.issue_key}"

## Step F: Open a draft PR
Use composio_execute GITHUB_CREATE_A_PULL_REQUEST:
  - title: "[{input.issue_key}] <summary>"
  - body: "Fixes {input.issue_key}\\n\\n## What changed\\n<describe>\\n\\n## Tests\\n<pass/fail>"
  - head: "fix/{input.issue_key}", base: "main", draft: true

## Export to scratchpad
scratchpad_write: branch_name, pr_url, files_changed, fix_applied, tests_passed
"""

# ── Name patterns to match the bug fixer recipe ─────────────────────

BUG_FIXER_PATTERNS = [
    "%bug fix%",
    "%bug triage%",
    "%jira bug%",
    "%bug fixer%",
]


def find_bug_fixer_recipe(db):
    """Find the bug fixer recipe by name pattern."""
    for pattern in BUG_FIXER_PATTERNS:
        recipe = (
            db.query(WorkflowTemplate)
            .filter(WorkflowTemplate.name.ilike(pattern))
            .first()
        )
        if recipe:
            return recipe
    return None


def main():
    parser = argparse.ArgumentParser(description="Update bug fixer recipe step 2 prompt")
    parser.add_argument("--dry-run", action="store_true", help="Print without DB changes")
    args = parser.parse_args()

    init_database()
    db = SessionLocal()

    try:
        recipe = find_bug_fixer_recipe(db)
        if not recipe:
            print("ERROR: Bug fixer recipe not found. Searched patterns:", BUG_FIXER_PATTERNS)
            sys.exit(1)

        print(f"Found recipe: '{recipe.name}' (id={recipe.id})")

        steps = list(recipe.steps or [])
        if len(steps) < 2:
            print(f"ERROR: Recipe has {len(steps)} steps, expected at least 2")
            sys.exit(1)

        print(f"  Steps: {len(steps)}")
        print(f"  Step 2 current prompt preview: {(steps[1].get('prompt_template', '') or '')[:120]}...")

        if args.dry_run:
            print("\n--- DRY RUN: New step 2 prompt ---")
            print(NEW_STEP_2_PROMPT)
            print("--- END DRY RUN ---")
            return

        steps[1]["prompt_template"] = NEW_STEP_2_PROMPT
        recipe.steps = steps
        flag_modified(recipe, "steps")
        db.commit()

        print(f"  Updated step 2 prompt ({len(NEW_STEP_2_PROMPT)} chars)")
        print("Done.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
