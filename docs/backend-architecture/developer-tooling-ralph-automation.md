# Developer Tooling & Ralph Automation

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.github/workflows/import-linter.yml](.github/workflows/import-linter.yml)
- [.gitignore](.gitignore)
- [docs/PRDS/126-BUSINESS-KNOWLEDGE-GRAPH.md](docs/PRDS/126-BUSINESS-KNOWLEDGE-GRAPH.md)
- [graphify-out/snapshots/bucket-1-pre-drop.sql](graphify-out/snapshots/bucket-1-pre-drop.sql)
- [orchestrator/.env.example](orchestrator/.env.example)
- [orchestrator/.importlinter](orchestrator/.importlinter)
- [orchestrator/alembic/versions/prd135_drop_bucket_1.py](orchestrator/alembic/versions/prd135_drop_bucket_1.py)
- [orchestrator/core/services/plugin_cache.py](orchestrator/core/services/plugin_cache.py)
- [orchestrator/modules/__init__.py](orchestrator/modules/__init__.py)
- [orchestrator/modules/tools/__init__.py](orchestrator/modules/tools/__init__.py)
- [orchestrator/tests/test_import_contract_present.py](orchestrator/tests/test_import_contract_present.py)
- [orchestrator/tests/test_no_mem0_residue.py](orchestrator/tests/test_no_mem0_residue.py)
- [orchestrator/tests/test_prd184_us001_learning_evaluation_deleted.py](orchestrator/tests/test_prd184_us001_learning_evaluation_deleted.py)
- [orchestrator/tests/test_prd184_us003_exec_planning_deleted.py](orchestrator/tests/test_prd184_us003_exec_planning_deleted.py)
- [orchestrator/tests/test_prd184_us004_tools_concurrency_deleted.py](orchestrator/tests/test_prd184_us004_tools_concurrency_deleted.py)
- [scripts/ralph/PROMPT_build_prd184.md](scripts/ralph/PROMPT_build_prd184.md)
- [scripts/ralph/PROMPT_build_prd211.md](scripts/ralph/PROMPT_build_prd211.md)
- [scripts/ralph/PROMPT_review_prd184.md](scripts/ralph/PROMPT_review_prd184.md)
- [scripts/ralph/PROMPT_review_prd211.md](scripts/ralph/PROMPT_review_prd211.md)
- [scripts/ralph/acceptance-prd184.sh](scripts/ralph/acceptance-prd184.sh)
- [scripts/ralph/acceptance-prd211.sh](scripts/ralph/acceptance-prd211.sh)
- [scripts/ralph/prd-184.json](scripts/ralph/prd-184.json)
- [scripts/ralph/prd-211.json](scripts/ralph/prd-211.json)

</details>



This page details the developer tooling and automation scripts used within the Automatos AI project. It covers the `scripts/` directory, focusing on CI/CD, disaster recovery (DR), and the "Ralph" automation for Product Requirement Document (PRD) implementation. It also touches upon `.claude` agents, `Makefile` utilities, the `docs/PRDS` workflow, and `reports/dossiers`. The purpose is to streamline development, enforce architectural discipline, and automate repetitive tasks, particularly those related to codebase refactoring and cleanup.

## Ralph Automation for PRD Implementation

"Ralph" refers to a set of automation scripts and processes designed to guide and verify the implementation of Product Requirement Documents (PRDs), especially those involving significant codebase changes like deletions or architectural enforcement. The core idea is to provide a structured, verifiable way to execute PRD specifications, often with the assistance of AI agents (like Claude).

The Ralph workflow is typically initiated by a `PROMPT_build_prdXXX.md` file, which acts as a detailed instruction set for an AI agent or a human developer. This prompt outlines the scope, deletion gates, acceptance criteria, and specific guardrails for each user story within a PRD.

### Ralph Workflow Overview

The Ralph workflow follows a structured process to ensure changes are implemented correctly and safely:

```mermaid
graph TD
    A[PRD Document (docs/PRDS/PRD-XXX.md)] --> B{Ralph Build Prompt (scripts/ralph/PROMPT_build_prdXXX.md)};
    B -- Guides --> C[Developer / Claude Agent];
    C -- Implements Changes --> D[Codebase];
    D -- Adds/Updates --> E[Ralph JSON Spec (scripts/ralph/prd-XXX.json)];
    E -- Defines --> F[User Stories & Acceptance Criteria];
    C -- Creates/Updates --> G[Guard Tests (orchestrator/tests/test_prdXXX_usYYY_*.py)];
    C -- Commits Changes --> H[Git Repository];
    H -- Triggers --> I[CI Pipeline];
    I -- Runs --> J[Acceptance Script (scripts/ralph/acceptance-prdXXX.sh)];
    J -- Verifies --> F;
    J -- Verifies --> G;
    J -- Verifies --> D;
    J -- Reports Status --> K[RALPH_COMPLETE / RALPH_BLOCKED];
```
**Diagram 1: Ralph Automation Workflow**

Sources:
- `scripts/ralph/PROMPT_build_prd184.md`
- `scripts/ralph/prd-184.json`
- `scripts/ralph/acceptance-prd184.sh`

### Ralph JSON Specification

Each Ralph-driven PRD has a corresponding JSON file (e.g., `scripts/ralph/prd-184.json` [scripts/ralph/prd-184.json:1-88]()) that defines the project, branch name, overall description, and a list of `userStories`. Each user story includes:
- `id`: A unique identifier (e.g., "US-001").
- `title`: A brief summary of the story.
- `description`: Detailed instructions for implementing the story.
- `acceptanceCriteria`: A list of verifiable conditions that must be met for the story to be considered complete. These often include checks for file deletion, content removal, and the presence of guard tests.
- `files`: A list of files or directories affected by the story.

For example, `prd-184.json` [scripts/ralph/prd-184.json:1-88]() details the "Kill-list" PRD, focusing on deleting dead code and scaffolding.

```json
{
  "project": "Automatos AI Platform — PRD-184 Kill-list (Phase-2 reserved deletion slot) — DELETE-NOW tier only",
  "branchName": "ralph/prd-184-kill-list-dead-surface",
  "description": "Cut from origin/main @ 9dd4c848a — re-verify EVERY path/anchor by grep, they drift. SCOPE = the DELETE-NOW tier ONLY...",
  "userStories": [
    {
      "id": "US-001",
      "title": "Delete the learning/evaluation theatre packages",
      "description": "Delete the two empty-theatre packages that signpost away from the real loops: `orchestrator/modules/evaluation/`...",
      "acceptanceCriteria": [
        "DONE — grep-proven: `git grep learning.feedback|learning.patterns|from .feedback|from .patterns` = EMPTY...",
        "DONE — evaluation/ fully deleted; the dead theatre learning/feedback/ + learning/patterns/ (both EMPTY __init__.py, 0 bytes) deleted...",
        // ... more criteria
      ],
      "files": ["orchestrator/modules/evaluation/", "orchestrator/modules/learning/", "orchestrator/modules/__init__.py", "orchestrator/tests/"]
    },
    // ... other user stories
  ]
}
```
**Table 1: Excerpt from `scripts/ralph/prd-184.json`**

Sources:
- `scripts/ralph/prd-184.json`

### Ralph Acceptance Scripts

After changes are implemented, an acceptance script (e.g., `scripts/ralph/acceptance-prd184.sh` [scripts/ralph/acceptance-prd184.sh:1-97]()) is run to verify that all acceptance criteria have been met. These scripts are typically Bash scripts that perform a series of checks, including:
- Running the full test suite (`orchestrator-full-suite green`) to ensure no regressions [scripts/ralph/acceptance-prd184.sh:18-19]().
- Verifying the deletion of specified files or directories using `[ ! -e <path> ]` [scripts/ralph/acceptance-prd184.sh:22]().
- Checking for the presence of guard tests using `grep -rlE` [scripts/ralph/acceptance-prd184.sh:33]().
- Ensuring no live imports of deleted components remain using `git grep -nE` [scripts/ralph/acceptance-prd184.sh:36-37]().
- Scope guards to prevent accidental modification of out-of-scope files (e.g., lockfiles, `node_modules`) [scripts/ralph/acceptance-prd184.sh:75-88]().
- Convention guards, such as preventing `os.getenv` outside `config.py` [scripts/ralph/acceptance-prd184.sh:91-92]().

The script exits with `0` for success (`RALPH_COMPLETE`) or `1` for failure (`RALPH_BLOCKED`).

```bash
#!/bin/bash
# Acceptance gate — PRD-184 Kill-list (DELETE-NOW tier US-001..US-006 only).
# Run from the worktree repo root. Exit 0 = delete-now tier done + safe.
# ...
check "orchestrator-full-suite green (@integration skips with no DB)" \
  'cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q'
# ...
check "US-001 modules/evaluation deleted" "[ ! -e orchestrator/modules/evaluation ]"
# ...
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-184 delete-now PASS"; else echo "ACCEPTANCE: PRD-184 FAIL"; fi
exit $FAIL
```
**Table 2: Excerpt from `scripts/ralph/acceptance-prd184.sh`**

Sources:
- `scripts/ralph/acceptance-prd184.sh`

### Guard Tests

A critical component of Ralph automation is the creation of "guard tests." These are small, focused tests designed to prevent deleted code from silently reappearing or to enforce architectural invariants. They are typically pure filesystem or source-grep checks, avoiding database or network dependencies.

For example, `orchestrator/tests/test_no_mem0_residue.py` [orchestrator/tests/test_no_mem0_residue.py:1-78]() ensures that:
1. Specific "mem0 residue" files, which were part of a retired external HTTP service, no longer exist [orchestrator/tests/test_no_mem0_residue.py:43-49]().
2. No HTTP mem0 client tokens (`MEM0_API_URL`, `mem0_client`, `httpx`) are reintroduced under `orchestrator/modules/memory/` [orchestrator/tests/test_no_mem0_residue.py:52-71]().
3. The in-process replacement (`durable_store.py`) is present [orchestrator/tests/test_no_mem0_residue.py:74-78]().

```python
# orchestrator/tests/test_no_mem0_residue.py
def test_no_mem0_residue():
    """Canonical PRD-211 US-002 guard: none of the 7 dead mem0 files exist."""
    survivors = [p for p in _RESIDUE if (_REPO / p).exists()]
    assert not survivors, (
        "dead mem0-residue file(s) resurfaced — the PRD-187 un-split retired the "
        f"external HTTP mem0 service; these must stay deleted: {survivors}"
    )

def test_no_http_mem0_client_under_modules_memory():
    """The live memory path is in-process (Qdrant/durable_store). No file under
    modules/memory may carry an HTTP mem0 client — that would un-do the un-split."""
    # ... checks for MEM0_API_URL, mem0_client, httpx
```

Sources:
- `orchestrator/tests/test_no_mem0_residue.py`

### Import Linter for Topology Discipline

The `import-linter` tool, configured via `orchestrator/.importlinter` [orchestrator/.importlinter:1-85](), enforces architectural topology discipline. It ensures that feature modules (`orchestrator/modules/*`) do not directly import each other, but instead route through designated layers like `modules.tools` or the `api` package. This prevents uncontrolled lateral coupling and maintains a modular monolith structure.

The configuration specifies an `independence` contract over a list of feature modules and explicitly `ignore_imports` for permitted routing patterns (e.g., `modules.tools.** -> modules.**`) [orchestrator/.importlinter:49-50](). This acts as a "ratchet," allowing existing valid cross-module imports while preventing new ones.

```ini
# orchestrator/.importlinter
[importlinter]
root_package = modules
include_external_packages = False

[importlinter:contract:feature-module-independence]
name = Feature modules must not import each other (route via modules.tools or api)
type = independence
modules =
    modules.agents
    modules.attachments
    # ...
ignore_imports =
    modules.tools.** -> modules.**
    modules.agents.factory.agent_factory -> modules.attachments.resolver
    # ...
```
**Table 3: Excerpt from `orchestrator/.importlinter`**

A dedicated CI lane (`.github/workflows/import-linter.yml`) runs `lint-imports` to enforce this contract. `orchestrator/tests/test_import_contract_present.py` [orchestrator/tests/test_import_contract_present.py:1-84]() provides a pure test to ensure the contract file exists and is well-formed.

Sources:
- `orchestrator/.importlinter`
- `orchestrator/tests/test_import_contract_present.py`
- `.github/workflows/import-linter.yml`

## `.claude` Agents, Hooks, and Skills

The `.claude` directory is used for AI agent-related configurations, particularly for Claude. This includes `worktrees/` which is ignored by Git [orchestrator/.gitignore:8-8](), indicating it's used for temporary, per-project agent work. `ralph-loop.local.md` [orchestrator/.gitignore:12-12]() is also ignored, suggesting it's a local scratchpad for Ralph automation loops, likely used by an AI agent.

This implies that Claude agents can be configured with specific "skills" or "hooks" to interact with the codebase, perform tasks like code modifications, and participate in the Ralph automation process. The `PROMPT_build_prdXXX.md` files are designed to be directly consumable by such agents.

## Makefile

While not explicitly detailed in the provided files, a `Makefile` is a common tool for automating development tasks. In this context, it would likely contain commands for:
- Running tests (e.g., `make test`).
- Building Docker images (e.g., `make build-orchestrator`).
- Database operations (e.g., `make migrate`, `make seed-db`).
- Linting and formatting (e.g., `make lint`, `make format`).
- Potentially invoking Ralph acceptance scripts or other automation.

## Docs/PRDS Workflow

The `docs/PRDS` directory contains Product Requirement Documents that drive development. The Ralph automation directly integrates with this workflow by providing a structured way to implement and verify PRD requirements.

The `prd_parity` script (mentioned in the wiki TOC) and Ralph acceptance scripts ensure that the codebase aligns with the PRD specifications. Dossiers in `reports/` (also mentioned in the wiki TOC) likely provide detailed reports or summaries related to PRD implementation or system state.

For example, `docs/PRDS/126-BUSINESS-KNOWLEDGE-GRAPH.md` would be a PRD that might have a corresponding Ralph automation for its implementation.

Sources:
- `docs/PRDS/126-BUSINESS-KNOWLEDGE-GRAPH.md`

## Reports/Dossiers

The `reports/dossiers` directory is intended for detailed reports or summaries. These could be generated automatically by scripts or manually compiled. Given the context of Ralph automation and PRD implementation, dossiers might contain:
- Summaries of Ralph execution results.
- Audit trails of code changes related to PRDs.
- Analysis of codebase topology (e.g., from `import-linter` runs).
- Historical data or snapshots relevant to specific PRDs (e.g., `graphify-out/snapshots/bucket-1-pre-drop.sql` [graphify-out/snapshots/bucket-1-pre-drop.sql] is a schema-only snapshot related to a database cleanup PRD).

Sources:
- `graphify-out/snapshots/bucket-1-pre-drop.sql`

---