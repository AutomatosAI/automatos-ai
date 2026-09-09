# 29.2. First-Run Seeding & Fresh Install

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.env.example](.env.example)
- [docker-entrypoint.sh](docker-entrypoint.sh)
- [frontend/components/activity/board/__tests__/blocked-reason.test.ts](frontend/components/activity/board/__tests__/blocked-reason.test.ts)
- [frontend/components/activity/board/__tests__/task-deliverables-panel.test.tsx](frontend/components/activity/board/__tests__/task-deliverables-panel.test.tsx)
- [frontend/components/activity/board/blocked-reason.ts](frontend/components/activity/board/blocked-reason.ts)
- [frontend/components/activity/board/task-deliverables-panel.tsx](frontend/components/activity/board/task-deliverables-panel.tsx)
- [frontend/components/settings/SystemLLMSettingsTab.tsx](frontend/components/settings/SystemLLMSettingsTab.tsx)
- [orchestrator/alembic/versions/prd222_veteran_skip_backfill.py](orchestrator/alembic/versions/prd222_veteran_skip_backfill.py)
- [orchestrator/api/workspaces.py](orchestrator/api/workspaces.py)
- [orchestrator/core/seeds/platform-management-skill.md](orchestrator/core/seeds/platform-management-skill.md)
- [orchestrator/core/seeds/seed_auto_agent.py](orchestrator/core/seeds/seed_auto_agent.py)
- [orchestrator/core/seeds/seed_local_first_run.py](orchestrator/core/seeds/seed_local_first_run.py)
- [orchestrator/tests/test_prd222_onboarding_reset.py](orchestrator/tests/test_prd222_onboarding_reset.py)
- [orchestrator/tests/test_prd226_doctrine.py](orchestrator/tests/test_prd226_doctrine.py)
- [orchestrator/tests/test_prd233_fresh_install_starts_onboarding.py](orchestrator/tests/test_prd233_fresh_install_starts_onboarding.py)
- [scripts/sync-auto-skill.py](scripts/sync-auto-skill.py)

</details>



This page details the mechanisms for first-run seeding and fresh installations within the Automatos AI platform. It covers how initial data, such as the `Auto` agent, platform-management skill, and starter workspace content, is provisioned. It also explains the "veteran-skip" backfill logic and the `fresh-install starts-onboarding` behavior, ensuring a consistent initial user experience.

Sources:
* [orchestrator/api/workspaces.py:1-166]()
* [frontend/components/settings/SystemLLMSettingsTab.tsx:1-175]()
* [orchestrator/core/seeds/seed_auto_agent.py:1-128]()
* [orchestrator/core/seeds/platform-management-skill.md:1-132]()
* [orchestrator/alembic/versions/prd222_veteran_skip_backfill.py:1-65]()
* [orchestrator/tests/test_prd233_fresh_install_starts_onboarding.py:1-57]()
* [orchestrator/core/seeds/seed_local_first_run.py:1-155]()
* [docker-entrypoint.sh:1-143]()
* [orchestrator/tests/test_prd226_doctrine.py:1-156]()
* [scripts/sync-auto-skill.py:1-118]()

## Purpose and Scope

The first-run seeding and fresh install process ensures that a newly deployed Automatos AI instance, particularly in a local development environment, is immediately functional and provides a guided onboarding experience. This involves:
1.  **Workspace Provisioning**: Creating the initial workspace and operator user.
2.  **Core Agent Seeding**: Ensuring the `Auto` agent and its essential `platform-management` skill are present.
3.  **Starter Content**: Populating the workspace with a basic set of agents, a demo playbook, and a welcome deliverable to showcase platform capabilities.
4.  **Onboarding State Management**: Setting the initial onboarding stage for new workspaces and handling existing ("veteran") workspaces appropriately.

This process is designed to be idempotent, meaning it can be run multiple times without causing data duplication or corruption, and it respects user modifications to seeded content.

Sources:
* [orchestrator/core/seeds/seed_local_first_run.py:1-39]()
* [orchestrator/core/seeds/seed_auto_agent.py:1-16]()
* [orchestrator/tests/test_prd233_fresh_install_starts_onboarding.py:1-8]()

## Workspace Seeding and Operator User

For local installations (`AUTH_EDITION == "local"`), the `docker-entrypoint.sh` script plays a crucial role in ensuring the default workspace and a local operator user exist.

### `docker-entrypoint.sh` Workflow

The `docker-entrypoint.sh` script orchestrates the database setup and initial data loading.
1.  **Wait for PostgreSQL**: Ensures the database is available before proceeding [docker-entrypoint.sh:22-39]().
2.  **Run Migrations**: Applies Alembic migrations to bring the database schema up to date [docker-entrypoint.sh:42-61](). This step is critical as it establishes the table structure.
3.  **Ensure Local Workspace**: If `AUTH_EDITION` is `local` and `DEFAULT_WORKSPACE_ID` is set, this function creates the default workspace and the local operator user [docker-entrypoint.sh:96-127]().
    *   The workspace is inserted with an `onboarding` JSONB field explicitly set to `{"stage": "not_started", "stages": {}, "segment": {}}` [docker-entrypoint.sh:112-113](). This is vital for triggering the onboarding flow on fresh installs (PRD-233).
    *   The local operator user (ID 1) is also created, which is used as a fallback in `api/chat.py` [docker-entrypoint.sh:115-118]().
4.  **Load Seed Data**: Executes `python -m core.database.load_seed_data` to load additional seed data [docker-entrypoint.sh:66-92](). This is where `seed_local_first_run` is invoked.

### `_ensure_workspace` Function

The `_ensure_workspace` function in `seed_local_first_run.py` is responsible for creating the default workspace and operator user if they don't already exist. This function is called by `core.database.load_seed_data` during the seed loading phase.

```mermaid
graph TD
    A[docker-entrypoint.sh] --> B{AUTH_EDITION == "local"?};
    B -- Yes --> C[ensure_local_workspace()];
    C --> D[INSERT INTO workspaces (id, name, slug, is_personal, is_active, onboarding) VALUES (...) ON CONFLICT (id) DO NOTHING];
    D -- onboarding: {"stage": "not_started", ...} --> E[INSERT INTO users (id, username, email, name, is_active) VALUES (...) ON CONFLICT (id) DO NOTHING];
    A --> F[python -m core.database.load_seed_data];
    F --> G[seed_local_first_run.py];
    G --> H[_ensure_workspace(db, DEFAULT_WORKSPACE_ID)];
    H -- If not exists --> D;
    H -- If exists --> I[Return "present"];
```

Title: Workspace and Operator User Seeding Flow

Sources:
* [docker-entrypoint.sh:96-127]()
* [orchestrator/tests/test_prd233_fresh_install_starts_onboarding.py:26-32]()
* [orchestrator/core/seeds/seed_local_first_run.py:80-88]()

## `Auto` Agent and Platform-Management Skill Seeding

Every workspace in Automatos AI has a special `Auto` agent. This agent acts as the default chat agent and is the single source of truth for the workspace's orchestrator LLM configuration and persona [orchestrator/core/seeds/seed_auto_agent.py:4-7]().

### `seed_auto_agent` Function

The `seed_auto_agent` function [orchestrator/core/seeds/seed_auto_agent.py:300-484]() is responsible for creating or updating the `Auto` agent for a given workspace.
*   It upserts the agent based on its slug (`auto-{workspace_id}`) [orchestrator/core/seeds/seed_auto_agent.py:10]().
*   It assigns a default persona, which includes the "Manager's Doctrine" [orchestrator/core/seeds/seed_auto_agent.py:99-106](). This doctrine is a set of principles guiding the agent's behavior.
*   It ensures the `platform-management` skill is associated with the `Auto` agent.

### Platform-Management Skill

The `platform-management` skill is an always-on skill that provides the `Auto` agent with a comprehensive set of tools for managing the platform [orchestrator/core/seeds/seed_auto_agent.py:31-35](). This skill is defined in `orchestrator/core/seeds/platform-management-skill.md` and includes tools for:
*   Marketplace browsing and installation (`platform_browse_marketplace_plugins`, `platform_install_skill`, etc.) [orchestrator/core/seeds/platform-management-skill.md:8-20]()
*   Agent management (`platform_list_agents`, `platform_create_agent`, `platform_update_agent`, etc.) [orchestrator/core/seeds/platform-management-skill.md:27-50]()
*   Skill management (`platform_get_skill_content`, `platform_create_workspace_skill`, etc.) [orchestrator/core/seeds/platform-management-skill.md:51-57]()
*   Playbook management (`platform_list_playbooks`, `platform_create_playbook`, `platform_execute_playbook`, etc.) [orchestrator/core/seeds/platform-management-skill.md:66-87]()
*   Task and Mission management (`platform_create_task`, `platform_list_missions`, `platform_approve_mission`, etc.) [orchestrator/core/seeds/platform-management-skill.md:88-132]()

The content of this skill is synchronized from the `automatos-skills` repository using the `scripts/sync-auto-skill.py` script [scripts/sync-auto-skill.py:1-16](). This script ensures that the version in the codebase is always up-to-date with the authoritative source.

### Manager's Doctrine

The "Manager's Doctrine" is a core component of the `Auto` agent's persona. It's a set of nine principles that guide the agent's decision-making and interaction style [orchestrator/core/seeds/seed_auto_agent.py:70-80](). This doctrine is embedded in the `Auto` agent's `custom_persona_prompt` via `compose_persona_with_doctrine` [orchestrator/core/seeds/seed_auto_agent.py:83-96]() and is also present in the `platform-management-skill.md` [orchestrator/tests/test_prd226_doctrine.py:45-47]().

```mermaid
graph TD
    A[seed_local_first_run.py] --> B[seed_auto_agent(db, workspace_id)];
    B --> C[Upsert Agent (slug=auto-{workspace_id})];
    C --> D[Compose Persona with Doctrine];
    D -- _FRIENDLY_FALLBACK + MANAGER_DOCTRINE_BLOCK --> E[Agent.custom_persona_prompt];
    C --> F[Ensure platform-management-skill assigned];
    F --> G[platform-management-skill.md];
    G -- Synchronized by --> H[scripts/sync-auto-skill.py];
    H --> I[automatos-skills/team/auto/SKILL.md];
```

Title: Auto Agent and Platform-Management Skill Seeding

Sources:
* [orchestrator/core/seeds/seed_auto_agent.py:1-128]()
* [orchestrator/core/seeds/platform-management-skill.md:1-132]()
* [orchestrator/tests/test_prd226_doctrine.py:38-47]()
* [scripts/sync-auto-skill.py:1-16]()

## Workspace Seeding (Starter Content)

The `seed_local_first_run` function in `orchestrator/core/seeds/seed_local_first_run.py` is responsible for populating a new local workspace with starter content. This content is designed to provide a "two-minute demo" experience [orchestrator/core/seeds/seed_local_first_run.py:2-3]().

The seeding process is idempotent-refresh, meaning:
*   Rows matching the current seed fingerprint are left alone.
*   Rows matching a *prior* seed fingerprint are updated to the current content.
*   User-modified rows (different fingerprint) are never overwritten.
*   Deleted seeded rows are not resurrected, as the workspace's `settings["local_first_run"]` ledger tracks what was seeded [orchestrator/core/seeds/seed_local_first_run.py:26-34]().

The starter content includes:
1.  **Auto Agent**: Ensured by calling `seed_auto_agent` [orchestrator/core/seeds/seed_local_first_run.py:12]().
2.  **Starter Roster**: Three agents (`Researcher`, `Writer`, `Analyst`) with predefined personas and responsibilities, using only native platform tools [orchestrator/core/seeds/seed_local_first_run.py:100-155]().
3.  **Demo Playbook**: A simple playbook that the starter roster can execute without requiring external integrations or a worker [orchestrator/core/seeds/seed_local_first_run.py:13]().
4.  **Welcome Deliverable**: A `BlogPost` entry that appears in the Deliverables tab, showcasing output without needing object storage or a worker [orchestrator/core/seeds/seed_local_first_run.py:14-16]().

```mermaid
graph TD
    A[core.database.load_seed_data] --> B[seed_local_first_run(db, workspace_id)];
    B --> C[Ensure Workspace & Operator User];
    B --> D[seed_auto_agent(db, workspace_id)];
    B --> E[Seed Starter Roster (Researcher, Writer, Analyst)];
    E -- Native tools only --> F[Agent Table];
    B --> G[Seed Demo Playbook];
    G --> H[WorkflowTemplate Table];
    B --> I[Seed Welcome Deliverable (BlogPost)];
    I --> J[BlogPost Table];
    K[Workspace.settings.local_first_run] -- Tracks seeded items --> B;
```

Title: Local First-Run Seeding Process

Sources:
* [orchestrator/core/seeds/seed_local_first_run.py:1-39]()
* [orchestrator/core/seeds/seed_local_first_run.py:95-155]()

## Veteran-Skip Backfill

The "veteran-skip" backfill is an Alembic migration (`prd222_veteran_skip_backfill`) designed to correctly initialize the `onboarding` state for existing workspaces.

### Problem Statement

Initially, the `onboarding` column was added to the `workspaces` table without a backfill for pre-existing workspaces. This caused veteran workspaces to default to a `not_started` onboarding stage, incorrectly triggering the new-user onboarding flow [orchestrator/alembic/versions/prd222_veteran_skip_backfill.py:1-7]().

### Solution

The `prd222_veteran_skip_backfill` migration addresses this by:
*   Updating workspaces where the `onboarding` field is `NULL` or `onboarding->>'stage'` is `NULL`.
*   Setting the `stage` to `skipped` and adding a `veteran: true` marker [orchestrator/alembic/versions/prd222_veteran_skip_backfill.py:32-44]().
*   Crucially, it only applies to workspaces created *before* a specific timestamp (`2026-08-29`) [orchestrator/alembic/versions/prd222_veteran_skip_backfill.py:45-46](). This prevents newly created workspaces (which should start onboarding) from being marked as `skipped` if the migration runs after their creation.

This ensures that veteran users do not encounter the onboarding wizard, while new users correctly start the onboarding process.

```mermaid
graph TD
    A[Alembic Migration: prd222_veteran_skip_backfill] --> B{Workspace.onboarding IS NULL OR Workspace.onboarding->>'stage' IS NULL?};
    B -- Yes --> C{Workspace.created_at < '2026-08-29'?};
    C -- Yes --> D[UPDATE Workspace SET onboarding = {stage: "skipped", veteran: true, ...}];
    C -- No --> E[Do Nothing (New workspace, should onboard)];
    B -- No --> F[Do Nothing (Already has onboarding state)];
```

Title: Veteran-Skip Backfill Logic

Sources:
* [orchestrator/alembic/versions/prd222_veteran_skip_backfill.py:1-65]()
* [orchestrator/tests/test_prd233_fresh_install_starts_onboarding.py:51-57]()

## Fresh Install Starts Onboarding Behavior

A fresh installation of Automatos AI is designed to immediately guide the user through an Auto-led onboarding flow. This behavior is achieved through a combination of the `docker-entrypoint.sh` script and the `prd222_veteran_skip_backfill` migration.

### Mechanism

1.  **Explicit `not_started` Stage**: When `docker-entrypoint.sh` creates the default local workspace, it explicitly sets the `onboarding` field to `{"stage": "not_started", "stages": {}, "segment": {}}` [docker-entrypoint.sh:112-113]().
2.  **Migration Exclusion**: The `prd222_veteran_skip_backfill` migration includes a `WHERE created_at < TIMESTAMP '2026-08-29'` clause [orchestrator/alembic/versions/prd222_veteran_skip_backfill.py:45-46](). This ensures that any workspace created *after* this date (which includes all fresh installs) is *not* touched by the veteran-skip logic.
3.  **Frontend Detection**: The frontend, upon fetching the current workspace via `GET /api/workspaces/current`, receives the `onboarding` snapshot [orchestrator/api/workspaces.py:68-69](). If `onboarding.stage` is `not_started`, it triggers the Auto-led onboarding wizard.

This coordinated approach guarantees that new users on a fresh install are immediately presented with the onboarding experience, while existing users are correctly identified as "veterans" and skip it.

Sources:
* [orchestrator/tests/test_prd233_fresh_install_starts_onboarding.py:1-8]()
* [docker-entrypoint.sh:109-113]()
* [orchestrator/alembic/versions/prd222_veteran_skip_backfill.py:45-46]()
* [orchestrator/api/workspaces.py:68-69]()

---