# Onboarding & First-Run Experience

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/admin/workspace-plan-select.tsx](frontend/components/admin/workspace-plan-select.tsx)
- [orchestrator/alembic/versions/users_last_sign_in_column.py](orchestrator/alembic/versions/users_last_sign_in_column.py)
- [orchestrator/modules/context/sections/onboarding.py](orchestrator/modules/context/sections/onboarding.py)
- [orchestrator/modules/tools/discovery/actions_onboarding.py](orchestrator/modules/tools/discovery/actions_onboarding.py)
- [orchestrator/modules/tools/discovery/handlers_onboarding.py](orchestrator/modules/tools/discovery/handlers_onboarding.py)
- [orchestrator/services/onboarding_content.py](orchestrator/services/onboarding_content.py)
- [orchestrator/services/onboarding_state.py](orchestrator/services/onboarding_state.py)
- [orchestrator/services/plan_tiers.py](orchestrator/services/plan_tiers.py)
- [orchestrator/tests/test_admin_plan_console.py](orchestrator/tests/test_admin_plan_console.py)
- [orchestrator/tests/test_f132_onboarding_sets_the_businesss_zone.py](orchestrator/tests/test_f132_onboarding_sets_the_businesss_zone.py)
- [orchestrator/tests/test_f188_b_stage_ids_stay_internal.py](orchestrator/tests/test_f188_b_stage_ids_stay_internal.py)
- [orchestrator/tests/test_f188_the_last_steps_fit_the_edition_and_the_apps.py](orchestrator/tests/test_f188_the_last_steps_fit_the_edition_and_the_apps.py)
- [orchestrator/tests/test_f314_b_no_onboarding_quiz_in_a_workspace_in_use.py](orchestrator/tests/test_f314_b_no_onboarding_quiz_in_a_workspace_in_use.py)
- [orchestrator/tests/test_f314_d_answers_alone_never_start_onboarding_in_a_workspace_in_use.py](orchestrator/tests/test_f314_d_answers_alone_never_start_onboarding_in_a_workspace_in_use.py)
- [orchestrator/tests/test_f337_d_no_quiz_in_a_working_workspace.py](orchestrator/tests/test_f337_d_no_quiz_in_a_working_workspace.py)
- [orchestrator/tests/test_prd222_onboarding_state.py](orchestrator/tests/test_prd222_onboarding_state.py)
- [orchestrator/tests/test_prd222_onboarding_tool.py](orchestrator/tests/test_prd222_onboarding_tool.py)
- [orchestrator/tests/test_prd222_onboarding_tool_arg_shapes.py](orchestrator/tests/test_prd222_onboarding_tool_arg_shapes.py)
- [orchestrator/tests/test_prd222_onboarding_tool_prior.py](orchestrator/tests/test_prd222_onboarding_tool_prior.py)
- [orchestrator/tests/test_prd222_segment_implied_advance.py](orchestrator/tests/test_prd222_segment_implied_advance.py)

</details>



This wiki page documents the PRD-222/233 conversational onboarding subsystem and first-run experience in Automatos AI. It details how the platform replaces rigid, static registration forms with an interactive, agent-led onboarding funnel. The architecture relies on the `OnboardingState` service [orchestrator/services/onboarding_state.py:1-34](), working workspace protections (`onboarding_content.py`) [orchestrator/services/onboarding_content.py:1-28](), dynamic context section assembly [orchestrator/modules/context/sections/onboarding.py:1-30](), dedicated platform action tools [orchestrator/modules/tools/discovery/actions_onboarding.py:1-28](), segment-implied stage advancement, trial/plan tier enforcement, and per-workspace Auto agent seeding.

---

## 1. OnboardingState Service & State Machine

The core state machine governing the onboarding journey resides in `orchestrator/services/onboarding_state.py` [orchestrator/services/onboarding_state.py:1-34](). Rather than using a separate relational table, every workspace stores its progress within a single JSONB column (`workspace.onboarding`), managed exclusively by this service [orchestrator/services/onboarding_state.py:3-9]().

### Document Schema and Immutability

The JSONB document tracks the active stage, per-stage funnel timestamps, structured business segment inputs, and metadata:

```json
{
  "stage": "not_started",
  "stages": {
    "questions": "2026-09-02T10:00:00Z"
  },
  "segment": {
    "business": "Cafe",
    "goal": "Wholesale orders",
    "comfort": "novice"
  },
  "started_at": "2026-09-02T10:00:00Z",
  "updated_at": "2026-09-02T10:05:00Z"
}
```[orchestrator/services/onboarding_state.py:23-33]()

Two hard rules govern the state machine implementation:

1. **Rebuild, Never Mutate:** `get_onboarding()` returns a deep copy of the document [orchestrator/services/onboarding_state.py:83-98](). All updates create a new dictionary object which then fully replaces `workspace.onboarding` on assignment. This pattern prevents SQLAlchemy's silent loss of in-place JSONB mutations [orchestrator/services/onboarding_state.py:10-17](). Callers must not mutate the dicts returned by `get_onboarding()` and must always persist changes through controlled methods like `advance_onboarding_stage()` or `set_segment()` [orchestrator/services/onboarding_state.py:83-90]().

2. **Monotonic Forward Stage Transitions:** The stages progress strictly in the order defined by `STAGE_ORDER` [orchestrator/services/onboarding_state.py:49-58]():

   ```
   not_started -> questions -> teach -> proposal -> building -> boom -> powerup -> completed
   ```

   The special stage `skipped` can be reached from any non-terminal stage [orchestrator/services/onboarding_state.py:46-59](). The stages `completed` and `skipped` are terminal (`TERMINAL_STAGES`) [orchestrator/services/onboarding_state.py:62](). Attempts to move backwards, to the same stage, or from a terminal stage raise an `InvalidStageTransition` exception [orchestrator/services/onboarding_state.py:71-73]().

### Onboarding Activity Check & Staleness

To detect if onboarding is actively consuming user attention (and thus tool routing should pin onboarding-specific tools), `is_onboarding_active(workspace)` returns True only if the current stage is recognized, non-terminal, and the run age is less than `ONBOARDING_STALE_DAYS` (14 days) [orchestrator/services/onboarding_state.py:107-129]().

### Diagram: Onboarding Stage Progression State Machine

```mermaid
graph TD
  NS["not_started"] --> Q["questions"]
  Q --> T["teach"]
  T --> P["proposal"]
  P --> B["building"]
  B --> Boom["boom"]
  Boom --> PU["powerup"]
  PU --> Comp["completed"]
  AnyNonTerminal["Any Non-Terminal Stage"] --> Skip["skipped"]

  click NS href "" "Stage: not_started"
  click Q href "" "Stage: questions"
  click T href "" "Stage: teach"
  click P href "" "Stage: proposal"
  click B href "" "Stage: building"
  click Boom href "" "Stage: boom"
  click PU href "" "Stage: powerup"
  click Comp href "" "Stage: completed"
  click Skip href "" "Stage: skipped (terminal)"
```
Sources: `orchestrator/services/onboarding_state.py:1-155`

---

## 2. Onboarding Context Section & Working Workspace Protections

The conversational experience is driven by a prompt context section coded in `orchestrator/modules/context/sections/onboarding.py` [orchestrator/modules/context/sections/onboarding.py:1-30](). This section programmatically injects stage-aware guidance for the workspace's current onboarding stage.

### Behavior, Triggers, and F314/F337 Protections

- **Trigger Rules:** The section renders only if the workspace's `onboarding.stage` is not terminal, or if the user explicitly re-triggers with a phrase from `_TRIGGER_PHRASES` such as `"set up my workspace"`, `"help me get started"`, or `"mission zero"` [orchestrator/modules/context/sections/onboarding.py:10-48]().
- **Working Workspace Guard (`quiz_not_wanted`):** Under F314 and F337 fixes, a workspace that is already in use (holding documents or board task cards) and whose onboarding has *not* started (or sat idle at `questions` for more than `QUIZ_IDLE_HOURS`) will not trigger the onboarding quiz [orchestrator/services/onboarding_content.py:43-98](). This prevents Auto from interrupting operational workflows with setup questions [orchestrator/services/onboarding_content.py:1-28]().
- **Decorator Wrapper (`not_for_a_workspace_in_use`):** Wraps `is_onboarding_active` to gate tool routing priors and AutoBrain Tier 0 pins in working workspaces [orchestrator/services/onboarding_content.py:108-117]().

Sources:
- `orchestrator/modules/context/sections/onboarding.py:1-138`
- `orchestrator/services/onboarding_content.py:1-120`

---

## 3. Platform Onboarding Tool Actions & Normalization

Auto interacts with the onboarding state machine exclusively through dedicated platform tool actions registered via `register_onboarding_actions()` [orchestrator/modules/tools/discovery/actions_onboarding.py:44-50]().

### Onboarding Tool Actions (`ONBOARDING_PRIOR_ACTIONS`)

- `platform_update_onboarding`: advances stage, records segment answers, sets plan and timezone [orchestrator/modules/tools/discovery/actions_onboarding.py:22]().
- `platform_scan_business_site`: website scanning (Firecrawl) [orchestrator/modules/tools/discovery/actions_onboarding.py:23]().
- `platform_search_packages`: finds matching bundles in the marketplace [orchestrator/modules/tools/discovery/actions_onboarding.py:24]().
- `platform_install_package` / `platform_install_marketplace_agent`: installs selected packages or agents [orchestrator/modules/tools/discovery/actions_onboarding.py:25-26]().
- `platform_submit_report`: submits the onboarding summary deliverable [orchestrator/modules/tools/discovery/actions_onboarding.py:27]().

These actions are folded into the dispatcher enum by `_apply_onboarding_prior` in `orchestrator/modules/tools/tool_router.py` while onboarding is active [orchestrator/tests/test_prd222_onboarding_tool_prior.py:11-14]().

### Tool Handler Flow & Normalization

The handler function `update_onboarding` in `orchestrator/modules/tools/discovery/handlers_onboarding.py` processes updates by invoking `_normalise_params()` to coerce LLM-supplied argument shapes [orchestrator/modules/tools/discovery/handlers_onboarding.py:1-165]():
- Parses JSON-encoded `segment` strings [orchestrator/modules/tools/discovery/handlers_onboarding.py:84-88]().
- Folds flat parameters (`business`, `goal`, `comfort`) into `segment` [orchestrator/modules/tools/discovery/handlers_onboarding.py:129]().
- Maps `value` carrying stage names to `advance_to` [orchestrator/modules/tools/discovery/handlers_onboarding.py:131-134]().
- Normalizes case and whitespace for `advance_to` and `plan` [orchestrator/modules/tools/discovery/handlers_onboarding.py:135-136]().

### Diagram: Natural Language to Code Entity Onboarding Execution Flow

```mermaid
graph TD
  UserMsg["UserChatInput"] --> OnboardSec["OnboardingSection.render()"]
  OnboardSec --> AutoBrain["AutoBrain / LLM"]
  AutoBrain --> ToolExec["PlatformActionExecutor"]
  ToolExec --> Handler["handlers_onboarding.update_onboarding()"]
  ToolExec --> Normaliser["handlers_onboarding._normalise_params()"]
  Normaliser --> StateMachine["services.onboarding_state.advance_onboarding_stage()"]
  StateMachine --> DB["Workspace.onboarding JSONB Column"]

  classDef codeEntity fill:#f5f5f5,stroke:#333,stroke-width:1px;
  OnboardSec,Handler,Normaliser,StateMachine class codeEntity;
```
Sources:
- `orchestrator/modules/tools/discovery/actions_onboarding.py:1-139`
- `orchestrator/modules/tools/discovery/handlers_onboarding.py:1-165`

---

## 4. Segment-Implied Advance & Build Enforcement

### Segment-Implied Advancement (`implied_stage`)

The state machine automatically advances stages based on recorded segment answers even when `advance_to` is omitted [orchestrator/tests/test_prd222_segment_implied_advance.py:1-8]():
- Any single answer recorded at `not_started` implies advancement to `questions` [orchestrator/tests/test_prd222_segment_implied_advance.py:53-55]().
- Providing all three core answers (`business`, `goal`, `comfort`) implies advancement to `teach` [orchestrator/tests/test_prd222_segment_implied_advance.py:57-59]().

### Build Completion Enforcement (Payoff Gate)

To prevent skipping build milestones, advancing from `building` to `completed` or `powerup` requires recorded build evidence (installed packages, created agents, or missions) [orchestrator/modules/tools/discovery/handlers_onboarding.py:55-75](). Attempting to bypass this raises an `InvalidStageTransition` error [orchestrator/tests/test_prd222_segment_implied_advance.py:116-123]().

Sources:
- `orchestrator/tests/test_prd222_segment_implied_advance.py:1-162`
- `orchestrator/services/onboarding_state.py:1-155`

---

## 5. Trial Ledger & Plan Tier Enforcement

Plan tier resolution and limit assignment are handled centrally by `services.plan_tiers` [orchestrator/services/plan_tiers.py:1-34]().

### Plan Tier Assignment (`assign_plan`)

When a user accepts a plan tier (such as `basic`, `pro`, or `business`) during the onboarding proposal stage, `assign_plan` writes the tier limits into `workspaces.plan_limits`:
- Enforces assignable tiers via `is_assignable()` (excluding placeholders like `enterprise`) [orchestrator/services/plan_tiers.py:59-64]().
- Derives resource caps and manages tier-owned budget ceilings (`budget`) with proper provenance (`source="tier"`), clearing stale tier ceilings upon tier changes while preserving admin custom budgets (`source != "tier"`) [orchestrator/services/plan_tiers.py:8-27]().

Sources: `orchestrator/services/plan_tiers.py:1-142`

---

## Summary Reference Table

| Component | Code Location | Role |
| :--- | :--- | :--- |
| Onboarding State Machine | `orchestrator/services/onboarding_state.py` | Manages workspace onboarding lifecycle and state |
| Working Workspace Protection | `orchestrator/services/onboarding_content.py` | Prevents quiz prompts in active working workspaces |
| Onboarding Context Section | `orchestrator/modules/context/sections/onboarding.py` | Injects dynamic chatbot guidance for onboarding |
| Onboarding Actions | `orchestrator/modules/tools/discovery/actions_onboarding.py` | Defines platform tool definitions for onboarding |
| Onboarding Handlers | `orchestrator/modules/tools/discovery/handlers_onboarding.py` | Implements tool action execution and normalization |
| Plan Tiers Service | `orchestrator/services/plan_tiers.py` | Resolves assignable tiers and plan limits |

Sources:
- `orchestrator/services/onboarding_state.py`
- `orchestrator/services/onboarding_content.py`
- `orchestrator/modules/context/sections/onboarding.py`
- `orchestrator/modules/tools/discovery/actions_onboarding.py`
- `orchestrator/modules/tools/discovery/handlers_onboarding.py`
- `orchestrator/services/plan_tiers.py`

---