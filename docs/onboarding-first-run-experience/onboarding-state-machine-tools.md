# Onboarding State Machine & Tools

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/__tests__/workspace-provider.onboarding.test.tsx](frontend/components/__tests__/workspace-provider.onboarding.test.tsx)
- [orchestrator/modules/context/sections/onboarding.py](orchestrator/modules/context/sections/onboarding.py)
- [orchestrator/modules/tools/discovery/actions_onboarding.py](orchestrator/modules/tools/discovery/actions_onboarding.py)
- [orchestrator/modules/tools/discovery/handlers_onboarding.py](orchestrator/modules/tools/discovery/handlers_onboarding.py)
- [orchestrator/services/onboarding_state.py](orchestrator/services/onboarding_state.py)
- [orchestrator/services/plan_tiers.py](orchestrator/services/plan_tiers.py)
- [orchestrator/tests/test_prd222_boom_build_evidence.py](orchestrator/tests/test_prd222_boom_build_evidence.py)
- [orchestrator/tests/test_prd222_onboarding_section.py](orchestrator/tests/test_prd222_onboarding_section.py)
- [orchestrator/tests/test_prd222_onboarding_state.py](orchestrator/tests/test_prd222_onboarding_state.py)
- [orchestrator/tests/test_prd222_onboarding_tool.py](orchestrator/tests/test_prd222_onboarding_tool.py)
- [orchestrator/tests/test_prd222_onboarding_tool_arg_shapes.py](orchestrator/tests/test_prd222_onboarding_tool_arg_shapes.py)
- [orchestrator/tests/test_prd222_onboarding_tool_prior.py](orchestrator/tests/test_prd222_onboarding_tool_prior.py)
- [orchestrator/tests/test_prd222_segment_implied_advance.py](orchestrator/tests/test_prd222_segment_implied_advance.py)

</details>



This page details the implementation of the onboarding state machine, the tools used to interact with it, and how the `OnboardingSection` context is injected into the LLM's prompt. It covers the core `onboarding_state` service, the `handlers_onboarding` and `actions_onboarding` modules, and the mechanisms for segment-implied stage advancement, resetting, and evidence-based transitions.

## Onboarding State Machine (`onboarding_state` service)

The `onboarding_state` service [orchestrator/services/onboarding_state.py]() is the single source of truth and the *only* writer for the workspace's onboarding JSONB document. This document tracks the user's progress through the onboarding journey, including the current stage, timestamps for each stage, and collected segment answers.

The service enforces two hard rules:
1.  **Rebuild, never mutate**: All writes deep-copy the current onboarding document, modify the copy, and reassign it to the `workspace.onboarding` attribute. This prevents silent data loss due to SQLAlchemy's JSONB change detection [orchestrator/services/onboarding_state.py:11-16]().
2.  **Monotonic forward**: Stage transitions can only move forward through the `STAGE_ORDER` [orchestrator/services/onboarding_state.py:17-20](). The `skipped` stage is reachable from any non-terminal stage, while `completed` and `skipped` are terminal stages. Backward, same-stage, unknown, or from-terminal moves raise an `InvalidStageTransition` error [orchestrator/services/onboarding_state.py:20-22,131-143]().

The onboarding document structure is as follows:
```json
{
  "stage": "not_started",              // current stage (one of ALL_STAGES)
  "stages": {"questions": "<iso>"},    // per-stage funnel timestamps
  "segment": {"business", "goal", "comfort", "team_size"},
  "started_at": "<iso>",               // first advance away from not_started
  "updated_at": "<iso>",               // every write
  "completed_at": "<iso>",             // set when reaching completed/skipped
  // "trial": {...}                     // added by W1S9 (US-004/005)
}
```
[orchestrator/services/onboarding_state.py:23-33]()

Key functions within the `onboarding_state` service include:
*   `get_onboarding(workspace)`: Returns a deep copy of the workspace's onboarding document, defaulting to a `not_started` state if unset [orchestrator/services/onboarding_state.py:76-91]().
*   `current_stage(workspace)`: Returns the current onboarding stage of the workspace [orchestrator/services/onboarding_state.py:94-96]().
*   `is_onboarding_active(workspace)`: Checks if the onboarding process is active (i.e., not in a terminal stage) [orchestrator/services/onboarding_state.py:99-101]().
*   `advance_onboarding_stage(db, workspace, target_stage, ...)`: Handles stage transitions, validating them against the `STAGE_ORDER` and applying funnel timestamps. It also incorporates logic for segment-implied advancement and evidence-based transitions [orchestrator/services/onboarding_state.py:200-260]().
*   `set_segment(db, workspace, segment_data)`: Records user segment answers, potentially triggering an implied stage advancement if enough information is gathered [orchestrator/services/onboarding_state.py:300-329]().
*   `implied_stage(current_stage, segment)`: A pure function that determines if the collected segment answers imply a stage advancement (e.g., from `not_started` to `questions`, or `questions` to `teach`) [orchestrator/services/onboarding_state.py:332-349]().
*   `build_evidence(db, workspace)`: Gathers evidence of "build" activities (e.g., installed packages, created agents, missions) to gate advancement to the `boom` stage [orchestrator/services/onboarding_state.py:352-390]().

```mermaid
graph TD
    subgraph "Onboarding State Machine"
        A[Workspace.onboarding JSONB] --> B{get_onboarding(workspace)}
        B --> C{current_stage(workspace)}
        B --> D{public_snapshot(workspace)}
        C --> E{is_onboarding_active(workspace)}

        subgraph "State Transitions"
            F[advance_onboarding_stage(db, ws, target_stage)] --> G{_validate_transition(current, target)}
            G -- "Valid (monotonic forward)" --> H[Update stage, timestamps]
            G -- "Invalid (backward, same, terminal)" --> I[InvalidStageTransition]
            H --> J[Persist (rebuild-don't-mutate)]
            K[set_segment(db, ws, segment_data)] --> L{implied_stage(current, segment)}
            L -- "Implied advance" --> F
            F -- "Target: boom" --> M{build_evidence(db, ws)}
            M -- "No evidence" --> I
            M -- "Evidence exists" --> H
        end

        subgraph "Data Persistence"
            J --> P[_persist(db, ws, new_doc)]
            P -- "db is None" --> Q[In-memory update only]
            P -- "db is not None" --> R[db.add(workspace); db.commit()]
        end
    end
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#f9f,stroke:#333,stroke-width:2px
    style R fill:#f9f,stroke:#333,stroke-width:2px
```
Title: Onboarding State Machine Data Flow
Sources: [orchestrator/services/onboarding_state.py:1-34]()

## Onboarding Tools (`handlers_onboarding` & `actions_onboarding`)

The interaction with the `onboarding_state` service from the LLM happens through a dedicated tool: `platform_update_onboarding`. This tool is defined in `actions_onboarding.py` and its handler is in `handlers_onboarding.py`.

### `platform_update_onboarding` Tool Definition

The `platform_update_onboarding` tool [orchestrator/modules/tools/discovery/actions_onboarding.py:55-142]() is the primary mechanism for the LLM to advance the onboarding stage, record user segment answers, and set the user's plan.

*   **`register_onboarding_actions(registry)`**: This function registers the `platform_update_onboarding` tool with the `ActionRegistry` [orchestrator/modules/tools/discovery/actions_onboarding.py:44-142]().
*   **`ONBOARDING_PRIOR_ACTIONS`**: A list of actions, including `platform_update_onboarding`, that are prioritized during tool discovery when onboarding is active. This ensures the LLM can always access these critical tools even if the user's prompt is not tool-shaped [orchestrator/modules/tools/discovery/actions_onboarding.py:21-28](). This mechanism is tested in [orchestrator/tests/test_prd222_onboarding_tool_prior.py:11-13]().

The tool's parameters include:
*   `advance_to`: The target onboarding stage (e.g., "questions", "teach", "completed", "skipped"). It explicitly excludes `not_started` as a target [orchestrator/modules/tools/discovery/actions_onboarding.py:32-41,76-81]().
*   `segment`: A JSON object containing user answers like `business`, `goal`, `comfort`, and `team_size` [orchestrator/modules/tools/discovery/actions_onboarding.py:82-116]().
*   `plan`: The plan tier the user accepted (e.g., "basic", "pro", "business") [orchestrator/modules/tools/discovery/actions_onboarding.py:118-126]().

Crucially, all parameters are optional, but *at least one* must be provided. The handler enforces this, not the schema's `required` field [orchestrator/modules/tools/discovery/actions_onboarding.py:129-131]().

### `update_onboarding` Handler

The `update_onboarding` handler [orchestrator/modules/tools/discovery/handlers_onboarding.py:180-209]() is responsible for processing calls to `platform_update_onboarding`.

Key aspects of the handler:
*   **Delegation**: It delegates all state changes to the `services.onboarding_state` module, ensuring the single-writer rule is maintained [orchestrator/modules/tools/discovery/handlers_onboarding.py:3-4]().
*   **Robust Parameter Normalization**: The `_normalise_params` function [orchestrator/modules/tools/discovery/handlers_onboarding.py:93-159]() handles various LLM-supplied argument shapes observed in production, such as JSON-encoded strings for `segment`, flat answers at the top level, or stage names under the `value` key. This ensures the handler can correctly interpret LLM output even if it deviates from the strict schema.
*   **Error Handling**: Invalid transitions (e.g., backward moves) are returned as clear errors to the LLM, preventing crashes and allowing the LLM to correct its behavior [orchestrator/modules/tools/discovery/handlers_onboarding.py:5-6]().
*   **Segment-Implied Advance**: If only segment data is provided, the handler uses `implied_stage` to determine if a stage advancement is warranted (e.g., after all three initial questions are answered) [orchestrator/tests/test_prd222_segment_implied_advance.py:104-112]().
*   **Same-Stage Re-assert**: If the LLM attempts to advance to the stage it's already in, the handler treats it as a benign no-op success, preventing infinite loops [orchestrator/tests/test_prd222_onboarding_tool.py:108-117](). It still processes any accompanying segment data [orchestrator/tests/test_prd222_onboarding_tool.py:119-130]().
*   **`_SAME_STAGE_HINTS`**: Provides specific guidance to the LLM when it re-asserts the same stage, helping it understand what actions are expected next [orchestrator/modules/tools/discovery/handlers_onboarding.py:38-47]().

```mermaid
graph TD
    A[LLM calls platform_update_onboarding] --> B{update_onboarding(db, ws_id, params)}
    B --> C[_normalise_params(params)]
    C -- "Normalized params" --> D{Extract advance_to, segment, plan}

    subgraph "Process Segment"
        D -- "Segment data present" --> E[set_segment(db, ws, segment_data)]
        E -- "Implied stage advance?" --> F{implied_stage(current_stage, segment)}
        F -- "Yes" --> G[Set target_stage to implied]
        F -- "No" --> H[Keep original target_stage]
    end

    subgraph "Process Advance"
        D -- "advance_to present" --> I[Validate target_stage]
        I -- "Valid & not same stage" --> J[advance_onboarding_stage(db, ws, target_stage)]
        I -- "Valid & same stage" --> K[record_same_stage_reassert()]
        I -- "Invalid transition" --> L[Return error to LLM]
        J -- "Target: boom" --> M{build_evidence(db, ws)}
        M -- "No evidence" --> L
        M -- "Evidence exists" --> N[Continue advance]
    end

    subgraph "Process Plan"
        D -- "plan present" --> O[assign_plan(db, ws, plan)]
    end

    G --> J
    H --> J
    K --> P[Return success with current snapshot]
    N --> P
    O --> P
    L --> Q[Return error to LLM]
```
Title: `platform_update_onboarding` Tool Handler Flow
Sources: [orchestrator/modules/tools/discovery/handlers_onboarding.py:1-30](), [orchestrator/services/onboarding_state.py:1-34](), [orchestrator/services/plan_tiers.py:1-30]()

## Onboarding Section Prompt Injection (`OnboardingSection`)

The `OnboardingSection` [orchestrator/modules/context/sections/onboarding.py]() is a critical component that injects stage-aware guidance into the LLM's prompt. This ensures that Auto (the AI agent) always knows the user's current position in the onboarding journey and what actions are expected next.

*   **Trigger**: The section is rendered if the workspace's `onboarding.stage` is not terminal (`completed` or `skipped`), or if the user explicitly triggers it with phrases like "set up my workspace" [orchestrator/modules/context/sections/onboarding.py:10-15,41-46]().
*   **Stage-Specific Guidance**: It renders a common set of rules (`_COMMON_RULES`, `_CAPABILITY_DOCTRINE`) and *exactly one* stage-specific guidance block (e.g., `_STAGE_QUESTIONS`, `_STAGE_TEACH`, `_STAGE_PROPOSAL`) based on the current `onboarding.stage` [orchestrator/modules/context/sections/onboarding.py:55-58,104-140]().
*   **Tool Usage Instructions**: The section explicitly instructs Auto to use the `platform_update_onboarding` tool to record progress and segment answers [orchestrator/modules/context/sections/onboarding.py:66-68]().
*   **Trust Rules**: It bakes in hard rules, such as never setting `skip_verification` or `auto_approve` for missions or tools, reinforcing security and user approval [orchestrator/modules/context/sections/onboarding.py:71-73]().
*   **Capability Doctrine**: The `_CAPABILITY_DOCTRINE` provides reflexes for Auto, guiding it on how to handle common scenarios like connecting apps, scanning URLs, or staffing from the marketplace [orchestrator/modules/context/sections/onboarding.py:79-97]().
*   **Dynamic Content**: The section can dynamically adjust its content, for example, by adding a `firecrawl_note` if the site scanning capability is not available [orchestrator/modules/context/sections/onboarding.py:117-117]().

```mermaid
graph TD
    A[User Interaction] --> B{ContextService}
    B --> C[Assemble Prompt]
    C --> D[OnboardingSection.render(ctx)]

    subgraph "OnboardingSection Logic"
        D --> E{Is onboarding active?}
        E -- "No (terminal stage)" --> F[Render ""]
        E -- "Yes (active stage)" --> G{Is user re-triggering?}
        G -- "Yes" --> H[Render re-trigger message + questions]
        G -- "No" --> I[Get current_stage from workspace.onboarding]
        I --> J{Select stage-specific guidance}
        J --> K[Combine _HEADER + _COMMON_RULES + _CAPABILITY_DOCTRINE + Stage Guidance]
        K --> L[Inject into prompt]
    end

    L --> M[LLM receives prompt]
    M --> N[LLM generates response/tool call]
    N -- "Calls platform_update_onboarding" --> O[update_onboarding handler]
    O --> P[onboarding_state service]
    P --> Q[Update workspace.onboarding]
    Q --> R[Next turn: OnboardingSection.render(ctx) reflects new state]
```
Title: OnboardingSection Prompt Injection Flow
Sources: [orchestrator/modules/context/sections/onboarding.py:1-30](), [orchestrator/services/onboarding_state.py:99-101]()

## Segment-Implied Advance, Reset, and Evidence Tests

The onboarding state machine includes sophisticated logic for advancing stages based on user input, handling resets, and requiring evidence for certain transitions.

### Segment-Implied Advance

The `implied_stage` function [orchestrator/services/onboarding_state.py:332-349]() is a pure function that determines if a stage advancement is implied by the collected segment answers.
*   If the current stage is `not_started` and any segment answer is provided, it implies a move to `questions`.
*   If the current stage is `not_started` or `questions` and all three core segment answers (`business`, `goal`, `comfort`) are provided, it implies a move to `teach`.
*   Later stages do not imply advancement based on segment answers alone [orchestrator/tests/test_prd222_segment_implied_advance.py:62-67]().

This logic is crucial for allowing the LLM to naturally progress the user through the initial questions without needing explicit `advance_to` calls for every single answer [orchestrator/tests/test_prd222_segment_implied_advance.py:70-86]().

### Resetting Onboarding

While not explicitly called "reset," the `skipped` stage acts as a terminal state that effectively ends the onboarding flow. It can be reached from any non-terminal stage [orchestrator/services/onboarding_state.py:138-139]().

### Evidence-Based Transitions

The transition to the `boom` stage (where the user is presented with their built team) requires concrete evidence that something has actually been built. This prevents the LLM from advancing the user to a "payoff" stage without delivering on the promise.

*   **`BUILD_EVIDENCE_STAGE`**: This constant is set to "boom" [orchestrator/services/onboarding_state.py:263-263]().
*   **`build_evidence(db, workspace)`**: This function checks for:
    *   A `package_installed` funnel stamp in the onboarding document.
    *   The presence of workspace-owned agents (excluding system or onboarding-role agents).
    *   The existence of orchestration runs (missions) for the workspace.
    *   If any of these are true, `any` is set to `True` [orchestrator/services/onboarding_state.py:352-390]().
*   **Gate Enforcement**: If the target stage is `boom` and `build_evidence` returns `any=False`, an `InvalidStageTransition` is raised, preventing the advance [orchestrator/services/onboarding_state.py:240-243]().
*   **Agent Definition**: The definition of a "built agent" for evidence purposes aligns with the `workspace_purge` survivor predicate, ensuring consistency across the codebase [orchestrator/tests/test_prd222_boom_build_evidence.py:142-157]().

These mechanisms ensure a robust and intelligent onboarding flow that adapts to user input and verifies progress before advancing.

Sources:
* [orchestrator/modules/context/sections/onboarding.py:10-15]()
* [orchestrator/modules/context/sections/onboarding.py:41-46]()
* [orchestrator/modules/context/sections/onboarding.py:55-58]()
* [orchestrator/modules/context/sections/onboarding.py:66-68]()
* [orchestrator/modules/context/sections/onboarding.py:71-73]()
* [orchestrator/modules/context/sections/onboarding.py:79-97]()
* [orchestrator/modules/context/sections/onboarding.py:104-140]()
* [orchestrator/modules/context/sections/onboarding.py:117-117]()
* [orchestrator/modules/tools/discovery/actions_onboarding.py:21-28]()
* [orchestrator/modules/tools/discovery/actions_onboarding.py:32-41]()
* [orchestrator/modules/tools/discovery/actions_onboarding.py:44-142]()
* [orchestrator/modules/tools/discovery/actions_onboarding.py:55-142]()
* [orchestrator/modules/tools/discovery/actions_onboarding.py:76-81]()
* [orchestrator/modules/tools/discovery/actions_onboarding.py:82-116]()
* [orchestrator/modules/tools/discovery/actions_onboarding.py:118-126]()
* [orchestrator/modules/tools/discovery/actions_onboarding.py:129-131]()
* [orchestrator/modules/tools/discovery/handlers_onboarding.py:3-4]()
* [orchestrator/modules/tools/discovery/handlers_onboarding.py:5-6]()
* [orchestrator/modules/tools/discovery/handlers_onboarding.py:38-47]()
* [orchestrator/modules/tools/discovery/handlers_onboarding.py:93-159]()
* [orchestrator/modules/tools/discovery/handlers_onboarding.py:180-209]()
* [orchestrator/services/onboarding_state.py:1-34]()
* [orchestrator/services/onboarding_state.py:11-16]()
* [orchestrator/services/onboarding_state.py:17-20]()
* [orchestrator/services/onboarding_state.py:20-22]()
* [orchestrator/services/onboarding_state.py:23-33]()
* [orchestrator/services/onboarding_state.py:76-91]()
* [orchestrator/services/onboarding_state.py:94-96]()
* [orchestrator/services/onboarding_state.py:99-101]()
* [orchestrator/services/onboarding_state.py:131-143]()
* [orchestrator/services/onboarding_state.py:138-139]()
* [orchestrator/services/onboarding_state.py:200-260]()
* [orchestrator/services/onboarding_state.py:240-243]()
* [orchestrator/services/onboarding_state.py:263-263]()
* [orchestrator/services/onboarding_state.py:300-329]()
* [orchestrator/services/onboarding_state.py:332-349]()
* [orchestrator/services/onboarding_state.py:352-390]()
* [orchestrator/tests/test_prd222_boom_build_evidence.py:142-157]()
* [orchestrator/tests/test_prd222_segment_implied_advance.py:62-67]()
* [orchestrator/tests/test_prd222_segment_implied_advance.py:70-86]()
* [orchestrator/tests/test_prd222_segment_implied_advance.py:104-112]()
* [orchestrator/tests/test_prd222_onboarding_tool.py:108-117]()
* [orchestrator/tests/test_prd222_onboarding_tool.py:119-130]()
* [orchestrator/tests/test_prd222_onboarding_tool_prior.py:11-13]()

---