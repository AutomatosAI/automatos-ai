# Watches & Autonomous Monitoring

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/agents/agent-readiness-badge.tsx](frontend/components/agents/agent-readiness-badge.tsx)
- [orchestrator/alembic/versions/governance_blueprints.py](orchestrator/alembic/versions/governance_blueprints.py)
- [orchestrator/core/models/blueprints.py](orchestrator/core/models/blueprints.py)
- [orchestrator/modules/tools/discovery/actions_governance.py](orchestrator/modules/tools/discovery/actions_governance.py)
- [orchestrator/modules/tools/discovery/handlers_governance.py](orchestrator/modules/tools/discovery/handlers_governance.py)
- [orchestrator/pytest.ini](orchestrator/pytest.ini)
- [orchestrator/scripts/probe_document_vectors.py](orchestrator/scripts/probe_document_vectors.py)
- [orchestrator/services/blueprint_validator.py](orchestrator/services/blueprint_validator.py)
- [orchestrator/services/escalation_service.py](orchestrator/services/escalation_service.py)
- [orchestrator/services/watch_actions.py](orchestrator/services/watch_actions.py)
- [orchestrator/services/watch_decider.py](orchestrator/services/watch_decider.py)
- [orchestrator/services/watch_notifications.py](orchestrator/services/watch_notifications.py)
- [orchestrator/services/watch_rerun.py](orchestrator/services/watch_rerun.py)
- [orchestrator/services/watch_service.py](orchestrator/services/watch_service.py)
- [orchestrator/services/watch_ticker.py](orchestrator/services/watch_ticker.py)
- [orchestrator/tests/golden/prd204_policy_table.json](orchestrator/tests/golden/prd204_policy_table.json)
- [orchestrator/tests/requirements.txt](orchestrator/tests/requirements.txt)
- [orchestrator/tests/test_prd204_rerun.py](orchestrator/tests/test_prd204_rerun.py)
- [orchestrator/tests/test_prd204_silent_holes.py](orchestrator/tests/test_prd204_silent_holes.py)
- [orchestrator/tests/test_prd204_watch_actions.py](orchestrator/tests/test_prd204_watch_actions.py)
- [orchestrator/tests/test_prd204_watch_decider.py](orchestrator/tests/test_prd204_watch_decider.py)
- [orchestrator/tests/test_prd204_watch_hooks.py](orchestrator/tests/test_prd204_watch_hooks.py)
- [orchestrator/tests/test_prd204_watch_registry.py](orchestrator/tests/test_prd204_watch_registry.py)
- [orchestrator/tests/test_prd204_watch_service.py](orchestrator/tests/test_prd204_watch_service.py)
- [orchestrator/tests/test_prd204_watch_ticker.py](orchestrator/tests/test_prd204_watch_ticker.py)
- [orchestrator/tests/test_rag_perf_pass.py](orchestrator/tests/test_rag_perf_pass.py)

</details>



## Purpose and Scope

The Watches & Autonomous Monitoring subsystem (introduced under PRD-204) provides continuous, autonomous supervision over long-running platform entities such as missions, agent runs, scheduled playbooks, and board tasks. Rather than relying solely on immediate execution feedback, watches actively supervise target progress against defined success and failure criteria, handle state transitions, execute automated corrective actions (such as replanning, reassigning, or rerunning), and escalate blocked or failing work to human operators via board cards and notifications.

Sources: `[orchestrator/services/watch_service.py:1-24]()`, `[orchestrator/services/watch_ticker.py:1-33]()`

---

## 1. Watch Subsystem Architecture & Registry (`WatchService`)

The `WatchService` acts as the single owner of watch-registry mutations, managing the creation, lookup, cancellation, and guarded status transitions of watches. It enforces the invariant that exactly one live watch may supervise a given target at any time via a partial unique index in PostgreSQL.

### Core Registry Operations

- **Creation (`create_watch`)**: Initializes a `Watch` entity, writing a `CREATED` event to the `watch_events` audit table and scheduling the initial check interval `[orchestrator/services/watch_service.py:149-181]()`.
- **Duplicate Prevention (`WatchAlreadyExistsError`)**: Rejects concurrent or duplicate live watches on the same target, backed by database-level integrity checks `[orchestrator/services/watch_service.py:63-73]()`.
- **Guarded Transitions (`transition`)**: Enforces state machine rules via `ALLOWED_WATCH_TRANSITIONS` and handles terminal state cleanups `[orchestrator/services/watch_service.py:6-7]()`, `[orchestrator/tests/test_prd204_watch_service.py:125-141]()`.
- **Idempotent Ingestion (`ingest` & `ingest_terminal`)**: Safely records lifecycle events using unique event keys swallowed via database savepoints, ensuring duplicate producer hooks do not corrupt telemetry `[orchestrator/services/watch_service.py:8-12]()`, `[orchestrator/tests/test_prd204_watch_service.py:161-187]()`.
- **Claiming Due Watches (`claim_due_watches`)**: Employs `FOR UPDATE SKIP LOCKED` semantics within an isolated transaction to coordinate concurrent watcher tickers safely `[orchestrator/services/watch_service.py:10-18]()`.

```mermaid
title "Natural Language Space to Code Entity Space: Watch Registry"
flowchart TD
    UserReq["User creates or schedules execution"] --> CreateWatch["WatchService.create_watch()"]
    CreateWatch --> DB[(PostgreSQL watches table)]
    DB --> UniqueCheck{{"Partial Unique Index:\none live watch per target"}}
    UniqueCheck -- "Duplicate" --> RaiseErr["Raise WatchAlreadyExistsError"]
    UniqueCheck -- "Valid" --> InsertRow["Insert Watch row + WatchEventType.CREATED"]
    
    Ticker["WatchTicker._tick()"] --> Claim["WatchService.claim_due_watches()"]
    Claim --> Lock["SELECT ... FOR UPDATE SKIP LOCKED"]
    Lock --> Check["WatchTicker._check_watch()"]

    subcode["Code Entities"]
    style subcode fill:none,stroke:none
    click CreateWatch href "orchestrator/services/watch_service.py"
    click RaiseErr href "orchestrator/services/watch_service.py"
    click Claim href "orchestrator/services/watch_service.py"
```

Sources: `[orchestrator/services/watch_service.py:1-182]()`, `[orchestrator/tests/test_prd204_watch_service.py:94-209]()`

---

## 2. Watch Ticker & Background Sweep (`WatchTicker`)

The `WatchTicker` operates as a background sweep loop registered on the unified scheduler (`TaskReconciler` pattern). While S3 event hooks provide a fast path for terminal states, the ticker serves as a robust fallback and trend analyzer.

### Ticker Mechanics

- **Scheduled Job**: Runs at configured intervals (`WATCHER_TICK_SECONDS`), driven by `AsyncIOScheduler` `[orchestrator/services/watch_ticker.py:85-99]()`.
- **Claim & Sweep**: Claims due watches where `next_check_at <= now` using `claim_due_watches` `[orchestrator/services/watch_ticker.py:123-128]()`.
- **State Refresh**: Performs cheap status reads against target tables (`_RUN_TERMINAL`, `_EXECUTION_TERMINAL`, `_BOARD_TERMINAL`) to catch terminal states missed by hooks `[orchestrator/services/watch_ticker.py:56-60, 111-140]()`.
- **Missed Run Detection**: For scheduled playbooks, compares expected cron fires against the latest `recipe_executions` rows, emitting missed-run events when thresholds are exceeded `[orchestrator/services/watch_ticker.py:17-19, 61-64]()`.
- **Deadline Enforcement**: Automatically transitions watches to `EXPIRED` status if `deadline_at` is breached without a verdict `[orchestrator/services/watch_ticker.py:15-16, 152-176]()`.

```mermaid
title "WatchTicker Sweep & Execution Flow"
flowchart TD
    Start["AsyncIOScheduler Tick Job"] --> OpenSession["SessionLocal()"]
    OpenSession --> ClaimDue["WatchService.claim_due_watches(db, now)"]
    ClaimDue --> Loop["Iterate Claimed Watches"]
    
    Loop --> TypeCheck{{"Target Type?"}}
    TypeCheck -- "scheduled_playbook" --> CheckCron["_check_scheduled_playbook()"]
    TypeCheck -- "run / mission / task" --> CheckRun["_check_run_target()"]
    
    CheckRun --> TerminalCheck{{"Is Target Terminal?"}}
    TerminalCheck -- "Yes" --> IngestTerminal["WatchService.ingest_terminal()"]
    TerminalCheck -- "No" --> DeltaCheck{{"Meaningful Change?"}}
    
    IngestTerminal --> Decider["WatchDecider.decide_terminal()"]
    DeltaCheck -- "Yes" --> WriteEvent["WatchService.ingest(event_type='status_change')"]
    DeltaCheck -- "No" --> Skip["No-op / Reschedule"]
```

Sources: `[orchestrator/services/watch_ticker.py:1-176]()`, `[orchestrator/tests/test_prd204_watch_ticker.py:1-204]()`

---

## 3. Watch Decider Policy Table & Verdict Evaluation (`WatchDecider`)

When a watch encounters a terminal target or triggers an evaluation threshold, it delegates to the `WatchDecider`. The decider evaluates verdicts based on configured quality thresholds, success criteria, and workspace autonomy policies.

### Evaluation Workflow

- **Deterministic Scoring**: Compares execution outputs or completion metrics against the `quality_threshold`. If deterministic criteria fail or succeed without ambiguity, LLM overhead is bypassed `[orchestrator/tests/test_prd204_watch_ticker.py:172-178]()`.
- **Policy Tables**: Evaluates matching rules defined in policy tables (e.g., `run_and_report`, `auto_replan`, `auto_rerun`, `escalate`) `[orchestrator/services/watch_service.py:166]()`.
- **Action Dispatch**: Based on the verdict, either closes the watch (`PASSED`/`FAILED`), triggers corrective actions via `watch_actions`, or hands off execution to an escalation service `[orchestrator/services/watch_ticker.py:23-29]()`.

Sources: `[orchestrator/services/watch_ticker.py:23-32]()`, `[orchestrator/tests/test_prd204_watch_decider.py:1-1]()`

---

## 4. Direction-Change & Corrective Actions (`WatchActions` & Rerun)

When a supervised execution fails or drops below quality thresholds, the subsystem initiates automated corrective actions.

### Corrective Action Mechanisms

- **Playbook Rerun & Tweak (`watch_rerun.py`)**:
  - Copies prior execution inputs into a new `RecipeExecution` row with `retry_of` lineage and `attempt_count + 1` `[orchestrator/services/watch_rerun.py:5-8, 153-184]()`.
  - Supports per-execution `step_overrides` (e.g., prompt template adjustments) merged dynamically at run start. The shared recipe definition (`workflow_recipes.steps`) is never mutated `[orchestrator/services/watch_rerun.py:8-10, 89-120]()`.
  - **Approval Gating**: Evaluates rerun costs against workspace approval policies. Full-auto paths launch immediately, while ask paths park a durable `ApprovalGrant(subject_type=SUBJECT_PLAYBOOK_RUN)` carrying the complete rerun specification `[orchestrator/services/watch_rerun.py:12-33]()`.
- **Direction Changes**: Supports replanning (`replan_mission`), agent re-assignment (`reassign`), agent spawning (`spawn_agent`), and human escalation (`escalate`) subject to action budgets (`action_budget`) `[orchestrator/services/watch_actions.py:1-13]()`, `[orchestrator/tests/test_prd204_watch_actions.py:1-12]()`.

```mermaid
title "Natural Language Space to Code Entity Space: Rerun & Approval Flow"
flowchart TD
    WatchFail["Watch detects failed execution"] --> RequestRerun["watch_rerun.request_rerun()"]
    RequestRerun --> EstimateCost["estimate_rerun_cost_usd()"]
    EstimateCost --> EvalPolicy["evaluate_approval()"]
    
    EvalPolicy -- "full_auto / under ceiling" --> Launch["create_rerun_execution()\nlaunch_execution()"]
    EvalPolicy -- "always_ask / over ceiling" --> ParkGrant["Park ApprovalGrant\n(subject_type=SUBJECT_PLAYBOOK_RUN)"]
    
    ParkGrant --> Inbox["Approvals Inbox UI"]
    Inbox -- "Admin Resume" --> ResumeExec["resume_playbook_run_grant()\nWATCH_GRANT_EXECUTORS['rerun']"]
    ResumeExec --> Launch

    subcode["Code Entities"]
    style subcode fill:none,stroke:none
    click RequestRerun href "orchestrator/services/watch_rerun.py"
    click EstimateCost href "orchestrator/services/watch_rerun.py"
    click EvalPolicy href "orchestrator/services/watch_rerun.py"
    click ResumeExec href "orchestrator/services/watch_rerun.py"
```

Sources: `[orchestrator/services/watch_rerun.py:1-184]()`, `[orchestrator/services/watch_actions.py:1-13]()`, `[orchestrator/tests/test_prd204_rerun.py:1-152]()`

---

## 5. Escalations & Human Handoff (`escalation_service`)

When corrective action budgets are exhausted, deadlines pass, or tasks experience repeated stalls, the `escalation_service` creates structured board tasks for human review.

### Key Escalation Functions

- **Blocked Task Escalation (`check_blocked_escalations`)**: Sweeps for board tasks blocked longer than `BLOCKED_ESCALATION_HOURS` (24 hours) and creates high-priority escalation items in the inbox `[orchestrator/services/escalation_service.py:22-44]()`.
- **Watch Escalation (`escalate_watch`)**: Creates a watch-flavored board task carrying target metadata, failure reasons, corrective action counts, and recommendations (`rerun`, `replan`, `reassign`, or accept), ensuring deduplication via workspace tags (`watch:{watch_id}`) `[orchestrator/services/escalation_service.py:104-165]()`.
- **Stalled Task Escalation (`escalate_stalled_task`)**: Automatically flags tasks experiencing repeated execution stalls (`stall_count >= 2`) for human intervention `[orchestrator/services/escalation_service.py:168-204]()`.

Sources: `[orchestrator/services/escalation_service.py:1-221]()`

---

## 6. Watch Notifications & Watchlist UI Tab

### Notification Routing
Watcher-only notifications (such as sweep-caught terminal events, missed cron runs, and deadline expirations) are dispatched through `services.watch_notifications.dispatch_watch_notification` or the centralized `NotificationDispatcher`. They integrate with workspace preferences for in-app delivery, webhooks, or chat reports (e.g., `_say_cancelled_in_chat`) `[orchestrator/services/watch_service.py:117-140]()`, `[orchestrator/services/watch_ticker.py:23-32]()`, `[orchestrator/tests/test_prd204_watch_ticker.py:99-118]()`.

### Watchlist UI Tab
The Watchlist tab in the Command Center frontend renders active watches, target lineages, timeline audit events, corrective action usage metrics, and direct links to approval grants or board escalation cards.

Sources: `[orchestrator/services/watch_service.py:117-140]()`, `[orchestrator/services/watch_ticker.py:23-32]()`

---