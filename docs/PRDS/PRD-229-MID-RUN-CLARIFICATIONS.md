# PRD-229 — Mid-Run Clarifications: `ask_orchestrator` and the Escalation Ladder

> **Status:** Draft for rollout planning — written 2026-08-27, not yet scheduled. **Staged last — depends on PRD-225; benefits from PRD-228.**
> **Origin:** Munder Difflin deep review (2026-08-27) — the deepest idea in their design: *workers ask the god; the god answers routine questions itself; only the critical reaches the human.* Review artifact:
> https://claude.ai/code/artifact/f31677a8-f2cb-47fe-b7dd-f705d764418b
> **Type (per CLAUDE.md §3):** Extension + consolidation (includes a decide-and-delete on dead code).

## 1. Overview

In Automatos today, an executing agent that hits ambiguity can ask nobody anything — it does its best or fails, and the human discovers the failure later. This PRD gives every executing lane one tool — `ask_orchestrator` — with a two-step ladder behind it: **Auto answers routine questions itself** from the work's own context (mission plan, upstream results, memory, intake corpus, fleet state), and only what Auto cannot answer escalates into the PRD-225 questions queue for a human. This is the autonomy multiplier: it converts silent failures into cheap clarifications, and clarifications into human questions only when they deserve to be.

## 2. Current reality (grounded)

- **No clarification path exists mid-run.** Task prompts are assembled once (`coordinator_service.py:2038 _prepare_task` — upstream digest + prompt build) and executed to a verdict (`:2212 _run_agent_io` → `AgentFactory.execute_with_prompt`); nothing in the tool surface lets the executing agent query anyone. Verification failure → retry/fail lanes, not questions.
- **The code that looks like it does this is dead.** `modules/agents/communication/inter_agent.py` — `AgentCommunicationProtocol` (`:94`), `SharedContextManager` (`:400`), `CollaborativeReasoner` (`:651`), `CollaborativeAgentFactory.execute_team_task` (`:978,991`) — has **zero callers** outside its own module (grep verified 2026-08-27; only the re-export in `modules/agents/__init__.py:36,60`). Per house rule §5 (delete what's superseded), this PRD carries the decide-and-delete.
- **The context to answer from already exists per run:** the mission field (`_process_run` creates it, `coordinator_service.py:1544`), upstream task results (the digest at `:2038`), memory (PRD-206 phase 1), the intake/business corpus, and — once PRD-228 lands — fleet state.
- **The escalation target exists once PRD-225 lands:** question-kind asks with park/resume semantics. Related plumbing already present: escalation ladder levels incl. L2 APPROVAL (`core/services/escalation.py:26-31`), watch escalation actions (`services/watch_actions.py:65,543`).

## 3. Goals

- G1: An executing agent (mission task, board task, playbook step) can call `ask_orchestrator(question, context_ref?)` and receive either an answer inline or "parked — continue what you can / stop cleanly," never a hang.
- G2: Auto's answering step is grounded and budgeted: it answers only from retrievable context (no fabrication — answers cite their source ref), spends a bounded number of answer attempts per run, and logs every Q&A onto the run's event trail.
- G3: Unanswerable or critical questions (destructive ops, spend, scope changes — the PRD-223 governance categories) auto-escalate into the PRD-225 queue with the subject parked; the resume path injects the human answer exactly as PRD-225 defines.
- G4: The dead `inter_agent.py` machinery is decided: deleted (default) or explicitly adopted as this PRD's transport — not left ambient.

## 4. Non-goals

- No free-form agent↔agent chat or mailboxes — the ladder is worker → orchestrator → human, nothing lateral (lateral coordination stays the planner's job via dependencies).
- No mid-turn *human* interactivity in the chat stream (the human path is PRD-225's queue, not a blocking prompt).
- No new autonomy semantics — the existing autonomy dial and approval gates keep governing what Auto may answer alone (a question in a governance category is always escalated regardless of confidence).

## 5. Design

- **Component A — the tool.** `ask_orchestrator` registered into the TASK_EXECUTION tool surface (`modules/context/modes.py` exposure; 3-file platform pattern) — available to executing agents, not to chat users. Synchronous with a hard budget: the call returns an answer, or a park decision, within one bounded round.
- **Component B — the answering service.** `services/orchestrator_answers.py`: given the run/task subject + question, retrieve (mission plan + upstream digest + field docs + memory + intake corpus + PRD-228 fleet state when relevant), answer with citations to the refs used, or return `cannot_answer(reason)`. Per-run answer budget (config `CLARIFICATION_BUDGET`, default e.g. 3) — past budget, everything escalates; the budget spend is logged on the run ledger (`progress_ledger.py` lane).
- **Component C — the ladder.** `cannot_answer` OR governance-category question → create a PRD-225 question against the subject (park semantics identical), notify per PRD-225 (in-app / Telegram), and return "parked" to the caller so the agent finishes what it can and stops cleanly. On human answer: PRD-225 resume injects the Q&A into the next run context; dependency-aware dispatch keeps unrelated tasks moving throughout.
- **Component D — decide-and-delete.** Default: delete `inter_agent.py` and its `__init__` re-exports in the same PR (zero callers; keeping it "just in case" is how 103 routers happened). Alternative considered and rejected unless Gerard overrides: adopting `AgentCommunicationProtocol` as transport — it models lateral agent messaging we are explicitly not building.

## 6. Waves & acceptance criteria

**Wave 0 — answering service + tool (no escalation yet).**
- [ ] `ask_orchestrator` visible only in TASK_EXECUTION context (context-mode test); chat surface unchanged.
- [ ] Grounded answer path: fixture run where the answer exists in upstream results → agent receives it with a source ref; fixture where it doesn't → `cannot_answer`, and (pre-escalation) the tool returns "proceed with stated assumption" guidance without inventing facts.
- [ ] Budget enforced; Q&A recorded on the run event trail (visible in the mission activity feed's existing event rendering).

**Wave 1 — escalation ladder (requires PRD-225).**
- [ ] `cannot_answer` creates a question-kind ask, parks the subject, notifies; human answer resumes with the Q&A in context; the full trail (agent question → Auto attempt → human answer) reads coherently on the subject.
- [ ] Governance-category questions skip Auto and escalate directly (category fixtures: spend, destructive op, scope change).

**Wave 2 — consolidation.**
- [ ] `inter_agent.py` deleted with re-exports removed, or formally adopted with callers — no third state. Import-lint green on the merged tip (declare-the-world files break on the merged tip, not the branch — known trap).

## 7. Technical considerations

- The synchronous round must respect the executing agent's `asyncio.wait_for` envelope (`coordinator_service.py:2237-2249`) — the answer budget is time-boxed as well as count-boxed so a slow retrieval can't blow the task timeout.
- Answer grounding uses the same retrieval stack as chat (RAG on S3 Vectors only — pgvector paths are legacy); no new retrieval infrastructure.
- Token economics: one clarification round is far cheaper than a failed-and-retried task; record both counters so the claim is measurable (success metric below).

## 8. Success metrics

- Reduction in mission task failure/retry rate attributable to ambiguity (baseline from current reconciler stats) after Wave 1.
- ≥ X% of `ask_orchestrator` calls answered by Auto without human escalation (target to set from Wave 0 telemetry).

## 9. Open questions (Gerard)

1. `CLARIFICATION_BUDGET` default per run (proposal: 3 Auto-answers; unlimited escalations — escalations are visible and cheap by design)?
2. When a question parks a task mid-run, should partial output be recorded as a draft result (visible on the card) or discarded to rerun clean after the answer? Proposal: record as draft — decision history beats cleanliness.
3. Confirm delete for `inter_agent.py` (Component D default), or do you want it adopted?
