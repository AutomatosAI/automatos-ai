# Research Mode

Write ONE research PRD section from the plan, validate, commit, exit.

## Phase 0: Orient

Study with subagents:
- @CLAUDE.md (project conventions)
- @docs/PRDS/100-RESEARCH-AUTONOMOUS-OPERATING-LAYER.md (master research document — this is your north star)
- Read the current loop's `prd.json` and `IMPLEMENTATION_PLAN.md`

### Determine Active Loop

```bash
cat scripts/ralph/research-100/meta.json | jq '.currentLoop'
```

Use that number to find your working directory:
```bash
LOOP_DIR="scripts/ralph/research-100/loop-${CURRENT_LOOP}-*"
```

Read `${LOOP_DIR}/prd.json` for your user stories and acceptance criteria.
Read `${LOOP_DIR}/IMPLEMENTATION_PLAN.md` for current progress.

### Check for completion

```bash
grep -c "^\- \[ \]" ${LOOP_DIR}/IMPLEMENTATION_PLAN.md || echo 0
```

- If 0: Run validation → commit → update meta.json (advance currentLoop, set status to "complete") → output **RALPH_COMPLETE** → exit
- If > 0: Continue to Phase 1

## Phase 1: Research & Write

1. **Study the plan** — Choose the FIRST unchecked task from the loop's IMPLEMENTATION_PLAN.md
2. **Read prd.json** — Find the matching US-XXX story and follow its acceptance criteria exactly
3. **Delegate research to subagents** — Spawn individual Agent tool calls for each research target. NEVER do research yourself — always delegate.
4. **Read existing code** — For design decisions, read the actual codebase to understand what exists today (can be delegated to subagents too)
5. **Synthesize & Write** — Take subagent findings and write the document section. ONE section/story only. Write completely — no placeholders or "TODO: fill in later"
6. **Validate** — All acceptance criteria must be met

### Research Agent Pattern (MANDATORY)

Every research task MUST be performed by subagents via the Agent tool. You are the **orchestrator** — you delegate, synthesize, and write. You do NOT research directly.

**One agent = one research focus. No exceptions.**

Each research target gets its OWN dedicated agent with a single-focus brief. An agent studying Temporal should know NOTHING about Prefect. An agent reading the Automatos codebase should not also be researching external repos. Clean context = deep research.

**Rules:**
1. **ONE topic per agent** — never ask an agent to research two systems or two concepts
2. **All agents in parallel** — spawn every research agent for a story in a SINGLE message with multiple Agent tool calls
3. **Focused prompts** — tell the agent exactly what to research, what format to return, and what to ignore
4. **No cross-contamination** — agents don't see each other's findings. YOU synthesize across agents.
5. **Codebase agents are separate** — reading Automatos code is a different agent from researching external systems

**Agent prompt template:**
```
You are researching ONE topic: [SPECIFIC SYSTEM/CONCEPT].

## Research Question
[Exact question to answer]

## Focus Areas
- [Specific aspect 1]
- [Specific aspect 2]
- [Specific aspect 3]

## Output Format
Return your findings as:
[Structured format — table, bullet points, code examples, etc.]

## Citation Requirements
- Include specific file paths, URLs, or paper titles for every claim
- If you can't find evidence, say so — don't speculate

## Context (for relevance only)
We're building [2-3 sentence description of what Automatos needs].
Your findings will be compared against other systems' approaches.
Do NOT research those other systems — stay focused on [YOUR TOPIC].
```

**Example: Story has 5 research targets + 1 codebase audit**

Spawn 6 agents in ONE message:
- Agent 1: "Research Temporal's workflow execution model. Focus: table schema, state transitions, activity lifecycle."
- Agent 2: "Research Prefect's flow run model. Focus: task dependencies, state machine, result passing."
- Agent 3: "Research Airflow's DAG run model. Focus: task instances, XCom, trigger rules."
- Agent 4: "Research Dagster's ops/jobs model. Focus: asset materialization, IO managers, run storage."
- Agent 5: "Research Symphony's task lifecycle. Focus: WORKFLOW.md, continuation vs retry, reconciliation."
- Agent 6: "Read orchestrator/core/models/board.py and alembic/versions/. Document existing table schemas that new orchestration tables must integrate with."

**After all 6 agents return:** YOU are the synthesizer. Compare findings. Build the comparison table. Add your judgment on what to adopt. Write the PRD section. This is where the real thinking happens — agents gather evidence, you make decisions.

### Research Tools Available (for subagents)

- `mcp__deepwiki__ask_question` — Ask about any GitHub repo's architecture/patterns
- `mcp__deepwiki__read_wiki_contents` — Read documentation for GitHub repos
- Web search — For academic papers, blog posts, prior art
- Codebase grep/read — For understanding what Automatos has today
- Context Engineering repo — `docs/context-engineering/00_foundations/` (symlinked into repo, 14 chapters of theoretical foundation: atoms → molecules → cells → organs → neural fields → symbolic mechanisms → unified field theory)

### Writing Rules (CRITICAL)

- **Loop 0 (Design):** Output is a structured outline per PRD — sections, key questions, research targets, acceptance criteria for the full PRD. Written to `docs/PRDS/outlines/` directory.
- **Loops 1-8 (Full PRDs):** Output is a complete research + design PRD written to `docs/PRDS/`. Each section must have substance — real analysis, real architecture decisions, real references.
- **Every claim needs evidence** — cite repos, papers, or codebase files
- **Every design decision needs rationale** — why this approach over alternatives
- **Every architecture choice references existing Automatos code** — show how it connects to what's built
- **Include code snippets** for data models, interfaces, and key algorithms (pseudocode or Python)
- **Include diagrams** as ASCII art or mermaid syntax where they aid understanding

### Quality Bar

A research PRD section is done when:
- Someone could implement from it without asking clarifying questions
- Prior art is cited with specific references (not "many systems do X")
- Design decisions have explicit tradeoffs listed
- Data models include field types and constraints
- Interfaces are defined with method signatures

## Phase 2: Update & Learn

**Update the loop's IMPLEMENTATION_PLAN.md:**
- Mark completed task `- [x] Completed`
- Add any discovered insights or risks
- Note connections to other PRDs discovered during research

**If this was the LAST task in the loop:**
- Update `scripts/ralph/research-100/meta.json`:
  - Set current loop's status to `"complete"`
  - Increment `currentLoop`

## Phase 3: Commit & Exit

```bash
git add -A && git commit -m "research(prd-10X): [description of what was researched/written]"
```

Check remaining:
```bash
grep -c "^\- \[ \]" ${LOOP_DIR}/IMPLEMENTATION_PLAN.md || echo 0
```

- If > 0: Say "X tasks remaining in loop Y" and EXIT
- If = 0: Output **RALPH_COMPLETE**

## Guardrails

1. Research depth matters — superficial summaries are worthless. Dig into actual implementations.
2. Connect everything back to PRD-100's vision and existing Automatos infrastructure.
3. ONE task per iteration. Research before writing. Validation MUST pass.
4. Keep IMPLEMENTATION_PLAN.md current with discoveries.
5. Cross-reference between PRDs — if PRD-103 research reveals something PRD-101 needs, note it.
6. Never output RALPH_COMPLETE if tasks remain.
