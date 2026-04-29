# AUTOMATOS 0.2 — Deliverables & Autonomous Flow

**Purpose:** Spec the end-to-end user journey so autonomy is visible. "Automatos made this for me" must be a single tab answer.

This is the product doc. Architecture gets captured in the other plan docs; this one is the narrative.

---

## 1. The unified `run` object

Every goal a user gives Automatos becomes a `run`. One table, one API, one UI card, four `kind`s.

```
run
├── id
├── workspace_id
├── kind           : chat | mission | recipe | plan
├── title          : user-facing label
├── goal           : the original input (text or structured)
├── status         : queued | running | paused | succeeded | failed | cancelled
├── coordinator_id : who's driving (sequential coordinator for most)
├── agent_ids      : who's on the crew
├── cost_usd       : total spend
├── tokens         : total tokens
├── parent_run_id  : runs-of-runs (e.g. plan → missions)
├── schedule_id    : recipe reference if recurring
├── created_at / started_at / finished_at
└── (+ JSONB: mode_config)  : mode-specific metadata
```

**Sub-tables:**
- `run_tasks` — sub-tasks the coordinator decomposed the goal into (mission/plan only).
- `run_events` — ordered event stream for debugging + UI live view.
- `deliverables` — the outputs (see §3).

### Four kinds, one object

| Kind | Surface | Coordinator | Produces |
|---|---|---|---|
| `chat` | compose box → streaming reply | none (direct agent) | one deliverable (message tape; file artifacts if any) |
| `mission` | goal → plan → approve → execute | sequential coordinator (PRD-82A) | N deliverables (one per task completion) |
| `recipe` | scheduled recurrence OR webhook trigger | scheduled → mission under the hood | N deliverables per run; recipe aggregates them |
| `plan` | wizard intake → workspace configuration | coordinator with BLUEPRINT+FORGE | 1 deliverable per configured-thing (agents, skills, dashboards, recipes) |

The same endpoint creates all four: `POST /api/goals/runs { kind, ... }`.

---

## 2. The end-to-end journey (3 personas)

### A. New user, Day 0 (the Mission Zero flow, upgraded for 0.2)

```
1. Sign up
   └─> workspace provisioned with default agent roster + default skill pack

2. Wizard (Business Intake, PRD-130) [Goals tab → Plan mode]
   └─> VOYAGER agent: "Tell me about your business"
   └─> BLUEPRINT drafts workspace template:
       - 4-16 agents recommended
       - 8-32 skills recommended
       - 3-5 recipes recommended
       - 1 dashboard layout recommended
   └─> User approves

3. Plan execution (still Goals tab, Plan mode)
   └─> FORGE configures the workspace:
       - creates agents
       - installs skills
       - installs tools
       - seeds recipes
       - assembles dashboard
   └─> emits one "plan" run with N deliverables (one per resource created)

4. First real goal (Goals tab → Mission mode)
   └─> "Draft a one-pager describing our Q2 strategy"
   └─> coordinator plans, agents execute
   └─> deliverable lands: [Deliverables tab → new card at top]

Time from sign-up to first deliverable: target ≤10 minutes.
```

### B. Returning user, any day

```
1. Open workspace
   └─> Goals tab is default
   └─> sees: active runs (top), today's deliverables (mid), due recipes (bottom)

2. Act:
   - Type a new goal → chat or mission (coordinator picks unless user overrides)
   - Approve a scheduled recipe that's pending a human gate
   - Check progress of an in-flight mission

3. Review:
   - Deliverables tab → today's outputs, grade what's good, promote to skill library if exceptional

4. Learn:
   - Knowledge tab → see what the graph accrued this week, what memories were consolidated

5. Manage:
   - Agents tab → swap model, adjust persona, install new skill
   - Settings → invite teammate, adjust budget, manage integrations
```

### C. Power user installing a vertical

```
1. Marketplace → Templates → Shopify
2. Preview: "16 agents, 32 skills, 8 recipes, 2 dashboards — $0 seat"
3. Install → plan run kicks off
4. 60 seconds later: workspace configured, first daily-digest recipe scheduled,
   roster visible in Agents tab
5. Goals tab → "Draft today's product spotlight for bestseller X" → mission runs
6. Deliverable lands; schedule as recurring recipe if useful
```

---

## 3. The deliverable object (the heart of Wave 4)

### Model

```
deliverable
├── id
├── workspace_id
├── run_id         : what run produced this (every deliverable has one)
├── run_task_id    : which sub-task (nullable; chat deliverables have run but no task)
├── agent_id       : who made it
├── title          : agent-set label
├── description    : short summary (agent-generated)
├── storage_uri    : s3://automatos-ai/workspaces/{ws}/deliverables/{id}/<filename>
├── mime_type      : application/pdf | text/markdown | text/html | image/* | ...
├── size_bytes
├── content_hash   : sha256 of content for dedup
├── tags           : jsonb array
├── grade          : 1-5 stars, nullable until graded
├── grade_comment  : nullable
├── graded_by      : user_id, nullable
├── graded_at      : nullable
├── promoted_to_skill : nullable skill_id (if user promoted this as a training example)
└── created_at
```

### API

```
GET    /api/deliverables                  list with filters (run, agent, type, grade, tag, date)
GET    /api/deliverables/{id}             detail
GET    /api/deliverables/{id}/download    presigned S3 URL
POST   /api/deliverables/{id}/grade       {stars, comment, tags[]}
POST   /api/deliverables/{id}/promote     {skill_id?: create new skill from this}
DELETE /api/deliverables/{id}             soft delete (file retained 30d)
GET    /api/deliverables/{id}/versions    history
```

### UI rendering (one component, every file type)

`<DeliverableView />` dispatches by mime type:

| MIME | Renderer | Source |
|---|---|---|
| `text/markdown` | `<MarkdownRenderer />` with GFM + mermaid | exists |
| `text/html` | sandboxed iframe | exists |
| `application/pdf` | pdf.js embed | exists (PRD-131 unified preview) |
| `image/*` | `<img>` with zoom | exists |
| `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` (xlsx) | SheetJS grid | exists |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` (docx) | mammoth.js → rendered HTML | exists |
| `application/json` | prism-highlighted | exists |
| `text/plain` | monospace | exists |

Per rendering feedback memory: **rendering = consumer's CSS job, API returns content.** `DeliverableView` does NOT transform content; it chooses the right render component and hands content through.

---

## 4. Collapse map — what's gluing deliverables today

Today, agent output lands in FOUR wrong places:

| Source | Model/table | Wave 4 action |
|---|---|---|
| Chat message with code block | `artifacts` table (+ chat message) | dual-write to `deliverables`; chat keeps the message; deliverable becomes the canonical handle |
| Agent heartbeat report | `agent_reports.file_path` → S3 markdown | dual-write to `deliverables`; new writes go to `deliverables` only after cut-over |
| Mission task output | written to `workspace_files` (`/mission-outputs/{task_id}.md`) | backfill as `deliverables`; mission task_id is the `run_task_id` |
| Generated image (PRD-63) | `generated_images` metadata | dual-write; mime type is `image/*`; storage URI points to S3 |

**Migration window:** 90 days dual-write. Telemetry shows when reads to legacy paths drop to zero. Then new writes stop on legacy; old data stays for audit.

---

## 5. Grade → skill promotion (the learning loop)

The whole point of grading deliverables is to feed the skill library.

```
1. Agent produces deliverable A.
2. User grades A = 5 stars, tags ["Q2 strategy", "one-pager"].
3. System optionally prompts: "Promote this as a skill training example?"
4. If yes → the deliverable + the task prompt + the run context become a `skill_source` row.
5. The skill's `training_examples` array gains this entry.
6. Next time the skill is invoked, the example is in context.
```

This closes the self-autonomous loop: **good outputs become how future goals get cheaper.**

(This feature spec is the seed for what will be a dedicated PRD post-0.2; 0.2's job is the data model that makes it possible.)

---

## 6. Event stream + live UI

Every run has a `run_events` table:

```
run_event
├── id
├── run_id
├── run_task_id  (nullable)
├── type         : plan_drafted | task_started | task_progress | tool_called |
                   llm_call | deliverable_created | task_completed | task_failed |
                   human_gate_required | run_completed | run_failed
├── payload      : jsonb (type-specific)
├── at           : timestamp
└── agent_id     : nullable (who emitted)
```

**Frontend:**
- `/ws/runs/{id}` WebSocket / SSE pushes events live.
- Goals tab → run detail page → live event list + deliverable carousel + task kanban.
- Event stream is the source of truth; the other views are projections.

**Backend:**
- Coordinator emits events.
- Heartbeat service emits events.
- Agent runtime emits `tool_called`, `llm_call` events.
- Notification service listens and pushes for `human_gate_required`, `run_failed`.

This model replaces the current fragmented activity feed + execution history + mission event stream — all three projections feed from `run_events`.

---

## 7. Human-gate pattern (where autonomy pauses for consent)

Autonomy without gates is a liability. Gates are first-class events:

```
run_event.type = "human_gate_required"
payload = {
  gate_kind: "approve_plan" | "approve_budget" | "approve_tool_call" | "escalation",
  message: "...",
  options: ["approve", "reject", "modify"],
  timeout_action: "reject" | "pause" | "escalate",
  timeout_seconds: 3600
}
```

When the event fires:
- Run status flips to `paused`.
- UI surfaces a prompt in the Goals tab run detail (and as a notification).
- User decision → event emitted → run resumes.

**Default gates 0.2 ships with:**
- Mission plan approval (already in PRD-82A).
- Budget threshold breach (PRD-105; may slip to 0.3 — if so, simple hard-stop at 90% of budget, no approval flow).
- High-trust-cost tool calls (e.g. anything writing to external systems) — per-tool config.

Per memory `mission-zero-flaws.md` P5 — HITL escalation — this is the pattern.

---

## 8. What changes in the UI (summary)

### Today
- "Where did that thing the agent made go?" → 4 answers.
- "What's my workspace doing?" → 3 screens.
- "How do I add a recurring job?" → Playbooks? Recipes? Workflows? Scheduled Tasks? (4 concepts.)

### After 0.2
- "Where did that thing the agent made go?" → **Deliverables tab.**
- "What's my workspace doing?" → **Goals tab.** Active runs on top.
- "How do I add a recurring job?" → **Goals → Recipes.** One concept.

### Before-and-after table

| Question | 0.1 answer | 0.2 answer |
|---|---|---|
| Start a quick task | `/chat` | Goals tab, compose box (mode=chat) |
| Start a multi-step task | `/missions/new` | Goals tab, compose box (mode=mission) |
| Schedule a recurring task | `/playbooks` OR `/workflows` OR `/scheduled-tasks` | Goals tab, mode=recipe |
| Configure whole workspace from a business plan | `/wizard` (PoC) | Goals tab, mode=plan |
| Find last week's outputs | 4 screens | Deliverables tab, filter by date |
| See agent performance | `/activity` + `/analytics` + agent detail | Analytics (admin) or Agents tab ranking |
| Install new capability | `/marketplace` (5 sub-routes) | Marketplace (one tab with 5 kinds) |
| Change agent config | `/agents/[id]` (multiple tabs) | Agents tab → detail modal → Skills/Tools/Config inline |

---

## 9. SDK / API consumer narrative

For headless / SDK use:

```python
from automatos import Workspace

ws = Workspace.connect(api_key="...")

# chat
reply = ws.goals.chat("Summarize this doc", attachments=[...])
print(reply.deliverables[0].download())

# mission
run = ws.goals.missions.create(
    goal="Draft a one-pager on Q2 strategy",
    approve_plan=True  # auto-approve; else returns with gate
)
run.wait(timeout=600)
for d in run.deliverables:
    print(d.title, d.storage_uri)

# recipe
recipe = ws.goals.recipes.create(
    goal="Daily Shopify bestseller digest",
    schedule="0 9 * * *",  # 9am daily
)
```

The SDK surface maps 1:1 to the API surface. No hidden concepts. The SDK is Wave 5 but is specced here so the API in Wave 2 doesn't paint us into a corner.

---

## 10. The grading of success (does the user believe it's autonomous?)

Post-0.2, user-facing tests:

1. **"Do one thing for me"** → Chat → deliverable in ≤60s.
2. **"Do a harder thing for me"** → Mission → deliverable in ≤10min.
3. **"Do this every day"** → Recipe → recurring deliverables without re-intervention.
4. **"Set up my whole business in here"** → Plan → workspace configured in ≤15min.

If all four flows land in the Deliverables tab without the user switching screens, 0.2 succeeded.

---

**Cross-references:**
- [02-NORTH-STAR.md](./02-NORTH-STAR.md) — vision
- [04-API-SURFACE.md](./04-API-SURFACE.md) — target endpoints
- [05-DATA-MODEL.md](./05-DATA-MODEL.md) — run + deliverable tables
- [06-FRONTEND-SURFACE.md](./06-FRONTEND-SURFACE.md) — four-tab shell
- PRD-82A — sequential coordinator (foundation)
- PRD-129 / PRD-133b — deliverables backbone work
