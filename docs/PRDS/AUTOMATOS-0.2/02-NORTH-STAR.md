# AUTOMATOS 0.2 — North Star

**Purpose:** One coherent story of what Automatos becomes after 0.2. Everything else in this plan serves this.

---

## 1. The one-sentence description

> **Automatos is a self-autonomous workspace OS: give it a goal, and a coordinated crew of AI agents plans, executes, verifies, and delivers — accruing persistent knowledge and skills so the next goal costs less than the last.**

Everything in 0.2 is reconciliation toward making that sentence true in the UI, the API, the data model, and the code layout.

---

## 2. The four-word brand

**Goal → Crew → Deliverable → Knowledge.** If a screen, a route, or a table doesn't map cleanly to one of those four concepts, it's either infrastructure (auth, billing, admin) or it's wrong.

That's the filter for 0.2: **does this concept belong to Goal, Crew, Deliverable, or Knowledge? If not, can it collapse into one?**

---

## 3. The four-tab shell

The workspace UI reduces to four tabs. Everything else — Activity, Missions, Board, Playbooks, Recipes, Heartbeats, Insights, Reports — folds into these four or retreats to "Advanced".

### Tab 1 — **Goals**

One input, four execution modes:

| Mode | Answers | Today's surface |
|---|---|---|
| **Chat** | "Do a quick thing for me" | `/chat` |
| **Mission** | "Deliver a multi-step outcome" | `/missions`, Mission Zero |
| **Recipe** | "Run this recurring workflow" | `/playbooks`, `/workflows`, `/recipes` |
| **Plan** | "Here's a business plan — configure the workspace" | Business Intake Wizard (PRD-130) |

In 0.2, the Goals tab is a single compose box. The user picks execution mode (or Coordinator picks for them). Everything downstream is a *run* of a *goal* — one unified object. [See DELIVERABLES-FLOW.md](./07-DELIVERABLES-FLOW.md).

### Tab 2 — **Deliverables**

Every artifact an agent produces, graded, linked back to the goal that spawned it. Preview inline (md, pdf, xlsx, docx, html, images — already unified in workspace previews as of 2026-04-22). Tag, search, promote to skill library.

This is the single answer to "where did that thing the agent made go?" Today that question has four wrong answers (reports, artifacts, chat messages, mission outputs). Wave 4 collapses them. [See DELIVERABLES-FLOW.md](./07-DELIVERABLES-FLOW.md).

### Tab 3 — **Knowledge**

The workspace's accrued intelligence:

- **Documents** — uploaded, generated, synced (Drive, Dropbox, S3)
- **Memory** — short-term, field, episodic, semantic
- **Graph** — knowledge graph + code graph nodes
- **Skills** — installed skill packs (sentinel, scout, shopify, ...)

One tab, four filterable views. The graph is the spine; everything else is a projection.

### Tab 4 — **Agents**

The crew:

- **Roster** — active agents with heartbeat state
- **Skills** — what each agent can do
- **Tools** — what each agent has connected (Composio, MCP, platform tools)
- **Reports** — auto-generated standups, performance over time

Configuration and coordination move to an expandable "Advanced" sub-tab — most users don't need to see it once defaults work.

---

## 4. The execution model (crew + coordinator)

AUTOMATOS 0.2 locks in the **Sequential Mission Coordinator** (PRD-82A, already shipped) as the canonical execution kernel for anything more than a chat reply. Chat remains chat; anything multi-step routes through the coordinator.

```
Goal  ─┬─→ chat    ─→ single-agent reply ─→ deliverable (light)
       ├─→ mission ─→ coordinator(sequential) ─→ N × deliverable
       ├─→ recipe  ─→ scheduled mission       ─→ N × deliverable (periodic)
       └─→ plan    ─→ mission-of-missions     ─→ entire workspace configured
```

**What changes in 0.2:** the four paths look like four surfaces today. They converge into one `run` object with four `kind`s, one coordinator entry point, one deliverable emitter. [See DELIVERABLES-FLOW.md section 3](./07-DELIVERABLES-FLOW.md).

### Parallel coordinator (PRD-82B) remains on the roadmap

Sequential is the default because it's reliable, debuggable, and budget-predictable. Parallel coordination (PRD-82B) is additive — same `run` object, same deliverable flow, different scheduler under the hood. It does not require another surface.

---

## 5. Skills & marketplace = the only extension point

**Kernel (platform code):**
- FastAPI + Postgres + Redis + S3 + S3 Vectors
- Coordinator + agent factory + tool router
- Auth, workspaces, billing

**Capability (skills + templates):**
- Every new vertical (Shopify, HubSpot, Salesforce, law firm, finance team) ships as a **workspace template** — bundle of agents + skills + tools + playbooks + dashboards (see `prd120-skills-marketplace.md` in memory).
- No new domain-specific code lands in `orchestrator/modules/` for vertical work in 0.2.
- The only reason to add a module is a new *primitive* (e.g. a new memory store, a new coordination primitive). Those are rare.

This is the cleanest architectural principle the platform has. AUTOMATOS 0.2 enforces it structurally by making `orchestrator/modules/` a short, stable list and pushing all new work into skills. [See DOMAIN-MODEL.md](./03-DOMAIN-MODEL.md).

---

## 6. End-to-end user journey (post-0.2)

**New user, Day 0:**
1. Sign up → workspace provisioned with default agent roster + default skill pack.
2. Mission Zero wizard (Business Intake — PRD-130) asks what the business does; Coordinator designs workspace template (which agents, which skills, which tools).
3. User approves → FORGE configures it → BLUEPRINT drafts first mission → user runs it.
4. First deliverable lands in Deliverables tab within minutes.

**Returning user, any day:**
1. Open workspace → Goals tab.
2. Type goal OR pick recurring recipe OR check scheduled mission status.
3. Coordinator runs; deliverables accrue; knowledge graph learns; next goal is cheaper.

**Power user:**
1. Marketplace → install vertical template (e.g. Shopify).
2. 16 agents + 32 skills appear in Agents tab; 8 recipes appear in Goals tab; dashboards appear in Knowledge tab.
3. Starts running missions native to their vertical on day 1.

This journey is what 0.2 makes achievable. Today it's about 60% achievable — Wave 4 (deliverables) and Wave 5 (autonomous flow) close the rest.

---

## 7. What Automatos is NOT (boundary marks)

- **Not an LLM.** We route to OpenRouter / Anthropic / OpenAI / DeepSeek. Model selection is a config, not a moat.
- **Not a chat app.** Chat is one of four goal modes, not the primary surface.
- **Not a workflow builder.** Recipes exist; no-code DAG editor is out of scope.
- **Not a vector DB.** We use S3 Vectors; lineage, not storage, is ours.
- **Not a Zapier.** Tools are how agents act; we don't expose "connect X to Y" as a user primitive.
- **Not an agent marketplace alone.** The agent catalog matters, but the template (workspace bundle) is the GTM unit.

These boundary marks kill duplicate code paths. If a module exists to make Automatos look like one of the above, it's dead in 0.2.

---

## 8. Success from a user's point of view

Post-0.2, a user should be able to answer these in one place each:

| Question | Answer lives in |
|---|---|
| "What's the workspace working on right now?" | Goals tab → active runs |
| "What has it made for me this week?" | Deliverables tab → filter by date |
| "What does it know about my business?" | Knowledge tab → graph + docs + skills |
| "Who's on my crew and what are they doing?" | Agents tab → roster + heartbeat |
| "How do I add a new vertical?" | Marketplace → install template |
| "How do I fix something that went wrong?" | Advanced → Activity Feed |

If any of those answers today requires visiting 2+ screens or using 2+ concepts, 0.2 fixes it.

---

## 9. One-line per domain ownership ("who owns what" after 0.2)

| Domain | Who owns it | Lives in |
|---|---|---|
| Goal definition & routing | Coordinator | `core/coordinator/` |
| Agent runtime | AgentFactory | `core/agent_factory.py` |
| Tool dispatch | ToolRouter | `core/routing/tool_router.py` |
| Skill injection | SkillContextSection | `modules/context/sections/skills.py` |
| Memory | Memory subsystems | `modules/memory/` |
| Knowledge graph | GraphStorage | `core/graph_storage.py` |
| Deliverables | DeliverableService | `services/deliverable_service.py` |
| Workspaces & multi-tenant | WorkspaceService | `core/workspaces/` |
| Billing, budget, governance | BudgetService | `core/budget/` (new) |
| Marketplace & templates | MarketplaceService | `core/marketplace/` (consolidated) |

Ten domains. Ten owners. Ten places a feature lives. That's the whole map. [Detailed in DOMAIN-MODEL.md](./03-DOMAIN-MODEL.md).

---

**The test for 0.2 done:** a new contributor, shown only the four-tab shell and this document, can find any feature in the code in under 60 seconds. Today they can't.
