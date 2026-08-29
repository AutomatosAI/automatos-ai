---
name: platform-operations
description: The tool-by-tool operations cookbook — exact JSON for marketplace installs, agent wiring, heartbeats, playbooks, board, scheduling, missions, governance, reports, HARNESS, workspace files, notifications, watches, and questions. LOAD THIS (platform_load_skill platform-operations) before executing any platform operation.
version: "1.0.0"
tags: [platform, operations, cookbook, reference, on-demand]
category: agent-role
---

<!-- GENERATED FILE — DO NOT EDIT IN THIS REPO. Source of truth:
     automatos-skills/team/auto-ops/SKILL.md (v1.0.0). Re-sync: python3 scripts/sync-auto-skill.py
     source-body-sha256[:12]=01a858ef74f1 -->

# Platform Operations Reference

Sections 0–16 below are the tool-by-tool operational reference. Use them when executing the actions described in the charter above. Tool names, parameter formats, and workflows are precise — follow the patterns exactly.

---

## 0. Command Centre — Mental Model & Glossary

The Command Centre is the operational hub of the workspace. It organises work across distinct concepts. Understanding what each concept represents — and what it is NOT — prevents confusion.

### Page IA — Use the Current Names

The user-facing navigation has these pages. Always reference them by these names:

| Current Page | What It Is | Old Names (do NOT use) |
|---|---|---|
| **Command Center** | Single pane of glass over the workforce — Summary · Board · Calendar · Activity · Watchlist · Governance · Questions tabs | ~~Activity~~ |
| **Deliverables** | Every artifact agents produce — reports, code, documents, images, PNGs. Three views: Outputs (gallery), Explorer (file tree + editor + terminal), Activity (execution timeline) | ~~Workspace~~, ~~Workspace Files~~, ~~Outputs~~ |
| **Assignments** | Where work gets handed to the crew — Playbooks (reusable routines), Missions (multi-step objectives), Plan (chat-mode plan-then-launch), Task (quick single action). Surfaces a Recommended carousel of marketplace + workspace items. | (new — no legacy name) |
| **Chat** | Auto + agent conversations | — |
| **Agent Management** | Roster, org chart, configuration, skills — plus the live **Fleet** tab (per-agent working/idle/blocked, queue, 24h cost) | — |
| **Tools & Integrations** | Composio + platform + internal tools | — |
| **Knowledge Base** | Documents, databases, code-graph | ~~Documents~~ (the page label is "Knowledge Base") |
| **Marketplace** | Catalog of agents, playbooks, skills, plugins, templates | — |
| **Analytics** | Performance, costs, insights | — |

When a user says "open Workspace" or "go to Activity", interpret it as **Deliverables** or **Command Center** respectively, and use the current name in your reply.

### Core Concepts

| Concept | What It Is | Time Horizon | What It Shows | What It Is NOT |
|---------|-----------|-------------|---------------|----------------|
| **Activity Feed** | Live and recent execution events across the workspace | Now + recent past | Running tasks, completed runs, failed executions, heartbeat cycles | Not task state (use Board), not future work (use Schedule) |
| **Board** | Kanban view of task workflow state | Persistent | Tasks in columns: Inbox → Assigned → In Progress → Review → Blocked → Done | Not an event log (use Activity), not a calendar (use Schedule) |
| **Schedule** | Upcoming planned work — tasks and playbooks set to run in the future | Future | Scheduled tasks (cron/one-shot), recurring playbook runs, heartbeat timers | Not what's running now (use Activity), not task state (use Board) |
| **Reports** | Structured outputs submitted by agents after completing work | Recent past | Heartbeat reports, research summaries, incident reports, audits | Not raw execution logs (use Activity Feed) |
| **Missions** | Higher-level coordinated initiatives that orchestrate multiple agents and tasks | Spanning | Goal, task DAG, step results, agent assignments, timing | Not single tasks (use Board) |
| **Tasks** | Discrete units of work — created manually or by playbooks, missions, heartbeats | Persistent | Title, description, status, priority, assigned agent, result | The atomic unit; everything else creates or tracks tasks |
| **Playbooks** | Reusable multi-step workflows that agents execute | Persistent (template) | Steps, agent assignments, execution history, schedule config | Not tasks — playbooks *create* tasks when they run |
| **Heartbeats** | Recurring autonomous agent check-ins | Recurring | Last run, next run, findings, reports | Not playbooks — heartbeats are agent-level, playbooks are workflow-level |

### Command Centre Widgets

The Summary tab shows 12 configurable widgets. Each widget has a specific data source:

| Widget | Data Source | Shows |
|--------|-----------|-------|
| **Active Now** | Activity Feed (status=working) | Currently running tasks, playbook executions, heartbeat cycles |
| **Status Overview** | Board Stats | Pie chart of task distribution across board columns |
| **Recent Activity** | Activity Feed (status=done+attention) | Completed and failed executions with duration |
| **Schedule** | Activity Schedule | Week calendar of upcoming scheduled routines and playbooks |
| **Agent Reports** | Agent Reports API | Latest reports from pinned agents |
| **Priority Breakdown** | Board Stats | Bar chart of tasks by priority (Urgent/High/Medium/Low) |
| **Types of Work** | Board Stats | Distribution of task sources (Routine/Task/Project) |
| **Team Workload** | Board Stats | Task count per agent as horizontal bars |
| **Cost Tracker** | KPI: cost-tracker | Total LLM spend, daily trend, top agent spenders |
| **Agent Performance** | KPI: agent-performance | Success rate per agent |
| **Playbook Metrics** | KPI: playbook-metrics | Playbook run counts, success %, avg duration |
| **Approval Gates** | KPI: approval-gates | Pending approvals, avg approval time, waiting missions |

### Tool-to-Concept Mapping

When using platform tools, know which concept you're operating on:

| Action | Tool | Concept |
|--------|------|---------|
| See what's running now | `platform_get_activity_feed` (status=working) | Activity |
| See recent completions | `platform_get_activity_feed` (status=done) | Activity |
| See board task counts | `platform_board_summary` | Board |
| Move a task to a new column | `platform_update_task_status` | Board |
| See upcoming scheduled work | `platform_list_scheduled_tasks` | Schedule |
| Schedule a one-off or recurring task | `platform_schedule_task` | Schedule |
| Schedule a playbook to run on cron | `platform_schedule_playbook` | Schedule |
| Run a playbook right now | `platform_execute_playbook` | Activity (creates execution) |
| Read an agent's latest report | `platform_get_latest_report` | Reports |

---

## 1. Understanding the Three Layers

There are three distinct layers. You MUST keep them separate in your thinking.

### Layer 1: Marketplace (the catalog)
The marketplace is a **catalog of available items** — agent templates, skills, plugins, tools, models. If something is in the marketplace, it is **possible to install**, not active. The marketplace is there to save time. It is NOT a limitation — if the user needs something that isn't in the marketplace, you build it custom.

### Layer 2: Workspace (what's enabled)
The workspace is what's **actually installed and active** for this specific workspace. Before anything can be used, it must be installed, enabled, or connected inside the workspace. If it's not in the workspace, treat it as unavailable to agents in this workspace.

### Layer 3: Agent (what's specifically equipped)
Even if something is installed in the workspace, an agent still may not have it. Tools, skills, and plugins must be **assigned to a specific agent** before that agent can use them.

**Agent capability is not:**
- "exists in marketplace" ← just means it's available to install
- "exists in workspace" ← means it's enabled but agent might not have it

**Agent capability IS:**
- "installed in workspace **AND** assigned to this agent" ← confirmed usable

### What This Means in Practice

A workspace starts **blank**. Nothing is installed. The owner decides what goes in:
- Which LLM models to install (cost control — only pay for what you use)
- Which plugins to enable (Shopify, GitHub, Slack, etc.)
- Which skills to install (agent capabilities from the marketplace)
- Which agents to create (each with a persona, model, tools, skills)
- Which playbooks to automate (recurring workflows)

**Principle:** Keep it clean. Only install what the business actually needs. A Shopify store doesn't need GitHub tools. A dev team doesn't need Shopify plugins.

### WHAT YOU SHOULD NEVER SAY

Never say "the agent has it", "the tool is ready", or "it can use web access" unless you have confirmed:
1. It is installed in the workspace
2. It is assigned to the agent
3. Ideally, it works in a real test

### THE MARKETPLACE IS OPTIONAL

The marketplace exists to save time — browse, find something close, install it. But:
- The user might not want anything from the marketplace
- They might have a very custom or unique requirement
- They might want to build from scratch using the APIs
- They might build it themselves without asking you

**If the marketplace doesn't have what the user needs, offer to build it custom.** Don't say "unavailable" — say "I don't see that in the marketplace, but I can build it for you."

---

### THE INSTALL CHAIN

**For individual items (single plugin, skill, or model) — three steps:**

| Step | Action | Tool | What It Does |
|------|--------|------|-------------|
| 1. **Find** | Browse the marketplace (optional) | `platform_browse_marketplace_*` | Discovers what exists — does NOT make it usable |
| 2. **Install** | Add to workspace | `platform_install_plugin`, `platform_install_skill`, `platform_install_model` | Makes it available workspace-wide |
| 3. **Assign** | Wire to a specific agent | `platform_assign_tool_to_agent`, `platform_assign_skill_to_agent`, `platform_assign_plugin_to_agent` | Agent can now actually use it |

**The safe sequence is always:** browse → install in workspace → assign to agent → verify → test

### CASCADING INSTALLS — Agents & Playbooks

**When a user installs an agent template or playbook from the marketplace, all dependencies are auto-installed.** This is the "I like this car, it comes with wheels" model.

**Installing a marketplace agent automatically:**
- Installs the agent's LLM model to the workspace
- Enables the agent's skills for the workspace
- Assigns all declared tools (Composio apps) to the cloned agent
- Warns about any tools that require an OAuth connection (Gmail, Slack, etc.)

**Installing a marketplace playbook automatically:**
- Clones all recommended agents from the marketplace
- Cascades each agent's dependencies (model, skills, tools)
- Remaps playbook steps to point to the newly cloned agents
- Warns about OAuth connections needed for required tools

**The only manual step is connecting OAuth apps.** If a tool like GMAIL or SLACK is auto-assigned, the user must still connect their account at Settings → Integrations. Always tell the user which connections are needed.

**The install response includes:**
- `cloned_items` — what was cloned (agent, recipe)
- `installed_dependencies` — what was auto-installed (models, skills, tools with status)
- `warnings` — what the user needs to do manually (OAuth connections)

### After Installing — What to Tell the User

After a cascading install, report clearly:
1. What was installed (agents, models, skills, tools)
2. What still needs manual action (OAuth connections)
3. Direct them to Settings → Integrations to connect OAuth apps

Example: "Installed Sales Prospector with model llama-3.3-70b-instruct, skills pattern_recognition and Data Analysis, and tools HUBSPOT, LINKEDIN, GMAIL. ⚠️ HUBSPOT, LINKEDIN, and GMAIL require OAuth connections — connect them at Settings → Integrations."

### Verify After Install

**After any install, always VERIFY:**
```json
{ "tool": "platform_get_agent", "params": { "agent_name": "AGENT_NAME" } }
```
Check the response shows the tool/skill/plugin in the agent's assignments. If it is not there, the assignment failed — do not tell the user it worked.

**NEVER claim an agent has a capability unless you have verified it with `platform_get_agent`.** Browsing the marketplace or knowing a tool exists is NOT the same as it being installed and assigned.

**When a user asks for an agent with specific capabilities:**
1. Check the marketplace for matching agent templates — if found, install (dependencies cascade automatically)
2. If no matching template exists, build custom: create agent, install items, assign tools
3. Either way: verify the agent's final configuration with `platform_get_agent`
4. Report what was installed, what was assigned, and what OAuth connections are needed
5. If you can't find or build what they need, ask them what they want — don't assume limitations

---

## 2. Marketplace — Browse, Evaluate, Install

The marketplace has five catalogs: **agents** (templates), **skills** (capabilities), **plugins** (tool bundles), **LLMs** (models), and **tools** (integrations). Always browse before building custom — reuse beats reinvention.

### 2a. Browse What's Available

**Search for agent templates:**
```json
{ "tool": "platform_browse_marketplace_agents", "params": { "search": "shopify", "limit": 20 } }
```
Returns pre-built agent templates with name, description, default model, and included tools. Use `category` to filter (e.g., "Marketing", "Engineering", "Sales", "Support").

**Search for skills:**
```json
{ "tool": "platform_browse_marketplace_skills", "params": { "search": "seo", "category": "marketing" } }
```
Skills are SKILL.md files that give agents expertise — workflow steps, tool usage patterns, output formats. Each skill lists the tools it needs.

**Search for plugins:**
```json
{ "tool": "platform_browse_marketplace_plugins", "params": { "search": "shopify" } }
```
Plugins bundle multiple tools into one installable package (e.g., Shopify plugin includes order management, inventory, products, customers).

### 2b. Check What's Already Installed

Before installing, check current state:
```json
{ "tool": "platform_list_workspace_plugins" }
{ "tool": "platform_list_workspace_skills" }
{ "tool": "platform_list_workspace_models" }
```
No parameters needed. Returns what's already enabled so you don't duplicate.

### 2c. Install into Workspace

**Install a plugin** (by ID or slug):
```json
{ "tool": "platform_install_plugin", "params": { "plugin_slug": "shopify-tools" } }
```

**Install a skill** (by ID or name):
```json
{ "tool": "platform_install_skill", "params": { "skill_name": "seo-specialist" } }
```

**Install an LLM model** (by OpenRouter model ID):
```json
{ "tool": "platform_install_model", "params": { "model_id": "anthropic/claude-sonnet-4-20250514" } }
```

**Decision framework for model selection:**
- **Free tier** (`meta-llama/llama-4-scout`): Background tasks, drafts, low-stakes work
- **Budget tier** (`deepseek/deepseek-chat`): Routine tasks, high volume, cost-sensitive
- **Mid tier** (`openai/gpt-4.1`, `anthropic/claude-sonnet-4-20250514`): Core agents, quality work, tool use
- **Premium tier** (`anthropic/claude-opus-4-20250514`): Strategy, complex reasoning, critical decisions

Use `platform_list_llms` to see all available models with capabilities and pricing:
```json
{ "tool": "platform_list_llms", "params": { "capability": "tools", "tier": "mid", "sort_by": "cost", "limit": 20 } }
```
Filter by `capability`: "tools" (function calling), "vision" (image input), "reasoning" (extended thinking), "json_mode" (structured output).

---

## 3. Tools & Integrations — Discovery

### 3a. List All Available Tools

```json
{ "tool": "platform_list_tools", "params": { "category": "all" } }
```
Returns platform tools (platform_*), Composio integrations (400+), and internal tools (search_knowledge, http_request, etc.). Filter by `category`: "platform", "composio", "all". Use `search` for fuzzy matching. Use `connected_only: true` to only show tools with active OAuth connections.

### 3b. Check Connected Apps

```json
{ "tool": "platform_list_connected_apps" }
```
Shows which Composio apps have active OAuth connections (Slack, Gmail, GitHub, Jira, etc.). **Only assign Composio tools to agents when the connection exists.** An agent with GMAIL tool but no Gmail OAuth connection will fail every call.

### 3c. List Data Sources

```json
{ "tool": "platform_list_datasources", "params": { "type": "all" } }
```
Shows document collections (RAG knowledge base) and database connections (NL2SQL). Filter by `type`: "documents", "databases", "all".

---

## 4. Agent Management — Full Lifecycle

### 4a. List Existing Agents

```json
{ "tool": "platform_list_agents", "params": { "status_filter": "active" } }
```
Returns all agents with name, type, status, description. Filter: "active", "inactive", "all".

### 4b. Get Agent Details

```json
{ "tool": "platform_get_agent", "params": { "agent_name": "SENTINEL" } }
```
Returns full config: model, tools assigned, skills, persona prompt, heartbeat config, activity stats. Use `agent_id` or `agent_name`.

### 4c. Create a New Agent

```json
{
  "tool": "platform_create_agent",
  "params": {
    "name": "SENTINEL",
    "agent_type": "worker",
    "description": "Infrastructure watchdog — monitors health, errors, costs",
    "model_id": "anthropic/claude-sonnet-4-20250514",
    "system_prompt": "You are SENTINEL, the infrastructure watchdog for this workspace. Your job is to detect problems before users do. You monitor platform health, error spikes, and LLM cost anomalies. You are precise, factual, and never speculate. When you find an issue, you report it with evidence and severity.",
    "temperature": 0.3,
    "tags": ["devops", "monitoring", "engineering"],
    "team": "Engineering & DevOps",
    "job_title": "Engineering Reliability & Security Lead",
    "reports_to_id": null
  }
}
```

**Agent types:** `"chatbot"` (interactive), `"worker"` (background tasks), `"researcher"` (analysis), `"coder"` (development).

**`system_prompt`** is the agent's persona — who they are, how they think, what they do, what they refuse to do. Make it specific and actionable. Never generic.

**`reports_to_id`** sets the org chart. Use the integer ID of the manager agent. Set to `null` for agents that report directly to the human CEO or to Auto.

**`team`** and **`job_title`** set organizational structure visible in the roster.

### 4d. Update an Agent

```json
{
  "tool": "platform_update_agent",
  "params": {
    "agent_name": "SENTINEL",
    "model_id": "deepseek/deepseek-chat",
    "temperature": 0.5,
    "status": "active"
  }
}
```
Any field can be updated independently. Use `agent_id` or `agent_name` to identify.

### 4e. Delete an Agent

```json
{ "tool": "platform_delete_agent", "params": { "agent_name": "OLD_AGENT" } }
```
**Permanent. Cannot be undone.** Only delete agents that are truly no longer needed.

---

## 5. Agent Assignments — Wiring Tools, Skills, Plugins

After creating an agent, you must assign their tools. An agent without tools is powerless.

### 5a. Assign a Composio Tool/App

```json
{ "tool": "platform_assign_tool_to_agent", "params": { "agent_name": "SCOUT", "app_name": "GMAIL" } }
```
`app_name` is the Composio app identifier: `"GMAIL"`, `"GITHUB"`, `"SLACK"`, `"JIRA"`, `"HUBSPOT"`, `"SHOPIFY"`, `"COMPOSIO_SEARCH"`, etc. **Check `platform_list_connected_apps` first** — only assign apps with active OAuth.

**COMPOSIO_SEARCH is special:** It gives the agent web search, news, finance, academic, shopping, and scraping capabilities via actions like `COMPOSIO_SEARCH_WEB`, `COMPOSIO_SEARCH_NEWS`, `COMPOSIO_SEARCH_TAVILY`, etc. Assign this to any agent that needs internet research.

### 5b. Assign a Skill

```json
{ "tool": "platform_assign_skill_to_agent", "params": { "agent_name": "SENTINEL", "skill_name": "sentinel" } }
```
Skills give agents expertise — structured workflows, tool usage patterns, output formats. The skill must be installed in the workspace first (via `platform_install_skill`).

### 5c. Assign a Plugin

```json
{ "tool": "platform_assign_plugin_to_agent", "params": { "agent_name": "SHOPIFY_OPS", "plugin_slug": "shopify-tools" } }
```
Plugins bundle multiple tools. The plugin must be installed in the workspace first.

### 5d. Complete Agent Wiring Example

To fully set up a new agent end-to-end:
1. Install required skill: `platform_install_skill` → `"sentinel"`
2. Install required model: `platform_install_model` → `"anthropic/claude-sonnet-4-20250514"`
3. Create the agent: `platform_create_agent` → name, persona, model, team, job_title
4. Assign skill: `platform_assign_skill_to_agent` → `"sentinel"`
5. Assign tools: `platform_assign_tool_to_agent` → `"COMPOSIO_SEARCH"` (if needed)
6. Configure heartbeat: `platform_configure_agent_heartbeat` → schedule, checks, proactive level
7. Verify: `platform_get_agent` → confirm everything is wired

### 5e. Skill Lifecycle — Read, Fork, Edit, Validate

Use the workspace skill tools to manage skill content directly. Never edit upstream marketplace content from inside a workspace — fork-on-edit is the correct path.

| Operation | Tool |
|---|---|
| Inspect a skill | `platform_get_skill_content` |
| Create a workspace skill | `platform_create_workspace_skill` |
| Edit a skill (forks if marketplace) | `platform_update_skill` |
| Delete a workspace-owned skill | `platform_delete_workspace_skill` |
| Browse marketplace catalogue | `platform_browse_marketplace_skills` |
| Install marketplace skill | `platform_install_skill` |
| Assign to agent | `platform_assign_skill_to_agent` |
| Unassign from agent | `platform_unassign_skill_from_agent` |

When reviewing a skill, check: name, description, current owner, workflow quality, tool list, stale references (retired agents, old role names), overlap with other skills, output format, guardrails, and origin (marketplace vs workspace fork).

---

## 6. Heartbeats — Agent Autonomous Cycles

Heartbeats are scheduled cycles where an agent wakes up, checks its domain, and reports findings. This is how agents work autonomously without being asked.

### 6a. Configure a Heartbeat

```json
{
  "tool": "platform_configure_agent_heartbeat",
  "params": {
    "agent_name": "SENTINEL",
    "enabled": true,
    "interval_minutes": 30,
    "prompt": "Run your full monitoring workflow: check platform health, scan for errors, audit LLM costs, compare against your last report, and submit findings.",
    "auto_act": false,
    "active_hours_start": "06:00",
    "active_hours_end": "22:00",
    "proactive_level": "notify",
    "notification_channel": "in_app",
    "checklist": "- Check platform health\n- Scan error logs (15min window)\n- Audit LLM cost vs 7-day average\n- Compare against previous SENTINEL report\n- Submit heartbeat report"
  }
}
```

**Parameters explained:**
- **`interval_minutes`**: How often the heartbeat fires. Options: 15, 30, 60, 120, 240, 480 (8h), 1440 (daily), 10080 (weekly).
- **`prompt`**: What the agent does each cycle. Be specific — reference their skill workflow.
- **`auto_act`**: If `true`, agent can take corrective action (restart services, reassign tasks). If `false`, agent only observes and reports.
- **`active_hours_start/end`**: HH:MM format. Heartbeats only fire during this window. Use `"00:00"` / `"23:59"` for 24/7.
- **`proactive_level`**: 
  - `"silent"` — run, log, no notification
  - `"notify"` — run, notify on findings
  - `"act_notify"` — run, take action, notify
  - `"autonomous"` — full autonomy within blueprint guardrails
- **`checklist`**: Markdown checklist shown in the heartbeat UI. Helps the human verify what the agent checks each cycle.

### 6b. Heartbeat Design Patterns

| Agent Role | Interval | Proactive Level | Active Hours |
|-----------|----------|-----------------|--------------|
| Infrastructure monitor | 15-30 min | act_notify | 24/7 |
| Business intelligence | 1440 (daily) | notify | Business hours |
| Content/marketing | 1440 (daily) | notify | Business hours |
| Sales/CRM | 480 (8h) | notify | Business hours |
| Knowledge curator | 1440 (daily) | act_notify | Off-hours |
| Cost watchdog | 240 (4h) | act_notify | 24/7 |

---

## 7. Playbooks — Automated Multi-Step Workflows

Playbooks are reusable, multi-step workflows that agents execute in sequence. Think of them as SOPs that run themselves.

### 7a. Create a Playbook

```json
{
  "tool": "platform_create_playbook",
  "params": {
    "name": "Daily CEO Briefing",
    "description": "Compile 24h activity into a concise briefing for the CEO",
    "tags": ["daily", "executive", "reporting"]
  }
}
```
Creates an empty playbook as a draft. You must add steps next.

### 7b. Add Steps to a Playbook

Each step has a `prompt_template` (what to do), an optional `agent_id` (who does it), and error handling.

**Step 1 — Gather metrics:**
```json
{
  "tool": "platform_add_playbook_step",
  "params": {
    "playbook_id": 1,
    "prompt_template": "Pull the last 24 hours of activity: platform_workspace_stats, platform_get_cost_breakdown, platform_board_summary. List key numbers: total chats, tasks completed, tasks blocked, total LLM spend, top 3 active agents.",
    "agent_id": 10,
    "order": 0,
    "error_handling": "retry",
    "output_key": "metrics"
  }
}
```

**Step 2 — Analyze and summarize:**
```json
{
  "tool": "platform_add_playbook_step",
  "params": {
    "playbook_id": 1,
    "prompt_template": "Using the metrics from the previous step, write a CEO briefing: max 150 words. Lead with the most important thing. Include: wins, blockers, spend, and one recommended action. Submit via platform_submit_report with report_type='summary'.",
    "agent_id": 10,
    "order": 1,
    "error_handling": "stop",
    "output_key": "briefing"
  }
}
```

**`error_handling` options:**
- `"stop"` — Halt playbook if this step fails
- `"skip"` — Skip this step and continue to next
- `"retry"` — Retry once, then stop if still failing

**`output_key`** — Names the output so later steps can reference it.

### 7c. Schedule a Playbook

Use `platform_schedule_playbook` to set a cron schedule directly:

```json
{
  "tool": "platform_schedule_playbook",
  "params": {
    "playbook_name": "Daily CEO Briefing",
    "cron_expression": "0 9 * * *",
    "timezone": "UTC",
    "enabled": true
  }
}
```

**Common cron patterns:**
- `0 9 * * *` — daily at 09:00
- `0 9 * * 1` — every Monday at 09:00
- `0 9 * * 1-5` — weekdays at 09:00
- `0 */4 * * *` — every 4 hours
- `0 9 1 * *` — 1st of each month at 09:00

Alternatively, use `platform_update_playbook` with `schedule_config` and `execution_config` for full control:

```json
{
  "tool": "platform_update_playbook",
  "params": {
    "playbook_id": 1,
    "schedule_config": {
      "type": "cron",
      "cron_expression": "0 9 * * *",
      "timezone": "UTC"
    },
    "execution_config": {
      "timeout_minutes": 15,
      "notify_on_complete": true
    }
  }
}
```

### 7d. Execute and Monitor

**Trigger manually:**
```json
{ "tool": "platform_execute_playbook", "params": { "playbook_name": "Daily CEO Briefing" } }
```
Returns an `execution_id` immediately (async).

**Check execution status:**
```json
{ "tool": "platform_get_playbook_execution", "params": { "execution_id": "abc-123" } }
```
Returns step-by-step results, timing, and final status.

### 7e. Manage Steps

**Update a step** (by 0-based index):
```json
{ "tool": "platform_update_playbook_step", "params": { "playbook_id": 1, "step_index": 0, "prompt_template": "Updated prompt...", "error_handling": "skip" } }
```

**Remove a step:**
```json
{ "tool": "platform_delete_playbook_step", "params": { "playbook_id": 1, "step_index": 2 } }
```

### 7f. Common Playbook Patterns

| Playbook | Steps | Schedule |
|----------|-------|----------|
| **Daily CEO Briefing** | Gather stats → Analyze → Submit report | 09:00 UTC daily |
| **Weekly Business Review** | Pull KPIs → Agent rankings → Cost analysis → Generate deck | Monday 09:00 |
| **Content Pipeline** | Research trends → Draft content → Review quality → Schedule publish | Daily 07:00 |
| **Knowledge Audit** | List docs → Check freshness → Reprocess failures → Report gaps | Weekly Sunday |
| **Sales Pipeline Review** | Pull CRM data → Score leads → Draft outreach → Update pipeline | Daily 08:00 |
| **Daily Social Post** | Brief → Build payload → Brand voice QA → Render to 4 sizes → Approval → Post to LinkedIn/X/IG | 09:00 daily (configurable) |

### 7g. Daily Social Post — Reference Pattern

This is the canonical multi-channel social pattern. Use it as a template when designing any daily-social playbook.

**Prerequisites (verify before designing):**
- Skills installed: `social-template-payloads`, `html-to-png`, `social-brand-voice`, `social-production-workflow`, `linkedin-content-creator`, `twitter-engager`, `instagram-curator`
- Composio connections: `LINKEDIN`, `TWITTER`, `INSTAGRAM` (warn the user about any missing OAuth)
- Workspace has the `automatos-social` repo cloned at `repos/automatos-social/` (renderer entry: `repos/automatos-social/render/index.html`, schema: `repos/automatos-social/schema.json`)

**Template & size catalog (defined in `schema.json` — never hardcode, always read):**
- Templates: `title`, `definition`, `stats`, `quote`, `announcement` (5)
- Sizes: `ig_post` (1080×1350), `ig_story` (1080×1920), `linkedin` (1200×628), `twitter` (1600×900) (4)

**Step structure:**

| # | Step | Skill / Tool | Output |
|---|------|--------------|--------|
| 1 | Source brief | content/research agent | `content/social/briefs/{YYYY-MM-DD}.md` |
| 2 | Build payload | `social-template-payloads` | `content/social/payloads/{YYYY-MM-DD}_{template}.json` (fields **must** match `schema.json` exactly) |
| 3 | Brand-voice QA | `social-brand-voice` | Pass-or-revise gate (blocks banned terms / off-brand tone) |
| 4 | Render to 4 sizes | `html-to-png` → `workspace_html_to_png` (×4, parallel) | `deliverables/social/{YYYY-MM-DD}/{template}_{size}.png` |
| 5 | Approval gate (optional) | `social-production-workflow` | Human approval task |
| 6 | Post to channels (parallel) | `linkedin-content-creator` (LINKEDIN_CREATE_POST), `twitter-engager` (TWITTER_CREATE_TWEET), `instagram-curator` (Instagram feed/story) | Posted IDs + URLs |

**Path conventions to bake into the playbook:**
- PNGs: `deliverables/social/{YYYY-MM-DD}/{template}_{size}.png`
- Briefs: `content/social/briefs/{YYYY-MM-DD}.md`
- Payloads: `content/social/payloads/{YYYY-MM-DD}_{template}.json`

**Decisions to confirm with the user before scheduling:**
1. **Cadence** — daily 09:00 default? Days-of-week filter (e.g. weekdays only)?
2. **Approval gate** — auto-publish, or always require human approval before Step 6?
3. **Channels** — all three every day, or rotate (e.g. LinkedIn weekdays, IG Mon/Wed/Fri)?
4. **Brand-voice fallback** — if `social-brand-voice` rejects, regenerate (loop back to Step 2) or skip the day?
5. **Caption ownership** — does `social-template-payloads` produce the caption, or does each per-channel poster write its own platform-tailored caption? (Default: payload owns the visual + a generic caption; per-channel skills lightly adapt for length/hashtags.)

**Rules Auto MUST follow when designing this playbook:**
- `workspace_html_to_png` is the **only** renderer — never invent a substitute.
- Payloads MUST match `schema.json` field names exactly — drift breaks rendering.
- The `social-brand-voice` step is non-optional — it's the gate that prevents off-brand posts.
- Never hardcode workspace IDs in the `file://` URL — the `html-to-png` skill knows how to construct paths against the worker's volume layout.
- Detailed render protocol (URL building, viewport pinning, encoded params) lives in the `html-to-png` SKILL.md — assign that skill to the rendering agent and trust it. Don't duplicate the protocol in playbook step prompts.

### 7h. Playbook Step Shape

Good playbook step shape:
- One clear outcome per step
- One responsible agent per step
- Short prompt — tell the agent what to produce, not how to think
- Clear output name (`output_key`)
- Explicit error behaviour (`stop`, `skip`, `retry`)
- Final step submits a report or writes a deliverable
- Validation where needed (brand voice, schema compliance)

Bad playbook smell: one giant step that says "read everything, analyse everything, create a campaign, update agents, write content, publish, and report back." Split into 5–7 small steps with names like `gather_reports`, `summarise_signals`, `rank_opportunities`, `draft_recommendations`, `submit_review_report`. Less magic, more control.

---

## 8. Board & Task Management

The board is a 6-column Kanban that tracks task workflow state:

| Column | Status | Meaning | Transition Effect |
|--------|--------|---------|-------------------|
| **Inbox** | `inbox` | New task, no agent assigned yet | None — waiting for triage |
| **Assigned** | `assigned` | Agent assigned but not started | None — waiting for dispatch |
| **In Progress** | `in_progress` | Agent is actively executing | **Triggers agent execution immediately** |
| **Review** | `review` | Execution complete, awaiting approval | Human or LLM review required |
| **Blocked** | `blocked` | Cannot proceed — needs intervention | Sets `blocked_at` + `blocked_reason` |
| **Done** | `done` | Completed successfully | Sets `completed_at` |

The board is NOT an event log (that's the Activity Feed) and NOT a schedule (that's the Schedule tab).

### 8a. Create a Task

```json
{
  "tool": "platform_create_task",
  "params": {
    "title": "Audit Q2 marketing spend",
    "description": "Pull all marketing-related LLM costs for Q2, compare against budget, flag any overspend.",
    "priority": "medium",
    "assigned_agent_name": "ATLAS",
    "tags": ["finance", "audit", "q2"],
    "status": "assigned"
  }
}
```

**Priority levels:** `"urgent"` (P0 — 4h SLA), `"high"` (P1 — 12h SLA), `"medium"` (P2 — 24h SLA), `"low"` (P3 — 72h SLA).

**Status options:** `"inbox"` (unassigned), `"assigned"` (ready), `"in_progress"` (executing — **triggers agent immediately**), `"review"` (needs human check), `"blocked"`, `"done"`.

**Parent tasks:** Use `parent_task_id` to create sub-tasks under an objective or project.

### 8b. Board Overview

```json
{ "tool": "platform_board_summary" }
```
Returns counts by status and priority, busiest agents, and recent completions. Use this to understand workload before assigning new tasks.

### 8c. Assign and Move Tasks

```json
{ "tool": "platform_assign_task", "params": { "task_id": 42, "agent_name": "SENTINEL" } }
{ "tool": "platform_update_task_status", "params": { "task_id": 42, "status": "in_progress" } }
```
**Important:** Moving a task to `"in_progress"` triggers the assigned agent to execute immediately. This is how you dispatch work.

**Supervision on ASSIGN (PRD-224):** when you create-and-start a ticket down the ASSIGN lane, a `run_and_report` watch is attached automatically (`AUTO_TICKET_WATCH`, default on) — done/failed/below-bar reports back into the thread where the work was asked for. The `platform_create_task` result carries a `supervision` field: **echo it honestly in your confirmation. Never claim a ticket is supervised unless that field says so** — if the watch failed to attach, say that.

**Agent-side statuses:** you can also set `"blocked"` (requires `blocked_reason`; sets `blocked_at`) and `"failed"` — same validation as the human path. A blocked ticket with an open question belongs in the Questions tab via `platform_ask_human`, not just a bare blocked status.

### 8d. Task Hierarchy

Tasks support parent-child relationships for organizing complex work:
- **Objectives** (epics) — high-level goals, no direct agent
- **Projects** — mid-level, assigned to a lead agent
- **Actions** — concrete tasks an agent executes

```json
{
  "tool": "platform_create_task",
  "params": {
    "title": "Reduce LLM spend by 20%",
    "description": "Objective: find cost optimization opportunities across all agents",
    "priority": "high",
    "status": "inbox"
  }
}
```
Then create child tasks with `parent_task_id` pointing to the objective.

---

## 9. Scheduling — Recurring & Deferred Tasks

### 9a. Schedule a Task

**One-shot (future execution):**
```json
{
  "tool": "platform_schedule_task",
  "params": {
    "task_type": "one_shot",
    "description": "Generate monthly KPI report for March",
    "schedule": "2026-04-01T09:00:00Z",
    "target_agent_name": "ATLAS"
  }
}
```

**Recurring (cron):**
```json
{
  "tool": "platform_schedule_task",
  "params": {
    "task_type": "recurring",
    "description": "Weekly pipeline review — score leads, update CRM, draft outreach",
    "schedule": "0 8 * * 1",
    "target_agent_name": "SCOUT",
    "max_runs": 52
  }
}
```

### 9b. Manage Scheduled Tasks

```json
{ "tool": "platform_list_scheduled_tasks", "params": { "status": "active" } }
{ "tool": "platform_cancel_scheduled_task", "params": { "task_id": 7 } }
```

---

## 10. Missions — Multi-Agent Orchestration

Missions are autonomous, multi-step operations where a coordinator decomposes a goal into tasks, assigns them to agents, and manages execution.

### 10a. Launch a Mission

```json
{
  "tool": "platform_create_mission",
  "params": {
    "goal": "Research our top 5 competitors, analyze their pricing, features, and market positioning. Produce a competitive intelligence report with actionable recommendations.",
    "config": {
      "max_retries": 2,
      "category": "research",
      "output_format": "markdown"
    }
  }
}
```
The coordinator automatically decomposes the goal into tasks, assigns agent roles, and executes sequentially.

### 10b. Control a Mission

Missions default to `awaiting_approval`; the chat renders an Approve & run card, and you also hold the verbs directly:
`platform_approve_mission` · `platform_reject_mission` · `platform_update_mission_plan` (edit before approval) · `platform_replan_mission` · `platform_pause_mission` · `platform_resume_mission` (optionally raises budget) · `platform_cancel_mission`.

Two things happen without you asking (do not duplicate them):
- **A watch is auto-created** on every mission you launch — supervision to a verdict, with corrective action and escalation.
- **The mission narrates its own lifecycle** into the thread that launched it (approved → started → task terminals → run terminal; task-level lines are suppressed above `MISSION_NARRATION_TASK_CAP` = 8 tasks; non-chat launches narrate to the Auto thread). Don't hand-report what narration already says.

### 10c. Monitor Missions

```json
{ "tool": "platform_list_missions", "params": { "state": "running", "limit": 5 } }
{ "tool": "platform_get_mission", "params": { "mission_id": 12 } }
```
Full details include: goal, state, task DAG, step results, timing, and any failures.

---

## 11. Governance — Blueprints & Budget

Blueprints define what agents are allowed to do. They enforce guardrails.

### 11a. Create a Blueprint

```json
{
  "tool": "platform_create_blueprint",
  "params": {
    "name": "Standard Agent",
    "description": "Default governance rules for workspace agents",
    "rules": {
      "max_tokens_per_request": 4000,
      "max_daily_spend_usd": 10.0,
      "allowed_tools": ["search_knowledge", "platform_submit_report", "platform_create_task"],
      "blocked_tools": ["platform_delete_agent", "workspace_exec"],
      "require_human_approval": ["platform_delete_agent", "platform_delete_playbook"],
      "enforce_mode": "strict"
    },
    "is_default": true
  }
}
```

**`enforce_mode`:**
- `"strict"` — Agent is blocked from violating rules
- `"advisory"` — Violations are logged but not blocked

### 11b. Validate an Agent

```json
{ "tool": "platform_validate_agent", "params": { "agent_id": 15 } }
```
Checks if agent config meets blueprint requirements (has required tools, within budget, correct permissions).

### 11c. Check Mission Budget

```json
{ "tool": "platform_check_budget", "params": { "run_id": "mission-uuid-here" } }
```
Returns remaining cost, remaining tokens, and budget status.

---

## 12. Reports — Agent Work Output

Every agent should submit reports after completing work. Reports create an audit trail and feed into dashboards.

### 12a. Submit a Report

```json
{
  "tool": "platform_submit_report",
  "params": {
    "title": "SENTINEL Heartbeat — 2026-04-12 14:00",
    "content": "## Platform Status: HEALTHY\n\n### Services\nAll 18 services responding normally...\n\n### Errors\n3 errors in last 15min (baseline: 5) — below threshold...\n\n### LLM Costs\nToday: $4.12 (7-day avg: $5.58) — 26% below average...",
    "report_type": "standup",
    "status": "ok",
    "summary": "All systems healthy, costs below average, no action needed",
    "metrics": {
      "services_checked": 18,
      "errors_found": 3,
      "cost_today_usd": 4.12,
      "cost_vs_average": -0.26
    }
  }
}
```

**`report_type`:** `"standup"` (routine check), `"research"` (analysis), `"incident"` (problem), `"summary"` (executive brief), `"delivery"` (completed work), `"audit"` (compliance check).

**`status`:** `"ok"`, `"warning"`, `"critical"`, `"info"`.

**Wave 1 — operating-signal fields.** A real report has *asks*, not just narrative. Use these structured fields so Auto can act on the report rather than reading prose:

```json
{
  "recommendations": [
    {"title": "Move PULSE to claude-haiku-4-5", "rationale": "70% cost cut, success_rate unchanged", "impact": "$120/mo saved"}
  ],
  "action_items": [
    {"title": "Patch HARNESS report attribution", "owner": "Platform Engineering", "due": "2026-05-12", "priority": "high"}
  ],
  "linked_task_ids": [42, 87],
  "requires_approval": true
}
```

If `requires_approval=true`, the report lands in the Decisions Needed queue. Recommendations and action_items are first-class — Auto can promote them to board tasks via `platform_create_task` and link the resulting ticket back with `platform_link_report_to_task`.

### 12b. Read Previous Reports

```json
{ "tool": "platform_get_latest_report", "params": { "agent_name": "SENTINEL", "report_type": "standup" } }
```
Use this for baseline comparison — compare current findings against the last report to identify trends.

### 12c. Acknowledge a Report (Wave 3)

```json
{ "tool": "platform_acknowledge_report", "params": { "report_id": "<uuid>" } }
```

Stamps `acknowledged_by/at`. Use after Auto has summarised the report for Gerard and routed any action_items into board tasks — that drops the row from the Decisions Needed queue.

### 12d. Link a Report to a Task (Wave 3)

```json
{ "tool": "platform_link_report_to_task", "params": { "report_id": "<uuid>", "task_id": 42 } }
```

Idempotent — won't duplicate. Keeps the trail intact: finding → ask → ticket.

### 12e. Memory Provenance (Wave 3)

When storing facts via `platform_store_memory`, always set `source_type` honestly:

| `source_type` | When to use |
|---|---|
| `platform_verified` | I queried + confirmed via tools (highest confidence) |
| `claude_reports` | The assistant claimed it, unverified |
| `current_status` | Read from a live source, transient |
| `inference` | Pattern-based guess (lowest confidence) |

```json
{
  "tool": "platform_store_memory",
  "params": {
    "content": "HARNESS audit run #3 found 2 cost regressions",
    "source_type": "platform_verified",
    "confidence": 1.0,
    "evidence_uri": "/harness/baselines/2026-05-08.json"
  }
}
```

Reading memory back: trust `platform_verified` first. Say "I think" or "from what I can see" when the source is `inference`.

---

## 13. Workspace Info & Health

### 13a. Workspace Overview

```json
{ "tool": "platform_get_workspace_info" }
```
Returns workspace name, member count, creation date, config summary.

### 13b. System Health

```json
{ "tool": "platform_get_system_health" }
```
Checks database, Redis, API, RAG pipeline, CPU, memory, disk. Use when something seems broken.

### 13c. Memory Management

**Store a fact for future conversations:**
```json
{ "tool": "platform_store_memory", "params": { "content": "Customer prefers Claude models over GPT for all writing tasks" } }
```

**Search memory:**
```json
{ "tool": "platform_search_memory", "params": { "query": "customer model preferences", "limit": 10 } }
```

**Browse all memories:**
```json
{ "tool": "platform_browse_memories", "params": { "limit": 20 } }
```

---

## 14. HARNESS — Team Optimization Loop

HARNESS is an automated optimization system that evaluates agent performance and suggests configuration improvements (model swaps, temperature adjustments, tool reassignments).

### 14a. Check Status

```json
{ "tool": "platform_harness_status" }
```

Returns one of these granular statuses (decide what to do next based on which one):

| `status` | Meaning | Auto's response |
|---|---|---|
| `disabled` | Workspace explicitly opted out (`orchestrator.harness.disabled = true`) | Don't run the cadence. Mention to Gerard if asked. |
| `dormant_insufficient_agents` | Fewer than 3 active agents (`active_agents`, `min_required_agents`) | Wait. Surface to Gerard if the count has been low for weeks. |
| `dormant_insufficient_data` | Heartbeat history < 7 days (`heartbeat_days_available`, `min_required_days`) | Wait. Confirm heartbeats are firing. |
| `scheduled_not_run_yet` | Eligible but Sunday cron hasn't fired yet | Normal pre-Sunday state. |
| `running` | Tick is in flight | Wait. Re-poll. |
| `failed` | Last tick raised — read `error` | Treat as **platform issue**. Surface to Gerard. |
| `completed` | Produced a baseline | Normal post-run state. Read `iteration_count`, `convergence`, `last_run_at`, `total_delta_magnitude`, and `artifacts`. |

When `status=completed`, also inspect the `artifacts` map. Each entry is either `"ok"` or `"failed: <reason>"`:

```
artifacts: {
  "baseline_latest": "ok",
  "baseline_archive": "ok",
  "trace": "ok",
  "prescriptions": "ok",
  "changelog": "ok",
  "audit_report": "ok"
}
```

If any entry is `"failed: ..."`, treat it as a platform issue (not agent issue) and surface to Gerard. Don't blame the agents for a writer that couldn't reach S3.

### 14b. Trigger Manual Run

```json
{ "tool": "platform_harness_trigger" }
```
Run after major changes (new agents, model swaps, post-incident) to re-evaluate configurations.

### 14c. Review History

```json
{ "tool": "platform_harness_history", "params": { "limit": 5 } }
```
Shows past optimization runs — what was prescribed, what was applied, convergence trend.

### 14d. Read the Audit Report

HARNESS files its audit under Auto (`report_type=audit`). Read the latest:

```json
{ "tool": "platform_get_latest_report", "params": { "agent_name": "Auto", "report_type": "audit" } }
```

The report is markdown with: convergence status, issues detected, applied changes, queued changes, failed prescriptions. Use this as the source-of-truth for the Monday cadence summary — don't re-derive findings from the raw status.

---

## 15. Workspace Files

Agents can read, write, and manage files in the workspace repository.

### 15a. File Operations

```json
{ "tool": "workspace_read_file", "params": { "path": "reports/sentinel/2026-04-12.md" } }
{ "tool": "workspace_write_file", "params": { "path": "playbooks/daily-briefing.md", "content": "# Daily Briefing Playbook\n..." } }
{ "tool": "workspace_list_dir", "params": { "path": "reports/" } }
{ "tool": "workspace_grep", "params": { "pattern": "TODO|FIXME", "path": ".", "include": "*.py", "max_results": 50 } }
```

### 15b. Shell & Git

```json
{ "tool": "workspace_exec", "params": { "command": "pytest tests/", "cwd": "repos/my-app", "timeout": 120 } }
{ "tool": "workspace_git", "params": { "operation": "status", "cwd": "repos/my-app" } }
{ "tool": "workspace_git", "params": { "operation": "commit", "cwd": "repos/my-app", "args": "-m \"fix: resolve login timeout\"" } }
```

**Allowed git operations:** status, diff, add, commit, push, pull, log, branch, checkout, stash, show, blame, fetch.

### 15c. HTML → PNG Renderer

`workspace_html_to_png` runs a headless Chromium inside the worker, screenshots the page, and writes the PNG into the workspace. The output **auto-registers as a deliverable** (`artifact_type=image`), so it appears in the Deliverables Gallery, the Workspace Explorer, and any Mission Outputs view with no extra plumbing — you don't need a separate "register deliverable" step.

```json
{
  "tool": "workspace_html_to_png",
  "params": {
    "url": "file:///workspaces/<id>/repos/automatos-social/render/index.html?template=title&headline=Hello",
    "viewport": { "w": 1080, "h": 1350 },
    "output_path": "deliverables/social/2026-04-29/title_ig_post.png",
    "wait_for": "[data-render-ready='true']",
    "full_page": false
  }
}
```

**Parameters:**
- **`url`** — `file:///workspaces/<id>/...` (must resolve inside the workspace) or `http(s)://` for public pages. URL-encode field VALUES, never the leading `?` or `&`.
- **`viewport`** — `{ "w": <px>, "h": <px> }`, exact pixel dimensions, max 4096 per side.
- **`output_path`** — workspace-relative, must end in `.png`.
- **`wait_for`** — CSS selector to await before screenshot. Default: `[data-render-ready='true']`. Templates that flip this attribute when ready guarantee a stable shot.
- **`full_page`** — `true` to capture the whole scrollable page, `false` (default) for viewport-only.

**Returns:** `{ "success": bool, "file_path": str, "file_size_bytes": int, "w": int, "h": int, "ms": int }`

**Use cases:**
- Social media cards (see Section 7g — assign the `html-to-png` skill to the rendering agent and let the skill handle the protocol).
- Report → image conversion (e.g. capture a chart-heavy HTML page as a PNG to embed in a Slack post).
- Site/landing screenshots for monitoring or audits.

**Don't do this:**
- Invent a different renderer — `workspace_html_to_png` is the only one.
- Hardcode workspace IDs in URLs from a playbook step prompt — let the assigned skill construct the path.
- Use `full_page: true` for fixed-size social cards — pin the viewport exactly to the size's pixels.

---

## 17. Auto Reporting & Escalation (Wave 2 + 3)

The platform now has a single configurable surface for *where* Auto reports
to and *how loud* each event should be. Use it.

### 17a. The auto_reporting Settings Shape

`workspace.settings.auto_reporting` is a JSONB blob:

```json
{
  "enabled": true,
  "primary_channel": "telegram",
  "fallback_channel": "in_app",
  "quiet_hours": {
    "enabled": true,
    "start": "22:00",
    "end": "08:00",
    "timezone": "Europe/Dublin"
  },
  "digest_frequency": "immediate",
  "digest_time": "09:00",
  "routes": {
    "agent_error": "primary",
    "agent_error:urgent": "telegram",
    "task_complete:info": "silent",
    "report_submitted": "fallback"
  }
}
```

Read with `platform_get_auto_reporting_prefs`. Update partials with
`platform_update_auto_reporting_prefs` (requires confirmation — always
restate the change before persisting).

### 17b. Routing Rules (in evaluation order)

1. `routes["{event_type}:{severity}"]` — most specific
2. `routes["{event_type}"]`
3. `routes["{severity}"]` — severity-only fallback
4. Otherwise: workspace `notification_preferences` (existing PRD-128 behaviour)

`primary` and `fallback` resolve to the configured channels. `silent` drops
the event. Quiet hours funnel non-urgent traffic to in_app — `urgent` and
`security` always pass through.

### 17c. Sending Notifications

```json
{
  "tool": "platform_send_notification",
  "params": {
    "event_type": "agent_error",
    "title": "HARNESS audit failed",
    "message": "Run #4 errored: report_submitted handler missing _agent_id",
    "severity": "urgent",
    "status": "error",
    "link_type": "report",
    "link_id": "<run_id>"
  }
}
```

`event_type` is one of the 9 events THIS tool accepts (the dispatcher itself recognises 21+ — asks, watches, and approvals notify through their own machinery, e.g. `question_pending` fires automatically when `platform_ask_human` parks work; you never raise those by hand). `severity` is one of
`info` / `warning` / `urgent` / `security`. The dispatcher returns a
`dispatched_to` list naming every destination that was actually fired.

### 17d. Escalation Levels (L0-L4)

A single ladder Auto uses to decide what flows where:

| Level | Severity | Meaning | Default destination |
|---|---|---|---|
| L0 | info | FYI, no action expected | in_app / digest |
| L1 | task | Needs work, no human decision | board task |
| L2 | approval | Needs Gerard's call | primary channel + board |
| L3 | urgent | Immediate attention | primary channel (bypass quiet hours) |
| L4 | security | Stop and escalate — no jokes | primary + Gerard direct |

Maps onto existing priorities: `critical/urgent` priority → L3 URGENT,
`high` → L2 APPROVAL, `medium` → L1 TASK, `low` → L0 FYI. BudgetStatus
`exceeded/critical` → L3, `warning` → L2.

Set `escalation_level` (0-4) on board_tasks, agent_reports, or
orchestration_runs to surface them in the right queue. Read it to triage
quickly — one query, all three surfaces.

### 17e. The Operating Pattern

When Auto detects something worth surfacing:

1. Classify the level (L0-L4) — explicit `escalation_level` beats
   inference. Use the security flag, requires_approval, status, budget,
   or priority signals in that order.
2. If L0: skip — it's noise, log to memory only if novel.
3. If L1: `platform_create_task` with `priority` mapped from the level.
4. If L2: create the task **and** `platform_send_notification` with
   `severity=approval`.
5. If L3: notification first (`severity=urgent`), then task. Bypass
   quiet hours.
6. If L4: notification + task + memory + Gerard direct chat. No jokes.

Auto-applied actions still file an `audit` report so Gerard can audit
later. Anything Auto did unilaterally goes in `recommendations` /
`action_items` of the report so the trail is intact.

### 17f. Heartbeat Completion (Wave 1)

Every heartbeat result now records `objective_met` (bool) and
`evidence_ref` (file path / report id / task id). When reading
heartbeats:

- `objective_met=True` + `evidence_ref` → real work happened.
- `objective_met=False` → fix the agent, not the report.
- `objective_met=NULL` (silent success, no observable output) → ask
  whether the heartbeat checklist actually has an objective worth
  recording.

This kills the "ran ≠ did the thing" class of nonsense.

---

## 18. Watches — Supervision to a Verdict

A watch follows one launched unit of work to a verdict: it detects the terminal state, scores the output against success criteria (0–1, shown ×10), takes bounded corrective action when the policy allows, and narrates the verdict into chat. Targets: `mission`, `playbook_execution`, `scheduled_playbook`, and — since PRD-224 — `board_task`.

```json
{ "tool": "platform_create_watch", "params": { "target_type": "board_task", "target_id": "42", "policy": "run_and_report", "success_criteria": "invoice list reconciled against Xero, discrepancies itemised" } }
```

- **Policies:** `run_and_report` (score + notify + close) · `score_and_improve` (one diagnose→tweak→rerun cycle below bar) · `watch_change` (compare a rerun to the prior run) · `persistent` (recurring supervision of a schedule).
- **`action_budget`** (default 2) caps corrective attempts before escalation.
- You rarely create ticket watches by hand — the ASSIGN lane auto-attaches one (`AUTO_TICKET_WATCH`). Missions get one at launch. Create manually when the user says "keep an eye on it" about something already running.
- Manage with `platform_list_watches` / `platform_get_watch` / `platform_cancel_watch`. The human surface is the Command Center **Watchlist** tab.

## 19. Questions & Asks — When Work Is Blocked on a Human

A question to the human is **task state, not a message**. The ask lives on the subject it blocks (question-kind approval grant), shows in the Command Center **Questions** tab with the cascade of downstream work stuck behind it, and the answer resumes the work automatically with the Q&A written into its context — permanent decision history.

```json
{ "tool": "platform_ask_human", "params": { "subject_type": "board_task", "subject_id": "42", "question": "**Which Xero account should refunds post to?**\n- `470 — Refunds` (current default)\n- `200 — Sales` (nets against revenue)", "options": ["470 — Refunds", "200 — Sales"] } }
```

- **Park, never wait.** The tool returns immediately; the subject goes `blocked`, the dispatch loop HOLDS it (no re-dispatch, no fail), and you move on to other work.
- **Answers flow from anywhere:** the Questions tab, or Gerard's Telegram (reply-to the delivered question, or `/answer <id> …`). On answer the subject re-queues itself — do not manually restart it.
- **Dismiss ≠ answered:** a dismissed ask leaves the subject blocked (the asker may re-ask); an explicit "use your judgment" answer is how Gerard unblocks without deciding.
- **Executing agents escalate here too:** an agent mid-task can raise a clarification; the platform answers routine ones itself from the work's own context (budget 3 per run) and escalates only the rest into this queue. (`platform_ask_orchestrator` is that execution-side tool — it is NOT on your chat surface and never should be.)
- Urgency: an ask that transitively blocks 3+ downstream tasks bypasses quiet hours automatically.

## 16. End-to-End Workspace Setup Pattern

When setting up a workspace from scratch for a new business:

### Phase 1: Assess Needs
1. Understand the business — industry, size, goals, pain points
2. `platform_list_connected_apps` — what integrations are already connected?
3. `platform_list_tools` with `connected_only: true` — what's actually usable?

### Phase 2: Install Infrastructure
1. `platform_install_model` for each required LLM (start with 2-3, expand later)
2. `platform_install_plugin` for each needed integration (Shopify, GitHub, etc.)
3. `platform_install_skill` for each agent capability needed
4. `platform_list_workspace_*` to verify everything installed correctly

### Phase 3: Build the Team
For each agent:
1. `platform_create_agent` — name, persona, model, team, job_title, reports_to
2. `platform_assign_skill_to_agent` — give them expertise
3. `platform_assign_tool_to_agent` — give them Composio access (only connected apps)
4. `platform_assign_plugin_to_agent` — give them plugin tools
5. `platform_configure_agent_heartbeat` — set up autonomous cycles
6. `platform_get_agent` — verify full wiring

### Phase 4: Set Up Operations
1. `platform_create_playbook` + `platform_add_playbook_step` for each SOP
2. `platform_create_blueprint` for governance guardrails
3. `platform_create_task` for initial board items
4. `platform_schedule_task` for recurring work

### Phase 5: Verify
1. `platform_list_agents` — all agents active, correct models
2. `platform_board_summary` — tasks created and assigned
3. `platform_list_playbooks` — SOPs configured
4. `platform_validate_agent` — each agent passes governance
5. `platform_get_system_health` — platform healthy

---

## Output Format

When reporting on workspace setup or management actions, use this structure:

```markdown
## Workspace Setup Report

### Infrastructure Installed
- Models: [list with tier]
- Plugins: [list]
- Skills: [list]
- Connected Apps: [list]

### Team Roster
| Agent | Role | Model | Skills | Tools | Heartbeat |
|-------|------|-------|--------|-------|-----------|
| ...   | ...  | ...   | ...    | ...   | ...       |

### Playbooks Configured
| Playbook | Steps | Schedule | Owner |
|----------|-------|----------|-------|
| ...      | ...   | ...      | ...   |

### Governance
- Blueprint: [name] — [enforce mode]
- Budget limits: [details]

### Verification
- [ ] All agents active and wired
- [ ] All heartbeats configured
- [ ] Governance blueprint applied
- [ ] Board initialized with tasks
- [ ] Playbooks scheduled
```
