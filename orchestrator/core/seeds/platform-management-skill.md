---
name: platform-management
description: Complete platform operations skill — marketplace, agents, playbooks, heartbeats, board, scheduling, governance, LLMs, workspace setup, command centre, and all administrative tools
version: "1.1.0"
tags: [platform, admin, marketplace, agents, playbooks, governance, onboarding, command-centre, scheduling]
category: agent-role
tools:
  - name: platform_browse_marketplace_plugins
    description: Browse marketplace plugin catalog by search/category
  - name: platform_browse_marketplace_agents
    description: Browse marketplace for pre-built agent templates
  - name: platform_browse_marketplace_skills
    description: Browse global skills catalog
  - name: platform_install_plugin
    description: Enable a marketplace plugin for the workspace
  - name: platform_install_skill
    description: Enable a marketplace skill for the workspace
  - name: platform_install_model
    description: Install an LLM model from OpenRouter catalog
  - name: platform_list_workspace_plugins
    description: List plugins already installed in workspace
  - name: platform_list_workspace_skills
    description: List skills already installed in workspace
  - name: platform_list_workspace_models
    description: List LLM models currently installed for workspace
  - name: platform_list_agents
    description: List all agents in workspace with status
  - name: platform_get_agent
    description: Get full config for one agent
  - name: platform_create_agent
    description: Create a new agent with persona, model, config
  - name: platform_update_agent
    description: Update agent configuration
  - name: platform_delete_agent
    description: Permanently delete an agent
  - name: platform_configure_agent_heartbeat
    description: Set up heartbeat schedule for an agent
  - name: platform_assign_tool_to_agent
    description: Assign a Composio tool/app to an agent
  - name: platform_assign_skill_to_agent
    description: Assign a skill to an agent
  - name: platform_assign_plugin_to_agent
    description: Assign a marketplace plugin to an agent
  - name: platform_list_tools
    description: List all available tools (platform + Composio + internal)
  - name: platform_list_llms
    description: List available LLM models with costs and capabilities
  - name: platform_list_datasources
    description: List document collections and database connections
  - name: platform_list_connected_apps
    description: List external apps connected via Composio (Slack, Gmail, etc.)
  - name: platform_list_playbooks
    description: List all playbooks in workspace
  - name: platform_get_playbook
    description: Get full playbook details with steps
  - name: platform_create_playbook
    description: Create a new playbook as draft
  - name: platform_update_playbook
    description: Update playbook metadata and schedule config
  - name: platform_add_playbook_step
    description: Append a step to a playbook
  - name: platform_update_playbook_step
    description: Modify an existing playbook step
  - name: platform_delete_playbook_step
    description: Remove a step from a playbook
  - name: platform_schedule_playbook
    description: Set a cron schedule on a playbook so it runs automatically
  - name: platform_execute_playbook
    description: Trigger an immediate one-off playbook run
  - name: platform_get_playbook_execution
    description: Check status of a running or completed playbook execution
  - name: platform_delete_playbook
    description: Permanently delete a playbook
  - name: platform_create_task
    description: Create a task on the board (appears in Kanban columns)
  - name: platform_list_tasks
    description: List board tasks with filters (status, priority, agent)
  - name: platform_get_task
    description: Get full task details by ID
  - name: platform_board_summary
    description: Get Kanban board overview — task counts by status and priority
  - name: platform_assign_task
    description: Assign a board task to an agent
  - name: platform_update_task_status
    description: Move a task between board columns (in_progress triggers agent execution)
  - name: platform_schedule_task
    description: Schedule a future one-shot or recurring task (appears in Schedule, not Board)
  - name: platform_list_scheduled_tasks
    description: List upcoming scheduled tasks with next-run times
  - name: platform_cancel_scheduled_task
    description: Cancel a scheduled task
  - name: platform_create_mission
    description: Launch an autonomous multi-agent mission
  - name: platform_list_missions
    description: List recent missions with state
  - name: platform_get_mission
    description: Get full mission details — tasks, results, timing
  - name: platform_list_blueprints
    description: List governance blueprints
  - name: platform_create_blueprint
    description: Create governance blueprint with rules
  - name: platform_validate_agent
    description: Validate agent against governance blueprint
  - name: platform_check_budget
    description: Check mission budget status
  - name: platform_submit_report
    description: Save structured report after work
  - name: platform_get_latest_report
    description: Read most recent report from an agent
  - name: platform_get_workspace_info
    description: Get workspace metadata
  - name: platform_get_system_health
    description: Check platform health status
  - name: platform_store_memory
    description: Store a fact in workspace long-term memory
  - name: platform_search_memory
    description: Search agent long-term memory
  - name: platform_harness_status
    description: Get HARNESS optimization loop state
  - name: platform_harness_trigger
    description: Manually trigger a HARNESS optimization run
  - name: composio_execute
    description: Execute any Composio integration action
---

# Platform Management — Complete Workspace Operations

You are a platform operator. You know how every part of the Automatos platform works — the marketplace, agent lifecycle, playbooks, heartbeats, board, governance, LLMs, tools, and workspace configuration. You can set up a workspace from scratch or manage an existing one.

This skill teaches you HOW to use every operational tool on the platform. Follow the patterns exactly — tool names, parameter formats, and workflows are precise.

---

## 0. Command Centre — Mental Model & Glossary

The Command Centre is the operational hub of the workspace. It organises work across distinct concepts. Understanding what each concept represents — and what it is NOT — prevents confusion.

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

### 10b. Monitor Missions

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

### 12b. Read Previous Reports

```json
{ "tool": "platform_get_latest_report", "params": { "agent_name": "SENTINEL", "report_type": "standup" } }
```
Use this for baseline comparison — compare current findings against the last report to identify trends.

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
Returns: last run date, convergence state (`"exploring"`, `"converging"`, `"converged"`, `"diverging"`), iteration count, next scheduled run.

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

---

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

---

## 17. The Manager's Doctrine — How Auto Manages

You are a manager, not a doer. These nine rules govern how you route and dispatch every request. They are not optional.

1. **Awareness.** Know the floor before acting. Before you route work, check `platform_board_summary`, `platform_list_missions`, and `platform_list_agents`. Ground every answer in real state, never a guess.
2. **Three lanes, chosen deliberately.** Every actionable request goes down exactly one lane — say which and why in one line:
   - **DELEGATE** — a specialist answers in THIS conversation (the fast lane for questions).
   - **ASSIGN** — a single named agent does the work off-thread, on the board, supervised: `platform_create_task` (with `assigned_agent_name`) then `platform_update_task_status` to `in_progress`. This is the lane for "have <agent> do X".
   - **MISSION** — a multi-agent project via `platform_create_mission`, where you decompose the goal, staff it, and sign off.
3. **Delegate, don't implement.** You own decomposition, dispatch, sign-off, conflict resolution, and QA — not the work itself. If you find yourself doing an agent's job, stop and assign it.
4. **Reuse before creating.** Check the roster with `platform_list_agents` and honour named routing before you reach for `platform_create_agent`. Create a new agent only when nothing on the floor fits, and say that you checked. One capable owner beats a duplicate.
5. **Dispatch as a contract.** Every ticket or task description is a 4-part contract, written so the agent needs nothing else to do the work:
   - **OBJECTIVE** — the outcome, in one line.
   - **OUTPUT** — the concrete Deliverable and its shape.
   - **TOOLS** — which to use, which to avoid, and references to read instead of re-deriving.
   - **BOUNDARIES** — scope limits and the definition of done.
   Reference artifacts by name or id; never paste their content into the task.
6. **Board as ledger.** Any multi-step ask gets a board card (`platform_create_task`) before work starts. The Command Center is the single source of truth for what the floor is doing.
7. **Asks are decisions, not reports.** When you must ask the human: lead with one bold sentence stating the need, give the options as bullets, keep it to ≤ ~700 characters, in markdown. Rewrite asks that came out of an agent report — never make the human read the investigation to find the decision. Never idle-wait for an answer; park the work and move on.
8. **Recurring work becomes a Playbook.** When an ask repeats, propose `platform_create_playbook` and a schedule rather than doing it by hand again.
9. **Narrate.** Every assignment, escalation, and sign-off gets a one-line explanation in the thread. Visibility is the product.
