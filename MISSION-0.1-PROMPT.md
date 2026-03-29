# Mission 0.1 — Build the A-Team

## Who We Are

Automatos AI is an AI-agent managed company. The CEO (human) sets direction. Auto (CTO) runs operations. Every other agent is a team member with a job title, department, tools, skills, and deliverables.

The platform is built: 98 tools, mission system, 6-column kanban board, heartbeats, playbooks, reports, analytics, governance blueprints, marketplace with agent templates/skills/plugins, 400+ Composio integrations, and 1000+ LLM models. What we DON'T have is a properly staffed, equipped, operational team.

**Your job: research the marketplace, hire the team, equip every agent, wire up governance and playbooks, and make them operational. No reviews. No assessments. Build it.**

---

## The Approved Roster — 14 Agents

The org chart is decided. You are hiring and configuring these roles:

### Executive & Core Control
| # | Name | Title | Dept | Reports To | Model | Purpose |
|---|------|-------|------|------------|-------|---------|
| 1 | **AUTO** | CTO / Chief of Operations | Executive | CEO (human) | `opus` | Day-to-day ops, agent oversight, mission coordination, final escalation |
| 2 | **ATLAS** | Business Intelligence Lead | Executive / Strategy | Auto | `anthropic/claude-sonnet-4-6` | KPI analysis, weekly reviews, trends, rankings, departmental scorecards |
| 3 | **ORACLE** | Knowledge Operations Lead | Platform Operations | Auto | `openai/gpt-4.1` | KB quality, document freshness, taxonomy, SOP integrity, memory hygiene |

### Growth & Marketing
| # | Name | Title | Dept | Reports To | Model | Purpose |
|---|------|-------|------|------------|-------|---------|
| 4 | **SCOUT** | Growth Marketing Specialist | Growth & Marketing | Auto | `deepseek/deepseek-chat` | SEO, growth experiments, channel performance, acquisition insights |
| 5 | **PULSE** | Lifecycle & Email Marketing Manager | Growth & Marketing | SCOUT | `haiku-4.5` | Email sequences, nurture campaigns, list segmentation, send optimization |

### Sales & Revenue
| # | Name | Title | Dept | Reports To | Model | Purpose |
|---|------|-------|------|------------|-------|---------|
| 6 | **CLOSER** | Revenue Ops & Pipeline Manager | Sales & Revenue | Auto | `openai/gpt-4.1` | CRM hygiene, deal stages, pipeline reporting, revenue workflows |
| 7 | **PROSPECT** | Lead Research & Outreach Specialist | Sales & Revenue | CLOSER | `deepseek/deepseek-chat` | Target lists, lead enrichment, outreach drafts, prospect prioritization |

### Finance & Operations
| # | Name | Title | Dept | Reports To | Model | Purpose |
|---|------|-------|------|------------|-------|---------|
| 8 | **LEDGER** | Finance & Cost Control Manager | Finance & Operations | Auto | `anthropic/claude-sonnet-4-6` | Spend tracking, model costs, budgets, invoice review, anomaly detection |

### Engineering & DevOps
| # | Name | Title | Dept | Reports To | Model | Purpose |
|---|------|-------|------|------------|-------|---------|
| 9 | **SENTINEL** | Engineering Reliability & Security Lead | Engineering & DevOps | Auto | `anthropic/claude-sonnet-4-6` | Monitoring, security posture, CI/CD, incident triage, technical escalation |
| 10 | **PATCH** | QA & Bug Triage Engineer | Engineering & DevOps | SENTINEL | `deepseek/deepseek-chat` | Bug sorting, issue reproduction, fix validation, test discipline |

### Content & Media
| # | Name | Title | Dept | Reports To | Model | Purpose |
|---|------|-------|------|------------|-------|---------|
| 11 | **COMMS** | Content Editor & Brand Voice Lead | Content & Media | Auto | `openai/gpt-4.1` | Editorial quality, newsletters, blog, tone consistency, social content |

### Customer Success
| # | Name | Title | Dept | Reports To | Model | Purpose |
|---|------|-------|------|------------|-------|---------|
| 12 | **GUIDE** | Customer Success & Onboarding Manager | Customer Success | Auto | `openai/gpt-4.1` | Onboarding, adoption, retention, support triage, feedback analysis |

### Research & Intelligence
| # | Name | Title | Dept | Reports To | Model | Purpose |
|---|------|-------|------|------------|-------|---------|
| 13 | **RADAR** | Market Research & Competitive Intel | Research & Intelligence | Auto | `anthropic/claude-sonnet-4-6` | Competitors, market shifts, pricing, positioning, strategic opportunities |

### Platform Operations
| # | Name | Title | Dept | Reports To | Model | Purpose |
|---|------|-------|------|------------|-------|---------|
| 14 | **NEXUS** | Platform Ops & Workflow Orchestrator | Platform Operations | Auto | `deepseek/deepseek-chat` | Board hygiene, task routing, playbook visibility, heartbeat exceptions |

---

## Phase 1: Research the Marketplace

Before equipping anyone, audit what's available. This is NOT optional — you must know the full catalog.

### 1A. Marketplace Deep Dive
Run ALL of these. Browse every category. Read every result:
- `platform_browse_marketplace_agents` — search each category: "Marketing", "Sales", "Engineering", "DevOps", "Finance", "Content", "Support", "Research", "Operations", "Design", "Product", "Quality"
- `platform_browse_marketplace_skills` — browse ALL skill categories. Note name, description, token cost, skill_type for each. These are what give agents their expertise.
- `platform_browse_marketplace_plugins` — browse ALL plugin categories. Note what capabilities each plugin bundles.
- `platform_list_tools` with `category="composio"` — the full integration catalog. This determines what each agent can connect to.
- `platform_list_llms` — all available models with capability flags (tools, vision, reasoning, json_mode) and cost tiers (free, budget, mid, premium)

### 1B. Current State Audit
- `platform_list_agents` — every existing agent
- `platform_get_agent` for each one — full config, tools, skills, activity
- `platform_list_connected_apps` — which OAuth connections are live (ONLY assign tools where connections exist)
- `platform_workspace_stats` — resource usage
- `platform_get_cost_breakdown` — spending by model and agent

### 1C. Read Operating Docs
- Read `PLATFORM-CAPABILITIES-DEFINITIVE.md` in the workspace root — all 98 tools documented
- Read `MISSION-ZERO-RESULTS.md` — the operating model blueprint (authority levels, comms matrix, review cadences)

**Output from Phase 1:** A complete catalog of available skills, plugins, tools, and models. A current-state map of what exists. Decisions on which existing agents to keep, reconfigure, or retire.

---

## Phase 2: Build Per-Agent Spec Sheets

For EACH of the 14 agents, produce a complete implementation spec:

1. **Persona / System Prompt** — Write a full persona. Who is this agent? What's their expertise? Communication style? What do they prioritize? What do they refuse to do? Make it specific and actionable, not generic fluff.

2. **Skills** — Pick from marketplace skills discovered in Phase 1. Match skills to role. Install with `platform_install_skill`, assign with `platform_assign_skill_to_agent`.

3. **Tools (Composio Apps)** — Pick from connected apps ONLY. Match to role:
   - COMMS needs: SLACK, GMAIL, maybe WORDPRESS
   - SENTINEL needs: GITHUB, JIRA, SLACK
   - CLOSER needs: HUBSPOT or SALESFORCE, GMAIL, SLACK
   - SCOUT needs: GOOGLE_ANALYTICS, GOOGLE_SHEETS, SLACK
   - LEDGER needs: STRIPE, GOOGLE_SHEETS, SLACK
   - etc. — you decide based on what's actually connected

4. **Plugins** — Pick from marketplace plugins. Install with `platform_install_plugin`, assign with `platform_assign_plugin_to_agent`.

5. **Tags** — Capability tags for task routing and matching. Be specific: `["growth", "seo", "analytics", "acquisition"]` not just `["marketing"]`.

6. **Heartbeat Config:**
   - **AUTO**: 15 min, proactive level 90, active hours 07:00-22:00
   - **Department leads** (ATLAS, SENTINEL, COMMS, CLOSER, LEDGER, GUIDE, RADAR): 60 min, proactive level 60, active hours 08:00-20:00
   - **Specialists** (SCOUT, PULSE, PROSPECT, PATCH, NEXUS, ORACLE): 120 min, proactive level 40, active hours 09:00-18:00
   - Each heartbeat prompt should describe what the agent checks, analyzes, and reports on every cycle

7. **Blueprint Class:**
   - `default` — standard agents (min 2 tools, require system prompt, require tags)
   - `strict` — LEDGER, SENTINEL (min 3 tools, restricted models, require system prompt, required_tags must include department)
   - `cost-tier` — PULSE, PATCH, NEXUS, PROSPECT (model restricted to haiku/deepseek only)

---

## Phase 3: Execute the Build

Execute in this order. Each step uses specific platform tools.

### Step 1: Clean House
- Review existing agents from Phase 1B audit
- Keep: SENTINEL, COMMS, ATLAS, ORACLE (if they exist and are healthy)
- Reconfigure: update model, persona, team, job_title, reports_to_id, tags
- Retire: delete or deactivate dead-weight agents (0 tasks, no heartbeat, no purpose)
- Tools: `platform_update_agent`, `platform_delete_agent`

### Step 2: Create New Agents
For each role that doesn't exist yet, create it:
- `platform_create_agent` with: name, agent_type, description, model, system_prompt, tags, team, job_title
- Set `reports_to_id` to the lead agent's ID
- Tools: `platform_create_agent`

### Step 3: Install & Assign Skills
- `platform_install_skill` for each required skill (workspace-level)
- `platform_assign_skill_to_agent` for each agent-skill pairing
- Verify assignment stuck with `platform_get_agent`

### Step 4: Assign Tools
- `platform_assign_tool_to_agent` for each Composio app per agent
- ONLY assign apps that showed up in `platform_list_connected_apps`
- If an app isn't connected, note it as "awaiting OAuth" and skip

### Step 5: Install & Assign Plugins
- `platform_install_plugin` for each required plugin (workspace-level)
- `platform_assign_plugin_to_agent` for each agent-plugin pairing

### Step 6: Configure Heartbeats
- `platform_configure_agent_heartbeat` for ALL 14 agents
- Use the intervals and proactive levels from the spec sheets
- Write a meaningful heartbeat prompt for each — not "check things" but specific checks, tools to call, and what to report on

### Step 7: Create Governance Blueprints
- `platform_create_blueprint` — create 3 blueprints:
  1. **"Standard Agent"** — `enforce_mode: "advisory"`, rules: `min_tools: 2, require_system_prompt: true`
  2. **"Sensitive Role"** — `enforce_mode: "strict"`, rules: `min_tools: 3, require_system_prompt: true, required_tags: ["department-tag"], allowed_models: ["anthropic/claude-sonnet-4-6", "openai/gpt-4.1"]`
  3. **"Cost-Controlled"** — `enforce_mode: "strict"`, rules: `require_system_prompt: true, allowed_models: ["anthropic/claude-haiku-4-5-20251001", "deepseek/deepseek-chat"]`
- Set "Standard Agent" as workspace default

### Step 8: Deploy Recurring Playbooks
Create with `platform_create_playbook` + `platform_add_playbook_step` + schedule config:

| Playbook | Cron | Owner | Steps |
|----------|------|-------|-------|
| **Daily CEO Briefing** | `0 8 * * *` | ATLAS | 1. Query 24h activity + costs 2. Summarize blockers + wins 3. Submit report 4. Post to Slack |
| **Weekly Business Review** | `0 9 * * 1` | ATLAS | 1. Pull 7-day KPIs 2. Agent rankings 3. Cost analysis 4. Create board task for CEO review |
| **Monthly KB Audit** | `0 10 1 * *` | ORACLE | 1. Scan docs for staleness 2. Check retrieval quality 3. Create cleanup tasks on board |
| **Daily Content Pipeline** | `0 10 * * 1-5` | COMMS | 1. Check content calendar 2. Draft social posts 3. Queue for review |
| **Weekly Cost Review** | `0 14 * * 5` | LEDGER | 1. Pull cost breakdown 2. Compare to baselines 3. Flag anomalies 4. Submit report |
| **Daily Engineering Sweep** | `0 7 * * *` | SENTINEL | 1. Check alerts + logs 2. Review open incidents 3. Triage new issues 4. Submit report |
| **Weekly Pipeline Review** | `0 10 * * 2` | CLOSER | 1. Pull CRM metrics 2. Stage analysis 3. Forecast update 4. Create board task |

### Step 9: Publish Operating Documents
Use `workspace_write_file` for each:
- `docs/communication-matrix.md` — Channel routing rules (Slack for routine, Telegram for urgent, Email for formal, Board for tracking)
- `docs/authority-boundaries.md` — L1 (all agents: read, create tasks, submit reports), L2 (Auto: assign tasks, adjust agents, trigger playbooks), L3 (human: budget, hiring, strategic changes)
- `docs/report-template.md` — Required sections: Summary, Metrics, Actions Taken, Blockers, Next Steps
- `docs/metric-baselines.md` — Snapshot current KPI values from analytics

### Step 10: Store Baselines & Configure Auto
- `platform_store_memory` — store current metric baselines (cost/week, requests/week, tokens/week, agent count, SLA %)
- `platform_configure_agent_heartbeat` for Auto — 15 min, proactive 90, active 07:00-22:00
- Auto's heartbeat prompt should: check board health, review costs, detect anomalies, assign unassigned high-priority tasks, submit CTO standup report

### Step 11: Validate Everything
For each agent:
- `platform_get_agent` — verify config is complete
- `platform_validate_agent` — verify blueprint passes (green badge)
- Confirm heartbeat is configured
- Confirm at least 1 skill and 2 tools assigned

Report any failures as a punch list with specific fixes needed.

---

## Hard Rules

1. **NO reviews or assessments as deliverables.** Every task produces a platform tool call, not a document about what should be done.
2. **Only assign Composio tools with active OAuth connections.** Check `platform_list_connected_apps` first. No exceptions.
3. **Budget-conscious model selection is non-negotiable.** haiku/deepseek for routine, sonnet for reasoning, opus for Auto only.
4. **Every playbook must create board tasks.** Work that doesn't hit the board doesn't exist.
5. **Every agent must submit reports via `platform_submit_report`.** Reports use `required_sections: ["Summary", "Metrics", "Actions Taken", "Next Steps"]`.
6. **Retire dead agents.** Any existing agent with 0 tasks, no heartbeat, and no clear role gets deactivated. No charity hires.

---

## Success Criteria

- [ ] 14 agents live with distinct roles across 8 departments
- [ ] Every agent has: persona, skills, tools (connected only), heartbeat, tags
- [ ] Org hierarchy set: `team`, `job_title`, `reports_to_id` populated for all
- [ ] 3 governance blueprints created and default set
- [ ] 7+ recurring playbooks deployed with cron schedules
- [ ] 4 operating docs published to workspace
- [ ] Metric baselines stored in memory
- [ ] Auto's 15-min heartbeat configured and running
- [ ] All agents pass `platform_validate_agent` (green readiness badge)
- [ ] Dead-weight agents from old roster deactivated

**This is a build mission. Ship it.**
