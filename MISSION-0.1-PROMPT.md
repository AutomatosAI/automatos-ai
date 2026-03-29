# Mission 0.1 — Build the A-Team

## Who We Are

Automatos AI is an AI-agent managed company. We build and operate a platform where AI agents work as employees — each with a role, tools, skills, and accountability. The CEO (human) sets direction. Auto (CTO agent) manages day-to-day operations. Every other agent is a team member with a job title, department, tools, and deliverables.

We have the platform. We have 98 tools, a mission system, a board, heartbeats, playbooks, reports, analytics, governance blueprints, a marketplace with agent templates/skills/plugins, and 400+ Composio integrations. What we DON'T have is a properly staffed, configured, operational team.

**Your job is to fix that.**

## Mission Objective

Design and implement a complete company team structure. Research what's available, hire the right agents, equip them properly, configure their operations, and make them productive.

## Phase 1: Research (Use Plan Mode)

Before hiring anyone, you MUST research what's available:

### 1A. Audit the Marketplace
- Run `platform_browse_marketplace_agents` with EVERY category (Marketing & Growth, Sales & Revenue, Engineering & DevOps, Design & Brand, Product & Strategy, Project Management, Operations & Support, Finance & Legal, Content & Media, Quality & Testing)
- Run `platform_browse_marketplace_skills` — browse ALL categories, note every skill name, description, and estimated token cost
- Run `platform_browse_marketplace_plugins` — browse ALL categories, note every plugin and what it provides
- Run `platform_list_tools` with category "composio" — see every available integration (Slack, Gmail, Jira, GitHub, Google Analytics, Stripe, HubSpot, LinkedIn, Twitter, etc.)
- Run `platform_list_llms` — see all available models, their capabilities, and costs

### 1B. Audit Current State
- Run `platform_list_agents` — who do we have now? What's their status, tools, skills?
- For each existing agent, run `platform_get_agent` — full config review
- Run `platform_list_connected_apps` — what integrations are already connected?
- Run `platform_workspace_stats` — current resource usage
- Run `platform_get_cost_breakdown` — current spending by model/agent

### 1C. Read the Operating Model
- Read `PLATFORM-CAPABILITIES-DEFINITIVE.md` — understand every system capability
- Read `MISSION-ZERO-RESULTS.md` — understand the operating model blueprint (authority levels, communication matrix, review cadences)

## Phase 2: Design the Org Chart

Based on your research, design the company structure. For EACH department, define:

### Departments Required
1. **Growth & Marketing** — SEO, content marketing, social media, email campaigns, analytics
2. **Sales & Revenue** — Lead gen, outreach, pipeline management, CRM
3. **Finance & Operations** — Cost tracking, invoicing, budgeting, expense management
4. **Engineering & DevOps** — Code review, CI/CD, monitoring, security, bug triage
5. **Content & Media** — Blog posts, newsletters, social content, design, copywriting
6. **Customer Success** — Support, onboarding, feedback collection, retention
7. **Research & Intelligence** — Market research, competitive analysis, trend spotting
8. **Platform Operations** — System health, performance monitoring, knowledge management

### For Each Agent, Specify:
- **Name** — Clear, memorable (e.g., "SCOUT" not "Agent 47")
- **Job Title** — Human-readable role (e.g., "Growth Marketing Specialist")
- **Department/Team** — Which team they belong to
- **Reports To** — Who they report to in the hierarchy (most report to Auto)
- **Model** — Choose wisely based on task complexity and cost:
  - `haiku-4.5` for high-frequency, simple tasks (cheapest)
  - `deepseek/deepseek-chat` for mid-tier work (great value)
  - `anthropic/claude-sonnet-4-6` for complex reasoning
  - `openai/gpt-4.1` for general purpose
  - Reserve `opus` for Auto only
- **System Prompt / Persona** — Who is this agent? What's their personality, expertise, communication style?
- **Skills** — Which marketplace skills to assign (install if needed)
- **Tools** — Which Composio apps they need (GMAIL, SLACK, GITHUB, GOOGLE_ANALYTICS, JIRA, STRIPE, HUBSPOT, LINKEDIN, TWITTER, etc.)
- **Plugins** — Which marketplace plugins to install and assign
- **Tags** — Capability tags for routing and matching
- **Heartbeat Config**:
  - Interval (how often they check in)
  - Proactive level (0-100, how autonomous)
  - Active hours (when they work)
  - What they do each heartbeat cycle
- **Playbooks** — What recurring workflows should they run? (cron schedule, steps, quality thresholds)
- **Blueprint Rules** — What governance rules apply? (min tools, required tags, model restrictions)

## Phase 3: Execute — Hire & Configure

For each agent in your plan:

1. **Create or update the agent** — `platform_create_agent` or `platform_update_agent`
2. **Install required skills** — `platform_install_skill` then `platform_assign_skill_to_agent`
3. **Install required plugins** — `platform_install_plugin` then `platform_assign_plugin_to_agent`
4. **Assign tools** — `platform_assign_tool_to_agent` for each Composio app
5. **Configure heartbeat** — `platform_configure_agent_heartbeat` with interval, proactive level, active hours
6. **Set org hierarchy** — Use `team`, `job_title`, `reports_to_id` fields

## Phase 4: Governance & Operations

1. **Create agent blueprints** — `platform_create_blueprint`:
   - Default workspace blueprint (min 2 tools, require system prompt)
   - Strict blueprint for sensitive roles (Finance, Security)
   - Cost-tier blueprint (restrict expensive models to senior agents)

2. **Deploy recurring playbooks** — `platform_create_playbook` with cron schedules:
   - **Daily CEO Briefing** (0 8 * * *) — Auto summarizes yesterday's activity, costs, blockers
   - **Weekly Business Review** (0 9 * * 1) — ATLAS pulls 7-day KPIs, trends, rankings
   - **Monthly KB Audit** (0 10 1 * *) — ORACLE audits documents for staleness
   - **Daily Social Post** — Content team publishes to social channels
   - **Weekly Newsletter** — Content team compiles and sends
   - Any other recurring workflows your team design requires

3. **Publish standard documents** — `workspace_write_file`:
   - `docs/communication-matrix.md` — Which channel for which message type
   - `docs/authority-boundaries.md` — What each authority level can do
   - `docs/report-template.md` — Required sections for all reports
   - `docs/metric-baselines.md` — Current KPI values as baseline snapshot

4. **Store metric baselines** — `platform_store_memory` with current values from analytics

5. **Configure Auto's own heartbeat** — 15-min interval, high proactive level, active hours 07:00-22:00, full CTO analysis loop

## Constraints

- **Budget-conscious**: Use cheaper models (haiku, deepseek) for routine work. Reserve expensive models for complex tasks.
- **Don't over-hire**: Better to have 12 excellent agents than 25 mediocre ones. Each agent must have a clear, distinct purpose.
- **Composio apps must be connected**: Only assign tools where the OAuth connection exists. Check `platform_list_connected_apps` first.
- **Existing agents**: Keep agents that are working well (SENTINEL, COMMS). Retire or reconfigure agents that are dead weight (0 tasks completed, no heartbeat).
- **Everything through the board**: All work flows through the Kanban board. Every playbook creates board tasks. Every agent reports via `platform_submit_report`.

## Success Criteria

When this mission is complete:
- [ ] Org chart with 10-20 agents across all departments, each with clear role
- [ ] Every agent has: skills, tools, persona, heartbeat, and blueprint validation passing
- [ ] At least 3 governance blueprints created (default, strict, cost-tier)
- [ ] At least 5 recurring playbooks deployed with cron schedules
- [ ] Standard docs published to workspace
- [ ] Auto's 15-min heartbeat configured and running
- [ ] Metric baselines stored
- [ ] All agents show green readiness badge (blueprint validation passing)

## Important

You are not writing a REVIEW or an ASSESSMENT. You are BUILDING. Every finding becomes an action. Every gap becomes a tool call. Research first, plan second, execute third. No circular reviews. Ship it.
