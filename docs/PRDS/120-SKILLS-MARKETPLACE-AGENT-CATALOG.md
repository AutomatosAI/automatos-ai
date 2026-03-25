# PRD-120: Skills Marketplace & Agent Catalog — Agency Import + Business Plan Template

**Status:** Draft
**Date:** 2026-03-24
**Authors:** Gerard Kavanagh + Claude
**Dependencies:** PRD-71 (Unified Skills), PRD-82A/B/C (Missions), PRD-76 (Agent Reporting)
**Branch:** TBD

---

## TL;DR

Import ~100 professional agent skills from the open-source `agency-agents` catalog, adapt them for Automatos, publish them in the Skills Marketplace, and build pre-configured agent templates users can deploy in one click. Then build the **Business Plan Mission Template** — a flagship mission that doesn't just write a document, it configures the user's entire workspace: agents, playbooks, heartbeats, schedules, and board tasks. Automatos dogfoods this as customer #1.

---

## 1. Problem Statement

### 1.1 Empty Marketplace

New users sign up, see an empty workspace, and have no idea what to do. The Skills Marketplace exists but has almost no skills. The Agent Marketplace has no pre-built agents. Users must configure everything from scratch — agent roles, skills, tools, personas — which requires expertise they don't have.

### 1.2 No Onboarding Path

There's no guided path from "I have a business idea" to "my workspace is running my business." Users who want AI to help them operationally have to:
1. Understand what agents are
2. Figure out which roles they need
3. Write skill prompts from scratch
4. Assign tools manually
5. Create playbooks and schedules

This is a 2-hour setup for an expert. For a new user, it's a dead end.

### 1.3 Missing Skill Library

We have the platform tools (`platform_create_agent`, `platform_install_skill`, playbook CRUD, heartbeat config) but no content. It's an app store with no apps.

---

## 2. Solution Overview

### Phase 1: Skill Import & Adaptation (2-3 days)
Cherry-pick ~80 relevant skills from `agency-agents`, rewrite for Automatos format, categorize and tag.

### Phase 2: Agent Templates in Marketplace (1-2 days)
Pre-configured agent definitions with assigned skills, tools, LLMs, and personas. One-click deploy to workspace.

### Phase 3: Business Plan Mission Template (2-3 days)
Flagship mission template that configures an entire workspace from a business goal.

### Phase 4: Dogfood — Automatos as Customer #1 (1 day)
Run the Business Plan template for Automatos itself. Validate the full loop.

---

## 3. Source Material: agency-agents Repository

**Repository:** `github.com/msitarzewski/agency-agents`
**License:** Open source
**Structure:** Markdown files with YAML frontmatter (name, description, color, emoji, vibe, tools, services)

### 3.1 Divisions & Agent Count

| Division | Agents | Automatos Relevance |
|----------|--------|-------------------|
| Engineering | ~22 | HIGH — Frontend, Backend, DevOps, Security, AI Engineer, SRE |
| Design | ~8 | HIGH — UI Designer, UX Researcher, Brand Guardian |
| Marketing | ~25 | HIGH — Growth Hacker, Content Creator, SEO, Social Media, LinkedIn |
| Sales | ~8 | HIGH — Outbound, Discovery, Deal Strategy, Pipeline |
| Product | ~4 | HIGH — Sprint Prioritizer, Trend Researcher, Feedback Synthesizer |
| Project Management | ~6 | HIGH — Project Shepherd, Studio Producer, Experiment Tracker |
| Testing | ~8 | MEDIUM — QA, Performance, API Testing, Accessibility |
| Support | ~6 | HIGH — Support Responder, Analytics, Finance, Legal Compliance |
| Paid Media | ~7 | MEDIUM — PPC, Tracking, Ad Creative |
| Game Development | ~20 | LOW — Niche (Unity, Unreal, Godot, Roblox) — import selectively |
| Spatial Computing | ~6 | LOW — XR/VR niche — skip for now |
| Specialized | ~23 | MIXED — Cherry pick: Document Generator, ZK Steward, Compliance, Recruitment, Supply Chain |

### 3.2 Import Priority

**Tier 1 — Import immediately (~50 agents):**
All of Engineering, Design, Marketing, Sales, Product, Project Management, Support

**Tier 2 — Import selectively (~15 agents):**
Testing (QA, Performance, API Tester), Paid Media (PPC, Ad Creative, Analytics), Specialized (Document Generator, Compliance, Recruitment, Supply Chain, Developer Advocate)

**Tier 3 — Skip or defer (~35 agents):**
Game Development (too niche), Spatial Computing (too niche), China-specific marketing (Xiaohongshu, WeChat, Baidu, Bilibili, Douyin, Kuaishou, Weibo)

---

## 4. Phase 1: Skill Import & Adaptation

### 4.1 Automatos Skill Format

Each skill lives in `/automatos-skills/skills/{category}/{skill-slug}/SKILL.md`:

```markdown
---
name: {Skill Name}
version: 1.0.0
category: {engineering|design|marketing|sales|product|project-management|testing|support|paid-media|specialized}
tags: [tag1, tag2, tag3]
description: {One-line description}
recommended_tools: [tool1, tool2]
recommended_model: {model suggestion — e.g. "fast" for simple tasks, "reasoning" for complex}
---

# {Skill Name}

## Identity
{Who this agent is — role, expertise, personality}

## Core Mission
{Primary responsibilities and what success looks like}

## Workflow
{Step-by-step process the agent follows}

## Deliverables
{What the agent produces — formats, templates, checklists}

## Rules
{Non-negotiable constraints and boundaries}
```

### 4.2 Adaptation Rules

When converting from agency-agents format to Automatos:

1. **Strip platform-specific references** — No mentions of OpenClaw, Cursor, Qwen, etc.
2. **Map tools to Automatos tools** — `WebFetch` → `workspace_exec` or Composio tools; `Read/Write/Edit` → `workspace_read_file`, `workspace_write_file`
3. **Add recommended_tools** — Map each skill to available Automatos tools (Composio integrations, workspace tools, platform tools)
4. **Add recommended_model** — `haiku` for simple/repetitive tasks, `sonnet` for standard work, `opus` for complex reasoning
5. **Simplify emoji headers** — Keep content, lose the emoji section markers
6. **Add Automatos context** — Skills should reference platform capabilities (reports, board tasks, playbooks) where relevant
7. **Trim aggressively** — Remove filler, keep actionable instructions. Target 200-400 lines per skill.
8. **Business-outcome focus** — Every skill should tie back to measurable business value

### 4.3 Category Structure

```
automatos-skills/
└── skills/
    ├── engineering/
    │   ├── frontend-developer/SKILL.md
    │   ├── backend-architect/SKILL.md
    │   ├── devops-automator/SKILL.md
    │   ├── security-engineer/SKILL.md
    │   ├── ai-engineer/SKILL.md
    │   └── ...
    ├── design/
    │   ├── ui-designer/SKILL.md
    │   ├── ux-researcher/SKILL.md
    │   ├── brand-guardian/SKILL.md
    │   └── ...
    ├── marketing/
    │   ├── growth-hacker/SKILL.md
    │   ├── content-creator/SKILL.md
    │   ├── seo-specialist/SKILL.md
    │   ├── social-media-strategist/SKILL.md
    │   └── ...
    ├── sales/
    │   ├── outbound-strategist/SKILL.md
    │   ├── discovery-coach/SKILL.md
    │   ├── deal-strategist/SKILL.md
    │   └── ...
    ├── product/
    │   ├── sprint-prioritizer/SKILL.md
    │   ├── trend-researcher/SKILL.md
    │   └── ...
    ├── project-management/
    │   ├── project-shepherd/SKILL.md
    │   ├── studio-producer/SKILL.md
    │   └── ...
    ├── support/
    │   ├── support-responder/SKILL.md
    │   ├── analytics-reporter/SKILL.md
    │   ├── finance-tracker/SKILL.md
    │   ├── legal-compliance/SKILL.md
    │   └── ...
    ├── testing/
    │   ├── qa-engineer/SKILL.md
    │   ├── performance-benchmarker/SKILL.md
    │   └── ...
    ├── paid-media/
    │   ├── ppc-strategist/SKILL.md
    │   ├── ad-creative-strategist/SKILL.md
    │   └── ...
    └── specialized/
        ├── document-generator/SKILL.md
        ├── compliance-auditor/SKILL.md
        ├── recruitment-specialist/SKILL.md
        └── ...
```

---

## 5. Phase 2: Agent Templates in Marketplace

### 5.1 Agent Template Schema

Each importable agent template stored in the DB (or seeded via migration):

```json
{
  "name": "Growth Hacker",
  "slug": "growth-hacker",
  "category": "marketing",
  "description": "Expert growth strategist specializing in rapid user acquisition through data-driven experimentation.",
  "persona": "You are a data-obsessed growth strategist who thinks in funnels, loops, and experiments...",
  "skill_slug": "growth-hacker",
  "recommended_model": "anthropic/claude-sonnet-4-6",
  "recommended_tools": ["GOOGLE_ANALYTICS", "GOOGLE_SHEETS", "SLACK"],
  "tags": ["growth", "marketing", "analytics", "acquisition", "conversion"],
  "icon": "📈",
  "tier": "free"
}
```

### 5.2 Marketplace Categories (UI)

| Category | Icon | Agent Count |
|----------|------|-------------|
| Marketing & Growth | 📈 | ~12 |
| Sales & Revenue | 💰 | ~8 |
| Engineering & DevOps | ⚙️ | ~10 |
| Design & Brand | 🎨 | ~6 |
| Product & Strategy | 🎯 | ~4 |
| Project Management | 📋 | ~6 |
| Operations & Support | 🏢 | ~6 |
| Finance & Legal | ⚖️ | ~4 |
| Content & Media | ✍️ | ~6 |
| Quality & Testing | 🔍 | ~4 |

### 5.3 One-Click Deploy Flow

1. User browses Marketplace → selects "Growth Hacker"
2. Frontend shows preview: skill summary, recommended tools, model
3. User clicks "Add to Workspace"
4. Backend: `platform_create_agent` → assigns skill, model, tools, persona
5. Agent appears in Roster, ready to use
6. User can customize name, model, tools after deploy

### 5.4 LLM Assignment Strategy

| Agent Complexity | Recommended Model | Examples |
|-----------------|-------------------|----------|
| Simple/repetitive | `haiku-4.5` | Support Responder, Finance Tracker, Report Distribution |
| Standard work | `sonnet-4.6` | Content Creator, SEO Specialist, Project Shepherd |
| Complex reasoning | `opus-4.6` or `deepseek-chat` | Backend Architect, Security Engineer, Deal Strategist |
| Code generation | `sonnet-4.6` | Frontend Developer, DevOps Automator, AI Engineer |

---

## 6. Phase 3: Business Plan Mission Template

### 6.1 The Vision

A new user says: **"I'm starting a coffee brand."**

The mission doesn't produce a PDF. It produces a **configured, operational workspace**:

1. **Business Plan Document** — Executive summary, market analysis, financial projections, go-to-market strategy
2. **Agents Created** — Marketing (Growth Hacker, Content Creator, Social Media), Sales (Outbound Strategist), Operations (Finance Tracker, Legal Compliance), Product (Trend Researcher)
3. **Playbooks Written & Scheduled** — Weekly social media calendar, monthly financial review, quarterly market analysis
4. **Heartbeats Configured** — Daily check-ins for active agents, KPI tracking
5. **Board Tasks Created** — Business plan milestones as trackable tasks with deadlines
6. **Orchestrator Role** — Assigned as "COO" — monitors plan execution, tracks targets

### 6.2 Mission Task Decomposition

```
Mission: "Write a business plan for [X] and set up my workspace"

Phase 1: Research & Analysis (parallel)
  ├── Task 1: Market research (Trend Researcher agent)
  ├── Task 2: Competitive analysis (Growth Hacker agent)
  └── Task 3: Financial modeling (Finance Tracker agent)

Phase 2: Document Generation (sequential, depends on Phase 1)
  ├── Task 4: Write executive summary
  ├── Task 5: Write market analysis section
  ├── Task 6: Write financial projections
  └── Task 7: Synthesize into full business plan document

Phase 3: Workspace Configuration (parallel, uses platform tools)
  ├── Task 8: Create agents from template catalog
  │   └── platform_create_agent × N (marketing, sales, ops, product)
  ├── Task 9: Write and schedule playbooks
  │   └── platform_create_playbook × N (social, reporting, analysis)
  ├── Task 10: Configure heartbeats
  │   └── platform_configure_heartbeat × N
  └── Task 11: Create board tasks from plan milestones
      └── platform_create_board_task × N

Phase 4: Verification & Handoff
  └── Task 12: Summary report — what was created, next steps, how to use your workspace
```

### 6.3 Platform Tools Required

All of these should already exist from PRD-71:

| Tool | Purpose | Status |
|------|---------|--------|
| `platform_create_agent` | Create agents with skill/model/tools | Verify exists |
| `platform_install_skill` | Assign skill to agent | Verify exists |
| `platform_assign_tool` | Assign Composio tool to agent | Verify exists |
| `platform_create_playbook` | Create playbook from template | **May need to build** |
| `platform_configure_heartbeat` | Set heartbeat schedule | Verify exists |
| `platform_create_board_task` | Create board task | **May need to build** |
| `platform_submit_report` | Write mission output report | Exists (PRD-76) |
| `workspace_write_file` | Save business plan document | Exists |

### 6.4 Template Configuration

```json
{
  "template_id": "business-plan",
  "name": "Business Plan & Workspace Setup",
  "description": "Generate a comprehensive business plan and configure your workspace with agents, playbooks, and schedules to run your business.",
  "category": "business",
  "complexity": "high",
  "estimated_budget_tokens": 500000,
  "estimated_cost_usd": 2.00,
  "required_input": {
    "business_name": "string",
    "business_type": "string",
    "industry": "string",
    "target_market": "string (optional)",
    "budget_range": "string (optional)",
    "goals": "string (optional)"
  },
  "output_types": ["document", "workspace_config"],
  "agent_templates_used": [
    "growth-hacker", "content-creator", "social-media-strategist",
    "outbound-strategist", "finance-tracker", "legal-compliance",
    "trend-researcher", "project-shepherd"
  ]
}
```

---

## 7. Phase 4: Dogfood — Automatos as Customer #1

Run the Business Plan template with:
- **Business:** Automatos AI Platform
- **Type:** B2B SaaS
- **Industry:** AI/Automation
- **Target Market:** SMBs, solopreneurs, agencies
- **Goals:** Pilot launch, first 10 paying users

Validate:
- [ ] Business plan document generated and downloadable
- [ ] Agents created in workspace with correct skills/tools/models
- [ ] Playbooks created and scheduled
- [ ] Heartbeats configured and firing
- [ ] Board tasks populated with milestones
- [ ] Orchestrator monitoring plan execution
- [ ] Total token spend within budget estimate

---

## 8. Other Mission Templates (Future)

Once the Business Plan template works, expand the library:

| Template | Output | Complexity |
|----------|--------|-----------|
| **Market Research Report** | PDF/MD report + data | Medium |
| **Competitive Analysis** | Comparison matrix + report | Medium |
| **Content Calendar** | 30-day calendar + draft posts | Medium |
| **SEO Audit** | Technical audit + recommendations | Medium |
| **Sales Pipeline Setup** | CRM config + playbooks | High |
| **Product Launch Plan** | Go-to-market doc + workspace config | High |
| **Quarterly Business Review** | KPI dashboard + report | Medium |
| **Recruitment Campaign** | Job descriptions + outreach playbooks | Medium |

---

## 9. Success Metrics

| Metric | Target |
|--------|--------|
| Skills imported and categorized | 65+ |
| Agent templates in marketplace | 50+ |
| One-click deploy success rate | 95% |
| Business Plan template completion rate | 80% |
| Average Business Plan token spend | <500K tokens |
| Time from signup to operational workspace | <10 minutes |
| Automatos dogfood: all agents + playbooks running | Yes |

---

## 10. Implementation Order

1. **Import skills** — Clone repo, cherry-pick, rewrite for Automatos format
2. **Seed marketplace** — Agent templates with skill/model/tool assignments
3. **Verify platform tools** — Confirm all needed `platform_*` tools exist, build missing ones
4. **Build Business Plan template** — Mission config + task decomposition
5. **Dogfood** — Run for Automatos, fix issues
6. **Polish UX** — Results page showing what was configured, download business plan doc
7. **Ship** — Ready for pilot users

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Skills too verbose, token-heavy | Aggressive trimming to 200-400 lines each |
| Platform tools missing for workspace config | Audit PRD-71 tools before Phase 3, build gaps |
| Business Plan mission too expensive | Budget tiers, allow user to skip workspace config |
| Agent templates become stale | Version skills, allow user edits post-deploy |
| License issues with source repo | Verify open-source license, attribute properly |
