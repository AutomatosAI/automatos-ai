# PRD-120 Phase 2: Skill Rewrites — Implementation Plan

## Overview
Rewrite all 81 imported skills to Automatos production quality. Each skill must have: proper persona, real platform/workspace/composio tool mappings with JSON call examples, structured workflows, output formats, and anti-patterns. Quality bar = Sentinel + Scout skills.

## Branch: ralph/prd-120-skill-rewrites

---

## Tasks

### Phase 1: Infrastructure
- [ ] US-001: Create SKILL-GUIDE.md reference + fix seed script frontmatter parsing for tools:{name,description} format

### Phase 2: Skill Rewrites by Category
- [ ] US-002: Engineering batch 1 (8): frontend-developer, backend-architect, devops-automator, security-engineer, ai-engineer, sre, data-engineer, database-optimizer
- [ ] US-003: Engineering batch 2 (8): code-reviewer, software-architect, technical-writer, mobile-app-builder, senior-developer, git-workflow-master, rapid-prototyper, incident-response-commander
- [ ] US-004: Engineering batch 3 (5): threat-detection-engineer, solidity-smart-contract-engineer, embedded-firmware-engineer, ai-data-remediation-engineer, autonomous-optimization-architect
- [ ] US-005: Marketing (16): growth-hacker, content-creator, seo-specialist, social-media-strategist, linkedin-content-creator, twitter-engager, tiktok-strategist, instagram-curator, podcast-strategist, reddit-community-builder, app-store-optimizer, carousel-growth-engine, short-video-editing-coach, ai-citation-strategist, book-co-author, cross-border-ecommerce
- [ ] US-006: Sales (8): outbound-strategist, discovery-coach, deal-strategist, pipeline-analyst, account-strategist, coach, engineer, proposal-strategist
- [ ] US-007: Design (8): ui-designer, ux-researcher, ux-architect, brand-guardian, visual-storyteller, image-prompt-engineer, inclusive-visuals-specialist, whimsy-injector
- [ ] US-008: Product + PM (11): sprint-prioritizer, trend-researcher, feedback-synthesizer, behavioral-nudge-engine, manager, project-shepherd, studio-producer, experiment-tracker, jira-workflow-steward, project-manager-senior, studio-operations
- [ ] US-009: Support + Testing (9): support-responder, analytics-reporter, finance-tracker, legal-compliance-checker, executive-summary-generator, infrastructure-maintainer, performance-benchmarker, api-tester, accessibility-auditor
- [ ] US-010: Paid Media + Specialized (8): ppc-strategist, creative-strategist, auditor, document-generator, compliance-auditor, recruitment-specialist, supply-chain-strategist, developer-advocate

### Phase 3: Validation
- [ ] US-011: Validate all 81 skills, run seed dry-run, commit to skills repo

---

## Key References

| File | Purpose |
|------|---------|
| `/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills/sentinel/SKILL.md` | Gold standard: monitoring skill with tool calls |
| `/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills/scout/SKILL.md` | Gold standard: sales/outreach skill with CRM tools |
| `scripts/seed_agent_catalog.py` | Seed script — parses SKILL.md, populates DB |
| `orchestrator/modules/tools/discovery/platform_actions.py` | All platform tool registrations |
| `orchestrator/modules/tools/discovery/workspace_actions.py` | All workspace tool registrations |
| `orchestrator/modules/context/sections/skills.py` | How skill content gets injected into agent prompts |
| `orchestrator/core/models/core.py` | Agent + Skill SQLAlchemy models |

## Quality Bar (CRITICAL)

Every rewritten skill MUST have:

1. **Frontmatter**: name, description (1 sentence), version, tags, category: agent-role, tools: [{name, description}]
2. **Identity**: 1-2 sentences — who this agent is in the Automatos context
3. **Workflow**: Numbered steps with ```json code blocks showing exact tool calls with realistic params
4. **Output Format**: Structured template (like Sentinel's status report)
5. **What NOT To Do**: 3-5 anti-patterns specific to the role
6. **Length**: 60-100 lines, no filler, every line actionable

## Tool Reference

### Platform Tools (agents call these via function calling)
- `platform_get_system_health` — service health + response times
- `platform_get_logs` — app logs by severity (params: severity, limit)
- `platform_get_llm_usage` — token usage + cost metrics
- `platform_get_cost_breakdown` — detailed cost analysis
- `platform_workspace_stats` — workspace metrics
- `platform_submit_report` — submit status/standup/audit report (params: title, report_type, status, content, metrics, summary)
- `platform_get_latest_report` — read previous report (params: agent_name)
- `platform_create_task` — create board task (params: title, description, priority, status)
- `platform_list_tasks` — list board tasks (params: status filter)
- `platform_board_summary` — board state overview
- `platform_search_memory` — search workspace knowledge
- `platform_search_chat_history` — search past conversations
- `platform_query_loki_logs` — LogQL query
- `platform_publish_blog_post` — publish content
- `platform_schedule_task` — schedule recurring work

### Workspace Tools (file/code operations)
- `workspace_read_file` — read file (params: path)
- `workspace_write_file` — write file (params: path, content)
- `workspace_list_dir` — list directory (params: path)
- `workspace_grep` — regex search (params: pattern, path, include, max_results)
- `workspace_exec` — run command (params: command, cwd, timeout)
- `workspace_git` — git operations (params: operation, args)

### Composio (external service actions)
- `composio_execute` — execute external action (params: action, app_name, ...)
  - HUBSPOT: LIST_CONTACTS, CREATE_CONTACT, UPDATE_CONTACT, CREATE_DEAL
  - GMAIL: SEND_EMAIL, LIST_EMAILS
  - LINKEDIN: SEND_MESSAGE, CREATE_POST
  - TWITTER: CREATE_TWEET
  - GOOGLE_SHEETS: READ_RANGE, WRITE_RANGE
  - GOOGLE_ANALYTICS: GET_REPORT
  - JIRA: CREATE_ISSUE, LIST_ISSUES, UPDATE_ISSUE
  - GITHUB: CREATE_ISSUE, LIST_REPOS
  - SLACK: SEND_MESSAGE

## Validation

Seed script dry-run:
```bash
python3 scripts/seed_agent_catalog.py --dry-run 2>&1 | tail -5
```

Skill count:
```bash
find /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills -name "SKILL.md" | wc -l
```

## Discovered Issues

(Ralph will log issues here during implementation)
