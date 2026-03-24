# PRD-120 Skills Marketplace & Agent Catalog — Implementation Plan

## Overview
Import ~65 professional agent skills from agency-agents, build agent templates marketplace with one-click deploy, and create the Business Plan Mission Template that configures entire workspaces.

## Branch: ralph/prd-120-skills-marketplace-agent-catalog

---

## Tasks

### Phase 1: Skill Import Infrastructure
- [x] US-001: Create automatos-skills/ directory structure + Python import/conversion script
- [x] US-002: Clone agency-agents, import Engineering + Design skills (29 skills: 21 eng + 8 design)
- [x] US-003: Import Marketing, Sales, Product, PM, Support skills (41 skills: 16 mkt + 8 sales + 5 product + 6 pm + 6 support)
- [x] US-004: Import Tier 2 selective skills (Testing, Paid Media, Specialized ~11 skills) + CATALOG.md

### Phase 2: Database & API
- [x] US-005: Add agent_catalog_templates DB table + Alembic migration + SQLAlchemy model
- [x] US-006: Seed agent catalog templates from imported SKILL.md files (55+ rows)
- [x] US-007: Marketplace API endpoints (browse, search, categories, deploy)

### Phase 3: Frontend
- [x] US-008: Marketplace page with category grid + agent cards + search
- [x] US-009: Agent template detail modal + one-click deploy button

### Phase 4: Business Plan Template & Wiring
- [x] US-010: Business Plan mission template in TEMPLATE_REGISTRY (4 phases, parallel groups)
- [ ] US-011: Mission template selector in create-mission-modal + backend template_id hint
- [ ] US-012: Integration tests + catalog validation

---

## Key References

| File | Purpose |
|------|---------|
| `orchestrator/api/marketplace.py` | Existing marketplace router at /api/marketplace |
| `orchestrator/api/marketplace_plugins.py` | Plugin marketplace (separate from agent catalog) |
| `orchestrator/api/skills.py` | Skill management API (PRD-22) |
| `orchestrator/api/agents.py` | Agent CRUD endpoints |
| `orchestrator/api/missions.py` | Mission create/approve/list endpoints |
| `orchestrator/core/models/core.py` | SQLAlchemy models — Agent, AgentTemplate (Pydantic, line ~863), BoardTask |
| `orchestrator/modules/coordination/templates.py` | Mission templates — TEMPLATE_REGISTRY, TaskTemplate, render_template |
| `orchestrator/modules/coordination/agent_matcher.py` | _ROLE_SYNONYMS for agent role matching |
| `orchestrator/modules/tools/discovery/platform_actions.py` | Platform tool registration (ActionDefinitions) |
| `orchestrator/modules/tools/discovery/platform_executor.py` | Platform tool handlers |
| `orchestrator/modules/tools/discovery/actions_playbooks.py` | platform_create_playbook (ALREADY EXISTS) |
| `orchestrator/modules/tools/discovery/actions_board_tasks.py` | platform_board_summary (board task tools) |
| `orchestrator/core/config.py` | ALL config constants |
| `frontend/components/missions/create-mission-modal.tsx` | Mission creation UI |
| `frontend/hooks/use-missions-api.ts` | Mission React Query hooks |
| `frontend/types/missions.ts` | Mission TypeScript interfaces |
| `docs/PRDS/120-SKILLS-MARKETPLACE-AGENT-CATALOG.md` | Full PRD |

## Architecture Rules (CRITICAL)

- Python 3.11+ with type hints on all public functions
- SQLAlchemy ORM with sync Session — follow existing patterns
- FastAPI endpoints with Pydantic BaseModel
- ALL config values go in orchestrator/core/config.py — NO os.getenv() anywhere else
- Agent roles in templates: use categories from _ROLE_SYNONYMS in agent_matcher.py
- NO hardcoded values — use config constants
- Frozen dataclasses for immutable data
- BEFORE DELETING ANY CODE: grep EVERY file for callers
- React Query v4 on frontend (isLoading not isPending)
- Skills go in automatos-skills/skills/{category}/{slug}/SKILL.md
- Agent catalog templates keyed by slug matching skill directory name

## Validation

Backend Python imports:
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai && python -c "
from orchestrator.modules.coordination import templates, planner, dispatcher, agent_matcher
from orchestrator.api import marketplace
print('All imports OK')
" 2>&1 | tail -5
```

Tests:
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai && python -m pytest orchestrator/tests/ -x -q --timeout=30 2>&1 | tail -20
```

Frontend:
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai/frontend && npx tsc --noEmit 2>&1 | grep -iE "marketplace|agent-template|create-mission" | head -10
```

Skill count check:
```bash
find automatos-skills/skills -name "SKILL.md" | wc -l
```

Note: Pre-existing errors may exist in other files. Only check for NEW errors introduced by your changes.

## Discovered Issues

- agency-agents filenames include division prefix (e.g., `engineering-backend-architect.md`). Import script strips this to produce clean slugs.
- agency-agents has `strategy` and `academic` divisions not listed in PRD — skipped (not in DIVISION_TO_CATEGORY).
- Total unfiltered count across all divisions: 121 agents. US-002/003/004 filtering will narrow to ~65.
- Python import validation requires venv with sqlalchemy etc. — pre-existing, not caused by this change.
- Many source agent files lack Workflow/Deliverables sections. Import script updated with fallback content generation for missing sections.
- Engineering yielded 21 skills (not ~15 as estimated) — no game-dev agents in engineering division to filter. All are professional software skills.
- Design yielded 8 skills (not ~6 as estimated) — includes whimsy-injector and inclusive-visuals-specialist which are niche but useful.
- SKIP_AGENTS needed post-strip slug variants — original slugs had division prefix, but skip check runs after stripping. Added 9 additional China-specific slugs.
- Marketing yielded 16 skills (not ~12 as estimated) — more non-China agents than expected. Filtered: baidu, bilibili, douyin, kuaishou, weibo, xiaohongshu, zhihu, china-ecommerce, private-domain, livestream-commerce, wechat-official-account.
- Product yielded 5 skills (not ~4) — includes behavioral-nudge-engine, a useful product psychology agent.
- Total skills after US-003: 70 (29 from US-002 + 41 from US-003).
- US-004: Testing yielded 3 skills (accessibility-auditor, api-tester, performance-benchmarker). Paid Media yielded 3 (auditor, creative-strategist, ppc-strategist). Specialized yielded 5 (compliance-auditor, developer-advocate, document-generator, recruitment-specialist, supply-chain-strategist).
- Total skills after US-004: 81 (70 + 11). Exceeds the 55+ target by 26.
- CATALOG.md generated with full table of all 81 skills across 10 categories.
