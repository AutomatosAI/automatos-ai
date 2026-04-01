# Test Playbook & Schedule

**Total: 376 tests | 53 files | 3 runners**

---

## Runners (How to Execute)

| Runner | Command | Tests | Runtime | Output |
|--------|---------|-------|---------|--------|
| **Nightly** | `python3 tests/run_nightly.py` | All 376 | ~15-20 min | `test-summary.json` |
| **Health Regression** | `python3 tests/run_health_regression.py` | ~120 high-signal | ~5-8 min | `qa-report.json` |
| **Gap Finder** | `python3 tests/run_gap_finder.py` | 0 (analysis only) | <1 sec | `coverage-gap-summary.json` |

Run individual files with: `python3 -m pytest tests/api/test_<name>.py -v`

---

## Schedule

### Every Deploy / PR Merge
Run the health regression suite — fast, high-signal, catches breakage.
```
python3 tests/run_health_regression.py
```

### Nightly (once per day)
Run the full suite — catches everything including performance regressions.
```
python3 tests/run_nightly.py
```

### Weekly
Run gap finder to check for new untested domains.
```
python3 tests/run_gap_finder.py
```

---

## Playbook 1: Smoke Tests (read-only, safe to run anytime)

Quick health check — no data created, no side effects.

| File | Tests | What it checks |
|------|-------|----------------|
| `test_health.py` | 5 | Platform health, system state |
| `test_health_bootstrap.py` | 3 | Bootstrap stages, no secret leaks |
| `test_analytics.py` | 5 | Dashboard, memory stats, metrics |
| `test_llm_analytics.py` | 5 | LLM usage, costs, projections |
| `test_documents.py` | 8 | Document list, queue, analytics |
| `test_memory.py` | 4 | Memory stats endpoints |
| `test_workflows.py` | 3 | Workflow list, templates, stats |
| `test_knowledge.py` | 10 | Multi-agent, field theory, templates, search |
| `test_llm_config.py` | 8 | LLM service categories, settings validation |

**Total: 51 tests | ~1 min | Schedule: every deploy**

---

## Playbook 2: Error Path Tests (safe — sends bad input, expects graceful failures)

Validates the platform returns 400/404/422 instead of 500 on bad input.

| File | Tests | What it checks |
|------|-------|----------------|
| `test_agent_errors.py` | 12 | Bad IDs, empty body, missing name, nonexistent sub-resources |
| `test_chat_errors.py` | 7 | Nonexistent agent, empty message, malformed body |
| `test_document_errors.py` | 5 | Nonexistent doc, empty search, no-file upload |
| `test_memory_errors.py` | 5 | Empty store/search, nonexistent delete |
| `test_heartbeat_errors.py` | 4 | Nonexistent agent, invalid ID, bad limits |
| `test_workflow_errors.py` | 6 | Nonexistent workflow/recipe, empty bodies |
| `test_channel_errors.py` | 5 | Nonexistent CRUD, invalid platform |
| `test_routing_errors.py` | 4 | Nonexistent rule, empty body, invalid agent |
| `test_persona_errors.py` | 5 | Nonexistent CRUD, missing name |
| `test_key_errors.py` | 5 | Empty body, invalid provider, nonexistent |
| `test_model_errors.py` | 4 | Empty recommend, fake model, negative tokens |
| `test_workspace_errors.py` | 5 | Empty exec, invalid workspace, missing path |
| `test_analytics_errors.py` | 4 | Invalid period, bad dates, negative limits |

**Total: 71 tests | ~2 min | Schedule: every deploy**

---

## Playbook 3: CRUD Journeys (creates + cleans up test data)

Tests full create → read → update → delete cycles per domain.

| File | Tests | What it checks |
|------|-------|----------------|
| `test_agents.py` | 11 | Agent CRUD + status, performance, logs, model-config |
| `test_agent_journeys.py` | 8 | Create → assign tools → execute → model config round-trip |
| `test_channels.py` | 6 | Channel CRUD + analytics + SQL column bug check |
| `test_personas.py` | 7 | Persona CRUD + assign to agent |
| `test_keys.py` | 6 | BYOK key add → test → platform status → delete |
| `test_routing.py` | 5 | Routing rule CRUD + cache stats |
| `test_recipes.py` | 6 | Recipe list, categories, featured, search, create |
| `test_recipe_cron.py` | 8 | Cron/manual recipe create → update schedule → delete |
| `test_chat.py` | 8 | Chat stream → followup → history → title update → delete |
| `test_tools.py` | 7 | Marketplace, stats, connected, search, refresh |
| `test_skills.py` | 5 | List, sources, agent skills, recommend, search |
| `test_models.py` | 5 | List, providers, recommend, cost estimate |
| `test_workspaces.py` | 5 | Current, integrations, exec, files |
| `test_webhooks.py` | 4 | Verify, send, settings, update |
| `test_permissions.py` | 8 | Permission matrix, assignments, assign/revoke, audit |
| `test_missions.py` | 11 | Mission list, detail, checkpoints, cost, events, cancel |

**Total: 110 tests | ~5 min | Schedule: nightly**

---

## Playbook 4: Deep Journeys (stateful multi-step, creates real data)

Deeper than CRUD — tests workflows across multiple steps with state carried between tests.

| File | Tests | What it checks |
|------|-------|----------------|
| `test_document_journeys.py` | 5 | Upload → list → detail → search → delete |
| `test_memory_journeys.py` | 6 | Store → search → recent → agent breakdown → delete |
| `test_workflow_journeys.py` | 6 | List → execute → status → execution list → templates |
| `test_heartbeat_journeys.py` | 6 | Status shape → orchestrator run → history → agent run → analytics |
| `test_mission_journeys.py` | 14 | Create → detail → events → cost → checkpoints → field → pause → resume → cancel → archive → delete |

**Total: 37 tests | ~5 min | Schedule: nightly**

---

## Playbook 5: User Journeys (cross-domain, mirrors real user flows)

Each file simulates a complete user persona workflow across multiple domains.

| File | Tests | Persona | Flow |
|------|-------|---------|------|
| `test_onboarding_journey.py` | 9 | **New user** | Workspace → agents → models → create agent → first chat → memory → dashboard → marketplace |
| `test_daily_workflow_journey.py` | 13 | **Power user** | Health → heartbeat → roster → performance → chat → LLM costs → memory → missions → recipes → metrics |
| `test_admin_config_journey.py` | 10 | **Admin** | Workspace → persona → agent → assign → BYOK key → routing rule → channel → verify all → cleanup |
| `test_mission_research_journey.py` | 11 | **Researcher** | Pick agent → create mission → detail → events → cost → checkpoints → field → cancel → stats → archive |
| `test_integration_setup_journey.py` | 13 | **Developer** | Marketplace → connected → credentials → skills → sources → recommend → agent skills → keys → refresh |
| `test_user_journeys.py` | 5 | **General** | Agent config round-trip, execute handle, chat title, workflow, heartbeat |

**Total: 61 tests | ~8 min | Schedule: nightly + pre-release**

---

## Playbook 6: Performance Baselines (latency SLO checks)

| File | Tests | What it checks |
|------|-------|----------------|
| `test_performance_baselines.py` | 15 | Response time thresholds across 3 tiers |

**SLO Tiers:**
- **FAST (< 500ms):** health, agent list, workspace current, models, keys
- **MEDIUM (< 2s):** chat history, memory stats, analytics, documents, marketplace, missions, heartbeat, recipes
- **SLOW (< 10s):** orchestrator run, model recommend

**Total: 15 tests | ~2 min | Schedule: nightly + after infra changes**

---

## Playbook 7: Regression Pins (known bugs that must never return)

| File | Tests | What it guards |
|------|-------|----------------|
| `test_memory_regressions.py` | 6 | Mem0 search POST (not GET), user_id format, workspace scoping |
| `test_agent_factory_regressions.py` | 3 | Tool source consistency, execution handle, tool loop |
| `test_document_sync_regressions.py` | 3 | Sync status accuracy, exception logging, S3 error handling |
| `test_workspace_isolation.py` | 5 | Invalid workspace header, stale data, spoofing prevention |

**Total: 17 tests | ~1 min | Schedule: every deploy**

---

## Playbook 8: Contract Tests (API shape validation)

| File | Tests | What it checks |
|------|-------|----------------|
| `test_api_response_contracts.py` | 5 | Agent list/detail, chat history, workspace, health response shapes |
| `test_artifact_schemas.py` | 4 | Coverage gap, QA report, audit domain schemas match PRD-78 |

**Total: 9 tests | ~30 sec | Schedule: every deploy**

---

## Quick Reference: What to Run When

| Scenario | Run These Playbooks | Time |
|----------|-------------------|------|
| **After a deploy** | 1 (Smoke) + 2 (Errors) + 7 (Regressions) + 8 (Contracts) | ~4 min |
| **Nightly** | All playbooks via `run_nightly.py` | ~20 min |
| **Before a release** | All + manual review of user journey results | ~20 min |
| **After infra change** | 1 (Smoke) + 6 (Performance) | ~3 min |
| **After new feature** | 2 (Errors) + 3 (CRUD) + 5 (User Journeys) | ~15 min |
| **Weekly audit** | `run_gap_finder.py` | <1 sec |
