# Phase 1: Pilot Readiness — Task List

**PRD:** 78 (Autonomous Test Coverage & Quality Mesh)
**Goal:** Flush bugs, pin regressions, deepen P0 coverage for 10-15 pilot users
**Baseline:** 124 API tests, 0 regression-pin tests, 0 contract tests, 0 unit tests
**Target:** ~160-200 tests with all P0 areas covered by stateful journeys

---

## 1. Infrastructure & Cleanup

### 1.1 Create directory structure
- [ ] `tests/regressions/` — dedicated regression-pin tests
- [ ] `tests/regressions/conftest.py` — shared fixtures
- [ ] `tests/contracts/` — artifact schema validation
- [ ] Update `audit_suite.py` EXPECTED_DOMAINS to include new directories

### 1.2 Test data cleanup strategy
- [ ] Add teardown fixtures that delete test-created resources (agents, chats, personas, channels, rules, keys, recipes) — **partially done** via session-scoped cleanup registries in conftest.py
- [ ] Add a dedicated test workspace ID (or create-on-demand) so pilot user data is never touched
- [ ] Document cleanup strategy in tests/README.md

### 1.3 Runner updates
- [ ] Add `tests/regressions/` to `run_health_regression.py` TARGETS
- [ ] Add `tests/contracts/` to `run_nightly.py` scope
- [ ] Verify `qa-report.json` artifact includes `source_files` for all failures (self-contained for Jira Admin)

---

## 2. Regression Pins (known bugs → tests)

These are bugs discovered during development that MUST stay fixed. Each gets a dedicated test.

### 2.1 Memory system regressions
- [ ] `test_mem0_search_uses_post.py` — Mem0Client.search() must use POST /search/, not GET list endpoint (fix-memory branch)
- [ ] `test_memory_stats_user_id_format.py` — api/memory_stats.py must use `ws_{id}` format, not `ws_{id}_agent_global`
- [ ] `test_memory_scoping_isolation.py` — memory queries must be scoped to workspace, no cross-workspace leakage

### 2.2 Multi-tenancy regressions
- [ ] `test_workspace_isolation.py` — new users must NOT see other workspace data (the dev-fallback bug)
- [ ] `test_workspace_header_validation.py` — X-Workspace-ID header must be validated against user membership (anti-spoofing)

### 2.3 Cloud document sync regressions
- [ ] `test_document_sync_status_accuracy.py` — cloud_documents.sync_status must reflect actual documents.status, not just document_id existence
- [ ] `test_document_processing_error_propagation.py` — S3 Vectors backend must raise exceptions, not return empty arrays

### 2.4 AgentFactory regressions
- [ ] `test_agent_tool_source_consistency.py` — all execution paths must use `get_tools_for_agent()`, never `_build_tool_schemas()`
- [ ] `test_agent_execute_tool_loop.py` — tool loop must support max_tool_iterations (was single-shot)

### 2.5 Existing embedded regressions (already in api/ — verify still passing)
- [ ] Verify `test_channel_analytics_source_query` still catches the SQL column bug
- [ ] Verify `test_create_recipe_with_null_created_by` still validates NOT NULL
- [ ] Verify `test_llm_settings_categories_exist` still validates all service categories
- [ ] Verify `test_invalid_schedule_type` returns 400 not 500

---

## 3. Deepen P0 Stateful Journeys

These are the critical paths pilot users will exercise. Each needs a full create → use → verify → cleanup flow.

### 3.1 Chat (currently 9 tests — good, needs error paths)
- [ ] `test_chat_with_invalid_agent` — chat against non-existent agent returns proper error
- [ ] `test_chat_empty_message` — empty message handled gracefully
- [ ] `test_chat_concurrent_messages` — two messages to same chat don't corrupt state
- [ ] `test_chat_history_pagination` — verify pagination params work on history endpoint

### 3.2 Memory (currently 4 tests — stats only, no mutations)
- [ ] `test_memory_store_and_search` — store a memory → search for it → verify found
- [ ] `test_memory_store_and_recent` — store → verify in recent list
- [ ] `test_memory_search_workspace_scoped` — search only returns current workspace memories

### 3.3 Workflows (currently 3 smoke tests — no execution journey)
- [ ] `test_workflow_create_execute_status` — create workflow → execute → poll status → verify completion
- [ ] `test_workflow_cancel` — execute → cancel → verify cancelled state
- [ ] `test_workflow_execution_history` — execute → verify appears in execution list

### 3.4 Heartbeat (currently 5 smoke tests — no lifecycle)
- [ ] `test_heartbeat_enable_run_verify` — enable heartbeat on agent → trigger → verify results stored
- [ ] `test_heartbeat_disable` — enable → disable → verify not running
- [ ] `test_heartbeat_response_shape_for_recipes` — verify heartbeat_results JSONB shape matches what recipes expect

### 3.5 Agents (currently 8 tests — CRUD good, needs tool assignment)
- [ ] `test_agent_assign_tools_and_verify` — create agent → assign tools → verify tools in agent config
- [ ] `test_agent_assign_skills_and_verify` — create agent → assign skill → verify
- [ ] `test_agent_execute_with_tools` — create agent → assign tools → execute prompt → verify tool use

### 3.6 Documents (currently 3 smoke tests — no upload journey)
- [ ] `test_document_upload_process_search` — upload file → wait for processing → search content → verify found
- [ ] `test_document_delete` — upload → delete → verify removed from list and search

### 3.7 Recipes (10 tests in recipe_cron, good — need execution verification)
- [ ] `test_recipe_execute_and_verify_output` — create recipe → execute → verify output artifact exists
- [ ] `test_recipe_schedule_fires` — create cron recipe → verify next execution scheduled

---

## 4. Contract Tests

### 4.1 Artifact schema validation
- [ ] `test_qa_report_schema.py` — validate qa-report.json matches the schema in PRD-78 section 6.2
- [ ] `test_health_regression_summary_schema.py` — validate health-regression-summary.json shape
- [ ] `test_coverage_gap_summary_schema.py` — validate coverage-gap-summary.json shape

### 4.2 API response contracts
- [ ] `test_agent_response_contract.py` — agent list/detail response shape is stable
- [ ] `test_chat_sse_contract.py` — SSE stream events follow AI SDK format

---

## 5. Internal Tool Validation

### 5.1 Platform executor
- [ ] `test_platform_tool_routing.py` — platform_* calls route to PlatformActionExecutor
- [ ] `test_workspace_tool_routing.py` — workspace_* calls route to WorkspaceClient
- [ ] `test_composio_tool_routing.py` — composio_execute routes to Composio SDK

---

## Estimated Test Counts

| Category | New Tests | Running Total |
|----------|--------:|--------:|
| Existing API tests | — | 124 |
| Regression pins (section 2) | ~13 | 137 |
| Deepened journeys (section 3) | ~22 | 159 |
| Contract tests (section 4) | ~5 | 164 |
| Internal tool tests (section 5) | ~3 | 167 |
| **Total** | **~43** | **~167** |

This puts us in the 160-200 range. Additional tests will come from bugs found during implementation.

---

## Priority Order

1. **Regression pins** (section 2) — these are known bugs, highest value per test
2. **Chat + Memory + Workflow journeys** (sections 3.1-3.3) — most exercised by pilot users
3. **Runner updates** (section 1.3) — ensure new tests are included in health regression runs
4. **Heartbeat + Agent journeys** (sections 3.4-3.5) — core platform differentiation
5. **Document upload journey** (section 3.6) — pilot users will upload docs
6. **Contract tests** (section 4) — locks artifact schemas for agent handoff
7. **Internal tool tests** (section 5) — validates the execution plumbing

---

## Definition of Done

- [ ] All P0 domains have at least one stateful journey test
- [ ] All known regressions from memory system have pin tests
- [ ] `run_health_regression.py` includes regression tests
- [ ] `qa-report.json` is self-contained (includes source_files for all failures)
- [ ] Health regression completes in < 8 minutes
- [ ] No test leaves behind orphaned test data
