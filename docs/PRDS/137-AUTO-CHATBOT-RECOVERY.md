# PRD-137 - Auto / Chatbot Recovery

**Status:** Review required before more implementation  
**Type:** Bug fix / architectural recovery  
**Owner:** Platform  
**Last updated:** 2026-04-30  
**Reviewer:** CODEX  

---

## Purpose

This PRD exists because Auto, the per-workspace orchestrator chat agent, regressed after the Shopify chatbot work and PRD-136 LLM tier split. The regression is not one bug. It is a pile-up of small assumptions in model selection, prompt assembly, memory enrichment, tool-loop control, and tool exposure.

The goal is to restore Auto as the workspace brain without adding another layer of patches. Every change in this PRD must either:

- remove a proven wrong assumption,
- delete a dead path,
- add a regression test around a proven bug, or
- produce evidence for an unproven hypothesis before coding the fix.

Important: a previous agent began implementing this PRD before review. Treat that worktree as a partial patch under review, not as accepted implementation.

---

## Current Code Review Snapshot

As of 2026-04-30, the worktree already contains partial changes for fixes #1, #2, #3, #5, #6, and #7.

| Area | Current state | Review verdict |
|---|---|---|
| #1 stale settings overwrite | `_STALE_FIXES` block removed from `seed_system_settings.py` | Directionally correct. Needs regression verification. |
| #2 `use_system_llm` rename | Renamed to `use_orchestrator_llm` in main chat, voice, service, factory | Directionally correct. Confirm zero runtime callers still use old kwarg. |
| #3 identity injection | `_inject_agent_identity` removed; `IdentitySection._build_chatbot_identity` now appends description/persona | Plausible. Needs tests for Shopify widget and Auto/CTO. |
| #5 skills cap | `SkillsSection.max_tokens` set to `None`; auxiliary skills hard-capped at 5000 | Fixes primary truncation. Does not implement configurable cap from earlier draft. |
| #6 Mem0 gating | API key gate and circuit breaker added | Incomplete: HTTP 5xx/429 responses are currently treated as breaker success unless fixed. |
| #7 retry limits | prefix defaults added for `platform_*` and `workspace_*` | Plausible. Needs unit tests and alignment with second loop-prevention cap. |
| #4 legacy stream path | Not implemented | Still live. Must delete or explicitly deprecate with proof. |
| #8 workspace tools | Not implemented | Previous root cause was unproven. Reframe as investigation. |
| #9 context sections | Not implemented | Previous root cause was wrong. Reframe as instrumentation/investigation. |

Generated/noisy files present in the worktree are out of scope for this PRD: `frontend/tsconfig.tsbuildinfo`, `frontend/graphify-out/`, and `orchestrator/graphify-out/`.

---

## Model Tier Contract

PRD-136 created two relevant LLM tiers:

| Tier | Settings key | Used for |
|---|---|---|
| Auto / Orchestrator | `system_settings.orchestrator_llm.*` plus Auto agent `model_config` | User-facing Auto chat orchestration |
| System | `system_settings.system_llm.*` | Internal services such as codegraph, RAG, NL2SQL, memory, and routing |

Auto on GPT-5.5 and System on Gemini Flash is valid. Any code that says it is using the "system LLM" for Auto should be renamed or corrected, because "system" is now a real independent tier.

---

## Recovery Plan

| # | Work item | Status | Severity |
|---|---|---|---|
| 1 | Remove stale orchestrator settings overwrite | Partially implemented | P0 |
| 2 | Rename misleading `use_system_llm` flag | Partially implemented | P1 |
| 3 | Make identity injection single-owner | Partially implemented | P1 |
| 4 | Delete legacy `stream_response` method | Not started | P2 |
| 5 | Stop truncating Auto's primary skill | Partially implemented | P0 |
| 6 | Gate Mem0 and make circuit breaker correct | Partially implemented, needs fix | P0 |
| 7 | Add platform/workspace tool-loop ceilings | Partially implemented | P1 |
| 8 | Investigate workspace tool exposure contract | Investigation only | P1 |
| 9 | Instrument CHATBOT context section assembly | Investigation only | P1 |

---

## Fix #1 - Remove Stale Settings Overwrite

### Problem

`seed_system_settings()` had a "one-time cleanup" block that rewrote existing `orchestrator_llm.provider` and `orchestrator_llm.model` values back to defaults on every container boot.

That defeats the Settings UI. If a user saves `openai/gpt-5.5`, the next boot can restore `google/gemini-2.5-flash`.

### Fix

Delete the stale-fix block entirely. Seed code may update setting metadata, but must never overwrite an existing `SystemSetting.value`.

### Current Code Review

The partial implementation has removed this block and left a PRD-137 comment. This is acceptable if tests or manual verification confirm:

- existing `value` is preserved,
- `default_value`, descriptions, and validation metadata can still be updated,
- a restart does not mutate `system_settings.orchestrator_llm.model`.

### Verification

- Add or update a unit test for `seed_system_settings()` preserving existing values.
- Manual: save Auto model in Settings, restart, confirm the value remains unchanged and chat logs show the selected model.

---

## Fix #2 - Rename `use_system_llm` to `use_orchestrator_llm`

### Problem

The flag name `use_system_llm` is now wrong. In the Auto chat path, `True` means "use `orchestrator_llm` settings," not the System tier.

### Fix

Rename the parameter and all callers to `use_orchestrator_llm`.

### Current Code Review

The partial implementation updates:

- `orchestrator/api/chat.py`
- `orchestrator/api/chat_voice.py`
- `orchestrator/consumers/chatbot/service.py`
- `orchestrator/modules/agents/factory/agent_factory.py`

`api/widgets/chat.py` does not currently pass the flag. That may be correct if widget agents should use their own model config. Do not add the flag there without a product decision.

### Verification

- `rg "use_system_llm" orchestrator --glob '*.py'` returns zero code matches.
- Existing chat, voice, route-to-agent, and Auto fallback tests pass.
- Logs clearly distinguish "agent model_config" from "orchestrator tier settings."

---

## Fix #3 - Make Identity Injection Single-Owner

### Problem

The old chatbot service injected agent description/persona as a second system message after ContextService built the system prompt. That can duplicate agent identity and cause widget agents to echo greetings or personas twice.

### Intended Ownership

`IdentitySection` owns identity content. `StreamingChatService` should not append a second identity system message.

### Fix

- Remove `_inject_agent_identity()` from `StreamingChatService`.
- Add agent `description` and persona text to `IdentitySection._build_chatbot_identity()`.
- Keep CTO/Auto soul override behavior separate. CTO prompt replacement remains owned by `_apply_cto_override()`.

### Current Code Review

The partial implementation follows this direction. It still needs tests because identity changes are easy to regress:

- Shopify/widget chatbot agent with description "Hello! Welcome..." should include the description once.
- Chatbot mode with custom persona should include persona once.
- Auto/CTO path should not get duplicate identity after the soul prompt override.
- ATOM path should be checked separately because it does not use the full ContextService prompt.

### Verification

- Add tests around `IdentitySection.render(... personality=True ...)`.
- Add a service-level test or fixture confirming `_inject_agent_identity` is not called/defined.
- Manual widget smoke test: greeting appears once.

---

## Fix #4 - Delete Legacy `stream_response`

### Problem

`StreamingChatService.stream_response()` is a legacy method that still constructs an LLM manager with `service_name="chatbot"`. Under PRD-136 this maps to the System tier, not the Auto/Orchestrator tier.

Current review shows this method still exists.

### Fix

Delete the legacy method and the stale example comment in `consumers/chatbot/integration.py`.

### Verification

- `rg "stream_response\\b" orchestrator --glob '*.py'` returns no legacy method/caller matches, excluding `stream_response_with_agent`.
- App boot and chat tests pass.

---

## Fix #5 - Stop Truncating Auto's Primary Skill

### Problem

`SkillsSection.max_tokens = 3000` capped every skill section. Auto's `platform-management` skill is much larger, so Auto could lose most of its operating manual before prompt assembly.

### Fix

Primary skill should not be section-capped. If an agent has multiple active skills, the primary skill is highest priority and auxiliary skills may share a bounded budget.

### Current Code Review

The partial implementation:

- sets `SkillsSection.max_tokens = None`,
- sorts active skills by priority,
- renders the first skill uncapped,
- caps auxiliary skill text at `aux_max_tokens = 5000`.

This is acceptable as a first recovery patch. Do not claim configurable settings support unless it is actually implemented.

### Follow-up Option

Later, add `orchestrator_llm.skills_aux_max_tokens` and/or `system_llm.skills_aux_max_tokens` if product needs runtime tuning.

### Verification

- Unit test: one large active skill is not truncated.
- Unit test: auxiliary skills are truncated to the configured class budget.
- Render Auto prompt and confirm the back half of `platform-management` appears.

---

## Fix #6 - Gate Mem0 and Correct the Circuit Breaker

### Problem

Mem0 is enrichment. It must not block chat startup when the API key is absent or the service is unhealthy.

### Fix

- If `MEM0_API_KEY` or `MEM0_API_URL` is missing, disable Mem0 calls immediately.
- Use a short timeout, default 3 seconds.
- Add a process-local circuit breaker.
- Record breaker failures for timeouts, connection errors, and transient HTTP failures.

### Required Correction To Current Patch

The partial implementation currently calls `_breaker.record_success()` immediately after `requests.request(...)` returns. That treats HTTP `401`, `429`, and `5xx` responses as circuit-breaker success.

Required behavior:

| Response | Breaker behavior | Retry behavior |
|---|---|---|
| 2xx/3xx | success | no retry |
| 400/401/403/404 | no retry; log as config/client error | do not open breaker unless product chooses otherwise |
| 429 | failure | retry once, then breaker failure |
| 5xx | failure | retry once, then breaker failure |
| timeout/connection error | failure | retry once, then breaker failure |

### Verification

- No API key: Mem0 methods return immediately and chat does not wait.
- Fake 500s: breaker opens after threshold.
- Fake 401: no timeout loop; clear log message.
- Valid response: breaker closes on success.

---

## Fix #7 - Add Platform/Workspace Tool-Loop Ceilings

### Problem

Auto can loop across platform or workspace tools. There are two loop controls:

- `ToolExecutionTracker`, which can skip execution before the tool runs.
- `_inject_loop_prevention`, which injects "stop calling this" guidance after repeated calls.

The two controls must agree.

### Fix

Add prefix-aware limits:

- `platform_*`: low cap, because most platform lookups are definitive.
- `workspace_*`: higher cap, because file/grep/git workflows may legitimately chain.

### Current Code Review

The partial implementation adds:

- `platform_default = 2`
- `workspace_default = 5`
- `_resolve_limit()`

This is plausible, but tests must confirm the exact boundary conditions. Also verify whether `platform_execute` should get a separate limit, because many platform/workspace actions are executed through the dispatcher as `platform_execute(action=...)`.

### Verification

- Unit test direct tool names: `platform_get_settings`, `workspace_grep`.
- Unit test dispatcher form: repeated `platform_execute` with same action and params.
- Integration trace: typical Auto platform question finishes within the configured max iterations.

---

## Investigation #8 - Workspace Tool Exposure Contract

### Previous Claim To Remove

The earlier draft claimed "10 workspace tools never reach the chat tool list." Current code review does not prove that.

The repo defines seven workspace actions:

- `workspace_read_file`
- `workspace_write_file`
- `workspace_list_dir`
- `workspace_grep`
- `workspace_exec`
- `workspace_html_to_png`
- `workspace_git`

These are registered by `register_workspace_actions()` through `register_all_actions()`.

### Real Question

Are workspace actions supposed to be directly callable as first-class tools, or called through `platform_execute(action="workspace_read_file", params={...})`?

Current code suggests:

- promoted actions get first-class schemas,
- non-promoted actions live behind the `platform_execute` dispatcher,
- `workspace_html_to_png` is promoted,
- most workspace actions are not promoted.

The `platform-management` skill examples should match the actual calling contract. If the model sees examples like `{ "tool": "workspace_read_file" }` but only has a `platform_execute` schema, Auto may appear to "miss" tools even though the registry is working.

### Investigation Tasks

- Dump the actual OpenAI `tools` array for an Auto chat.
- Dump the `platform_execute.action` enum.
- Compare both against `platform-management/SKILL.md`.
- Decide whether to promote more workspace actions or rewrite skill examples to use `platform_execute`.

### Acceptance Criteria

- One documented contract for workspace actions.
- Skill examples match the contract.
- Tests or startup probe confirm all expected workspace actions are registered.

---

## Investigation #9 - CHATBOT Context Section Assembly

### Previous Claim To Remove

The earlier draft said sections were skipped because `applies_to(mode)` was wrong. Current code does not use an `applies_to()` mechanism.

CHATBOT declares 10 sections in `modules/context/modes.py`:

- `identity`
- `onboarding`
- `skills`
- `composio`
- `plugins`
- `platform_actions`
- `memory`
- `business_graph`
- `datetime_context`
- `conversation`

`ContextService` instantiates that list and renders each section. If only some appear in the final prompt, the cause is likely one of:

- section rendered empty because no data existed,
- section render failed and returned empty,
- token budget dropped the section,
- a later prompt override replaced the prompt,
- the inspected path was ATOM mode, which bypasses full ContextService.

### Investigation Tasks

- Add temporary or test-only instrumentation that records per-section: rendered, empty, token estimate, dropped, exception.
- Dump `ContextResult.sections_included` and `sections_trimmed` for Auto CHATBOT mode.
- Compare full path vs ATOM path.
- Confirm CTO soul override behavior separately.

### Acceptance Criteria

- A test asserts expected CHATBOT sections under a seeded fixture.
- Empty sections are understood and documented.
- No code fix is written until the missing-section cause is proven.

---

## Rollout Order

Each item should land in a small reviewable commit. Do not batch all fixes.

1. Stabilize or revert the partial worktree so only one fix is under review at a time.
2. Fix #1 with test.
3. Fix #6 fully, including HTTP status breaker behavior, with tests.
4. Fix #4 by deleting the dead method.
5. Fix #5 with skill truncation tests.
6. Fix #7 with retry-limit tests.
7. Fix #3 with identity tests and Shopify widget smoke verification.
8. Fix #2 rename, or keep the partial rename if all callers/tests pass.
9. Run Investigation #8.
10. Run Investigation #9.

Fixes #8 and #9 must not begin as implementation work. They begin as evidence gathering.

---

## Code Review Checklist

- Root cause is proven by local code, logs, or a failing test.
- The fix removes a wrong assumption rather than masking a symptom.
- No dead compatibility path remains unless explicitly justified.
- No direct `os.getenv()` outside `config.py` for new config.
- Settings UI remains source of truth for `orchestrator_llm.*`.
- Auto / System / Embeddings tier separation is preserved.
- Tests cover every changed behavior.
- Generated artifacts are not included in the PR unless intentionally produced.

---

## Acceptance Criteria

After validated fixes ship:

1. Saving Auto model to `openai/gpt-5.5` survives restart.
2. Auto chat logs show the selected orchestrator model.
3. Chat startup does not wait on Mem0 when no key is configured.
4. Mem0 breaker opens on transient failures and does not treat 5xx as success.
5. Auto's primary skill renders without the old 3000-token cap.
6. Shopify/widget agent identity appears once.
7. Legacy `stream_response()` is gone.
8. Tool-loop limits have tests for platform, workspace, and dispatcher paths.
9. Workspace tool exposure contract is documented and reflected in the skill.
10. CHATBOT section inclusion is measured before any section-assembly fix is attempted.

---

## Non-Goals

- Shipping a full Mem0 product rollout.
- Reworking cost attribution.
- Renaming every `service_name` concept in `LLMManager`.
- Changing widget model-selection behavior without a separate product decision.
- Promoting all workspace actions without evidence that first-class schemas are needed.
