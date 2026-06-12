# PRD-155 — Route Contract & Mount Honesty (WS-13a)

**Chain:** Night 1, stacked on `ralph/prd-154-wave0-quick-wins` tip. Branch `ralph/prd-155-route-contract`. Size **S**.
**Source:** report §2.12 ("single highest-leverage fix"), §4 WS-13.

## Overview

The platform rotted silently because nothing asserts that (a) the frontend only calls endpoints that exist, and (b) every router actually mounts. Global search, skills CRUD, the Interactive Terminal, and analytics Overview all broke this way. This PRD builds the net that protects every subsequent PRD in the chain — it lands before the big remediations on purpose.

## Goals

- CI fails when the frontend references a backend path that doesn't exist.
- Boot fails loudly when a router import/mount fails (today: ~25 routers mount inside `try/except ImportError: pass`; two already fail silently every boot — `main.py:115,123`).
- A reachability test asserts every registered platform tool/action resolves to a live route.

## User Stories

### S1: Backend route manifest
Script dumps `app.routes` (method+path) to `reports/route-manifest.json` via a pytest fixture importing the app — no server boot needed.
**Acceptance:**
- [ ] `python -m scripts.dump_routes` (or pytest collect) produces deterministic sorted manifest
- [ ] Runs in CI without Postgres-dependent side effects (app import must not connect at import time — fix lazily if it does)

### S2: Frontend path extraction + contract test
Statically extract API paths from `lib/api-client.ts`, hooks, and any remaining raw `fetch` (template-literal aware: `${id}` → path-param wildcard). Contract test: extracted paths ⊆ manifest (method-aware where extractable). Known-dead paths are NOT allowlisted — they were deleted or fixed in PRD-154; anything still failing is a real finding to fix here.
**Acceptance:**
- [ ] `npm run test:contract` (vitest) fails on a fabricated path (negative fixture test)
- [ ] Suite passes on the PRD-154 tip — every remaining mismatch fixed or its caller deleted in this story (no suppression file)

### S3: Startup mount assertions
Replace silent `try/except` router mounts in `main.py` with a manifest of expected routers; boot raises on any import/mount failure (env `ALLOW_DEGRADED_BOOT=true` escape hatch for emergencies, default off — read via `config.py`).
**Acceptance:**
- [ ] Test: simulated router ImportError fails app startup with the router name in the message
- [ ] The two currently-failing imports (`main.py:115,123`) are fixed or their routers deliberately deleted (decision recorded in PR description)
- [ ] `pytest -q` green

### S4: Tool reachability test
For every action in the registry, assert its route exists in `unified_executor.tool_routes` and a smoke invocation of the dispatch layer resolves a handler (no LLM, no external calls).
**Acceptance:**
- [ ] Test enumerates the registry; stale tool names (e.g. the `workspace_file_read` vs `workspace_read_file` class of drift) fail with a diff-style message
- [ ] Green on PRD-154 tip

### S5: CI wiring
Add contract + reachability jobs to `.github/workflows/test.yml` (same Postgres service pattern, per-test timeout). Non-required initially, mirroring the Wave-2 test-net precedent; flip-to-required noted as the repo admin's follow-up.
**Acceptance:**
- [ ] Both jobs run on push; red on seeded violation branch, green on chain tip

## Non-Goals

Dead-code deletion sweep (PRD-168), api-client split (PRD-168), honest-metrics beyond what PRD-154 fixed.

## Success Metrics

- A PR adding a frontend call to a non-existent endpoint cannot pass CI.
- `grep -rn "except ImportError" orchestrator/main.py` returns nothing.

## Testing

New: `orchestrator/tests/test_route_contract.py`, `test_router_mounts.py`, `test_tool_reachability.py`; frontend `tests/contract/`. Full suite + lint/typecheck green per chain policy.
