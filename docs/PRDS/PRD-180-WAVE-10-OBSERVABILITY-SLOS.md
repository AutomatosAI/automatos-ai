# PRD-180: Wave 10 — Observability and SLOs

**Phase:** D — Enterprise hardening (weeks 24–32)
**Branch:** `feat/w10-observability-slos` · **Worktree:** `automatos-ai-prd180`
**Dependencies:** Wave 1 (spine) — **merged to main (`557857576`)**
**Build size:** M · **Risk:** Low
**OS Review refs:** §5, §7 (Command Centre reality), §12.7, roadmap Phase D, §13 pass/fail "Observability/SLOs" bar

---

## Overview

The UI misrepresents system state and real-time push does not exist — "Streaming live" is decorative while every ops surface polls on 8–60s intervals and the board SSE has zero subscribers. This wave makes observability honest: replace polling with a real `LISTEN/NOTIFY`-backed SSE, delete the fabricated metrics and placebo model selector, surface `composio_execute` in the running/error indicators, and define three tracked SLOs.

**Owner decision (locked 2026-07-03): DELETE both honesty zombies** — the fabricated sidebar stats and the placebo model selector. A control that visibly does nothing corrodes trust in every real control.

---

## Ownership boundary (parallel-safe)

Runs concurrently with W12 (PRD-182, CI/test bar). They share **zero files** — W10 is frontend components + the board SSE endpoint; W12 is CI configs.

- **W10 OWNS:** `orchestrator/api/board_tasks.py` (SSE endpoint), `frontend/components/command-center/*`, `frontend/components/layout/studio-sidebar.tsx`, `frontend/components/chatbot/message.tsx`, `frontend/components/chatbot/chat-page-content.tsx` + the model-selector UI (`agent-selector.tsx`), and any new SLO surface.
- **W10 MUST NOT TOUCH:** `.github/workflows/*`, `orchestrator/pytest.ini`, `orchestrator/requirements.txt`, `frontend/next.config.js` (W12 owns the CI/gating surface).

Note: W12 will add a `tsc` CI gate baselined to today's error count. Keep your changed `.tsx` files type-clean (run `tsc --noEmit` on them) so you don't add to the baseline.

---

## Findings & Scope

| Finding | Issue (verified) | Fix |
|---|---|---|
| **F090** | `board_tasks.py:426 stream_board_events` is a timed ping (not `LISTEN/NOTIFY`); zero frontend subscribers; Command Centre polls (`command-center-shell.tsx`) | Real `LISTEN/NOTIFY`-backed SSE + a frontend subscription that replaces the poll |
| **F038** | Fabricated sidebar metrics render by default (`studio-sidebar.tsx` — `tick 5s`, `$/dec $0.0027`, `cache 68%`, `v0.11`) | **Delete** the fabricated stats block |
| **F035** | Placebo model selector — `chat-page-content.tsx:71` hardcodes `initialChatModel="gpt-4"` and the picker doesn't switch the real model | **Delete** the non-functional selector; chat uses the real configured default (`LLM_DEFAULTS`) |
| **F037** | `composio_execute` is name-filtered out of running/error indicators (`message.tsx:~278-280`) → every external-app action is invisible while it runs | Un-filter `composio_execute` so it shows a running chip + error chip |
| **SLOs** | No tracked SLIs/dashboards | Define + expose **three** SLOs |

---

## Stories (test-first where testable; frontend via vitest + tsc)

### S1 · Real-time board SSE via LISTEN/NOTIFY (F090) — M
**Files:** `orchestrator/api/board_tasks.py` (`stream_board_events`), the board-task write path (emit `NOTIFY` on insert/status-change), `frontend/components/command-center/command-center-shell.tsx` (subscribe, drop the poll).
**Backend test:** `test_board_sse_listen_notify` asserts the stream yields an event when a board task changes status (NOTIFY path), not merely a timed ping. Mock the PG connection's notify channel at the boundary.
**Frontend test:** `command-center-shell` subscribes to the SSE and stops polling — assert via vitest (the poll interval is removed / the EventSource is wired). 
**Notes:** Emit `NOTIFY board_events, <payload>` on board-task insert/status transitions; the SSE `LISTEN`s and forwards. Keep a heartbeat comment for connection liveness but drive real events off NOTIFY. Remove the polling interval in the shell once subscribed.

### S2 · Delete the fabricated sidebar stats (F038) — S
**Files:** `frontend/components/layout/studio-sidebar.tsx`.
**Test:** `studio-sidebar` vitest asserts the hardcoded `tick/cache/$dec/v0.11` block is gone (no fabricated literals rendered).
**Notes:** Delete the block and any dead props/state feeding it. Do not replace with real metrics in this story (that would need real sources) — just remove the lie. If a real metric is trivially available (e.g. a genuine version string from config), it may stay only if truthful.

### S3 · Delete the placebo model selector (F035) — S
**Files:** `frontend/components/chatbot/chat-page-content.tsx:71`, the selector UI (`frontend/components/chatbot/agent-selector.tsx`), `chat.tsx`.
**Test:** vitest asserts no hardcoded `"gpt-4"` initial model and no non-functional selector; chat initialises from the real default (`LLM_DEFAULTS.model_id`).
**Notes:** Remove the hardcoded `initialChatModel="gpt-4"` and the selector control that doesn't drive the backend. Delete what you remove (dead state/handlers). The backend already ignores `selectedChatModel` — so removing the control is the honest fix, not wiring a lie.

### S4 · Surface `composio_execute` in running/error indicators (F037) — S
**Files:** `frontend/components/chatbot/message.tsx:~278-280`.
**Test:** vitest asserts a message with a `composio_execute` tool call renders a running indicator (and an error chip on failure), i.e. it is no longer filtered out.
**Notes:** Remove the name-filter that excludes `composio_execute`; show the resolved action (W7 now logs the real action name) in the chip where available.

### S5 · Three SLOs, defined and dashboarded — M
**Files:** a new SLO module/endpoint (reuse the existing metrics/monitoring surface if one exists — grep `handlers_monitoring`/`actions_monitoring` first), a small dashboard surface.
**Deliverable:** define **three** concrete SLIs with targets and expose them (e.g. tool-call success rate, board-task dispatch latency p95, SSE event-delivery freshness). Pick three that are measurable from existing telemetry (`ToolExecutionLog`, board tasks). Document target + measurement window.
**Test:** `test_slo_metrics` asserts each SLI computes from seeded telemetry and returns a value + target.
**Notes:** Reuse existing monitoring plumbing (there is a monitoring handler/actions surface already — do not build a parallel metrics stack). Keep it honest — real numbers only.

---

## Verification (NO servers, NO dev-browser — hard rule)

Frontend: `tsc --noEmit` on changed files, `vitest` for the component tests, `eslint` on changed files. **Never** run `next dev`, `next start`, or a headless browser (it kills the user's Chrome). `next build` is acceptable only if quick and needed; prefer tsc.
Backend: `python -m py_compile` + pure pytest (mock the PG NOTIFY channel).

```
# frontend (from frontend/)
npx tsc --noEmit <changed files> ; npx vitest run <changed test files> ; npx eslint <changed files>
# backend
python -m py_compile orchestrator/api/board_tasks.py
python -m pytest orchestrator/tests -k "board_sse or slo" -q
```

## Conventions (see automatos-ai/CLAUDE.md)
- No `os.getenv()` outside `config.py`; no backward-compat shims — **delete** the zombies fully (dead state, props, handlers, imports); immutable patterns; reuse the existing monitoring surface (no parallel metrics stack); orange-*→warning tokens if you touch styles.
- Commit to `feat/w10-observability-slos` in conventional commits (feat(prd-180): ...). **Do not push or open a PR.**

## Success metrics
- Board SSE pushes real events via LISTEN/NOTIFY; Command Centre no longer polls.
- Zero fabricated UI metrics; no placebo model selector.
- `composio_execute` visible in running + error indicators.
- Three SLIs computed from real telemetry with targets + a dashboard surface.
