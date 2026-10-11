# Ralph Build Prompt — PRD-258 Auto Live, back on track, Wave 1 (measure the voice turn)

You are executing **PRD-258 Wave 1**, ONE story per session, unattended: every story gets a fresh session. Branch **`feat/prd-258-w1-voice-measure`** ← `main`, in its own worktree. One PR for the whole wave, one CI run at the end: nothing in a story waits on CI.

**Why this wave exists (owner, 11 Oct): Auto Live is "just slow", and nobody has a number.** Prod logs hold no voice lines in 30 days, the local `voice_turns`/`voice_calls` tables are empty, and `voice_turns.total_ms` stops the clock at the END of the reply. Every spoken turn rebuilds the whole text-chat turn before the model is called (PRD-258 §1). This wave builds the instrument only: per-turn timings, Retell's own per-call latency, arming on the local stack through a tunnel, and a simulator that times the turn without a person on a call. Making it faster is Wave 2, after PRD-256 merges. Do not speed anything up in this wave.

Wave 1 builds, in the JSON's priority order:
1. **US-001** move helpers out of the oversized `voice_retell.py`; ONE migration; per-turn timings in `voice_turns`.
2. **US-002** Retell's per-call latency in `voice_calls.vendor_latency`.
3. **US-003** one `PUBLIC_API_HOST`; a re-arm re-points the Retell agent; the local agent's name; the ngrok tunnel README and script.
4. **US-004** the voice-latency runner in `orchestrator/scripts/eval/voice_latency/`.

## Read first, every iteration

1. `scripts/ralph/prd-258w1.json` is the BINDING contract. Its **`decisions`** are binding (D2 one migration; D3 ngrok; D4 nothing runs here; D5 one PR per wave; D6 `voice_retell.py` never grows).
2. `docs/PRDS/PRD-258-AUTO-LIVE-VOICE-LATENCY.md` is the spec (US-001..US-004, FR-1..FR-3, §7).
3. `CLAUDE.md` and `AGENTS.md`, which it imports: reuse over build, delete what you replace, no `os.getenv` outside `config.py`, no hardcoded values, migrations chain onto the single head and survive `create_all`.

## The code this wave builds on (RE-VERIFY each by grep before building on it)

- The voice lane: `orchestrator/api/voice_retell.py` (1,005 lines — `mint_web_call`, `_apply_call_event`, `retell_events_webhook`, `arm_voice_live`, `_agent_retell_stream` (the per-turn worker thread), `_agent_retell_stream_inner` (binding, `stream_response_with_agent(spoken_mode=True)`, `record_voice_turn`), `retell_llm_websocket` with its `respond()` closure (the one-time 'One moment.' ack, `voice_live_ws_first_frame` log)).
- `orchestrator/modules/voice/`: `providers/retell.py` (`retell_response_frames`, `extract_agent_text`, the speech units, `parse_llm_request`, `verify_webhook_signature`), `retell_api.py` (`build_agent_tuning`, `create_custom_llm_agent`, `update_agent`, `create_web_call`, the URL constants), `telemetry.py` (`voice_turn_fields`, `record_voice_turn`), `call_binding.py` (`resolve_call_binding`, `stamp_assistant_voice_source`, `create_voice_chat`), `live_settings.py`, `voice_meter.py`.
- Models: `orchestrator/core/models/voice_turns.py`, `voice_calls.py`; `messages.context_trace` in `orchestrator/core/models/core.py` (~L1259); where the trace is built: `consumers/chatbot/smart_orchestrator.py` (`to_assembly_trace`).
- The stream format: `orchestrator/consumers/chatbot/streaming.py` (`StreamingHandler`, the AI-SDK line prefixes).
- Config: `orchestrator/config.py` (`PUBLIC_API_HOST` is defined twice, ~L900 and ~L1939; the `VOICE_LIVE_*` block ~L1829–L1935; `AUTH_EDITION`).
- Migrations: `orchestrator/alembic/versions/`, the head pins in `orchestrator/tests/test_prd209_alembic_single_head.py` and `orchestrator/tests/test_prd236_w1_routes.py`.
- Tests to copy patterns from: `orchestrator/tests/test_prd207_voice_live.py`, `test_prd203_voice_retell.py`, `test_voice_quality.py`.
- The eval layout to copy: `orchestrator/scripts/eval/tool_routing/`.

## Code rules (AGENTS.md; CI checks them on every line you change)

- **Functions:** a Python function you touch is at most **50 code lines** and nests at most **4 levels**; cyclomatic complexity at most 10. Touching an existing function already over a limit means splitting it, not adding to it. `retell_llm_websocket` is long: if you touch it, extract, don't add.
- **Files:** a new file at most 800 lines (aim 200–400). **`voice_retell.py` never ends a story longer than on the base (D6).** Move code out to make room.
- **Config only via `orchestrator/config.py`**; a new name goes in `orchestrator/reports/config-surface.json` in sorted position. No `os.getenv` / `os.environ` elsewhere.
- **No hardcoded values:** URLs, names, paces and timeouts are named constants or config.
- **Errors:** never swallow — `logger.exception` / a logged warning with the call id, then a clear failure or the documented fallback. Telemetry never fails a voice turn.
- **Routes:** no new route in this wave. The route manifest does not change; if it would, stop and say why.
- **Both editions** keep working; the local-only agent name is gated by `AUTH_EDITION`, never by role.
- **ONE migration (D2),** in US-001 only.

## The execution contract

- **This session ends when your turn ends.** Nothing reports back later.
  - Never end your turn while anything runs in the background (an agent, a background shell).
  - Never use ScheduleWakeup, and never reply that you will "pick up when it reports".
  - **A code review runs in the FOREGROUND, before the commit,** only for **US-001** (the migration) and **US-003** (arming touches credentials and outbound URLs); use the code-reviewer agent (database-reviewer too for US-001's migration) and wait for it.
  - Commit as soon as the story's code and tests are written.
- **RE-VERIFY every anchor by grep before building on it.** If an anchor moved, adapt and say so in the commit body. If a story's premise is gone, reply `RALPH_BLOCKED` with the grep.
- **Order is the design.** Follow `priority` in the JSON.

## Hard rules for this wave

- **Measure, don't change.** No latency change, no new filler, no change to what Auto says or which tools it gets, no change to Retell tuning values. Moving code is allowed only to keep `voice_retell.py` from growing.
- **One `voice_turns` row per live turn,** still lengths-only for text (never the transcript or the reply).
- **The webhook stays fast and fail-closed:** HMAC first; a get-call failure never turns a 2xx into an error.
- **The runner never writes a file and never ships data:** the ask set comes on stdin; results go to stdout; its folder's `.gitignore` keeps `results.md` and `asks*.json` out.

## Tests: CI is the gate, once per wave

**On the owner's machine nothing runs**: no pytest, npm, tsc, docker, compose, server, browser, ngrok, a Retell call or the runner, and no `pip install` / `npm install`. The runner exports `DATABASE_URL`, `POSTGRES_*` and `REDIS_*` pointing at a dead port, on purpose. `python3 -m py_compile` on changed Python files, `ruff check` on changed paths (if ruff is on PATH), `bash -n` on shell scripts and `bash scripts/ralph/acceptance-prd258w1.sh --no-ci` are the only local checks.

- **You write the tests. CI runs them, once, at the end of the wave.** Backend: `orchestrator/tests/test_prd258_*.py`. **A test must fail without your change.** No test calls Retell, a model or the network; httpx mocks use a byte-stream body (a `content=` response is pre-read and breaks streaming code).
- **A story NEVER waits on CI.** No `gh run watch`, no `gh run list`, no polling of `test.yml`.
- **An AC marked `→ OWNER:` and everything in `ownerTest` is the owner's:** never mark it DONE, never claim it.

## Commits

- **SIGN EVERY COMMIT: `git commit -s`.** CI enforces DCO. Messages: `type(prd-258): description`.
- **Every story commit message ends with ` [skip ci]`** (D5).
- **STAGING DISCIPLINE:** stage explicit paths only. NEVER `git add -A`, `git add .` or `git add -u` (`node_modules` is untracked and NOT gitignored). NEVER `git stash` (shared with every worktree).

## Hard NOs

- NO merging, NO PRs (the runner opens the wave's draft PR when the build is complete), NO pushing anywhere except `origin feat/prd-258-w1-voice-measure`. Never touch `main`.
- NO reading or writing `~/.automatos-analyst`, NO results or ask files in the repo (D4).
- NO second migration, NO new route, NO new Python or npm dependency (`websockets` and `httpx` are already in requirements).
- NO growth of `voice_retell.py` (D6); NO edit to `consumers/chatbot/service.py` (PRD-256 is rewriting it).
- NO `os.getenv`/`os.environ` outside `orchestrator/config.py` (tests excepted), NO hardcoded values, NO secret, token or ngrok domain committed.
- NO weakening, skipping or deleting a test or a CI check.

## Per-iteration protocol

**Owner, 6 Oct: ONE CI run per wave.** Do NOT wait for CI on a story.

1. Pick the first story (by `priority`) with ACs not yet DONE, and re-verify its anchors by grep.
2. Implement it with its tests. Run `python3 -m py_compile` on the changed Python files, `ruff check` on the changed paths if ruff is on PATH, `bash -n` on changed shell scripts. Run the foreground review where the contract says. Commit (signed, ` [skip ci]`).
3. Mark its ACs `→ DONE — <evidence: the test names and files>` in `scripts/ralph/prd-258w1.json`. Commit that alone (signed, ` [skip ci]`): `chore(prd-258): <US-id> ACs DONE [skip ci]`.
4. **STOP.** One story per session. Unless that was the last story, end your reply with one line, `STORY_DONE <US-id>`. Do not start the next story.

## Completion

- **All ACs DONE** (this session finished the last story): run `bash scripts/ralph/acceptance-prd258w1.sh --no-ci` and fix what is red in signed ` [skip ci]` commits. Then make ONE signed commit WITHOUT `[skip ci]` (`chore(prd-258): Wave 1 built, run CI`), and end with the completion sentinel. The runner pushes it, opens the wave's draft PR (that starts the wave's one CI run), runs the gate against that run, then the review.
- **A story can't be built without breaking a Hard NO:** end with the blocked sentinel, after one line of why and the grep evidence.
- **Sentinels:** `RALPH_COMPLETE`, `RALPH_BLOCKED` and `STORY_DONE <US-id>` are read ONLY from the last line of your reply. Write the one you mean as that last line, alone, and never write any of those words anywhere else in the reply.
