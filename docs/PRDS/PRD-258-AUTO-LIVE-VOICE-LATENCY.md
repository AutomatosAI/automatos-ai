# PRD-258: Auto Live, back on track: measure the voice turn, then make it fast

**Status:** Decided 11 Oct (§9 taken at the recommendations; Wave 1 = US-001..US-004 by Ralph, US-005 is the owner's run) · **Owner:** Gerard · **Written:** 11 Oct 2026, grounded on `main @ 9b792f873` and the PRD-256 branches (#1036 ← #1039 ← #1048)
**Type:** Extension. Every story extends a seam that exists: the Retell socket (`api/voice_retell.py`), `voice_turns` / `voice_calls` telemetry, `messages.context_trace`, the prompt-cache helpers (`core/llm/prompt_cache.py`) and the chat turn in `consumers/chatbot/service.py`. The fixed "One moment." filler is replaced, not kept beside the new lead-in. No new voice vendor and no self-hosted model.

## 1. Introduction

Auto Live (PRD-207) works, but it is slow: Gerard's word on 11 Oct was "just slow". Retell does speech-to-text, turn detection and text-to-speech; the time we control is between Retell's `response_required` and the first words we send back. Today, every spoken turn rebuilds the whole text-chat turn before the model is called, one step after another (`StreamingChatService._stream_response_with_agent_scoped`, `service.py:2466-2800` on main):

1. `agent_factory.activate_agent`: Auto's runtime is built from scratch.
2. `_load_agent_context` and `_get_tools` plus narrowing. A code comment (`service.py:2620-2622`) puts full tool discovery at about 12 s. Voice skips Composio (`VOICE_LIVE_SKIP_COMPOSIO`), which cuts that, but it was never measured.
3. `_prepare_messages`: memory retrieval and prompt assembly.
4. `_retrieval_first` (`service.py:1415`): any utterance that reads as a question is searched in the knowledge base before the first model call.
5. The model call, which may call tools before it says anything.

PRD-256 measured one sampled night-11 text turn at **8,215 ms of context assembly before the first model call** (its US-013). A voice turn pays the same.

Three more things add to it:

- **We wait while the owner talks.** Retell streams the transcript as it grows (`update_only`). We count those frames and do nothing with them (`voice_retell.py:984-985`).
- **Each turn starts cold.** `_agent_retell_stream` creates a new thread and event loop for every turn (`voice_retell.py:553-628`), so nothing built for one turn can be reused by the next.
- **Silence after the first slow turn.** "One moment." plays at 2.5 s, once per call (`voice_retell.py:827-885`, `config.py:1847-1851`). It was limited to once because it stacked during utterance storms. Every later slow turn is dead air.

**We also have no numbers.** In the last 30 days, prod logs hold no `voice_live` lines, and the local `voice_turns` and `voice_calls` tables are empty. `voice_turns.total_ms` stops the clock when the reply finishes (`voice_retell.py:763-775`). The figure the owner feels, time to the first word, is only logged (`voice_live_ws_first_frame`), never stored. Retell reports its own per-call latency (`latency.e2e/asr/llm/tts`, with p50/p90) and we don't keep it.

**Assessed and rejected on 11 Oct:** NVIDIA PersonaPlex and other native speech-to-speech models. They bring their own LLM, with no tool calling and no way to run Auto's agent loop, and they need a GPU we decommissioned in #625. PRD-207's design stands: Retell for the audio, our own Auto for the words.

## 2. Goals

- Every spoken turn stores where its time went: prep, model, tools, first frame. Every call stores Retell's own latency figures.
- A baseline is measured before any latency work, on the local stack through a tunnel, with a scripted simulator and a real call.
- Context assembly is cached across turns (this PRD takes over PRD-256 US-013) and kept warm for the whole call.
- Retrieval starts while the owner is still talking.
- The owner hears a relevant first sentence quickly on every slow turn, not just the first.
- The bars below, set before the baseline, are met on the same script, with Auto's answers no worse on the PRD-256 eval pack.

## 3. Sequencing

- **Part 1 (US-001 to US-005, measure)** starts now, on main. It touches telemetry, arming and a local runner, not the chat turn.
- **Part 2 (US-006 to US-010, faster)** starts after PRD-256's #1036 and #1039 merge. Both rewrite `service.py`, and the step order above is the same on every PRD-256 branch, so stories cite function names, not line numbers. #1048 joins the base if it merges first.
- **US-011** re-runs the Part 1 script and compares against the baseline.

## 4. User stories

### Part 1: Measure

#### US-001: Each spoken turn records where its time went
**Description:** As Gerard, I want each voice turn to store its timings, so a slow turn says whether it was prep, the model or a tool.

**Acceptance criteria:**
- [ ] `voice_turns` gains nullable columns:
  - `first_frame_ms`: `response_required` to the first content frame sent, counting the lead-in.
  - `first_auto_frame_ms`: the same, to Auto's first frame, excluding the lead-in.
  - `prep_ms`: copied from the assistant message's `context_trace` (`core/models/core.py:1259-1264`).
  - `tools_before_speech`: tool calls made before the first Auto frame.
  - `lead_in` and `prefetch_hit`: booleans, written false until US-008 and US-009 ship.
- [ ] The migration chains onto the single head at build time and survives a `create_all`-built database (`ADD COLUMN IF NOT EXISTS`). Both head pins move.
- [ ] `voice_turn_fields` stays pure, and its unit test covers the new fields.
- [ ] A turn superseded by barge-in records `first_frame_ms` if a frame was sent, and nothing invented if not.
- [ ] Test: a turn driven through `respond()` with a stubbed agent stream (300 ms delay, then two units) writes `first_auto_frame_ms` ≥ 300 and `audio_delivered` true.

#### US-002: Each call stores Retell's own latency
**Description:** As Gerard, I want Retell's per-call latency kept with the call, so speech-to-text and text-to-speech time sit beside ours.

**Acceptance criteria:**
- [ ] `voice_calls` gains `vendor_latency` (JSONB, nullable). It holds `p50`, `p90` and `num` for each of `e2e`, `asr`, `llm`, `llm_websocket_network_rtt` and `tts`, from the call object's `latency`.
- [ ] Filled from the `call_analyzed` event. If that event has no `latency`, one `GET /v2/get-call/{call_id}` is made. A failure is logged with the call id and never fails the webhook.
- [ ] `_apply_call_event` stays idempotent: a replayed event writes the same values.
- [ ] Test: a signed `call_analyzed` fixture with a `latency` block fills `vendor_latency`. The same fixture without one triggers exactly one mocked get-call.

#### US-003: Auto Live runs on the local stack through a tunnel
**Description:** As Gerard, I want to arm Auto Live on my local stack, so voice can be measured and tested on main without deploying.

**Acceptance criteria:**
- [ ] `PUBLIC_API_HOST` is defined once in `config.py`; today it is defined twice (`:900` and `:1939`). The kept definition is the Railway-aware one.
- [ ] Re-arming an existing agent also points `response_engine.llm_websocket_url` and `webhook_url` at the current `PUBLIC_API_HOST`, through `update_agent` (Retell's PATCH accepts both). Today a re-arm only re-tunes (`voice_retell.py:484-492`), so an agent created against one host can never be moved.
- [ ] On the local edition, a newly created agent is named `Auto Live (local)`, so it is never confused with prod's agent in the shared Retell dashboard.
- [ ] `scripts/voice/` holds a short README and a script that starts the tunnel, prints the `PUBLIC_API_HOST` to put in the compose override (never in `.env`), and lists the steps: restart the backend, then arm. The tunnel tool is settled in §9-Q2.
- [ ] Test: re-arm with an existing agent id sends one PATCH carrying the tuning and both URLs built from a patched `PUBLIC_API_HOST`.

#### US-004: A voice eval that runs without speaking
**Description:** As Gerard, I want a scripted runner that speaks Retell's socket protocol to our backend, so every change can be timed the same way without a person on a call.

**Acceptance criteria:**
- [ ] The runner's code is committed at `orchestrator/scripts/eval/voice_latency/`, following the `tool_routing` eval's layout, with its pure parts tested in CI. The ask set and the results stay local in `~/.automatos-analyst/voice-eval/`, like PRD-256's auto-eval.
  - It runs inside the backend container and reads the ask set on stdin: `docker exec -i automatos_backend python -m scripts.eval.voice_latency.run --workspace <id> --user <id> < ~/.automatos-analyst/voice-eval/asks.json`.
  - It inserts a `minted` `voice_calls` row with provider `sim` for a test chat in that workspace.
  - It opens `ws://localhost:8000/api/voice/retell/llm-websocket/{call_id}` and answers `config` and `ping_pong` as Retell does.
  - For each ask, it sends `update_only` frames as the utterance grows word by word at speaking pace (about 150 words a minute), then `response_required`.
- [ ] The ask set has 20 asks in four kinds, 5 each: small talk; a knowledge question; a board or work read (for example "what's on my board today?"); and a read that needs a tool. The owner keeps it; the runner only reads it.
- [ ] Per ask, it records `first_frame_ms`, `first_auto_frame_ms` and the full reply time, plus `prep_ms` and `tools_before_speech` from `voice_turns`.
- [ ] Per run, it writes one line to `results.md`: build SHA, p50/p90 per measure, per kind and overall, and the count of turns with no audio.
- [ ] A real-call script is included: the same 20 asks for Gerard to read on one live call. Its report joins `voice_turns` with that call's `vendor_latency`.

#### US-005: The baseline, before any latency work
**Description:** As Gerard, I want today's numbers on record, so every later story is judged against them.

**Acceptance criteria:**
- [ ] On a fresh local build from main with US-001 to US-004 merged, run three simulator runs and one real call (Gerard reads the script).
- [ ] The results line and a short note go in `results.md`. They record whether any turn failed or went silent; this confirms or rules out "flaky" beyond "slow".
- [ ] **Update-only spike:** the real call's log records how often `update_only` arrives during speech, and whether the owner's last utterance grows word by word or arrives only when complete. If it arrives only complete, US-008 is dropped and the note says so.
- [ ] Gerard confirms or moves the bars in §8 before Part 2 starts.

### Part 2: Faster (after #1036 and #1039 merge)

#### US-006: Context assembly under two seconds (takes over PRD-256 US-013)
**Description:** As an owner, I want Auto to start answering within a couple of seconds, typed or spoken, so a turn isn't eight seconds of setup.

**Acceptance criteria:**
- [ ] The per-workspace prompt sections (identity, product facts, skills, platform actions) are cached across turns. The cache is cleared by the writes that change them: agent, skill, brand-kit and workspace-settings saves. A test covers each clearing write.
- [ ] Board and document reads in the turn run in parallel, not one after another.
- [ ] The cached static part of Auto's prompt goes through `build_cached_system` (`core/llm/prompt_cache.py`). On the PRD-256 eval pack, the second turn of a conversation reports cache-read tokens > 0.
- [ ] Median `prep_ms` on the PRD-256 eval pack is under 2 s, with the pack's four rates no worse than the last run.
- [ ] A timing test runs context assembly against a fixed workspace, measuring the first build against a cached rebuild.
- [ ] PRD-256 US-013 is marked moved to PRD-258 in that PRD's file, in the same PR.

#### US-007: One warm session per call
**Description:** As an owner on a call, I want Auto ready before I finish my first sentence, so the first answer is as quick as the later ones.

**Acceptance criteria:**
- [ ] Each call gets one worker thread and event loop, created at `call_details` and stopped when the socket closes. This replaces the thread and loop created per turn in `_agent_retell_stream`. The ping loop stays isolated, as it is today.
- [ ] On `call_details`, the worker builds Auto's runtime, agent context and tool catalog once and keeps them for the call. A turn only adds history, per-turn narrowing, memory and retrieval.
- [ ] If a turn arrives before warm-up finishes, it waits on the warm-up rather than starting a second one.
- [ ] Barge-in cancels the turn but never the warm session. Socket close stops the worker within `_WORKER_STOP_GRACE_SECONDS`.
- [ ] A turn's `prep_ms` on a warm call, measured on the simulator, is reported against US-005's baseline.
- [ ] Tests:
  - Three turns on one call activate the agent once.
  - A barge-in mid-turn leaves the next turn on the same warm runtime.
  - Socket close stops the worker loop.

#### US-008: Retrieval starts while the owner is talking
**Description:** As an owner, I want Auto to look things up while I'm still talking, so the answer doesn't wait for a search that could have started earlier.
*(Built only if the US-005 spike shows `update_only` carries the utterance as it grows.)*

**Acceptance criteria:**
- [ ] On `update_only`, once the owner's latest utterance has been stable for a configured pause, or has grown by a configured number of words, the warm session starts memory retrieval and the knowledge prefetch for that text.
  - At most one prefetch runs per call; a newer one cancels the older.
  - All dials are config entries listed in `reports/config-surface.json`.
- [ ] On `response_required`, the turn reuses the prefetch only if the final text starts with the prefetched text and adds no more than the configured number of words. Otherwise it searches again. `voice_turns.prefetch_hit` records which happened.
- [ ] Only reads run early: the model and tools never start before `response_required`. This avoids LiveKit's open bugs where early generation doubles cost (livekit/agents#4219) and runs while a tool is still in flight (livekit/agents-js#1365).
- [ ] Tests:
  - A growing utterance triggers one prefetch, and the matching final text reuses it.
  - A final text that diverges searches again.
  - No model call is made before `response_required`.

#### US-009: A relevant first sentence on every slow turn (replaces "One moment.")
**Description:** As an owner, I want to hear something useful within a second, such as "Let me look at your board", so I know Auto heard me while it works.

**Acceptance criteria:**
- [ ] If Auto's first frame hasn't arrived by a configured delay (starting at 700 ms), Claude Haiku 4.5 (`claude-haiku-4-5`) writes one short lead-in from the utterance and the last two messages, and it is sent as the turn's first frame.
  - The call goes through `create_llm_manager`, so it is metered on the chat lane and priced.
  - Its prompt allows only an acknowledgement or a statement of what Auto is about to do. It never states a fact, and never says anything was done.
- [ ] The lead-in is dropped if Auto's first frame arrives first.
- [ ] If the lead-in was spoken, Auto's model call receives it as already said, so Auto doesn't repeat it. If Auto's first model call starts before the lead-in is ready, the lead-in is dropped.
- [ ] Storm guard: at most one lead-in per configured window per call, and never two without Auto speaking in between. This is the "five people talking" failure the once-per-call rule was guarding against.
- [ ] If Haiku fails or times out, the configured fallback phrase is spoken. `VOICE_LIVE_FIRST_FRAME_ACK_*` and the once-per-call flag are deleted. The new dials are in `config.py` and `reports/config-surface.json`.
- [ ] The thread shows what was heard: lead-in, then Auto's reply. PRD-256's honesty rule is evaluated on Auto's text only.
- [ ] Tests:
  - A slow stubbed turn speaks the lead-in first.
  - A fast turn speaks no lead-in.
  - Two `response_required` frames within the window produce one lead-in.
  - A Haiku timeout speaks the fallback.
  - Auto's messages include the already-said note when a lead-in was spoken.

#### US-010: Tune Retell's turn-taking by numbers
**Description:** As Gerard, I want the Retell knobs set from measured calls, so Auto responds quickly without cutting me off.

**Acceptance criteria:**
- [ ] `enable_backchannel` becomes a config dial; it is hard-coded off at `retell_api.py:54`. `backchannel_frequency` is added beside it.
- [ ] Two real calls on the US-004 script: backchannel off and on, with `responsiveness` at 0.9 (today) and 1.0.
- [ ] Each call's `vendor_latency` and the count of cut-offs (Gerard's tally) go in `results.md`. Gerard picks the setting, and it becomes the default in `build_agent_tuning`.

#### US-011: The after run
**Description:** As Gerard, I want the same script run after Part 2, so the change is a number, not an impression.

**Acceptance criteria:**
- [ ] Three simulator runs and one real call on a fresh local build from main with Part 2 merged. The line in `results.md` is set against US-005's baseline.
- [ ] Each bar in §8 is marked met or missed with its number. A missed bar is reported with the turn breakdown and left for Gerard to decide.
- [ ] The PRD-256 eval pack is re-run: its four rates are no worse than its last line.

## 5. Functional requirements

- **FR-1:** every Auto Live turn writes one `voice_turns` row with `first_frame_ms`, `first_auto_frame_ms`, `prep_ms` and `tools_before_speech`, and the lead-in and prefetch flags once US-008 and US-009 ship.
- **FR-2:** every Auto Live call keeps Retell's own latency figures in `voice_calls.vendor_latency`.
- **FR-3:** re-arming moves an existing Retell agent to the current `PUBLIC_API_HOST`, for both the socket and the webhook.
- **FR-4:** a call builds Auto's runtime, agent context and tool catalog once, at `call_details`, and reuses them for every turn of that call.
- **FR-5:** the per-workspace prompt sections are cached across turns, and a write that changes them clears the cache.
- **FR-6:** reads (memory, knowledge prefetch) may start on `update_only`. The model and tools never start before `response_required`.
- **FR-7:** a turn with no Auto frame by the lead-in delay speaks one Haiku-written lead-in, subject to the storm guard. The lead-in never states a fact or a completed action.
- **FR-8:** voice keeps one Auto: the same agent, memory, knowledge and core tools as typed chat. `force_text_only` is never set on a voice turn.
- **FR-9:** every new dial is a `config.py` entry listed in `reports/config-surface.json`. There are no literals in the voice modules.

## 6. Non-goals

- No new voice vendor, no self-hosted speech stack and no speech-to-speech model (PersonaPlex assessed 11 Oct).
- No change to Retell's speech-to-text, turn detection or text-to-speech beyond the agent knobs in US-010.
- Composio stays off for spoken turns (`VOICE_LIVE_SKIP_COMPOSIO`); that decision is unchanged.
- No phone calls, widget voice, or languages other than `en-GB`.
- Haiku never answers in Auto's place (§9-Q1).
- No front-end change. The orb and call panel are untouched.

## 7. Technical considerations

- **Where the code lands:**
  - `api/voice_retell.py` is already 1,005 lines. The per-call session (US-007), the prefetch (US-008) and the lead-in (US-009) go in new modules under `modules/voice/` (`call_session.py`, `prefetch.py`, `lead_in.py`).
  - `voice_retell.py` only wires them in, and it ends the PRD shorter than it started.
  - `service.py` gains at most a hook for already-said text and a way to accept a prebuilt runtime and catalog. PRD-256 is shrinking that file, and this PRD must not grow it.
- **Thread safety:** the warm session lives on the call's worker loop. Database sessions stay per turn (`get_db_session`). Only immutable objects (runtime, catalog, prompt sections) are shared across turns.
- **Cost:**
  - The lead-in is one Haiku call per slow turn, under 50 output tokens.
  - A prefetch is one embedding plus one vector search, at most one at a time per call.
  - The simulator spends model tokens like any chat turn. The real-call runs spend Retell minutes, within the workspace cap.
- **Tests run in CI only.** The simulator and real-call runs are measurements on the local stack, not CI gates.
- **Public repo:** the ask set and results stay in `~/.automatos-analyst/voice-eval/` and are never committed.

## 8. Success metrics (bars set before the baseline; Gerard confirms or moves them in US-005)

| Measure | Where it comes from | Bar |
|---|---|---|
| Time to first spoken words, end of speech to first audio | real call, `vendor_latency.e2e` | p50 ≤ 1.5 s (PRD-207's target), p90 ≤ 2.5 s |
| Our first frame, counting the lead-in | simulator `first_frame_ms` | p50 ≤ 900 ms, p90 ≤ 1.5 s |
| Auto's first frame on small talk and knowledge asks | simulator `first_auto_frame_ms` | p50 ≤ 2.0 s |
| Turn prep on a warm call | `voice_turns.prep_ms` | p50 ≤ 500 ms |
| Typed-chat prep | PRD-256 eval pack | median < 2 s |
| Silent turns, excluding superseded ones | `audio_delivered = false` | 0 |
| Auto's quality | PRD-256 eval pack, four rates | no worse than its last line |

## 9. Open questions — taken at the recommendations, 11 Oct 2026

Gerard launched Wave 1 on 11 Oct with each question at its recommendation (Decisions in `scripts/ralph/prd-258w1.json`): (1) no, Haiku never answers; (2) an ngrok static domain; (3) the lead-in shows in the thread; (4) US-013 moves here. The questions stay below for the record.

1. **Small talk:** may Haiku answer pure small talk ("hi", "thanks") outright and skip Auto? *Recommendation: no.* It would be a second Auto, and a warm call (US-007) should make those turns fast anyway.
2. **Tunnel:** an ngrok free static domain (one account, a stable host, no DNS) or a cloudflared named tunnel on an Automatos domain (needs the domain on Cloudflare)? A cloudflared quick tunnel changes host every start, which means a re-arm each time. *Recommendation: the ngrok static domain.*
3. **The lead-in in the thread:** show it as part of Auto's message (what was heard), or only in the voice transcript? The draft shows it.
4. **PRD-256 US-013:** confirm it moves here, as answered on 11 Oct ("build on 256").

## Related

- PRD-207 (Auto Live), PRD-203 V·S4/S6 (the Retell seam and `voice_turns`), PRD-201 S1/S4 (`context_trace`, prompt cache).
- PRD-256 and its PRs #1036, #1039 and #1048: the chat turn this PRD builds on.
- Retell docs: [LLM WebSocket](https://docs.retellai.com/api-references/llm-websocket), [Get call](https://docs.retellai.com/api-references/get-call) (latency fields), [Update agent](https://docs.retellai.com/api-references/update-agent) (`response_engine`, backchannel).
- LiveKit [turn handling](https://docs.livekit.io/agents/logic/turns/tuning/), whose preemptive generation is the source of US-008's read-only version.
