# PRD-207 — Auto Live: real-time voice + the presence orb ("Auto is in the room")

> **Status:** DRAFT for review — spec only, no build yet. Authored from Gerard's 2026-07-17 direction, grounded file:line against `main @ 115e90bfe`.
> **North star (Gerard's words):** *"an almost real speech-to-speech convo and Auto to look alive on the Chat screen… I still want to keep my existing screen… we drop what we have lower (when the Auto voice is on) and we have this cool effect like Auto is in the room."*
> **Research basis (2026-07-17, 24 sources, 19 claims verified 3-0):** the streamed cascade (streaming STT → OUR agent loop → streaming TTS over WebRTC) beats native speech-to-speech models on latency, cost AND tool-calling reliability — and tool-calling is Auto's whole act. Target ≤1.5s voice-to-voice, interruptible. Users will *experience* it as speech-to-speech; architecturally it is the cascade over the merged Retell seam.

---

## 1. What this is

Turn the merged-but-unarmed Retell lane (PRD-203 V·S4) into a product: a **live, interruptible, two-way voice conversation with Auto on the existing chat screen**, with a brand-orange **presence orb** that reacts to both voices, transcripts that land in the visible chat thread (so PRD-206 memory/continuity covers spoken conversations too), **per-workspace metering and caps** so voice is a sellable, cost-bounded add-on, and a settings surface that fits the platform's existing BYOK philosophy.

**Framing (CLAUDE.md §3): Extension / activation — heavily.** The webhook that drives Auto's brain from Retell is MERGED (`api/voice_retell.py:116`). The live-call UI shell EXISTS (PRD-74's `VoiceCallPanel` — transcript, state machine, controls) and needs its transport swapped from the dead pod WebSocket to Retell's web SDK. The settings tab EXISTS (`VoiceProfilesSettingsTab`). The telemetry table EXISTS (`voice_turns`, PRD-203 V·S6). Net-new is confined to: (a) the web-call session mint + call-lifecycle metering, and (b) the orb.

**Build size:** M–L across 10 stories (phased §5). · **Risk:** Medium — the webhook already runs the agent loop; the risky parts are cost control (metering must land WITH arming, not after) and the browser audio UX (echo/permissions/Safari — largely handled by Retell's WebRTC SDK).

**Day-one journey (the shared target for build + QA):** open `/chat` → click **Live** → browser asks mic permission → welcome content animates lower, the orange orb mounts, Auto speaks a one-line opener → "Where did we leave off with social?" → Auto answers from the CURRENT thread/workspace context (PRD-206 resume, spoken) → transcript appears in the same chat with voice badges → user interrupts mid-answer; Auto stops and takes the correction → a gated action gets a spoken ask + a tap-to-confirm card → end call → the settings meter increments within a minute.

---

## 2. Current reality (grounded — what EXISTS vs what is NET-NEW)

**Exists, armed by config only:**
- **The brain bridge.** `POST /api/voice/retell/llm` (`api/voice_retell.py:116`) — HMAC fail-closed (`:124`), streams Auto's reply as Retell frames via the SAME `StreamingChatService` as text chat (`:104-113`). Keys exist and are EMPTY: `RETELL_API_KEY` / `RETELL_WEBHOOK_SECRET` / `RETELL_AGENT_ID` (`config.py:1271-1273`).
- **Voice telemetry.** `voice_turns` (`core/models/voice_turns.py:36`) with decomposed latencies, lengths-only privacy discipline — but written ONLY by the HTTP push-to-talk path (`modules/voice/telemetry.py`); Retell turns are currently unmetered.

**Exists, wrong transport / wrong era:**
- **The live-call UI.** `VoiceCallPanel` (PRD-74 Phase 3, `frontend/components/voice/VoiceCallPanel.tsx`) — full state machine (`disconnected/connecting/connected/speaking/processing/responding`), live transcript, duration, controls — riding `useVoiceStream` (`frontend/hooks/use-voice-stream.ts:4,39`), a **WebSocket to the idle Pipecat GPU pod** (`NEXT_PUBLIC_VOICE_PIPELINE_URL`). This panel is the REWIRE target, not a rebuild.
- **Push-to-talk.** The input-bar mic (`multimodal-input.tsx:11-15`) records and POSTs `/api/chat/voice` — the serial STT→full-LLM→CPU-TTS path (TTS allowed 120s, `modules/voice/client.py`), the "slow as hell" experience this PRD replaces for live mode.
- **Voice settings.** `VoiceProfilesSettingsTab.tsx` — Kokoro (pod) voice catalogue; becomes the home of the Auto Live card.

**Gaps in the merged webhook that Phase 1 must close (all grounded):**
- **No human attribution:** webhook chats are created with `user_id=0` (`voice_retell.py:75,108`) → voice memories carry no owner, PRD-206's Q7 private-scope and viewer-guard are inert for voice, and the thread belongs to nobody.
- **No thread binding:** every call keys a NEW chat by `retell:{call_id}` (`voice_retell.py:70-79`) — a live call on the chat screen would NOT land in the thread the user is looking at.
- **No metering, no caps:** nothing counts live minutes; nothing refuses a call over budget.
- **No browser entry point:** Retell web calls require a server-minted access token (Retell invalidates it if unused for ~30s) — no such endpoint exists.

**Net-new (nothing fits):** the web-call mint endpoint, a `voice_calls` lifecycle/metering table (`voice_turns` is per-turn telemetry, not per-call duration — §4-S3 justifies the new table), and the orb.

---

## 3. Findings → fix → story

| # | Finding (grounded) | Fix | Story |
|---|---|---|---|
| 1 | No browser entry to the merged lane; Retell needs server-minted tokens | `POST /api/voice/web-call` mint with dynamic vars (workspace/agent/chat/user) | **S1** |
| 2 | Webhook: `user_id=0`, new chat per call (`voice_retell.py:70-79,108`) | Attribute the real user; bind to the on-screen chat when provided | **S2** |
| 3 | Live minutes uncounted; no caps | `voice_calls` lifecycle rows + monthly per-workspace meter + fail-closed cap at mint | **S3** |
| 4 | No workspace enable/gate; a chatty workspace could run the platform bill | Default-OFF toggle + monthly cap + platform kill-switch | **S4** |
| 5 | `VoiceCallPanel` rides the dead pod WS; no presence effect | Rewire onto `retell-client-js-sdk`; the orange orb; welcome content drops lower in live mode | **S5** |
| 6 | Voice turns invisible in the thread UI | Voice badge via the existing `messages.source` plane (PRD-205) + live receive via `chat_changed` | **S6** |
| 7 | No user-facing control surface | Auto Live card in `VoiceProfilesSettingsTab`: toggle, voice, meter | **S7** |
| 8 | Retell path writes no `voice_turns` | Telemetry parity for live turns | **S8** |
| 9 | Platform pays for every minute | BYOK: per-workspace Retell credentials (existing `byok_overrides` philosophy) | **S9** |
| 10 | Embed-widget customers can't go live | Widget/embed voice, per-key gated (the #564 CORS lesson) | **S10** |

---

## 4. Stories (test-first; CI is the only gate — no local runs)

### S1 · Web-call session mint — S/M
`POST /api/voice/web-call` (workspace-scoped router; hybrid auth): resolves the caller to INTEGER `users.id` (the #513 idiom), checks — in order, fail-closed — the platform `voice.live_enabled` settings toggle (S4), Retell credentials present (S4 settings), workspace toggle (S4), cap gate (S4 formula), then calls Retell's create-web-call API with dynamic variables `{workspace_id, agent_id?, chat_id?, user_id}`, **inserts the `voice_calls` row at mint** (`status='minted'` — the `call_id → workspace/user/chat` mapping exists BEFORE any event arrives; S2's trust boundary, S3's attribution and S9's BYOK secret-resolution all key off it), sets the vendor-side max call duration where supported (S4), and returns `{access_token, call_id}` (token dies in ~30s unused — the client connects immediately). No Retell key ever reaches the browser. Route-manifest +1 (772→773).
**Test:** `test_web_call_mint_gates_in_order` (each gate refuses with an honest reason; Retell client mocked); `test_mint_passes_dynamic_vars`; `test_mint_inserts_minted_call_row`.

### S2 · The webhook learns who is talking and where — S/M
Accept `user_id` and `chat_id` dynamic vars in `parse_llm_request`: when present (web calls), messages land in the EXISTING chat (`chat_id`) attributed to the real user — so the on-screen thread IS the call transcript, PRD-206 owner/scope stamping works (`viewer_subject_id` = the caller), and thread checkpoints cover spoken conversations. Phone/ownerless calls keep today's `user_id=0` + per-call chat fallback. `stream_response_with_agent(user_id=<real>)` replaces the hardcoded `0` (`voice_retell.py:108`).
**Trust boundary (binding):** HMAC proves the request came through Retell — it does NOT authorise the dynamic vars. Before binding, the webhook cross-validates server-side: an S1 mint row exists for this `call_id` AND its stored workspace/user/chat mapping MATCHES the vars; the chat belongs to the workspace; the user belongs to the workspace with access to that chat. Any mismatch → fail closed to the safe per-call-chat fallback (never write into a thread the mint mapping doesn't prove).
**Confirmation in voice mode (binding):** gated actions (`requires_confirmation` / PRD-193 grants) keep the UI confirmation card — Auto SPEAKS the ask ("I need a tap to confirm that — it's in the chat") and the card lands in the bound thread; **no spoken approval is trusted in Phase 1** (STT mishears; grants hash exact params for a reason). Spoken-confirm for low-risk actions is §8-Qk.
**Test:** `test_webhook_binds_to_existing_chat_and_user`; `test_phone_fallback_unchanged`; `test_webhook_rejects_chat_id_outside_workspace`; `test_webhook_rejects_user_without_chat_access`; `test_webhook_rejects_vars_mismatching_mint_row`. Mocked services.

### S3 · Call lifecycle + the minute meter — M
New table `voice_calls` (justified §2: per-CALL duration/billing grain; `voice_turns` stays per-turn telemetry): `id, workspace_id, user_id (nullable), chat_id (nullable), call_id (unique), provider ('retell'), status ('minted'|'started'|'ended'|'failed'), minted_at, started_at, ended_at, duration_seconds, disconnect_reason, created_at`. **The row is BORN at S1 mint**; Retell lifecycle events (`call_started/call_ended/call_analyzed`) arriving on `POST /api/voice/retell/events` (HMAC-verified like the LLM webhook) idempotently UPDATE it by `call_id`. An event for an unknown `call_id` is logged LOUD and stored as an orphan row (phone-lane compat) — never silently dropped. A pure `monthly_minutes_used(workspace_id)` reader powers the cap gate and the S7 meter. Structured observability events on the existing logging plane: `voice_live_mint_requested/denied(reason)` · `voice_live_call_started/ended/failed` · `voice_live_cap_exceeded` · `voice_live_webhook_rejected(reason)`. Migration chains on the current single head. Route-manifest +1 (→774).
**Test:** `test_call_lifecycle_updates_minted_row_idempotently`; `test_monthly_minutes_rollup`; `test_unknown_call_id_is_loud_orphan`; HMAC-refusal test.

### S4 · Caps, kill-switch, plan gate — S
**The platform kill-switch is a Settings TOGGLE, not an env var** (PRD-143 replaced `.env` with DB `system_settings`; CLAUDE.md bans `os.getenv` outside config.py — a runbook env flag would fight the platform). `voice.live_enabled` is a `system_settings` row (category `voice`), **default OFF**, flipped by the super-admin from the Settings page via the existing `platform_update_system_setting` surface — arm/disarm live, no redeploy, no SSH. Likewise the Retell platform credentials (`RETELL_API_KEY`/`RETELL_WEBHOOK_SECRET`/`RETELL_AGENT_ID`) live in `system_settings` (masked, the PRD-143 S11 "sensitive values always masked" rule) so arming is entirely a UI act. `config.py` reads these through the canonical DB-settings accessor with an OFF/empty default — the only "variable" is the safe default, never the on-switch.
Tuning constants stay config (not operational switches): `VOICE_LIVE_DEFAULT_MONTHLY_CAP_MINUTES` (**100**, §8-Qj — but the effective cap is a settings value, see below), `VOICE_LIVE_ACTIVE_CALL_RESERVE_MINUTES` (**10**), `VOICE_LIVE_MAX_CALL_MINUTES` (**30** — passed to the vendor call config at S1 mint where supported).
**Cap gate formula (explicit — the two-tabs race is bounded, not implicit):** refuse a mint when `ended_minutes_this_month + active_calls × RESERVE ≥ cap`. The second simultaneous mint sees the first call's reservation.
Workspace: `settings.voice_live = {enabled: bool (default false), monthly_cap_minutes?: int, retell_voice_id?: str}` — the per-workspace enable + cap, written through the PRD-143 S11 fail-closed whitelist pattern (extend `OPERATOR_WORKSPACE_SETTINGS_KEYS` + the settings PUT surface; never a free-form write). Two gates, both UI-flippable: super-admin's platform `voice.live_enabled` (the master switch) AND the workspace's own `voice_live.enabled`. Over-cap mint refusals say exactly why ("Voice budget used: 100/100 min this month").
**Test:** `test_platform_toggle_off_refuses_mint` (system_settings OFF → 503, no env var involved); whitelist-refusal test; cap-boundary test; `test_active_call_reservation_blocks_second_mint`.

### S5 · The orb + "drop lower" live mode on the chat screen — M
The screen Gerard keeps is kept: entering live mode ANIMATES the existing welcome/messages content downward and mounts the **presence orb** top-center — Auto in the room. Mechanics:
- **Transport:** new `use-retell-call` hook wrapping `retell-client-js-sdk` (S1 mints the token) — REPLACES `useVoiceStream` inside `VoiceCallPanel`'s state machine (labels/states reused; the panel becomes the docked/expanded variant, the orb is the ambient variant on the chat page).
- **Orb, both voices:** agent side from the SDK's `emitRawAudioSamples` `audio` events (raw PCM Float32Array — verified against Retell docs) reduced to RMS; user side from `getUserMedia` → Web Audio `AnalyserNode`; `agent_start_talking`/`agent_stop_talking` flip speaking/idle. States: idle-breathing / listening (user levels) / thinking (tool latency shimmer) / speaking (agent levels). Brand `warning` orange (the orange→warning codemod rule), canvas-rendered, `prefers-reduced-motion` honoured (static glow).
- **Live captions** under the orb from Retell transcript updates (accessibility + trust), barge-in works because Retell handles interruption vendor-side.
- **Dead-air honesty:** Auto's tool calls can take seconds; while the webhook streams nothing, the orb shows *thinking* and the caption shows the tool-status line (the same signal text chat gets) — never silent limbo.
- **Controls + states (explicit, not implied):** visible mic-live indicator, mute/unmute, end call — all keyboard-accessible with screen-reader labels for orb state. Every failure state has designed copy: mint refused (reason shown), mic permission denied, unsupported browser (Safari fallback copy), mid-call disconnect (orb drops to idle + "call ended" line in thread), Retell unavailable (503 → the Live button explains, chat keeps working).
**Test:** pure RMS/state reducer tests; vitest render of orb states incl. reduced-motion + every failure state; keyboard-navigation test on controls; hook mocked (no real SDK in CI).

### S6 · Voice turns visible in the thread — S
Messages written during a live call carry `source = {origin:'voice', label:'Auto · voice'}` (the PRD-205 `messages.source` plane — additive, never in `parts`); the chat UI badge-renders them like background messages. Live receive on the open thread rides the existing `chat_changed` SSE + CustomEvent bridge (the PRD-205 useState-only-chat lesson — reuse, don't rebuild).
**Test:** `test_voice_messages_stamped_with_source`; vitest badge render.

### S7 · The Auto Live settings card — S/M
Two surfaces, matching the two gates. **Super-admin (system settings):** the platform `voice.live_enabled` master toggle + the masked Retell credentials ride the existing system-settings UI (`platform_update_system_setting`) — arm/disarm the whole platform from a switch, no redeploy. **Workspace (`VoiceProfilesSettingsTab`) gains an Auto Live card:** enable toggle (writes the S4 whitelisted `voice_live.enabled`), Retell voice picker (`retell_voice_id`, §8-Qc default), this-month meter (`{used}/{cap} min` from S3 — honest-UI, no fake counts), and arming status ("Platform voice: on/off" + "voice key configured" — read-only truth reflecting the super-admin state). Copy states the pricing model per §8-Qb once decided.
**Implementation note (verify before building the picker):** confirm Retell supports per-web-call voice override. If it does not, the voice is bound to the Retell AGENT config — then either manage per-workspace agent variants or hide the picker behind platform config until agent/voice management exists. A dropdown wired to nothing does not ship ("vibes are not an API").
**Test:** vitest — toggle writes the whitelisted key; meter renders empty/used/capped states honestly.

### S8 · Telemetry parity for live turns — S
The Retell webhook writes `voice_turns` rows per turn (fields it can honestly measure server-side: response length, turn total from webhook receipt→stream close, `conversation_id`, `call_id` linkage; STT/TTS component latencies are vendor-side → 0/null, honestly absent not faked). PRD-203's "is voice any good?" number now covers the mode users actually feel.
**Test:** `test_retell_turn_persists_voice_turn` (mocked db).

### S9 · BYOK voice (Phase 2) — M
Per-workspace Retell credentials following the platform's BYOK philosophy (`byok_overrides` precedent): workspace stores its own key + webhook secret (secrets storage per the existing integrations pattern — never plaintext in `settings`); the mint endpoint uses the workspace key when present (platform key otherwise); webhook verification becomes two-stage (resolve call→workspace→try workspace secret, fall back platform secret; fail-closed). Settings card gains "Use your own Retell account" + a doc link. BYOK workspaces bypass the platform-key cap (their meter still records).
**Test:** multi-tenant signature-resolution tests; byok-key-selection test.

### S10 · Widget/embed voice (Phase 2) — M
The embedded widget (`api/widgets/` SDK plane — NOT the in-app widget) gets live voice gated **per API key** (opt-in flag on the key, like `allowed_domains`), honouring the #564 CORS preflight lesson. Pilot target: the drgreen/Academy-style tutors. Scope decision is §8-Qh.
**Test:** key-flag gate test; preflight test for the mint route from an allowed origin.

---

## 5. Sequencing / phases

- **Phase 1 = S1+S2+S3+S4+S5+S6+S7+S8** — one shippable act: arm → talk → orb → transcript in thread → capped, metered AND measured from minute one. (Metering is NOT deferrable: caps without S3's meter are decorative. S8 is in Phase 1 because §6 declares `voice_turns` the judging metric — you cannot referee a mode you don't measure; Auto's own review caught this contradiction.)
- **Phase 2 = S9+S10** — BYOK, embed voice.
- **Human runbook (§6, Gerard) — all in the UI, no redeploy:** create the Retell account/agent → in **Settings** paste the Retell key/secret/agent-id (masked `system_settings`) and flip `voice.live_enabled` ON → point Retell at `/api/voice/retell/llm` + `/api/voice/retell/events` → pick Auto's voice (§8-Qc) → toggle your own workspace ON → make one test call → watch `voice_calls`/`voice_turns` move. Disarm anytime = flip the same toggle OFF (instant platform-wide kill).
- The **existing push-to-talk mic and the chatterbox pod stay untouched in Phase 1** — their retirement is §8-Qd/Qe (Gerard's call, one is cross-repo), not a silent deletion.

## 6. Verification (CI is the only gate — no local runs)
Pure/mocked tests throughout (HMAC fixtures for both webhooks; Retell SDK never invoked in CI); migrations (`voice_calls`) chain single-head and self-apply on boot; two new routes → route-manifest (772→774); vitest for orb states/settings card/badges; grep-guard: the live path never imports `modules/voice/client.py` (the 120s-TTS pod client). The number that judges the feature: `voice_turns`/`voice_calls` latency + usage, against the research bar (≤1.5s voice-to-voice).

## 7. Safety, privacy, scope (binding)
Mic access is browser-consented; no Retell key in the browser (server-minted 30s tokens). Voice transcripts flow into memory under the PRD-206 write contract — **the Q3 exclusion validator covers SPOKEN secrets too** (say a password on a call and it still never becomes a memory), owner stamping via S2 makes Q7 private-scope real for voice. Telemetry stays lengths-only (no transcript text — the `voice_turns` discipline). Call recordings/retention live at Retell — §8-Qf decides the account setting (rec: transcripts yes, audio retention OFF); EU data-residency for the pilot clients is §8-Qg (verify on the account — the research did not confirm Retell's EU posture; treat as unknown until checked). Per-workspace caps + platform kill-switch bound the spend; BYOK moves it to the customer entirely.

## 7.5 Non-negotiable implementation constraints (the build agent's checklist)

1. Live voice launched from `/chat` binds to the CURRENT chat and the authenticated user — never an anonymous side-channel.
2. `chat_id` / `user_id` / `workspace_id` are cross-validated server-side against the S1 mint row; HMAC alone never authorises binding.
3. No mint unless ALL pass: platform `voice.live_enabled` settings toggle → Retell credentials present → workspace toggle → cap formula. Fail-closed, honest refusal reasons. The master switch is a super-admin Settings toggle (DB `system_settings`), never an env var.
4. Metering (`voice_calls` + the meter) ships WITH first arming, never after.
5. Retell credentials never reach the browser; tokens are server-minted and short-lived.
6. Spoken content follows the SAME memory exclusion rules as text (PRD-206 Q3 validator).
7. Gated actions require UI confirmation in Phase 1 — no spoken approvals (§8-Qk decides any relaxation).
8. Audio retention at the vendor defaults OFF (§8-Qf can only tighten, not loosen silently).
9. Live Retell turns write `voice_turns` in Phase 1 (S8) — the judging metric exists from day one.
10. Push-to-talk and the chatterbox pod are retired by EXPLICIT decision (§8-Qd/Qe), never by accidental deletion.

---

## 8. Open questions — Gerard's call (decide, don't let me defer — CLAUDE.md §12)

> Auto's in-platform review (2026-07-17) concurred with every recommendation below and its required-changes are folded into S1–S8 above. These remain GERARD's boxes — a reviewer's concurrence is not a decision.

1. **Qa · Vendor**: Retell (recommended — the seam is merged, web SDK exposes raw audio for the orb). Confirm, or name a rival worth a bake-off.
2. **Qb · Pricing model**: metered add-on ("$X per 100 min") vs plan-bundled minutes + top-ups. **Rec: bundle ~100 min into the paid tier, sell top-ups** — platform cost ≈ $0.11–0.15/min all-in, so $20–25/100min retails with real margin.
3. **Qc · Auto's voice**: pick the Retell voice id (this is brand — needs your ear). Per-workspace override exists (S4).
4. **Qd · Push-to-talk fate**: keep alongside live mode, or retire once Live proves itself? **Rec: retire after 2 green weeks** — two voice paths is a parity tax.
5. **Qe · Chatterbox pod retirement** (the PRD-203 §8-Qa leftover, cross-repo `automatos-voice`): **Rec: retire with Qd** — it is paid-for, idle, and superseded.
6. **Qf · Recordings at Retell**: audio retention ON (debuggability) vs OFF (privacy). **Rec: OFF; transcripts only.**
7. **Qg · EU/UK data residency**: does the pilot (InBuild UK) require EU processing? Verify Retell's posture on the account BEFORE arming for that workspace — unknown, not assumed.
8. **Qh · Embed-widget voice (S10)**: in scope for Phase 2, or wait for tutor demand?
9. **Qi · BYOK pricing**: do BYOK workspaces pay a platform fee for voice orchestration, or is BYOK free-riding acceptable at this stage? **Rec: free at pilot; revisit with billing.**
10. **Qj · Default monthly cap**: **Rec: 100 min/workspace** when enabled (≈$15 worst-case exposure per workspace), raisable per workspace — plus the S4 per-call soft max (30 min default) so a zombie call can't quietly eat the pantry.
11. **Qk · Spoken confirmation (new, from Auto's review)**: Phase 1 is UI-confirm-only for gated actions (binding, §7.5-7). Later: allow SPOKEN confirmation for low-risk actions? **Rec: revisit only after Live has two green weeks and the transcript-accuracy numbers exist** — an approval channel earns trust with data, not vibes.

---

*Traceability: consumes PRD-203 V·S4 (the Retell webhook — `api/voice_retell.py`), V·S5 (no truncation), V·S6 (`voice_turns`); rewires PRD-74 Phase 3 (`VoiceCallPanel`, `useVoiceStream` → retired transport); reuses PRD-205 (`messages.source` badge + `chat_changed` live receive), PRD-206 S1/S2 (write contract — spoken secrets excluded; checkpoints cover voice threads via S2 binding), PRD-143 S11 (fail-closed settings whitelist), PRD-196 (GDPR subject tags — voice memories now attributable), #513 (Clerk-string user id), #564 (widget CORS per-key allowlist). Research: 2026-07-17 deep-research run (24 sources; cascade-over-speech-to-speech verdict; Retell web SDK raw-PCM claim verified 3-0). The orb is the product; the meter is the guardrail; the auntie now speaks.*
