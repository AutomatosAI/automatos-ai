# Internal Audit v1 — Auto's inside view paired with server-side logs (2026-07-09)

**What this is:** the first run of the evidence-only internal audit prompt, executed by Auto (agent 337) at 21:16–21:20 UTC, paired line-by-line with Railway logs for the same request (`req=e6cc7b7f0c76`). Inside view = what Auto's tools told it. Outside view = what the server actually did and how long everything took. Companion to `AUTO_LATENCY_BENCHMARK_2026-07-09.md`; re-run both together in future phases.

Code under test: `perf/tool-router-latency` @ `833338303` (PR #519 complete: async+bounded narrowing, Redis query-embed cache, latency-sorted provider routing).

---

## 1. The run itself, decoded

| Layer | Measured |
|---|---|
| Total request span | **190s** |
| LLM (orchestrator, gpt-5.5) | 6 calls · p50 **10.6s** · max **73.3s** · input p50 **37,465 tokens** (max 41,263) |
| Tool loop | 5 iterations (1+7+6+1+10 = 25 calls), ended by design — see §2 |
| Tool executions | p50 ~0ms; `platform_field_query` 5s once; memory searches ≤1s; everything else instant |
| Narrowing | 1 live embed **2,138ms** (in bound, no timeout) + **2 Redis cache hits (3ms, 6ms)** · enum **15 ACTIVE** both loads · 0 bridge lines |
| Tool-load wall | 224ms and 2,783ms |
| Post-turn memory writes | transcript 18,332 chars + **35 distilled facts, all tier `global`** (§4.6) |

**Reading: the platform layer cost ~5s of the 190s. The other ~97% is GPT-5.5 reasoning over 37–41k-token contexts.** PR #519's mechanics all behaved exactly as designed under a real workload.

## 2. Why Auto's audit was cut short ("tool-stop")

Server log, 21:17:47:

```
[tool-loop] Multi-step tool platform_execute hit hard cap (8 calls) — forcing synthesis
```

The chat tool-loop has a per-tool hard cap (8 `platform_execute` calls) as a runaway guard. The audit legitimately needed more, hit the cap on iteration 5, and the loop forced synthesis — which is why probes 5b/6/7-budget/8-SLA/9 were left NOT CHECKED. **Not a bug; a designed chat guardrail.** Consequence: deep audits need either (a) the remaining probes chunked one-per-message, (b) a sanctioned diagnostic mode with a raised cap, or (c) mission context — which is exactly Auto's own "Runtime Integrity Probe mission" suggestion.

## 3. Fix verification (PR #519, live)

- Narrowing **active** (enum 15 vs the 129-action fallback) on both tool-loads.
- The repeated loads in one message now hit the **query-embed Redis cache** (3ms/6ms) — the triple-load's embedding cost collapsed to ~0 for repeat loads, as designed.
- **Zero** `TIMED OUT`, **zero** thread-bridge lines, no `AsyncClient.aclose` noise.
- Latency-sorted provider routing booted at 21:17:01; the one live embed in this window (2,138ms) likely predates the new container, and subsequent loads were cache hits — first clean post-routing live-embed measurement comes with the next fresh query. Watch `[perf] rank_actions` `query_embed` on fresh phrasings.

## 4. Findings from the paired run (ranked)

1. **The Efficiency-D / "capacity 95.5%" mystery is solved — it's one number wearing two hats.** `platform_get_efficiency_score` returned `agent_efficiency=95.5`, `workflow_efficiency=0`, score 57/D. The earlier "predictive alert: agent capacity 95.5%" is the same `agent_efficiency` figure re-narrated, and the D grade is 95.5 blended with a `workflow_efficiency` of **zero** — almost certainly an unfed metric (no workflow telemetry), not bad execution. Same family: `platform_get_error_rates` returned `{}` for the second run straight. **Metric attribution, not execution pain — as suspected.**
2. **Memory scan sampling is nondeterministic and shrank**: `partial=true, scanned 5/22` this run vs `10/22` hours earlier — and skipped agents aren't identified in the response. Memory answers are quietly sample-based.
3. **Retrieval phrasing sensitivity confirmed 3-for-3**: every natural-language phrasing returned 0 hits; every keyword phrasing returned 5–6 (`playbooks workflows…how they work`→0 vs `playbook`→6; same for agent-assignment and failure pairs). Clean, repeatable eval case for the retrieval-quality thread.
4. **Playbook step drift, memory vs live**: memory says Instagram Carousel (186) has 4 steps; live `platform_get_playbook` says 2. Playbook 288's remembered permission-wall remains unverified (cut off by the cap). Step persistence deserves the live check.
5. **Roster off-by-one confirmed live**: `workspace_info.agent_count=22` vs `workspace_stats.resources.agents=21` in the same minute. Attribution/definition difference between surfaces.
6. **Memory write-amplification**: this single audit turn distilled **35 facts, all `global` tier** — audit instructions becoming global memories is pollution that will feed finding #3's noise. Worth a look at distill tiering on tool-heavy turns.
7. **Auto tried to introspect itself** — `http_request GET http://localhost:8000/openapi.json` (failed, connection refused). Direct evidence for the tool-manifest ask: give it `platform_list_actions` instead of letting it improvise HTTP calls.
8. Playbook execution counts came back live (Jira Bug Fixer 102 runs, Nightly Test Pipeline 179, …) — the fleet genuinely runs; last-run status/time is not exposed by `platform_get_playbook`, which blocks failure spot-checks from chat.

## 5. Follow-ups this report seeds (decisions, not deferrals)

- Merge #519 (all mechanics proven here; CI green).
- Finish the cut-off probes: chunk 5b/6/7/8/9 one-per-message, or build the Integrity-Probe mission Auto proposed.
- Small PRDs worth cutting from §4: workflow_efficiency/error-rates feeding (metric truth), memory-scan completeness + skipped-agent disclosure, distill-tier guard on tool-heavy turns, `platform_list_actions` manifest, playbook last-run exposure.

## 6. Re-run recipe

1. Paste the Internal Audit v1 prompt (rules + 11 probes) to Auto in a fresh chat.
2. Capture: `railway logs --since 30m -n 5000 > /tmp/audit.log`
3. `python3 scripts/analyze_auto_latency.py /tmp/audit.log`
4. Match Auto's appendix call-list against the `[tool-loop]`/`execute_tool` lines; diff §1/§4 tables against this file.
