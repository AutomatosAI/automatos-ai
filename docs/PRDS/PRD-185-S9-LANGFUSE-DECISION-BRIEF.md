# PRD-185 · S9 — Eval substrate (Langfuse) — Decision Brief

**For:** Gerard · **Date:** 2026-07-04 · **Status:** DECISION NEEDED (blocks S9 build)
**Wave:** Phase-2 Wave-0 (observability & feed-the-loops) · **Story:** S9 (the one "M")
**Related:** S10 memory eval (offline number, shipped in #507) · S1/S2 telemetry (shipped in #504/#505)

---

## 1. The decision in one line

**Which trace backend do we point the two chokepoint hooks at** — Langfuse **Cloud** (managed), self-host **v3** (heavy), self-host **v2** (light but deprecated), or **hold the backend and ship only the vendor-neutral seam now?** The *code* (the two hooks) is the same in every case; only the backend and its ops/data trade-offs differ.

The PRD's stated intent was **self-host Langfuse (MIT)**. Since that was written, v3's self-host footprint grew materially (ClickHouse + Redis + S3, not Postgres-only) — worth re-confirming before we commit the infra, hence this brief (§12 — surface, don't defer).

---

## 2. What S9 actually needs (and what it does NOT)

**Needs:** a place where a **live trace/score lands at each of the two chokepoints**, so "was the tool call good / was retrieval grounded" becomes a *queryable number over real traffic* instead of a synthetic one. This is the live complement to S10 (which already emits recall@5/MRR **offline**).

**Does NOT need** (explicitly, per the PRD): an eval *platform* we build; a parallel telemetry stack; or trace coverage of everything. Two hooks, adopt-don't-build, reuse the existing chokepoints.

**Already in place** (so S9 is small on the code side):
- Tool-dispatch funnel `modules/tools/tool_router.py:497 execute_tool`, where `write_telemetry`/`fire_telemetry` (`modules/tools/execution/telemetry.py:89`) **already fires per call** (the S1/S2 repair). A trace hook slots in beside that write — same chokepoint, no new path.
- Retrieval funnel `modules/rag/service.py:677 RAGService.retrieve_context` — the single place a turn's docs + scores are known.
- Internal metrics already exist (`core/monitoring/automatos_metrics.py`, `services/slo_metrics.py`); **no external tracing/OTel today**; Langfuse is not yet a dependency.

**Estimate (backend-agnostic):** the instrumentation is **~2 hook points + a thin config-gated client wrapper** (~S, not M — the "M" in the PRD was the backend stand-up, not the code). Tests mock the client at the boundary → pure, CI-safe.

---

## 3. The options

| # | Option | Infra / effort | Data residency | Ongoing ops | Fit for a solo-founder pilot |
|---|--------|----------------|----------------|-------------|------------------------------|
| **A** | **Langfuse Cloud** (Hobby free tier) | **Zero infra** — SDK + API keys | ⚠️ Trace payloads (prompts, completions, **retrieved client doc content**) leave the box to a US SaaS | None | **Fastest to a live number.** Best if pilot data → 3rd-party is acceptable now |
| **B** | **Self-host v3** | **Heavy** — Postgres + **ClickHouse** + Redis/Valkey + S3/blob + 2 containers; ~4 vCPU / 8 GB / 100 GB; ClickHouse ≥24.3, all-UTC | ✅ Stays in our infra | **Real** — 5+ services to run/upgrade/back up on Railway | Highest control, highest burden. Justified once trace volume or client data-residency demands it |
| **C** | **Self-host v2** | **Light** — Postgres-only (reuse the DB we already run) | ✅ Stays in our infra | Low | Tempting, but **v2 is maintenance-mode** — adopting it is buying known tech debt + a forced v3 migration later |
| **D** | **Seam-only now, backend later** | The 2 hooks + a vendor-neutral tracer interface behind config; **no backend wired this wave** | n/a (nothing emitted yet) | None | De-risks: writes the hard part once; the backend becomes a config swap. Ships S9's *code* without the infra decision |

---

## 4. Recommendation

**Ship the vendor-neutral seam (D) wired to Langfuse Cloud free tier (A) behind a default-OFF config flag** — i.e. **A over D's inert form, but built as D so the backend is swappable.**

Rationale:
- **Feed-the-loops-first thesis:** the binding constraint is *signal at the chokepoints*, not the sophistication of the store. Cloud free tier gets a **live number this week** with zero infra — the point of Wave-0.
- **The hard part is the hooks, and they're backend-agnostic.** Writing them behind a thin interface means choosing Cloud now costs us nothing later: self-host v3 becomes an env swap when we outgrow the free tier or a client mandates residency.
- **Avoid v3's 5-service lift during a pilot.** Standing up ClickHouse + Redis + S3 + 2 containers on Railway is real ops for a solo-founder, to store traces we're barely generating yet. Premature.
- **Avoid v2.** Light today, forced migration tomorrow — the platform already carries too much of that.

**The one axis that flips this:** **data residency.** Cloud sends prompts/completions/**retrieved client document content** to a US SaaS. For *your own pilot data* that's almost certainly fine. The day a real client's documents flow through it, self-host v3 (B) becomes the answer — and because we built the seam, it's a config change, not a rewrite.

**Config gate:** default **OFF** (`TRACING_ENABLED=false`), keys via `config.py` only (no `os.getenv` — the 86-file rule). Off = zero overhead, zero data egress. You flip it on when you want to watch.

---

## 5. If you approve, the S9 build is

1. Vendor-neutral tracer interface + a Langfuse-Cloud implementation behind it; config-gated (`config.py`), default OFF.
2. Hook **trace** at `execute_tool` (beside `write_telemetry` — reuse, don't parallel).
3. Hook **score/trace** at `retrieve_context` (grounding: #docs, top score, empty-vs-error from S3's typed result).
4. Pure tests: assert a trace/score is emitted at each chokepoint with the client mocked at the boundary (`test_dispatch_emits_trace` / `test_retrieval_emits_score`).
5. No new tables, no parallel telemetry, no eval-platform build.

**Effort:** S (2 hooks + thin client + pure tests). CI is the gate.

---

## 6. Open question for you

1. **Backend:** Cloud-free-tier-behind-the-seam (recommended) · self-host v3 now · hold at seam-only? →
2. **Data residency:** is trace payload → Langfuse Cloud acceptable for pilot data? (This is the real decision; everything else follows.) →

*Once you pick, S9 is a small PR. It's the last Wave-0 story with a code path; S8 still needs AWS, S4-breaker is your scheduler-scope call.*

---

*Sources (Langfuse self-host, verified 2026-07-04): [Self-hosting overview](https://langfuse.com/self-hosting) · [ClickHouse requirement](https://langfuse.com/self-hosting/deployment/infrastructure/clickhouse) · [v2→v3 migration](https://langfuse.com/self-hosting/upgrade/upgrade-guides/upgrade-v2-to-v3) · [v2 deployment (Postgres-only)](https://langfuse.com/self-hosting/v2/deployment-guide).*
