# Platform OS Hardening — 14-Wave Roadmap (PRD-171 … PRD-184)

**Status:** Active — Phase A PRDs authored, Phases B–E just-in-time
**Type:** Program roadmap / index (parent of PRD-171 … PRD-184)
**Priority:** P0 — the coherence + enterprise-grade path for the platform
**Owner:** Gerard Kavanagh
**Author:** Gerard Kavanagh + Claude (Opus 4.8)
**Date:** 2026-07-02
**Source:** [`reports/PLATFORM_OS_REVIEW_2026-07-01.md`](../../reports/PLATFORM_OS_REVIEW_2026-07-01.md) — §13 (the waves), §4 (fixes), §5 (missing), §9.3–9.5 (loop + policy design), §12 (north-star), §14 (owner decisions)
**Findings register pinned to commit:** `37fdecc4e` (94 verified findings, 13 critical — re-confirm each `file:line` on current `main` before editing)

---

## Operating Principle

> **Coherence first, then policy, then moat, then hardening, then polish.** Make the orchestration
> engine run at all (Wave 1); wrap every surface in one policy plane and make the product deployable
> two ways (Waves 4–6); compound the only defensible moat (Wave 7); harden to enterprise grade
> (Waves 10–12); ship the flagship (Waves 13–14). Every wave item ships with a **failing test written
> before the code**, a build size (S/M/L), a risk flag, and its dependencies. Nothing here needs a new
> subsystem — the hard artifact (a correct model-driven loop) already exists and is proven live in chat.

---

## 1. What this is

The review takes Automatos from **nineteen sophisticated capabilities that mostly do not join up** to a
**multi-tenant SaaS-grade operating system with Auto as the orchestrator**, buildable by a solo founder
driving agent fleets in **fourteen PRD-sized waves over ~8 months**, with Wave 1 startable in a single
afternoon.

The one fact that governs everything: on `origin/main` today **Auto can converse but cannot orchestrate**.
`agent_factory.py:1070` passes `content_truncate_chars=0` to a `ToolLoopExecutor` constructor that PRD-157
renamed to accept only `content_truncate_tokens` (`tool_loop.py:154`). Chat uses the correct kwarg and
works; **every non-chat path** — board tasks, missions, scheduled/heartbeat runs, webhooks, inter-agent
delegation — raises `TypeError`, burns budget, returns `status:error`, and the board spine then closes the
task as `done`. PRD-161's exactly-once dispatch and PRD-163/164's lifecycle work all deliver into a dead
engine. **Fixing one line (Wave 1) flips "Auto operates everything end-to-end" from false to true** — the
precondition for every wave that follows.

This roadmap is the **index**; each wave is a standalone PRD (PRD-171 … PRD-184). This file is not a PRD —
it sequences and gates them.

---

## 2. Definition of enterprise-ready (7 pass/fail bars)

Enterprise-ready = multi-tenant SaaS grade, expressed as pass/fail bars, not adjectives (review §13):

| Pillar | Bar (pass/fail) | Current gap | Closed by |
|---|---|---|---|
| **Security** | Zero unauth endpoints touching tenant data; gitleaks-clean history; tokens encrypted at rest; SAST + dependabot + gitleaks lanes green; 3rd-party/MCP output treated as untrusted | ≥4 unauth tenant endpoints (F003/F007/F039/F045); committed Clerk JWTs (F012); plaintext 147-scope Shopify token (F058); no SAST/gitleaks (F092) | W2, W3 |
| **Tenancy / RBAC** | Cross-tenant matrix (A→B read/write/delete, every domain) 100% denied; `super_admin ⊇ admin ⊇ user` everywhere; SDK-key perms == session perms; isolation enforced in CI | Global skill delete lobotomises all tenants (F002); S3 vector search ignores workspace filter (F005); 7 routers 403 super_admin (F043); god-key vs null-key fork (F042) | W2, W4 |
| **Observability / SLOs** | 100% of mutating tool calls in a per-tenant audit log; real-time push (not polling); zero fabricated UI metrics; ≥3 tracked SLIs w/ dashboards | Board SSE has zero subscribers, UI polls (F090); fabricated sidebar stats + placebo model selector (F035/F038); no audit-log completeness | W10, W11 |
| **Governance (EU-AI-Act, staged)** | One policy config at one choke point; budgets + approvals cover every engine; deterministic pre-call admission gate; SOC 2 then EU-AI-Act Art. 12/14 mapped onto audit log + approval cards | No unified plane (F085); no pre-call budget gate (F086); governance is missions-only (F060); model-blind dollar gate (F059) | W4, W11 |
| **Reliability** | Alembic replays from zero in CI, exactly one head; exactly-once execution proven under lease expiry; tested restore w/ stated RPO/RTO | 4 alembic heads, no from-zero replay (F010); double-exec on >10-min runs + kanban drag (F024/F025); no backup/DR (F050) | W1, W6 |
| **Deployability (local + SaaS)** | `git clone && docker compose up` → working local instance, no login, zero external SaaS; one core two editions behind a flag; local object store | Clerk mandatory (F008); fresh-clone boot broken (F009); 9 `railway.internal` defaults hardcoded (F068); no local MinIO (F089) | W5, W6 |
| **Supportability** | Structured errors-as-data to the model (no schema/handler drift); runbooks for top-5 incident classes; one canonical execution path | 5 execution engines coexist (F060); recurring schema-drift (F031); no restore/red-main runbooks | W4, W6, W14 |

Plus two working-method bars: **test-first discipline** (failing test before code, per wave) and an
**honest CI coverage ratchet** (no aspirational 80% — measure the real baseline on code that actually runs,
then ratchet; Wave 12).

---

## 3. The 14 waves at a glance

| Wave | PRD | Phase | Title | Findings | Deps | Size | Risk |
|---|---|---|---|---|---|---|---|
| **W1** | [PRD-171](./171-EXECUTION-SPINE-INTEGRITY.md) | A · Coherence | Execution Spine Integrity | F001, F023, F024, F025 | — | S–M | low (critical-impact) |
| **W2** | [PRD-172](./172-TENANT-ISOLATION-CLOSURE.md) | A · Coherence | Tenant Isolation Closure | F002–F007, F039, F045, F019 | — (∥ W1) | M | medium |
| **W3** | [PRD-173](./173-SECRET-SUPPLY-CHAIN-HYGIENE.md) | A · Coherence | Secret & Supply-Chain Hygiene | F011, F012, F058, F092 | — | M | medium (history rewrite) |
| **W4** | PRD-174 | B · Policy & deploy | Unified Policy Plane v1 | F085, F086, F059, F040, F014, F042, F043, F060 | W1, W2 | L | **high** (touches every path) |
| **W5** | PRD-175 | B · Policy & deploy | Auth Decoupling (open-core) | F008, F075 | — (helps W6) | high | one-function seam |
| **W6** | PRD-176 | B · Policy & deploy | Deployability & Reliability Baseline | F009, F010, F051, F089, F068, F050 | W5 | M–L | medium |
| **W7** | PRD-177 | C · Moat | Operating-Graph Learning Loop Closure | F015, F016, F017, F018 | W1, W4 | M–L | medium |
| **W8** | PRD-178 | C · Moat | Field Memory Correctness & Promotion | F020, F062, F063 | W1 | M | low–med |
| **W9** | PRD-179 | C · Moat | Planning Intelligence Completion | F021, F048, F049, F070 | W1 | M | low |
| **W10** | PRD-180 | D · Hardening | Observability & SLOs | F090, F035, F037, F038 | W1 | M | low |
| **W11** | PRD-181 | D · Hardening | Governance & Compliance Staging | F060, F013(GDPR) | W4 | L | mostly additive |
| **W12** | PRD-182 | D · Hardening | CI & Test Enterprise Bar | F034, F056, F057, F044, F092 | W5 | M | low |
| **W13** | PRD-183 | E · Vertical | Shopify Pilot Hardening | F032, F033, F087, F088, F076, F013 | W1, W2, W7 | L | medium |
| **W14** | PRD-184 | E · Flagship | Code Canvas + hygiene tail | PRD-170 push, F036, dead-code, api-client split | everything stable | L | merge-last |

---

## 4. Phase detail

**Phase A — Coherence (weeks 1–6).** Make the engine run and close the tenant leaks. W1 has no deps and
is the highest-leverage work in the plan (fix F001; stop closing failed runs as `done`, F023; renew leases,
F024; exclude mirror rows from drag, F025). W2 runs in parallel — per-router auth/scoping additions, not a
change to the 657-site shared hybrid auth — and is **WS-1 done properly** (tenancy is the single
differentiator vs OpenClaw). W3 rewrites history to purge the committed Clerk artifact, merges the mem0 fork
security patch, encrypts the Shopify token, and adds the supply-chain lanes.

**Phase B — Policy plane & deployability (weeks 6–16).** W4 is the centrepiece and where the Claude Code
harness reference transfers most directly — one typed pre-tool/pre-LLM gate (event bus) evaluating tenancy,
role, budget, rate, and approval in a single place, `deny > ask > allow`, errors-as-data. Build behind a
flag with characterization tests. W5 mounts Clerk only in SaaS behind an `AppAuth` facade (the
`setClerkTokenGetter` seam already exists). W6 makes `docker compose up` boot from a fresh clone and Alembic
replay from zero with exactly one head.

**Phase C — Moat compounding (weeks 16–24).** W7 closes the write-mostly learning loop (per-action Composio
telemetry, learned edges reaching default routing, intent threading) — **acceptance is a business gate**:
publish an eval showing ≥5–10-point edge-uplift per tenant, or the moat claim fails honest review. W8 makes
field memory correct and adds field→durable promotion. W9 completes PRD-164's verified remainders (it is
merged, PR #457 — this is not an unbuilt PRD).

**Phase D — Enterprise hardening (weeks 24–32).** W10 wires real-time push (LISTEN/NOTIFY) and kills
fabricated metrics. W11 adds audit-log completeness, extends governance to board + playbook, then the staged
EU-AI-Act layer (only *after* the policy plane and audit log exist — compliance without the substrate is
theatre). W12 adds the frontend CI lane, collects the orphaned test trees, turns on `strict=true` branch
protection, and installs the measured coverage ratchet.

**Phase E — Vertical & flagship (weeks 32+).** W13 turns Shopify into a reference customer (catalog webhooks
actually update the graph; sync/reindex exposed as platform tools; a generic vertical-provision abstraction
so vertical #2 doesn't fork `api/shopify.py`). W14 pushes the PRD-170 WIP **this week** (loss risk — it lives
on an unpushed single-machine branch), then builds Code Canvas and executes the merge-last hygiene tail
(dead-code kill, api-client megafile split).

---

## 5. Build model

- **Model: Opus 4.8 (`claude-opus-4-8`) for the PRDs *and* the builds.** The frontier reasoning is already
  done (this review); PRD-writing and the fixes are synthesis + grounding against live code, which Opus 4.8
  is SOTA at, at half Fable's cost. Fable 5 is *not* used: it buys nothing when the hard thinking is
  upstream, costs 2×, and its cybersecurity classifiers risk false-positive **refusals** on exactly this
  content (W2 tenant isolation, W3 secret hygiene, W4 policy plane). Tune effort within Opus — `xhigh` for
  the W4 policy-plane PRD, `medium` for the rest.
- **Test-first.** Every wave item ships its failing test first (the acceptance criterion names that test).
- **Honest CI coverage ratchet.** Do not adopt an aspirational 80%; measure the real baseline (W12) and
  fail-close only below the measured floor.
- **Author PRDs phase-by-phase, just-in-time.** Phase A PRDs are authored now (171–173). Phases B–E are
  written as their predecessors land — W7–W14 written today would be fiction (they depend on the codebase
  *after* Phase A/B, and three are gated on the decisions in §6). Writing them early = rework.
- **Ralph option.** Mechanical waves (e.g. much of W3, W12) can drop into the `scripts/ralph/prd-NNN.json`
  overnight kit; design-heavy waves (W4) are driven interactively.
- **Repo discipline (CLAUDE.md §4/§5/§12):** no backward-compat shims, delete what you supersede in the same
  PR, and **do not descope unilaterally** — the wave boundaries here are the review's dependency order, not
  silent deferral. If a piece is needed for a wave to work end-to-end, it is that wave's work.

---

## 6. Gating owner-decisions (lock before the named wave)

These are inputs the PRD needs, not PRD content (review §14). Three are hard gates:

1. **Act-vs-ask → before W4.** What may Auto do without asking vs always ask? Recurs in three concrete
   places: template/brand-kit authoring (human-only or not), the board create-task affordance, and the
   default approval tier on side-effecting Composio actions (refunds, discounts). Encode the answer **once**
   as the workspace policy document the W4 plane evaluates — not as per-surface toggles. **Until decided, the
   policy plane has no target semantics and Auto's autonomy stays all-or-nothing.**
2. **Global vs per-tenant learned edges → before W7.** The graph writes edges per workspace but reads them
   with no workspace filter. Review recommends **per-tenant** (the tenancy-correct answer and the only
   defensible moat). Blocks W7 and the moat pitch.
3. **Shopify distribution model → before W13.** Single App Store app vs per-merchant Dev Dashboard clones —
   the code and the runbooks currently describe incompatible strategies. Plus: build the embedded Remix admin
   (real webhook + GDPR handlers) or retire it and stay CDN-only (F013). Blocks W13.

Lighter §14 calls (recommendations given, decide as they surface): commit to one execution loop and retire
the legacy engines (blocks the coherence thesis); memory's place in the IA (KB tab vs own page); classic-shell
sunset date; fix-or-delete the four honest-signal zombies (F036/F038/F039 + broken docx template); canonicalise
one package manager; the moat pitch framing; open-core messaging timing; **production unknowns to confirm now**
(is `SHOPIFY_INTERNAL_API_KEY` set (F003/F004), does every `S3_VECTORS_BUCKET` carry `{workspace_id}` (F005),
were the committed Clerk material (F012) and the flagged AWS key `AKIA3ZLYFH2WTHW2CMN6` rotated — review
recommends rotating both regardless); branch-protection `strict=true`; read-only DB role for NL2SQL sources (F019).

---

## 7. Do not do (review §13)

- Do **not** market semantic top-N tool routing as the moat — it's commoditised (Tool Search Tool, ToolNet,
  Zep). The moat is the per-tenant outcome-labeled edge dataset, and only after W7 proves 5–10-point uplift.
- Do **not** chase channel breadth (12 advertised, 5 with drivers) — ship 4 business channels with per-tenant
  auth, retire the driverless adapters.
- Do **not** build vertical #2 by forking `api/shopify.py`.
- Do **not** keep five execution engines — converge on the board spine.
- Do **not** launch Code Canvas before the spine, policy plane, and worker provisioning are stable — but push
  its local WIP this week.
- Do **not** attempt the alembic single-baseline squash first — author one merge revision now, squash later
  with a from-zero test.
- Do **not** do the api-client split or classic-shell sunset early — they are merge-last refactors.
- Do **not** reopen merged PRDs 157–169 wholesale — fix only the named findings.
- Do **not** add SOC 2 / EU-AI-Act tooling before the policy plane and audit log exist.
- Do **not** implement the policy plane as shell hooks — in-process typed event bus with tenant-scoped handlers.

---

## 8. Reconciliation with the prior review (WS-1 … WS-14)

PRDs 154–169 have landed and map onto the prior workstreams:

- **Keep as landed, patch surgically:** PRD-157 (RAG), 158 (Teams), 159 (Memory — finish the mem0 fork in W3),
  160 (NL2SQL), 162 (Calendar — stop here), 165 (Graph), 166 (Field), 167 (Templates).
- **Supersede the "done" status:** PRD-161 (delivers into a dead engine until W1); PRD-163's dollar budget
  folds into the W4 policy plane.
- **PRD-164 is merged (PR #457, all four stories)** → W9 completes its verified remainders (F021, F048, F049,
  F070), not an unbuilt PRD.
- **Partial, to finish:** PRD-168 (route contract → W12; dead-code + api-client split → W14) and PRD-169
  (honest UI → W10; KB IA + classic-shell sunset → W14).
- **PRD-170** is local-only WIP → Code Canvas flagship, W14 (push this week).
- **Net-new (no trace on main):** PRDs 150–153 — the open-core deployability chain (W5–W6) and the unified
  policy plane (W4). WS-1 did tenancy piecemeal but never abstracted a plane.

---

## 9. PRD index

| PRD | Wave | Title | Status |
|---|---|---|---|
| [PRD-171](./171-EXECUTION-SPINE-INTEGRITY.md) | W1 | Execution Spine Integrity | Draft |
| [PRD-172](./172-TENANT-ISOLATION-CLOSURE.md) | W2 | Tenant Isolation Closure | Draft |
| [PRD-173](./173-SECRET-SUPPLY-CHAIN-HYGIENE.md) | W3 | Secret & Supply-Chain Hygiene | Draft |
| PRD-174 | W4 | Unified Policy Plane v1 | authored when W1–W2 land |
| PRD-175 | W5 | Auth Decoupling (open-core) | Phase B |
| PRD-176 | W6 | Deployability & Reliability Baseline | Phase B |
| PRD-177 | W7 | Operating-Graph Learning Loop Closure | Phase C |
| PRD-178 | W8 | Field Memory Correctness & Promotion | Phase C |
| PRD-179 | W9 | Planning Intelligence Completion | Phase C |
| PRD-180 | W10 | Observability & SLOs | Phase D |
| PRD-181 | W11 | Governance & Compliance Staging | Phase D |
| PRD-182 | W12 | CI & Test Enterprise Bar | Phase D |
| PRD-183 | W13 | Shopify Pilot Hardening | Phase E |
| PRD-184 | W14 | Code Canvas + hygiene tail | Phase E |
