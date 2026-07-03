# PRD-181: Wave 11 — Governance and Compliance Staging

**Phase:** D — Enterprise hardening (weeks 24–32)
**Branch:** `feat/w11-governance-compliance` · **Worktree:** `automatos-ai-prd181`
**Dependencies:** Wave 4 (policy plane + bus) + Wave 8 (field provenance) — **merged to main (`557857576`)**
**Build size:** L (the biggest single wave) · **Risk:** Medium (touches governance + deletion — build carefully, audit everything)
**OS Review refs:** §5, §9 (risks #3/#4/#10), §12, §13 "Governance" bar, roadmap Phase D

---

## Overview

The enforcement substrate exists (W4 `PolicyGate` + typed bus; W8 provenance) but governance is **missions-only** and there is **no data-subject erasure**, which the review flags as pilot-binding (risk #4 — UK right-to-erasure must cascade through derived data). This wave stages governance + compliance on top of the merged substrate: complete the per-tenant audit log, extend approval/budget coverage to board + playbook, ship GDPR export + erasure-with-cascade, and lay the EU-AI-Act scaffold.

**Owner decisions (locked 2026-07-03):**
- **Depth = foundation + EU-AI-Act scaffold.** Audit + F060 + GDPR are full; EU-AI-Act ships as Art.12 traceability + Art.14 oversight mapping + an Annex-IV **doc scaffold**. The formal risk-classification write-up is an explicit fast-follow, **not** silently dropped.
- **Scope = automatos-ai only.** Platform-side GDPR erasure (including wiping Shopify-**derived** data from field/graph/vectors) is in scope. The sibling `automatos-shopify` Remix GDPR webhook handlers (F013) are **flagged for a dedicated Shopify-pod session** — do NOT touch that repo. Expose the platform erasure entrypoint the Remix webhook will later call.

---

## What already exists (reuse — do not rebuild)

- **Policy plane (W4):** `orchestrator/modules/policy/` — `PolicyGate.check()` at the `unified_executor.execute_tool` chokepoint (deny > ask > allow); a typed **bus** (`bus.py`) explicitly designed for an audit handler to attach ("audit + compaction policy can attach later", `bus.py:18`); `budget.py` `check_budget`/`BudgetDecision.audit_snapshot()`.
- **Audit substrate:** `AuditLog` table + `AuditService.log()` (`core/workspaces/audit.py`) — extend coverage, don't recreate.
- **Approval primitive:** `core/services/approval_policy.py` (`evaluate_approval`, used by missions + W9 HARNESS). Extend it to board/playbook, don't fork.
- **Erasure foothold:** `services/workspace_purge.py` — **self-maintaining** for SQL (`_discover_scoped_tables` purges every `workspace_id` table + S3). The gap is the **non-SQL derived stores** and **subject** granularity.

---

## Stories (test-first; commit per story)

### S1 · Audit-log completeness — every tool call + policy verdict, per tenant (also EU-AI-Act Art.12) — M
**Files:** `modules/policy/bus.py` (attach an audit handler), `core/workspaces/audit.py` (extend `AuditService` if needed), the `PolicyGate.check` path.
**Test:** `test_audit_completeness` asserts an allow, an ask, and a deny each write an `AuditLog` row with tenant, actor, tool, verdict, and reason. `test_audit_is_per_tenant` asserts rows carry `workspace_id`.
**Notes:** Attach an audit handler to the policy bus (the seam it was built for) so *every* verdict is recorded — this is the Art.12 record-keeping substrate the rest of the wave reads. Do not double-log; the bus is the single write point.

### S2 · F060 — approval + budget coverage for board tasks & playbook runs — L
**Files:** `core/services/approval_policy.py` (extend), a new durable **approval-grant** table + model (`core/models/`), the board dispatch + playbook/recipe execution paths, `services/coordinator_service.py` budget-ceiling helper (generalise beyond `OrchestrationRun`).
**Test:** `test_board_task_requires_approval` — a board task invoking an `ask`-tier tool creates a **durable, revocable, expiring** approval-grant and blocks until granted (not a hard block, not auto-allow). `test_playbook_budget_ceiling` — a playbook run has a dollar ceiling enforced like missions. `test_grant_is_revocable_and_expiring`.
**Notes:** This is the **deferred W4 slice** (the mission-only approval primitive generalised into a scoped, expiring, tool-agnostic grant so non-chat agents — board/scheduled/webhook — hitting `ask` get a real workflow). Route through the existing `PolicyGate`/`approval_policy`; do not build a parallel plane. Extend the mission dollar-ceiling to board + playbook.

### S3 · GDPR data export — M
**Files:** new `services/gdpr_service.py` + a super-admin/tenant-scoped `api/gdpr.py` endpoint.
**Test:** `test_gdpr_export` — export returns a portable bundle of a workspace's (and, where the data-subject tag exists, a subject's) data across primary + derived stores.
**Notes:** Reuse `_discover_scoped_tables` for the SQL side; add the vector/mem0 reads. Machine-readable (JSON) output.

### S4 · GDPR erasure with derived-data cascade (risk #4) — L
**Files:** `services/gdpr_service.py` + extend `services/workspace_purge.py`; erasure hooks in the **non-SQL derived stores**: Qdrant `field_memory` + RAG doc vectors (by `workspace_id` payload filter, and by data-subject where tagged), mem0 durable memories (`modules/memory/unified_memory_service.py`), and confirm S3 objects. Learned-edge tables (`core/models/tool_routing.py`) are SQL-scoped so `_discover_scoped_tables` already covers them — verify.
**Test:** `test_gdpr_erasure_cascade` — after erasure, the subject/workspace has zero rows in SQL **and** zero vectors in Qdrant field + RAG **and** zero mem0 durable memories; `test_erasure_is_audited` — every erasure writes an `AuditLog` row.
**Notes:**
- Erasure = the real cascade. A delete that leaves the subject in field memory/vectors is not a GDPR delete (§12 — build the thing that actually works).
- **Subject granularity:** implement subject-level erasure where the data-subject/provenance tag exists (W8 added provenance). Where a store lacks a data-subject tag, **flag it as a documented gap in the PR** (a `# GDPR-GAP:` marker + the report) — do NOT silently skip. Add the tag at write where cheap.
- Expose a single `erase_data_subject(...)` / `erase_workspace(...)` entrypoint the future Shopify `customers/redact` webhook will call.

### S5 · EU-AI-Act Art.14 — human oversight on the approval cards — M
**Files:** the approval-card surface (frontend + the approval payload from S2), `core/services/approval_policy.py`.
**Test:** `test_approval_card_shows_risk_tier` — an `ask` verdict's approval card/payload carries the autonomy risk tier + the oversight rationale (why a human is in the loop).
**Notes:** Map the existing autonomy tiers onto the approval surface so a human approver sees the risk classification and rationale. Frontend verify: tsc/vitest, no dev-server/browser.

### S6 · EU-AI-Act scaffold — Annex-IV doc + autonomy-tier risk classification — S/M
**Files:** `docs/compliance/EU-AI-ACT-ANNEX-IV.md` (scaffold mapping the system's components), a lightweight autonomy-tier risk-classification (config/enum in `config.py` + a small module), `docs/compliance/README.md`.
**Deliverable:** the Annex-IV **scaffold** (sections mapped to real components, marked TODO where the formal write-up is pending) + a risk-tier classification the approval cards (S5) read. **This is a scaffold per the owner decision — do not attempt the full formal technical file.** Clearly mark the formal risk-classification write-up as the fast-follow.

---

## Verification (NO servers, NO dev-browser)
Backend: `py_compile` + pure pytest (mock Qdrant/mem0/PG at the boundary; seed a `workspaces` row in DB tests). Frontend (S5): `tsc --noEmit` + `vitest` only — never `next dev`/browser (kills the user's Chrome).
```
python -m py_compile <changed files>
python -m pytest orchestrator/tests -k "audit or approval or gdpr or erasure" -q
```

## Conventions (see automatos-ai/CLAUDE.md)
- No `os.getenv()` outside `config.py`; reuse PolicyGate/bus/AuditService/approval_policy/workspace_purge — **no parallel planes**; no new tables where one fits (the approval-grant table is genuinely new — justified). SQLAlchemy 2.0: `CAST(:p AS type)`. Immutable patterns; comprehensive error handling; **audit every governance + erasure action**.
- **Do NOT touch the `automatos-shopify` repo** (F013 flagged for its own pod). Do NOT silently defer any story — if blocked, surface it (§12).
- Commit per story to `feat/w11-governance-compliance` (feat(prd-181): ...). **Do not push or open a PR.**

## Success metrics
- Every tool call + policy verdict audited per tenant (Art.12 substrate).
- Board tasks + playbook runs carry durable, revocable, expiring approval grants + a dollar ceiling (F060 closed).
- GDPR export + erasure cascade proven across SQL, Qdrant field/RAG vectors, and mem0 durable; every erasure audited; gaps flagged not skipped.
- Approval cards show the AI-Act risk tier + oversight rationale; Annex-IV scaffold + tier classification in place, formal write-up flagged as follow-up.
