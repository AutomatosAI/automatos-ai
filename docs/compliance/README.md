# Compliance

Governance and compliance documentation for the Automatos AI platform. Staged in
PRD-181 (Wave 11) on top of the merged enforcement substrate (Wave 4 policy plane
+ Wave 8 field provenance).

> **Guiding principle (OS review §13):** *do not add SOC 2 or EU-AI-Act tooling
> before the policy plane and audit log exist — compliance without the enforcement
> substrate is theatre.* Wave 11 builds the substrate (audit completeness, approval
> coverage, GDPR export/erasure) as real code, and stages the EU-AI-Act layer as a
> **scaffold** with the formal write-up flagged as a fast-follow.

## What is real vs. scaffold

| Area | Status | Where |
|---|---|---|
| **Audit-log completeness (Art.12)** | **Real** | `orchestrator/modules/policy/audit_handler.py`, `orchestrator/core/workspaces/audit.py` (`audit_logs`). Every tool call + policy verdict, per tenant. |
| **Approval + budget coverage (F060)** | **Real** | `orchestrator/core/models/approval_grants.py`, `orchestrator/services/board_approval.py`, `orchestrator/services/budget_ceiling.py`, `orchestrator/api/approval_grants.py`. Board tasks + playbook runs, durable/revocable/expiring grants + a dollar ceiling. |
| **GDPR export** | **Real** | `orchestrator/services/gdpr_service.py` (`export_workspace`), `GET /api/v1/gdpr/export`. SQL + Qdrant field + mem0. |
| **GDPR erasure with derived-data cascade** | **Real** | `gdpr_service.erase_workspace` / `erase_data_subject`, `POST /api/v1/gdpr/erase`, `/erase-subject`. Cascades to SQL (+ S3 objects + learned edges), Qdrant field memory, mem0. |
| **EU-AI-Act Art.14 oversight → approval cards** | **Real (mapping)** | `orchestrator/modules/policy/ai_act.py`; the card reads the risk tier + rationale. |
| **EU-AI-Act Annex-IV technical file** | **Scaffold** | [`EU-AI-ACT-ANNEX-IV.md`](./EU-AI-ACT-ANNEX-IV.md) — sections mapped to real components; formal write-up is a **flagged fast-follow**. |
| **Formal risk-classification (is the system high-risk?)** | **Fast-follow** | Not done. See the Annex-IV §5 and its checklist. |

## Known GDPR gaps (documented, not silently skipped)

Subject-level erasure is implemented where a data-subject tag exists. Three stores
carry only workspace/mission/agent scoping and **no data-subject tag**, so
subject-level erasure there returns 0 and the gap is surfaced (in code as
`# GDPR-GAP:` markers and in `erase_data_subject(...)["gaps"]`):

- **Qdrant field memory** — `workspace_id`/`mission_id`/`task_id`/`agent_id` only.
- **mem0 durable memories** — namespaced by workspace/agent/recipe.
- **SQL** — workspace-scoped, not subject-scoped (per-table subject resolution is
  domain-specific, e.g. a Shopify customer id, and belongs to the Shopify redact
  handler).

Workspace-level erasure fully covers all three. Adding a data-subject tag at write
time is the follow-up that unlocks subject-level erasure in these stores.

## Scope boundary — Shopify

Platform-side GDPR erasure of Shopify-**derived** data (customer data promoted
into field/graph/vectors/mem0) is **in scope** and handled by the entrypoints
above. The sibling `automatos-shopify` Remix GDPR webhook handlers
(`customers/redact`, `customers/data_request`, `shop/redact`) are **out of scope**
for this repo — they call `gdpr_service.erase_data_subject(...)` /
`export_workspace(...)` and are flagged for a dedicated Shopify-pod session.
