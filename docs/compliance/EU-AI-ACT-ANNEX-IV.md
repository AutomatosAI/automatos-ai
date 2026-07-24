# EU AI Act — Annex IV Technical Documentation (SCAFFOLD)

> **Status: SCAFFOLD (PRD-181 W11, S6).** This document maps the Automatos AI
> platform's components to the Annex IV headings and points each heading at the
> *real* code / mechanism that backs it. It is **not** the full formal technical
> file. The formal risk-classification write-up and the completed narrative for
> each section are a **flagged fast-follow** (owner decision, 2026-07-03) — every
> `TODO (fast-follow)` below marks where that formal prose is still to be written.
>
> Do not treat a scaffolded section as a compliance attestation. Where a section
> says the mechanism exists, that is verifiable in code; where it says `TODO`, the
> formal assessment has not been done.

**System:** Automatos AI — multi-tenant agent orchestration platform ("Auto").
**Provider:** Automatos AI.
**Applicable framework:** Regulation (EU) 2024/1689 (EU AI Act), Annex IV.
**Note on classification:** an autonomous merchant-assistant agent is *likely not*
high-risk under the Act (see OS review §6, blind spot #10), and the paying pilot is
UK (UK GDPR, no AI Act). This file is staged now so the substrate exists before it
is needed — **not** because high-risk status has been determined. Determining
risk classification is itself a `TODO (fast-follow)`.

---

## 1. General description of the AI system

- **Intended purpose:** Auto composes capabilities (tools, missions, playbooks,
  board tasks) on a tenant's behalf under a per-workspace policy. Reads and chat
  are auto; side-effecting and destructive actions route to human approval.
- **Provider & versions:** Automatos AI. Version = the deployed git SHA of
  `automatos-ai` main. *TODO (fast-follow): pin the release/version scheme used
  for the AI-system-of-record.*
- **How the system interacts with hardware/software:** FastAPI orchestrator +
  Postgres/pgvector + Qdrant (field memory) + mem0 (durable memory) + a
  third-party tool intermediary (Composio) for external actions.
- **Deployment forms:** SaaS (Railway) and (target) self-host / open-core.

## 2. Detailed description of elements and development process

- **Methods/steps for development:** PRD-driven; each capability lands as a PRD
  wave with tests. *TODO (fast-follow): summarise the SDLC + change-approval
  evidence (see OS review §6 #8 — SOC 2 evidence automation).*
- **Design specifications & architecture:** the "one policy plane, one chokepoint"
  design — `orchestrator/modules/policy/` (`PolicyGate.check` at
  `unified_executor.execute_tool`). System architecture: OS review §12.
- **Data governance & datasets:** per-tenant isolation on `workspace_id`; learned
  edges and field memory are per-workspace. *TODO (fast-follow): data provenance
  + training-data governance narrative.*
- **Human oversight assessment (Art.14):** **mapped and live.** The policy plane
  classifies every action into a risk class and routes destructive / external /
  publish actions to a human-in-the-loop approval card carrying the risk tier and
  rationale — see §Human oversight below.

## 3. Monitoring, functioning and control

- **Capabilities & limitations:** Auto acts only through the tool registry; the
  policy plane is the single admission boundary (budget, role, approval).
- **Expected accuracy / robustness:** *TODO (fast-follow): the operating-graph
  routing eval + accuracy baseline (OS review §8, Wave 7) feeds this.*
- **Foreseeable risks & mitigations:** blast-radius (correlated tool actions),
  memory poisoning, and right-to-erasure — see OS review §6 #3/#4/#9. Mitigations
  staged in this wave: approval grants (S2), audit completeness (S1), GDPR erasure
  cascade (S3/S4).

## 4. Record-keeping — automatic logs (Art.12)

- **Substrate: LIVE (S1).** Every tool call and every policy verdict (allow / ask
  / deny) is recorded per tenant via the policy bus's audit handler
  (`orchestrator/modules/policy/audit_handler.py` → `audit_logs`). Rows carry
  tenant, actor (user / agent / system), tool, verdict, reason, risk tier, and
  (for a block) the policy error code.
- **Retention:** *TODO (fast-follow): define the log retention window + export
  cadence for an audit.*

## 5. Risk classification of the autonomy tiers

- **Mechanism: LIVE (S6 scaffold).** `orchestrator/modules/policy/ai_act.py` maps
  the policy risk classes onto Art.14 oversight tiers:

  | Policy risk class      | Oversight tier         | Human approval before action? |
  |------------------------|------------------------|-------------------------------|
  | `read`                 | `monitor`              | No (logged, reviewable)       |
  | `internal_write`       | `human_on_the_loop`    | No (reversible, monitored)    |
  | `publish`              | `human_in_the_loop`    | Yes                           |
  | `external_side_effect` | `human_in_the_loop`    | Yes                           |
  | `destructive`          | `human_in_the_loop`    | Yes                           |

  Unknown/new risk classes fail safe to `human_in_the_loop`.

- **Formal high-risk determination:** **`TODO (fast-follow)` — NOT DONE.** This
  scaffold provides the *tier* classification the approval cards read; it does not
  assert whether the system is "high-risk" under Annex III. That formal
  determination (and the resulting obligations) is the flagged follow-up.

## 6. Human oversight (Art.14)

- **LIVE (S5).** An `ask` verdict surfaces an approval card that shows the AI-Act
  risk tier + the oversight rationale (why a human is in the loop) and the
  estimated cost. A human approves / rejects; a durable, revocable, expiring
  approval grant (S2) records the decision. Board tasks and playbook runs are in
  scope, not missions-only.

## 7. Changes through the lifecycle

*TODO (fast-follow): describe versioning of the policy document + how a change to
the risk→oversight mapping is reviewed and recorded.*

## 8. Standards & specifications applied

*TODO (fast-follow): list harmonised standards once risk classification is done.*

---

### Fast-follow checklist (explicitly deferred — owner decision 2026-07-03)

- [ ] Formal risk-classification technical file (is the system high-risk? which
      Annex III category, if any?).
- [ ] Completed prose for every `TODO (fast-follow)` above.
- [ ] Log retention + audit-export policy (§4).
- [ ] Accuracy/robustness baseline narrative (§3), fed by the Wave 7 routing eval.
- [ ] Data-governance + provenance narrative (§2).
