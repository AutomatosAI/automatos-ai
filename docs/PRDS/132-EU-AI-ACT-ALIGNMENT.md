# PRD-132: EU AI Act Alignment

> **Status:** Draft (2026-04-18)
> **Positioning:** "EU AI Act aligned" — *not* "certified" (no certification body exists for most categories yet)
> **Dependencies:** PRD-44 (Security Hardening), PRD-70 (Pentest Remediation), PRD-103 (Verification & Quality), PRD-105 (Budget & Governance), PRD-106 (Outcome Telemetry), PRD-37 (SaaS Foundation)
> **Design principle:** Opt-in, progressive disclosure. Compliance features do **not** appear in the default UX. They activate only when a workspace enables "Compliance Mode" or a customer provisions the Compliance Pack SKU.

---

## 1. Problem

The EU AI Act entered into force 2024-08-01. Key dates already in effect:

- **2025-02-02** — Prohibited practices banned (Art. 5)
- **2025-08-02** — GPAI obligations apply (Art. 53–55)
- **2026-08-02** — High-risk AI system obligations fully apply (Art. 6, 9–15, 43, 50)
- **2027-08-02** — Full application for embedded high-risk systems

Automatos deploys autonomous agents that take real-world actions via 856+ Composio apps (Slack, Gmail, HR systems, CRM, etc.). Depending on the deployer's use case, Automatos-built agents may qualify as **high-risk AI systems** (e.g. HR screening, credit scoring, critical infrastructure monitoring) or **limited-risk** (customer-facing chatbots, content generation).

EU enterprise buyers — especially regulated industries (finance, healthcare, HR, legal, public sector) — are adding AI Act readiness to procurement gates **now**. Most US-first agent platforms (LangChain, CrewAI, n8n) are not leading with compliance. This is a competitive wedge.

**We do not want compliance theatre cluttering the default product.** The goal is to be *aligned by design* so that when a compliance-conscious customer asks "show me your AI Act posture," we can demonstrate it — without every other user seeing checkbox noise.

## 2. Non-goals

- Not pursuing formal CE marking or notified-body certification (no body exists for most categories yet)
- Not claiming "compliant" — only "aligned" and "conformity-ready"
- Not replacing tenant-level responsibilities — deployers are still the legal "deployer" under the Act; Automatos is a "provider" of a GPAI-adjacent platform
- Not enabling any of this by default for existing workspaces — opt-in via `workspace.compliance_mode`
- Not building a full GRC tool — we integrate with Vanta/Drata/SecureFrame where possible
- No UI clutter for non-compliance workspaces — all surfaces are feature-flagged
- Not in scope: SOC2, ISO 27001, HIPAA (separate tracks)

## 3. Success criteria

1. **A compliance-conscious buyer can self-serve**: land on the marketing site → read "EU AI Act posture" page → request Compliance Pack → provisioned in < 24h
2. **Risk tiering is automatic for most agents**: when a user creates a Shopify customer-service agent, it's auto-classified `limited` without user input; high-risk templates (HR screening, credit) require explicit acknowledgement
3. **Annex IV-style technical documentation auto-generates** per deployed agent, exportable as PDF for the deployer's auditor
4. **Prohibited-practice guardrails are platform-wide** (not opt-in) — social scoring, biometric categorization, manipulation patterns are blocked in `tool_router` regardless of workspace
5. **Default UX unchanged**: non-compliance workspaces see zero new UI elements, zero new required fields, zero onboarding friction
6. **Customer auditor dashboard**: deployer can show their auditor a read-only view of agent risk tiers, logs retention, human-oversight events, and incident history

## 4. Architecture at a glance

```
Platform (always-on, invisible to default users)
  ├─ Prohibited-practice guardrails       → tool_router pre-execution check
  ├─ AI disclosure transparency           → chatbot/voice/widget emit "AI" marker
  ├─ Logging retention policy             → outcome_telemetry + audit_service
  └─ GPAI model-card metadata             → llm_provider registry

Opt-in (workspace.compliance_mode = true OR Compliance Pack SKU)
  ├─ Risk tier metadata                   → agents, recipes, missions
  ├─ Human oversight contract             → approval flows formalized as Art. 14
  ├─ Annex IV doc generator               → per-agent PDF export
  ├─ Incident reporting workflow          → /api/compliance/incidents
  ├─ Data governance tracking             → training/fine-tune data lineage
  └─ Auditor dashboard                    → /compliance/posture read-only

Compliance Pack SKU (paid add-on)
  ├─ Auditor dashboard (shareable link with time-boxed tokens)
  ├─ Vanta/Drata connector
  ├─ Quarterly conformity report (auto-generated + reviewed)
  └─ Dedicated compliance support SLA
```

## 5. Mapping: AI Act articles → Automatos features

| Article | Obligation | Automatos implementation |
|---|---|---|
| **Art. 5** — Prohibited practices | Ban social scoring, manipulative AI, biometric categorization, real-time biometric ID | **Always-on** `tool_router` pre-check against prohibited-intent classifier + explicit deny list |
| **Art. 6 + Annex III** — High-risk classification | Classify if system used in HR, credit, education, critical infra, law enforcement, etc. | `agent.ai_act_risk_tier` enum: `minimal \| limited \| high \| prohibited` — auto-inferred from recipe templates and Composio apps in use |
| **Art. 9** — Risk management system | Continuous risk assessment over lifecycle | Extend PRD-103 (Verification & Quality) with risk events; surface in auditor dashboard |
| **Art. 10** — Data governance | Training/validation/test data quality, bias checks | Lineage tracking for any fine-tuned agents; document provenance for RAG corpora (via PRD-46 Cloud Doc Sync) |
| **Art. 12** — Logging | Automatic, tamper-evident logs with retention | Extend PRD-106 Outcome Telemetry with retention policy (default 6 months, configurable to 10 years for high-risk) + signed log chain |
| **Art. 13** — Transparency to deployer | Instructions for use, capabilities, limitations | Auto-generated "Model/Agent card" per deployed agent |
| **Art. 14** — Human oversight | Meaningful human oversight capability | Formalize existing approval flows (PRD-105 Budget & Governance); add mandatory `requires_human_approval` flag for `high` tier |
| **Art. 15** — Accuracy, robustness, cybersecurity | Documented performance + adversarial robustness | PRD-103 verification outputs fed into auto-generated conformity evidence |
| **Art. 43** — Conformity assessment | Self-assessment for Annex III systems | Annex IV-style doc generator; exportable PDF |
| **Art. 50** — Transparency to end users | Users told they're interacting with AI | **Always-on** disclosure in chatbot, voice, widget SDK (overridable only with compliance-mode off-switch + audit event) |
| **Art. 53–55** — GPAI obligations | Technical documentation, copyright summary, training data summary | GPAI model registry: each LLM in PRD-54 marketplace carries provider-supplied AI Act disclosures |
| **Art. 73** — Serious incident reporting | Report serious incidents within 15 days | `/api/compliance/incidents` workflow, integrates with existing monitoring (PRD-74) |

## 6. Data model additions

```python
# agents table (additive — nullable, no migration impact on default tier)
ai_act_risk_tier: Enum("minimal", "limited", "high", "prohibited") | None
ai_act_risk_tier_source: Enum("auto_inferred", "user_declared", "template_default") | None
ai_act_risk_tier_acknowledged_at: datetime | None  # required for "high"

# workspaces table
compliance_mode: bool = False                      # master switch
compliance_pack_enabled: bool = False              # paid SKU flag
log_retention_days: int = 180                      # default; 3650 for high-risk tenants

# new table: compliance_incidents
id, workspace_id, agent_id, mission_id, severity,
reported_at, reported_by, description, status,
ec_notification_sent_at, resolution

# new table: annex_iv_documents
id, agent_id, version, generated_at, pdf_s3_key,
auditor_share_token, token_expires_at
```

## 7. API surface (opt-in only)

```
GET  /api/compliance/posture              → full workspace posture snapshot
POST /api/compliance/incidents            → report a serious incident
GET  /api/compliance/agents/{id}/annex-iv → generate/fetch technical doc PDF
POST /api/compliance/auditor-share        → create time-boxed read-only link
GET  /api/compliance/prohibited-log       → history of blocked prohibited-practice attempts
PUT  /api/workspaces/{id}/compliance-mode → toggle compliance mode (admin only)
```

All routes gated by `workspace.compliance_mode = true` **or** admin role. Invisible otherwise.

## 8. Phased rollout

**Phase 1 — Always-on foundation (no UI changes)**
- Prohibited-practice deny list in `tool_router`
- AI disclosure markers in chatbot/voice/widget response metadata
- Log retention config (default 180d; honors existing telemetry)
- GPAI metadata on LLM provider registry

**Phase 2 — Opt-in compliance mode**
- Workspace toggle: `compliance_mode`
- Risk tier auto-inference on agent/recipe creation
- Human oversight formalization (wraps existing approvals)
- Annex IV PDF generator
- Auditor dashboard read-only view

**Phase 3 — Compliance Pack SKU**
- Time-boxed auditor share links
- Vanta/Drata connector
- Quarterly conformity report
- Incident reporting with EU notification workflow
- Dedicated support SLA

**Phase 4 — Market-facing**
- `automatos.ai/eu-ai-act` posture page
- Sales enablement: one-pager, RFP response template, SOC2-style trust center entry
- Partner with an EU-based legal advisor for the marketing claim

## 9. What's explicitly NOT building new

- Not building a new audit service — extend PRD-44's `audit_service.py` stub
- Not building a new logging pipeline — extend PRD-106 Outcome Telemetry
- Not building new approval UI — formalize PRD-105 flows
- Not building a GRC suite — integrate, don't replicate

## 10. Commercial angle

**Compliance Pack SKU** — add-on to existing SaaS tiers:

- **Starter / Pro**: compliance mode available, basic auditor dashboard
- **Compliance Pack (+$X/mo)**: auditor share links, Vanta/Drata connector, quarterly report, dedicated SLA
- **Enterprise**: everything + custom retention, on-prem log export, named compliance contact

Prospective buyers: EU finance, EU healthcare, EU HR SaaS, EU public sector, any US company selling agent-driven services into the EU.

## 11. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | "Aligned" claim misread as "certified" | HIGH | Marketing review; disclaimer on posture page; legal-approved copy |
| 2 | Prohibited-practice classifier false positives block legitimate flows | MED | Conservative deny list; override path logged and reviewable |
| 3 | Risk tier auto-inference mis-classifies | MED | User can re-declare; high tier requires explicit ack; template defaults conservative |
| 4 | Feature creep — compliance UI bleeds into default workspace | HIGH | Strict feature flags; design review gate; non-compliance workspace smoke test in CI |
| 5 | EU regulatory landscape shifts before Phase 3 ships | MED | Architecture is additive; articles map is a living doc |
| 6 | Deployer assumes Automatos covers their obligations | HIGH | Clear provider-vs-deployer language in TOS and auditor dashboard |

## 12. Success metrics

- **Leading**: `compliance_mode=true` adoption rate in EU workspaces; Compliance Pack trial → paid conversion
- **Product**: 0 new required fields in default agent creation flow (UX regression gate)
- **Commercial**: # of RFPs won citing EU AI Act alignment; # of EU enterprise logos
- **Risk**: 0 successful prohibited-practice invocations; incident report SLA (< 15 days)

## 13. Open questions

1. Do we pursue a named EU legal partner for the marketing claim (e.g. Bird & Bird, Taylor Wessing)? Cost vs. credibility tradeoff.
2. GPAI thresholds — if we ever fine-tune or host our own model, do we cross the 10^25 FLOP GPAI-with-systemic-risk threshold? Unlikely near-term but worth a footnote.
3. Should the prohibited-practice classifier be ML-based (slower, smarter) or rule-based (fast, brittle)? Phase 1: rules. Phase 3: consider ML.
4. Which of Vanta / Drata / SecureFrame do we partner with first? Most EU enterprise buyers prefer Vanta.
5. Do we surface "AI Act tier" publicly on the agent marketplace (PRD-45) as a filter? Probably yes for credibility.
