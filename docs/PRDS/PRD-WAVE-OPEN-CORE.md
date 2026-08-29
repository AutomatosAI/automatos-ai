# PRD-WAVE-OPEN-CORE — Basic + SaaS from one codebase, then sessions

> **Status:** rollout overview — owner decisions locked 2026-08-29. Companion PRDs: **PRD-209** (updated: S7–S9 added) · **PRD-233** (new) · **PRD-234** (new) · **PRD-210** (existing — the promotion gate). Research record: artifact `automatos.../artifact/5ca2751a-e9eb-4153-b44d-43fceb337db5`.

## Decision record (2026-08-29, Gerard)

1. **Single codebase — FINAL.** Basic (open source, local), SaaS (automatos.app), Enterprise are three deployment profiles of `automatos-ai`. No wrapper repos tracking product code, ever (GitLab 2011-2019 is the documented failure; own dossier `thesis-T2-repo-topology.md` concurs). A private control-plane repo (deploy config, billing when built, platform keys) arrives only when billing work starts — it consumes public images, never forks them.
2. **Enterprise — PARKED.** Cloud-enterprise = top SaaS tier via existing PRD-222 machinery when wanted; self-host license gating waits for a real deal. Nothing built now.
3. **Order — Phases 0–2 first (quickest win: Basic + SaaS co-exist, both working), then Phase 3** (session mode).
4. **Session mode = Basic's headline**: agents run as Claude Code sessions on the user's own subscription, zero API keys; Automatos is the manager/board/memory above them. Structurally local-only — the differentiator SaaS cannot offer.
5. **Teams-connect (local workspaces linked via SaaS subscription) = later, nice-to-have, funnel to SaaS.** Recorded so nothing built now precludes it (the workspace-scoped runtime contract is the future seam).
6. **Licensing — closed topic.** Apache-2.0 already in the repo makes community contributions shippable in all editions. Only rider: DCO check + two CONTRIBUTING paragraphs (PRD-233 S5).

## Phases → PRDs → order

| Phase | Goal | PRD | State | Blocks |
|---|---|---|---|---|
| **0** | `docker compose up` true; smoke lane armed; SaaS provably untouched | **PRD-209** (S1–S9) | Drafted, build-ready | everything |
| **1** | Basic worth downloading: worker on the laptop, honest tool degrade, seeded first-run, one storage factory, self-host docs + DCO | **PRD-233** | Drafted, build-ready after 209 | Phase 3 |
| **2** | Promotion safety: git-history secrets scrub + branch-protection arming | **PRD-210** (exists) | Drafted | any public promotion of Basic |
| **3** | Session mode: tickets execute as subscription sessions; fleet = the session console; Codex lane | **PRD-234** | Drafted, build-ready after 233 | — |

**Merge order: 209 → 233 → 234.** 210 is order-independent but **gates promotion** (no launch push, README blitz, or announcement before the scrub runs). Shared-file collision to respect: `docker-compose.yml` is touched by 209 S7 and 233 S1 — sequence, don't parallel, those two PRs.

**SaaS-safety invariant (all phases):** every change is compose-only (Railway never reads compose), fresh-clone-only (alembic stamp/squash), or gated behind `AUTH_EDITION` (default `saas`, saas path byte-identical). The armed smoke lane + existing required lanes make both editions PR-gated from Phase 0 onward.

## Activation dials

| Dial | Default | Meaning |
|---|---|---|
| `AUTH_EDITION` | `saas` | `local` ⇒ no-login single-workspace edition (PRD-175, live) |
| `DEFAULT_WORKSPACE_ID` | set in `envs/api.defaults` (209 Q6) | the local workspace; boot-guard-required in `local` |
| `S3_VECTORS_ENABLED` | `false` | local RAG = pgvector leg (197 S5); SaaS sets `true` |
| `COMPOSIO_API_KEY` | unset locally | absent ⇒ honest tool degrade (233 S2); BYO documented |
| `AUTOMATOS_WORKSPACE_DIR` | `./workspaces` | worker host-access root (233 S1, scope = 233 Q1) |
| `SESSION_RUNTIME_ENABLED` | `false` | session mode; boot-guarded to `AUTH_EDITION=local` (234 S1) |

## Owner decisions still open (asked in the PRDs, not blocking scheduling)

- **209:** Q1 stamp-vs-squash (rec: squash) · Q2 make lanes required (repo-admin) · Q6 workspace-id convention (rec: fixed well-known id).
- **233:** Q1 host-access scope (rec: designated dir) · Q3 storage sweep as separate PR (rec: yes) · Q4 MCP-in-router deferral to 234 (confirm).
- **234:** Q1 session concurrency cap · Q2 per-task vs long-lived sessions (rec: per-task) · Q3 Auto stays API-side v1 (confirm) · Q4 Codex lane timing · Q5 tickets-only v1 (confirm) · Q6 runtime fields in agent settings UI (rec: yes).

## Launch notes

Ralph-ready: each PRD is a normal wave (worktree + branch + PR; CI the gate; commit-not-push per convention). Launch is human-only; pin the model on unattended runs. 209's S1+S2 coupled-pair rule is the one sequencing trap inside a wave; 233/234 are internally parallel-safe as marked.
