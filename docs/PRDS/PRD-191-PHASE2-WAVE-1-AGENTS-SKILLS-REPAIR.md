# PRD-191: Phase 2 · Wave 1 — Agents-Skills Data Repair (Stop Paying the Duplicate-Skill Tax)

**Phase:** Phase 2 — Module Deep-Review remediation · **Wave 1** (resurrect the dead client-facing loops) · report id **P2-10**
**Branch:** `feat/p2-w1-agents-skills-repair` · **Worktree:** `automatos-ai-p2w1-skills`
**Dependencies:** **PRD-185** (Wave 0, `649482aa3`) merged — this wave is judged against Wave 0's numbers. Wave 0 gates Wave 1.
**Build size:** S–M (one alembic migration that *extends* an existing one; three small code repairs + their pure tests) · **Risk:** Low–Medium (one migration touches production `agent_skills`/`skills` rows — dedupe-before-constrain ordering is load-bearing; no rebuilds)
**Source:** `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` §6 Wave 1 (P2-10); primary dossier `reports/dossiers/agents-skills.md` §C defects 1–2, §J upgrades 1/2/5/8, §G integrity/efficiency triad. Security §3.2.a folded in (skill-attach visibility).

---

## Overview

The Phase-2 review's one-line finding is **"good bones, open loops."** For the agents-skills module the bones are the execution spine — `AgentFactory.execute_with_prompt`, the F001-hardened single path all ten surfaces ride. The open loop here is not a severed nerve; it is **live data corruption on the flagship agent's prompt.** In production, 205 active skill names are duplicated, `platform-management` exists five times as a global-active skill, and the workspace Auto agent is linked to it **four times**. `SkillsSection` does no dedup, so the same ~26.5 KB body renders once uncapped and then again to fill the 5,000-token auxiliary budget — roughly **~5k tokens of pure duplication injected into every Auto turn**, crowding memory, Knowledge Graph and tool sections out of a priority-trimmed budget.

Judged against the **North Star** — *does this make Auto more autonomously capable and the agents' output higher-quality for clients?* — this is a direct per-turn quality repair. Auto's prompt is the product surface every client turn passes through; ~5k duplicated tokens is ~5k tokens *not* spent on the client's memory, documents and context. **No moat framing; no new capability.** The deliverable is that Auto's prompt stops carrying dead weight, the corruption cannot recur, and the "primary skill" that renders uncapped is the *right* one instead of load-order luck.

**Three things are broken, and this wave fixes each with a constraint or a real value — not a workaround:**
1. **Duplicate skill rows and duplicate agent↔skill links** reach the prompt because the seeders race on hot request paths and nothing at the schema level forbids the duplication. *(J1)*
2. **The seeders are not concurrency-safe** — query-then-insert with a swallowed `except`, running from `hybrid.py`, `chat.py` and `workspaces.py` across multiple workers. *(J1)*
3. **"Skill priority" is a phantom** — `SkillsSection` sorts on `getattr(s, "priority", 0)` against a `Skill` model that has no `priority` column, so the uncapped-primary slot is arbitrary relationship-load order (standing finding F054). *(J2 / J5)*

**Drift note — read this before estimating (CLAUDE.md §2 REUSE-first).** The dossier was pinned to `77bc9c6d5`; this worktree is `649482aa3` and the surface moved. A migration named `dedupe_skills_unique_workspace_name.py` **already exists and is in the mainline chain** (`wave1a_agent_responsibilities` → `wave1b_heartbeat_completion` chain onto it). It already deduped `skills` and added `UNIQUE(workspace_id, name)` plus a partial `UNIQUE(name) WHERE workspace_id IS NULL`. **What it did *not* do — and what this PRD is — is the dossier's headline constraint: `UNIQUE(agent_id, skill_id)` on `agent_skills`.** That table still has no PK/unique (bare `agent_id`/`skill_id` columns, `core.py:29-32`); the existing migration's own comment claims an *"implicit unique (agent_id, skill_id)"* (`dedupe_skills_unique_workspace_name.py:52`) that **does not exist** — its link-dedupe is best-effort with nothing backing it. This PRD **extends the existing chain**, it does not duplicate a dedupe. See **S1** and the **Gerard's-call** box below.

**PILOT lens (locked):** 5 lifetime marketplace installs, an 8-months-stale `skill_sources` sync, a 43-persona library bypassed 199:20 — these are **cold-start / adoption facts, not defects, and are not in scope** to "fix by driving usage" (`feedback-pilot-usage-not-quality-signal`). What *is* in scope is the genuine data-integrity bug: duplicate rows are demonstrably degrading Auto's prompt **today**, and the fix is the constraint + concurrency-safe seeder + real priority, verified by pure tests.

---

## Findings & Scope (all `file:line` per the review tree `77bc9c6d5`; **confirmed by grep against `649482aa3`** — numbers below are the *current* worktree)

| Finding | Issue (verified in worktree `649482aa3`) | Fix | Story |
|---|---|---|---|
| **J1-a** | `agent_skills` is a bare junction — `Column('agent_id'…), Column('skill_id'…)`, **no PK, no UNIQUE** (`core/models/core.py:29-32`). No migration ever created one (grep: only FK-drops on `agent_skills`). The `dedupe_skills_unique_workspace_name.py:52` comment asserts an *"implicit unique (agent_id, skill_id)"* that does not exist. Auto agent 340 is linked to `platform-management` **4×** in prod. | New migration **chained onto the current head**: (1) collapse residual duplicate `(agent_id, skill_id)` links keeping the lowest `ctid`/no survivor row needed, then (2) `op.create_unique_constraint("uq_agent_skills_agent_id_skill_id", "agent_skills", ["agent_id","skill_id"])`. **Dedupe THEN constrain — order is load-bearing.** | **S1** |
| **J1-b** | `SkillsSection._build` sorts + renders **every** attached skill with no dedup by id or name (`modules/context/sections/skills.py:48-100`). Even with S1's link-unique, two *different* skill rows with the same `name` (e.g. the 26 KB builtin `platform-management` and the 70 KB github import, both global-active) can still both attach and both render. | Assembly-side guard: dedup `active_skills` by `skill.id` **and** by `(name, content_hash)` before the primary/aux split, so identical bodies never render twice regardless of data state. Deterministic — protects the prompt independent of row state. | **S2** |
| **J1-c** | `_upsert_platform_management_skill` is query-then-`.first()`-then-`db.add` with **no `ON CONFLICT`/advisory lock** and a bare `except Exception` (`core/seeds/seed_auto_agent.py:81-134`); `_assign_skill_to_agent` is SELECT-then-INSERT (`:137-147`). Both run on hot paths across workers: `core/auth/hybrid.py:369`, `api/chat.py:124`, `api/workspaces.py:296`. Now that a `UNIQUE(name) WHERE workspace_id IS NULL` is **live**, a concurrent seed **raises `IntegrityError` that the bare `except` swallows** → silent seed failure. | Make the **existing** seeder idempotent-under-concurrency: a Postgres advisory lock around the platform-management upsert (keyed on a stable name hash) **or** `INSERT … ON CONFLICT DO NOTHING` then re-select; convert `_assign_skill_to_agent` to `INSERT … ON CONFLICT (agent_id, skill_id) DO NOTHING` (backed by S1's constraint). **Reuse these two functions — do not add a parallel seeder.** Remove the swallow-all `except`. | **S3** |
| **J2 / J5 (F054)** | `SkillsSection` sorts by `getattr(s, "priority", 0) or 0` (`sections/skills.py:63-67`) but `Skill` has **no `priority` column** (`core.py:340-384`) — the sort key is always `0`, so "primary skill uncapped" is relationship-load order. Existing `tests/test_skills_section.py:87-94` sets `priority` on a `MagicMock`, so it passes against a phantom attribute. Precedent exists: `agent_assigned_plugins` carries a real `priority` (`core.py:314`, `order_by="AgentAssignedPlugin.priority.asc()"`). | Add a real `priority` column to the **`agent_skills` association** (per-attachment, mirroring the plugin precedent), default `0`, surfaced later in the skills tab as drag-to-order; make `SkillsSection` read the association-level priority via the ORM relationship. Closes F054 **with a decision, not a phantom.** (Same migration as S1.) | **S4** |
| **Sec §3.2.a** | The **attach** operation on the canonical agents router filters only `Skill.id.in_(ids), is_active` with **no global-or-own-workspace visibility check** — create (`api/agents.py:368-375`), bulk (`:438-445`), `POST /{id}/skills` (`:785-791`). Another workspace's private skill can be attached (and thus prompt-injected) by id. `api/skills.py` already gates this via `_skill_visible_to` (`api/skills.py:160-170`). | Apply the **existing** `_skill_visible_to` helper to the three attach sites in `api/agents.py`. Reject non-visible ids (drop or 404, matching `skills.py` semantics). No new helper. | **S5** |

---

## Stories (test-first — write the failing test, make it green, refactor)

> All tests are **pure**: no DB, no network. The migration's *logic* is verified by a pure test of the dedupe SQL builder / a fixture-driven dedupe function and a schema-shape assertion on the model metadata; the seeder is verified with a mocked session at the boundary; priority ordering and assembly dedup are verified with in-memory skill fixtures. See **Verification**.

### P2-10 · Constrain the data (dedupe → UNIQUE → real priority, one migration)

**S1 · `UNIQUE(agent_id, skill_id)` on `agent_skills` — dedupe THEN constrain — S · _the dossier headline; load-bearing ordering_**
**Files:** new migration `orchestrator/alembic/versions/prd191_agent_skills_unique_and_priority.py` with `down_revision = "wave1b_heartbeat_completion"` (the current head — **confirm the head at build time with `alembic heads`/grep; chain onto it, never `down_revision = None`**, which is what left the existing dedupe migration a floating fixup); model edit `core/models/core.py:29-32` (add the `UniqueConstraint`).
**Test:** `test_agent_skills_dedupe_keeps_one_link_per_pair` feeds an in-memory list of `(agent_id, skill_id)` rows with a known 4× duplicate (mirroring Auto 340 → platform-management ×4) to the migration's pure dedupe helper and asserts exactly one survivor per pair, lowest row retained; `test_agent_skills_model_has_unique_constraint` asserts the `agent_skills` Table metadata carries a `UniqueConstraint` on `('agent_id','skill_id')` (schema-shape assertion — no DB).
**Notes:** **Dedupe existing rows first, add the constraint second — reversing the order makes the `CREATE UNIQUE` fail on the live 4× Auto links.** Mirror the existing migration's conflict-safe `DELETE … USING … WHERE EXISTS` shape (`dedupe_skills_unique_workspace_name.py:54-73`) for the link collapse. The constraint is what makes S3's `ON CONFLICT (agent_id, skill_id)` and S2's link-dedup durable rather than best-effort. This migration is the single most valuable line in the wave — it converts "duplicated four times" into "cannot duplicate."

**S4 · Real skill priority on the attachment — S · (closes F054 with a decision) — same migration as S1**
**Files:** same migration (`prd191_agent_skills_unique_and_priority.py`, add `priority INTEGER NOT NULL DEFAULT 0` to `agent_skills`); `core/models/core.py:29-32` (add `Column('priority', Integer, nullable=False, server_default='0')`); `modules/context/sections/skills.py:63-67` (read the association-level priority, drop the phantom `getattr(s, "priority", 0)`).
**Test:** `test_skills_section_orders_by_real_priority` builds two in-memory attached skills with association priorities `(10, 1)` and asserts the priority-10 skill is the uncapped primary and the priority-1 skill is the aux — and that flipping the priorities flips which renders uncapped (proving the sort key is *load-bearing*, not phantom). Update the existing `tests/test_skills_section.py` fixtures (`:87-94`) to set priority on the **attachment**, not a bare `MagicMock` attribute on the skill.
**Notes:** Per-attachment priority (not a `Skill`-level column) mirrors the plugin precedent (`AgentAssignedPlugin.priority`, `core.py:314`) — the same skill can be primary for one agent and auxiliary for another. This is the decision F054 asked for; do **not** re-file it as a phantom or defer it. Multi-skill agents (the Shopify roster carries ~10 skill attachments each) get the *right* skill uncapped.

### P2-10 · Stop the corruption recurring (assembly + seeder)

**S2 · `SkillsSection` deterministic dedup — S · _protects the prompt independent of row state_**
**Files:** `modules/context/sections/skills.py:48-100` (dedup `active_skills` before the primary/aux split).
**Test:** `test_skills_section_dedupes_by_id` attaches the same skill row twice (same `id`) and asserts its body renders **once**; `test_skills_section_dedupes_identical_bodies` attaches two *different* ids with identical `(name, content_hash)` (the 26 KB-vs-itself case) and asserts the body appears once and the aux budget is **not** consumed by the duplicate.
**Notes:** Belt-and-braces with S1: the constraint stops duplicate *links*, this stops duplicate *content* reaching the prompt even if two distinct rows share a name/body. Dedup by `skill.id` first, then by `(name, content_hash)`; preserve the highest-priority instance (depends on S4's real priority for the tie-break). This is the one change that would have blunted the ~5k-token tax months ago regardless of the DB state.

**S3 · Concurrency-safe seeders (idempotent upsert) — S**
**Files:** `core/seeds/seed_auto_agent.py:81-134` (`_upsert_platform_management_skill`), `:137-147` (`_assign_skill_to_agent`). **Extend these two — no new seeder.**
**Test:** `test_platform_skill_upsert_idempotent_under_race` mocks the session so the first `.first()` returns `None` and the `INSERT` raises a unique violation (simulating a second worker winning the race), and asserts the function **re-selects and returns the existing row** rather than swallowing the error and returning `None`; `test_assign_skill_uses_on_conflict` asserts the assign path emits an `ON CONFLICT (agent_id, skill_id) DO NOTHING` insert (or takes the advisory-lock branch) and never creates a second link for the same pair. Mirror the `test_seed_telemetry.py` "idempotent seed (mock DB)" pattern.
**Notes:** The seeder runs on `hybrid.py:369` / `chat.py:124` / `workspaces.py:296` across workers — this is *the* root cause of the 5 duplicate global rows and the 4× Auto link. Advisory lock (`pg_advisory_xact_lock` on a stable hash of `"platform-management"`) **or** `ON CONFLICT DO NOTHING` + re-select — pick one, keep it inside the existing transaction. **Remove the blanket `except Exception` that turned a live `IntegrityError` into a silent no-seed** (a Wave-0 lesson: silent excepts are why loops died for months). No `os.getenv` — any tunable goes through `config.py`.

### P2-10 / Security §3.2.a · Close the attach-visibility hole

**S5 · Skill-attach visibility parity on the canonical router — S · (Security §3.2.a)**
**Files:** `api/agents.py:368-375` (create), `:438-445` (bulk), `:785-791` (`POST /{id}/skills`); reuse `_skill_visible_to` from `api/skills.py:160-170`.
**Test:** `test_attach_rejects_foreign_workspace_skill` asserts attaching a skill whose `workspace_id` is another workspace's UUID is rejected (not silently attached), while a global (`workspace_id IS NULL`) skill and an own-workspace skill both attach — with `ctx` mocked at the boundary (pure).
**Notes:** This is the exact probe the capability map flagged and it is a *correctness* hole as well as a security one — a foreign private skill attached by id is prompt-injected into your agent. Reuse the shipped helper; do not write a second visibility check. Full authorization analysis remains the Opus hardening pass's domain — this story closes the concrete attach gap on the canonical router only.

---

## Sequencing

- **S1 (constraint) → S3 (`ON CONFLICT (agent_id, skill_id)`)** — the seeder's conflict target needs the unique to exist. Author both in the same PR; the migration lands with the code.
- **S1 and S4 are the same migration** (unique + `priority` column added together) — one alembic revision, `down_revision` = the current head (**re-confirm at build time**).
- **S4 (real priority) → S2 (dedup tie-break)** — S2's "keep the highest-priority instance" reads S4's priority; if S2 lands first, tie-break on lowest `id` and tighten once S4 is in.
- **S5 is independent** — no ordering constraint; can land in any order / a parallel worktree (disjoint files: `api/agents.py` only).
- The only shared file across stories is `modules/context/sections/skills.py` (S2 dedup + S4 priority read) — land them in one commit or coordinate the two edits.

---

## Verification (CI is the only gate — no local runs)

Per current project convention (`feedback-no-local-servers`, tightened 2026-07-03): **do not run servers, builds, `alembic upgrade`, `pytest`, `tsc`, or installs on the dev machine.** Write the code + **pure** tests (no DB / network — mock the session at the boundary; assert on the dedupe function, the seeder's conflict handling, the priority ordering, and the `agent_skills` Table metadata), commit, push, and let **CI (the PR checks) verify.**

- The **migration is verified by a pure test** of its dedupe helper (in-memory `(agent_id, skill_id)` rows → one survivor per pair) **plus a schema-shape assertion** that the model's `agent_skills` Table carries the `UniqueConstraint` and the `priority` column — **not** by running the migration locally.
- The **seeder idempotency** is a pure test with a mocked session where the INSERT raises a unique violation on the second call.
- The **priority ordering** and **assembly dedup** are pure tests over in-memory skill fixtures (extend `tests/test_skills_section.py`, which already loads `SkillsSection` under an isolated fake module graph — reuse that harness).
- Every new test must run in CI with no external service. CI (the PR checks) is the sole gate.

---

## Conventions (non-negotiable — see `automatos-ai/CLAUDE.md`)

- **REUSE first (§2).** This wave *extends* the existing `dedupe_skills_unique_workspace_name` migration chain and the existing `_upsert_platform_management_skill` / `_assign_skill_to_agent` seeders and the existing `_skill_visible_to` helper. Do **not** add a parallel dedupe migration, a parallel seeder, or a second visibility check.
- **No new alembic head.** Chain `down_revision` onto the current head (`wave1b_heartbeat_completion` at time of writing — re-confirm with `alembic heads`). A floating `down_revision = None` is exactly the deployability hazard that left the existing dedupe migration a loose fixup.
- No `os.getenv()` outside `config.py`; any advisory-lock key or threshold goes through the canonical config module.
- **No backward-compat shims** — delete the phantom `getattr(s, "priority", 0)` path when the real column lands; delete the swallow-all `except Exception` in the seeder. No `_legacy` retention.
- Immutable patterns; small focused functions; comprehensive error handling; **no silent `except` swallows** (this wave exists partly because of one).
- **No new tables where an existing one fits, no new tools where an existing one extends** — `priority` goes on the existing `agent_skills` association; the fix rides existing seeders/helpers.
- Canonical vocab: **Playbook** (not Recipe), **Deliverable**, **Knowledge Graph**, **Command Center**, **Auto**.
- Branch `feat/p2-w1-agents-skills-repair`; commit, push, open a PR; CI is the gate.

## Success metrics (the definition of "the duplicate-skill tax is paid off")

- **`agent_skills` carries `UNIQUE(agent_id, skill_id)`;** a second link for the same pair is impossible at the schema level (S1).
- **Auto's prompt renders each skill body once** — `SkillsSection` dedupes by id and by `(name, content_hash)`; the ~5k duplicated-token load is gone from the affected workspace (S2).
- **The seeders are idempotent under concurrency** — running them from every worker on the hot paths creates exactly one platform-management global row and one Auto link, with no swallowed `IntegrityError` (S3).
- **The uncapped primary skill is chosen by a real priority,** not relationship-load order; F054 is closed with a column, not a phantom (S4).
- **A foreign workspace's private skill cannot be attached** to an agent on the canonical router; global and own-workspace skills still attach (S5).
- **Integrity metric moves the right way** (dossier §G1): duplicate active `(name)` global skills and duplicate agent↔skill links both trend to 0; prompt-efficiency skill-tokens-per-turn (§G2) drops from ~43% waste toward <5% on affected Auto turns.

## What this wave gates

This is a **Wave 1** client-facing repair judged against Wave 0's numbers — it directly serves the North Star (Auto's per-turn prompt quality) and unblocks any later skills work. It is the **prerequisite for adopting the Agent Skills open standard** (the dossier's §J4 strategic move — trigger-based L1-only loading, L3 script execution): you cannot switch to on-demand skill loading while the DB still holds duplicate rows and the association has no identity guarantee. Repair the data and give the association a real priority now; the standard-adoption PRD builds on a clean base.

---

*Traceability: every story cites its dossier ref (`reports/dossiers/agents-skills.md` — §C defects 1–2, §J upgrades 1/2/5/8) and report id **P2-10** (`reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md:203`). Security §3.2.a (skill-attach visibility) folded into S5. `file:line` refs are **grep-confirmed against the current worktree `649482aa3`** (not the pinned review tree `77bc9c6d5`, which had drifted — see the Overview drift note and the Gerard's-call box). North-Star framed; PILOT lens applied; no moat framing.*

---

## ⚠️ Gerard's call (surfaced, not deferred — CLAUDE.md §6, §12)

The review tree the dossier was pinned to (`77bc9c6d5`) has drifted from this worktree (`649482aa3`). A `skills` dedupe + `UNIQUE(workspace_id, name)` + partial `UNIQUE(name) WHERE workspace_id IS NULL` **already shipped** in `alembic/versions/dedupe_skills_unique_workspace_name.py`, and it is **in the mainline chain** (`wave1a` → `wave1b` chain onto it). So the dossier's J1 "dedupe skills + partial unique on global `(name, skill_source)`" is **partly already done — but with a different key** (`name` alone, not `(name, skill_source)`). Three decisions are yours, not mine to make silently:

1. **`UNIQUE(agent_id, skill_id)` is genuinely still missing** (grep-confirmed: no PK/unique on `agent_skills` in the model or any migration; the existing migration's line-52 "implicit unique" comment is wrong). **S1 builds this — this is the real, un-done core of P2-10.** Confirmed in scope unless you say otherwise.

2. **Should the global-skill unique be re-keyed to `(name, skill_source)`** as the dossier specified, or is the shipped `(name)`-only marketplace unique sufficient? The `(name)`-only key means the 26 KB builtin `platform-management` and the 70 KB github import (both global) **cannot coexist** — one blocks the other on insert. If you want both provenances to be able to exist (builtin-core *and* a github variant), the dossier's `(name, skill_source)` key is needed and the existing migration would have to be amended. **Default I've scoped: leave the shipped `(name)` key as-is** (simpler, and S2's assembly dedup covers the render side regardless) — flag if you want the re-key.

3. **The existing migration's `agent_skills` link-dedupe is best-effort with nothing backing it** (it relies on the non-existent "implicit unique"). S1's constraint makes it durable. Confirm you want S1 to (a) re-run a defensive link-dedupe before adding the constraint (safe even if the earlier migration already ran) — **my default** — or (b) assume the earlier dedupe is complete and only add the constraint (fails if any dup link survived).
