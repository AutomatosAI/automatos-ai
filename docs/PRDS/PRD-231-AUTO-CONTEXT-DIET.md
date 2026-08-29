# PRD-231 — Auto's Context Diet: the Identity / Operations Split

> **Status:** Draft — approved direction (Gerard, 2026-08-29: "I like this... 60% improvement and Auto is less cluttered so he can think clearer"). Scope deliberately tight per his instruction — the broader "how Auto takes on load" review is a named follow-up, not this PRD.
> **Type (per CLAUDE.md §3):** Refactor/consolidation — every mechanism already exists; this PRD re-shapes content and flips one loading flag. **Zero new machinery.**

## 1. Overview

Auto carries **~28,000 tokens of soul + skill on every single turn** — his entire operations manual plus his personality notes, where the personality notes duplicate half the manual. This PRD splits *identity* (always-on) from *operations reference* (on-demand), and removes the soul↔skill duplication, cutting Auto's standing context cost by **~63% (measured)** while keeping every capability reachable.

This is the same pattern Gerard already applied to the tool registry (semantic pre-prompt narrowing so Auto isn't flooded with tool options) — applied to the two biggest always-on blobs.

## 2. Current reality (measured 2026-08-29, chars÷3.6)

| Blob | Tokens | Loaded |
|---|---|---|
| Soul (`custom_soul` default, `auto-cto-custom-soul.txt`) | **2,375** | every turn |
| Skill charter (identity/doctrine/authority/routing, §A–§H of `platform-management`) | **9,138** | every turn |
| Skill ops reference (tool-by-tool cookbook, §0–§19) | **16,483** | every turn |
| **Total standing cost** | **27,996** | **every turn** |

- `platform-management` is pinned in `SKILL_CORE_ALWAYS_ON` (`modules/context/sections/skills.py:33`) → full L2 body renders unconditionally. Every *other* skill already renders as a one-line L1 catalog entry with on-demand activation.
- **The on-demand mechanism exists and is live:** `platform_load_skill` (`modules/tools/discovery/actions_skills.py:30`, handler `handlers_skill_runtime.py:65` — "S2: trigger-based L2 activation"). Non-core skills are listed L1; the model calls `load_skill <name>` to pull the body for that turn. Activation is already logged (`[skills] activation: core_always_on=… l1_offered=…`).
- **The soul duplicates the skill.** Five soul sections restate skill charter content, verified 5/5 present: **My Role** (≈ charter identity), **My Authority** (≈ §B), **How I Think** (≈ §F), **My Operating Rhythm** (≈ §D), **My Routing Rules** (≈ §E/§H). Two always-on copies of the same rules that can drift — and since the soul is *tenant-editable* (Settings → Orchestrator → Soul & Personality, `custom_soul` via `PUT /api/workspaces/current/orchestrator`), a tenant edit can silently contradict platform routing today.
- Skills repo (`automatos-skills`) is the authoring source for the skill; the seed is a generated copy (PR #640's `scripts/sync-auto-skill.py` + `_refresh_builtin_if_stale` propagation). The soul is tenant-owned; its seed file is only the default for new/uncustomized workspaces (PRD-226 backfill skips customized souls by design).

## 3. Goals

- G1: Auto's standing soul+skill cost drops from ~28.0k to **~10.4k tokens/turn** (soul ~1.3k identity-only + charter ~9.1k), with the ops reference loading only on turns that perform platform operations.
- G2: **One home per rule.** The soul contains no routing/authority/cadence content; operational rules live only in the skill. A tenant soul edit can no longer contradict platform routing.
- G3: No capability regression: Auto always *knows what exists* (L1 index line + the charter's pointers) and pulls the cookbook when doing the work.
- G4: Both new artifacts stay repo-authored with the drift guard (the source-of-truth rule made structural).

## 4. Non-goals

- The broader context-assembly review (agent factory prompt, platform_actions section, composio overlay, memory/business-graph budgets — the "how Auto takes on load" pass) — **named follow-up, out of this PRD by Gerard's instruction.**
- No changes to the tool registry's semantic narrowing (already good).
- No per-domain splitting of the ops reference in v1 (one ops skill; split further only if activation data says so — see §8).
- No forced rewrite of customized tenant souls — the slim soul is the new *default*; the PRD-226 backfill continues to skip customized personas (their owners can adopt the slim default from Settings if they wish).

## 5. Design

### Component A — slim the soul to identity-only (repo default + seed)

Remove from the default soul the five sections that duplicate the skill charter (**My Role / My Authority / How I Think / My Operating Rhythm / My Routing Rules**). Keep everything that is genuinely *identity*: Who I Am, Personality, How I Treat People, Strong Opinions, Memory & Recall honesty, Under Pressure, Ambitions, Sacred Ground, My Promise, the technical-communication override. Result ≈ 1.3k tokens of pure character. One cross-reference line replaces the removed content: *"How I manage — lanes, authority, cadence, routing — lives in my platform-management skill; that is the single source."*

### Component B — split the skill: charter stays core, ops becomes an on-demand skill

- `automatos-skills/team/auto/SKILL.md` keeps **§A–§H (the charter)** — this remains `platform-management`, stays in `SKILL_CORE_ALWAYS_ON`, always-on at ~9.1k.
- The **Operations Reference (§0–§19)** moves to a new repo skill **`team/auto-ops/SKILL.md`** → skill name **`platform-operations`**, seeded and **assigned to Auto as a NON-core skill**: it renders as one L1 catalog line every turn, and Auto calls `platform_load_skill platform-operations` when actually performing operations. The L1 description is the trigger and must be written to bite: *"The tool-by-tool cookbook — exact JSON for marketplace installs, agent wiring, heartbeats, playbooks, board, scheduling, missions, governance, reports, HARNESS, files, notifications. LOAD THIS before executing any platform operation."*
- The charter keeps a 10-line **index** of what the ops skill contains (section names only), so Auto never forgets what exists.
- The charter's §H doctrine, the 226 lock-step (contract fragment verbatim), and the doctrine anchors stay in the charter — all existing 226 tests keep passing against `platform-management`.

### Component C — plumbing (all existing patterns)

- `scripts/sync-auto-skill.py` → syncs **both** repo files to **two** seed files (`platform-management-skill.md`, `platform-operations-skill.md`), same banner + self-checks.
- `seed_auto_agent.py` → upserts + assigns both skills (same advisory-lock + `ON CONFLICT` pattern).
- `_BUILTIN_PATHS` gains the second entry so `_refresh_builtin_if_stale` propagates ops edits to existing workspaces too.
- `SKILL_CORE_ALWAYS_ON` unchanged (`platform-management` only) — that's the flag doing the work.

### Component D — the drift guard (folds in finding #3)

CI check: run `sync-auto-skill.py --check` (no-write mode) asserting both seed files match the repo sources — the source-of-truth rule becomes structural instead of remembered. (Repo checkout for CI: vendor the check against the seed's recorded source hash if cross-repo fetch is unwanted — decide in implementation; the check must fail loud, not skip silent.)

## 6. Stories

- **US-001** — Repo: split `team/auto/SKILL.md` (charter) / create `team/auto-ops/SKILL.md` (ops, frontmatter `name: platform-operations`); slim soul default; charter gains the ops index line + soul gains the single-source cross-reference. *(automatos-skills PR)*
- **US-002** — Sync script handles both files + `--check` mode; regenerate both seeds. *(automatos-ai)*
- **US-003** — `seed_auto_agent` seeds+assigns `platform-operations` (non-core); `_BUILTIN_PATHS` second entry; existing-workspace assignment backfill for the new skill (same idempotent pattern as the platform-management assignment).
- **US-004** — Verification: 226 doctrine/contract tests green against the slimmed charter; new tests — ops skill renders L1-only for Auto (not full body), `platform_load_skill platform-operations` returns the cookbook, soul default carries no routing/authority sections (anti-duplication guard: the five removed headers must NOT reappear in the soul seed).
- **US-005** — CI drift guard (Component D).
- **US-006** — Measure, don't guess: capture the `[skills] activation` log line + context-size telemetry for a week; record actual per-turn savings and ops-skill activation hit-rate in the PRD's results section.

## 7. Risks & mitigations

- **Auto doesn't load the cookbook when needed** → fumbled operation. Mitigations: biting L1 trigger text; the charter's index; §H doctrine point 1 already tells Auto to ground before acting. US-006's activation telemetry tells us within days if the trigger under-fires; the rollback is one line (add `platform-operations` to `SKILL_CORE_ALWAYS_ON`).
- **A choppy seam** — an ops section that's actually decision-relevant (e.g. canonical page names in §0). Accepted: answering "where do I find X" *is* an ops lookup; loading the skill for it is correct behavior, not a failure.
- **Customized souls keep the old fat default** — by design (tenant-owned). The duplication *risk* for them also shrinks anyway, because the skill side stops being contradictable-by-default going forward.

## 8. Follow-ups (explicitly parked, per Gerard)

1. **"How Auto takes on load"** — the full context-assembly review: agent-factory prompt, `platform_actions` section, composio overlay, memory/business-graph budgets, per-section token telemetry. The CHATBOT mode loads ~10 sections (`modules/context/modes.py:40-50`); tools alone were measured ~11k in the 2026-07-23 tool-surface review. This PRD's US-006 telemetry is the entry point.
2. Per-domain split of `platform-operations` (5 domain skills ≈ 3–4k each) if activation data shows Auto loading the full 16.5k frequently.
3. The `/skill` direct-reference escape hatch from PRD-71's open questions.

## 9. Expected result (to be replaced by US-006 measurements)

| | Before | After |
|---|---|---|
| Always-on soul+skill | 27,996 tok | **~10,400 tok** |
| Ops cookbook | every turn | only on operating turns |
| Rule homes | 2 (drift-prone) | 1 |
