# PRD-174 — Unified Policy Plane v1 (Wave 4)

**Status:** Draft v1 — pending approval
**Type:** Architecture / Security (the governance plane; P0 centerpiece)
**Priority:** P0 — the "conductor" that lets Auto act autonomously *within bounds*; every autonomy story depends on it
**Owner:** Gerard Kavanagh
**Author:** Gerard Kavanagh + Claude (Opus 4.8)
**Date:** 2026-07-02
**Phase:** B — Policy plane & deployability · **Size:** L · **Risk:** **HIGH** (touches every execution path — build behind a flag, characterization tests first)
**Depends on:** Wave 1 (execution spine, ✅ merged) + Wave 2 (tenant isolation, ✅ merged)
**Parent:** [PLATFORM-OS-ROADMAP.md](./PLATFORM-OS-ROADMAP.md)
**Source:** review §9.3 (the buildable target), §9.4 (Claude-Code primitive replacements), §9.5 steps 2 & 4 (migration order), §12.4 (Ring 2), §13 Wave 4, §14 (act-vs-ask decision)
**Findings register pinned to:** `37fdecc4e` — re-confirm each `file:line` on current `main` before editing
**Findings in scope:** F085, F040, F086, F059, F014, F042, F043 (+ F060 governance-asymmetry, closed by the chokepoint)
**Owner decision (LOCKED 2026-07-02):** act-vs-ask default posture = **Balanced** (see §5)

---

## Operating Principle

> **One typed gate, evaluated in one place, for every tool call — on every surface.** Auto's blueprint
> tells it what to *attempt*; the policy plane, and only the policy plane, decides what *executes*. Keep
> Claude Code's separation of model *intent* from *authorization*, but move the authority from the local
> filesystem to the **tenant control plane**. Guardrails become policy (DB-configured, one config, one
> chokepoint), not code scattered across router dependencies. **Deny outranks ask outranks allow**, and
> every denial returns a structured reason the model can read (errors-as-data).

---

## 1. Purpose

Today there is **no policy plane as a plane** (F085). Budgets, approvals, rate-limits and roles live in
three partial, contradictory mechanisms — per-router auth dependencies, action-registry flags, and the
mission approval engine — with no single configuration evaluated in one place. Guardrails stop at the
mission boundary; the API surface has no awareness of `approval_policy`. So the choice today is
**babysit-everything or trust-blindly** — the all-or-nothing autonomy the whole "Jarvis acting out" thesis
needs to escape.

Concretely, the deterministic gate stack that *does* exist — super-admin fail-closed, the autonomy dial,
confirmation, hierarchy, and a destructive backstop (`platform_executor.py:606-812`) — guards **`platform_*`
actions only**. Composio, workspace, and registry tools **route around it** (`unified_executor.py:334-364`).
Meanwhile the rate limiter is a placebo (F040), there is no pre-call budget admission gate (F086), the
dollar ceiling is model-blind across four hardcoded price tables (F059), `admin_only` is a no-op in agent
context (F014), and RBAC forks — empty permissions mean *allow-all* on one plane and *deny-all* on another
(F042), while seven routers 403 the super-admin (F043).

**W4 builds the single enforcement point** that folds all of this into one place, so Auto can act
autonomously *within bounds*, and so audit-log completeness and staged compliance (W11) have a substrate to
attach to. It is the review's centerpiece and where the Claude Code harness reference transfers most directly.

---

## 2. Background

### 2.1 What's working today (reuse, don't reinvent)

- **The deterministic gate stack exists** — super-admin fail-closed, autonomy dial (`platform_executor.py:626`),
  confirmation, hierarchy, and a non-overridable destructive backstop (`platform_executor.py:606-812`). W4
  **refactors these into shared `PolicyGate` functions**, it does not re-invent them.
- **The convergence point exists.** `UnifiedToolExecutor.execute_tool` (`unified_executor.py:243+`) is where
  platform, workspace, Composio-per-action, and registry tools already converge — the natural single chokepoint.
- **The approval primitive exists.** PRD-163 already makes approval a state transition with an in-chat card;
  `ask` reuses it (card for chat, approval row for headless).
- **The autonomy dial exists.** `auto` mode is the existing full-autonomy dial that correctly never satisfies
  the super-admin gate and never bypasses the destructive backstop.
- **Per-agent ACL exists** (`tool_router.py:265-277`) — pre-filter at surface time; the plane reserves
  call-time gates for `ask`/conditional verdicts.
- **A model-aware price registry exists** in the DB (unused) — F059's fix wires the dollar gate to it.

### 2.2 What's broken / missing

- **F085 — no policy plane.** Guardrails scattered across router deps, action-registry flags, and the mission
  approval engine; the API surface has no awareness of `approval_policy` (`approval_policy.py`, mission-runtime only).
- **F060 — governance asymmetry.** Budget/approval gates exist for missions but **not** board tasks or
  playbooks; Composio/workspace/registry tools bypass the gate stack (`unified_executor.py:334-364`).
- **F086 — no pre-call budget admission gate.** No `BudgetExceeded` class exists; the tool loop, chat,
  agent-factory and recipe loops all issue LLM/tool calls with **no spend check** (`tool_loop.py:326`); the
  LLM manager only cost-logs *after* the call.
- **F059 — model-blind dollar ceiling.** Four independent hardcoded price tables coexist with an unused
  model-aware DB registry, so the approval-budget primitive approves/blocks on wrong dollars.
- **F040 — placebo rate limiter.** `SlowAPIMiddleware` is never added despite the `Limiter` being
  constructed; the stack's only fail-*open* gate (`platform_executor.py:794`).
- **F014 — admin_only no-op in agent context.** `is_admin` auto-flips true whenever the *workspace* has any
  admin member (true for every workspace), not the calling principal (`platform_executor.py:641-645`).
- **F042 — RBAC god-key/null-key fork.** Empty permissions = allow-all on the widget plane, deny-all on the
  board plane; one no-permission key is a god-key on one and a null-key on the other.
- **F043 — super-admin locked out.** Seven routers gate admin functions with `system_role == 'admin'`, which
  403s super-admin entirely.

### 2.3 Why now

W1 (spine) and W2 (tenancy) are merged, so the plane has a working loop to wrap and a tenant boundary to
enforce. The act-vs-ask decision is **locked (Balanced)** — the review is explicit that *until* it's decided
the plane "has no target semantics and Auto's autonomy stays all-or-nothing." Everything after W4 (moat loop
W7, governance/compliance W11) depends on this enforcement + audit substrate existing. Build it once, on the
converged spine, rather than three times.

---

## 3. Findings in scope

| ID | Sev | Location (pinned `37fdecc4e`) | Defect | Fix |
|---|---|---|---|---|
| **F085** | Critical (missing) | `core/services/approval_policy.py` (mission-runtime only) | No policy plane; guardrails in 3 partial mechanisms | New `modules/policy/` package + one `PolicyGate` chokepoint |
| **F060** | High | `unified_executor.py:334-364`; `api/tasks.py`, `main.py:885` | Composio/workspace/registry bypass the gate; governance is missions-only | Route every tool through `PolicyGate.check()` from `UnifiedToolExecutor` |
| **F086** | High (missing) | `tool_loop.py:326` | No pre-call budget admission gate; no `BudgetExceeded` | `on_pre_tool` dollar admission gate + `BudgetExceeded` |
| **F059** | High | 4 hardcoded price tables + unused DB registry | Model-blind dollar ceiling | Model-aware pricing from the DB registry |
| **F040** | High | `platform_executor.py:794`; middleware unregistered | Placebo rate limiter (fail-open) | Register `SlowAPIMiddleware`; fail **closed** |
| **F014** | High | `platform_executor.py:641-645` | `admin_only` no-op — `is_admin` from workspace, not caller | Derive from caller's own role (PRD-168 actor identity); explicit default-off "agents inherit admin" |
| **F042** | High | widget vs board planes | Empty perms = allow-all one plane, deny-all other | One empty-permission semantic (**empty = deny**), shared helper |
| **F043** | High | 7 routers `system_role=='admin'` | Super-admin 403'd | Shared `super_admin ⊇ admin ⊇ user` role helper |

---

## 4. Design & changes

Built as **two migration steps** (review §9.5 steps 2 & 4), behind an `AUTOMATOS_POLICY_PLANE` flag with
characterization tests written *first* (high risk — it touches every execution path).

### 4.1 Step 2 — one policy chokepoint (ships independently)

- **Extract the platform gate stack into `modules/policy/` `PolicyGate`.** Move the super-admin fail-closed,
  autonomy-dial, confirmation, hierarchy, and destructive-backstop functions (`platform_executor.py:606-812`)
  into shared, tenant-scoped `PolicyGate` functions.
- **Invoke `PolicyGate.check()` from `UnifiedToolExecutor.execute_tool`** (`unified_executor.py:243+`), so
  Composio, workspace, and registry tools **stop bypassing** it (F060, F085). All three loops (chat, recipe,
  heartbeat) inherit universal guardrails on their platform paths immediately.
- **F040 — make the rate limiter fail closed:** register `SlowAPIMiddleware`; a limiter that can't evaluate
  denies rather than allows.
- **F014 — owner-fallback behind an explicit default-off setting:** `is_admin` derives from the caller's own
  role via the PRD-168 actor identity; "agents inherit admin" becomes an explicit, default-off workspace policy.
- **F042/F043 — one role + permission semantic:** a shared `super_admin ⊇ admin ⊇ user` helper (fixes the 7
  routers) and **empty-permissions = deny** everywhere (kills the god-key).

### 4.2 Step 4 — the policy plane v1

- **A typed event bus** (`modules/policy/`): `RunStart`, `PreToolUse`, `PostToolUse`, `PostToolBatch`,
  `RoundEnd`, `RunEnd`, `PreCompact`. Handlers return a verdict object
  `{decision: allow|deny|ask|defer, updated_input, injected_context, reason}` — the exact Claude Code verdict
  semantics, re-keyed for tenancy. **deny > ask > allow.**
- **Add the `on_pre_tool` seam** beside the dedup check in `_execute_round` (`tool_loop.py:340`). Today only
  post-tool hooks exist (`tool_loop.py:176-182,254`); the pre-tool seam is the natural attach point for
  blueprint, approval, and budget policy (F086, ADJUSTED).
- **F086 + F059 — pre-call dollar admission, model-aware:** inside `on_pre_tool`, a `BudgetExceeded` admission
  gate checks `Workspace.plan_limits` (cost/tokens, not just concurrency) using the **model-aware DB price
  registry** — retiring the four hardcoded tables.
- **Permission modes = act-vs-ask, mapped onto existing primitives (not invented):**
  - `plan` → tools filtered to `permission_level=='read'` (also fixes `service.py:807` stripping *all* tools
    in plan mode so Auto can research while planning).
  - `default` → `requires_confirmation` routes to **ask** — the PRD-163 in-chat card for chat, an approval row
    for headless.
  - `auto` → the existing full-autonomy dial (`platform_executor.py:626`) — never satisfies the super-admin
    gate, never bypasses the destructive backstop.
  - `dontAsk` → auto-denies anything unapproved (the right default for webhooks and scheduled runs).
- **Structured errors-as-data:** formalize the half-existing envelope (`tool_router.py:715-736`) to
  `{code, message_for_model, remediation, retryable}`; **every policy denial returns its reason as tool
  content** so the model can adapt instead of erroring.
- **Prerequisite gates (harness-refuses-until-X, not prompt-remembers-X):** read-before-write on
  documents/records; plan-approved-before-dispatch (extend PRD-163 to destructive board Run-Now);
  integration-trusted-before-exposed.

### 4.3 What replaces the Claude Code primitives (§9.4 — the design's north star)

The single most transferable Claude Code pattern is hooks-as-policy that lives *outside* the model. Each
dangerous single-user primitive gets a deliberate tenant-safe replacement — this is what the plane implements:

| Claude Code primitive | Why it doesn't transfer | Replacement in Automatos (this PRD) |
|---|---|---|
| `bypassPermissions` / skip-prompts | "isolated VM" justification; no tenant-safe equivalent | autonomy dial bounded by non-overridable deny rules + super-admin gate + destructive backstop (shape exists `platform_executor.py:606,626,796`) |
| Model sees every tool schema | wasteful/leaky across tenants | pre-filter at surface time (per-agent ACL `tool_router.py:265-277`); call-time gates only for ask/conditional |
| Local trust roots (`~/.claude`, repo `.claude/`) | a repo can grant itself permissions | config authority in the **tenant control plane** — DB rows with provenance; tenant content cannot self-grant |
| Hooks as shell commands | RCE by configuration | **keep the event taxonomy + verdict semantics; execution = in-process handlers + signed webhooks** (NOT shell) |
| Bash string-prefix matching | fragile (docs say so) | typed-tool registry with `permission_level`; `workspace_exec` sandbox-first |
| Session-local "don't ask again" | ephemeral, unauditable | durable, auditable, revocable approval grants with scope + expiry (extend PRD-163 rows) |
| CLAUDE.md-as-policy | shapes intent, never enforces | blueprints/prompts shape intent; enforcement lives **only** in the policy plane |
| Single-user credentials | no tenant isolation | per-tenant vaulted, scoped, rotated secrets at the execution boundary (Composio per-workspace pattern) |

---

## 5. The Balanced policy — the workspace policy document the plane evaluates

Encode the **act-vs-ask decision once** (LOCKED: **Balanced**), as the DB-configured workspace policy the
plane reads — not per-surface toggles (the three partial planes today already contradict each other). Defaults:

- **Auto (no ask):** reads; low-risk internal writes (draft docs, update own board task status, internal
  memory writes, research/plan).
- **Ask (PRD-163 approval card / row):** **destructive ops** (deletes, board **Run-Now**); **external
  side-effects** (Composio sends/refunds/discounts, email/channel posts, Shopify writes); **over-budget** spend
  (workspace `plan_limits`).
- **Templates / brand-kits:** Auto **drafts**; a human **approves publish** (not human-only, not auto-publish).
- **Board:** mutations stay **chat/assignment-driven** — no free "create-task" affordance in v1.
- **Composio side-effect tier:** refunds/discounts/sends default to **ask**; tunable per-workspace.

All of the above are **workspace-policy rows, tunable** — the PRD ships the Balanced defaults; changing them is
config, not a deploy.

---

## 6. Test-first acceptance

Write these **failing first**, then implement to green (characterization tests before the refactor, per the
high-risk flag):

1. **Headline (review §13):** a **characterization test proves a denied tool call never executes** and returns
   a **structured denial the model can read** (`{code, message_for_model, remediation, retryable}`).
2. **F060/F085:** a Composio (and a workspace, and a registry) tool call is evaluated by `PolicyGate` — it no
   longer bypasses the gate; a denied one doesn't execute.
3. **F086/F059:** a tool call that would exceed the workspace `plan_limits` (priced **model-aware**) raises
   `BudgetExceeded` **before** the call; a within-budget call proceeds.
4. **F040:** with the limiter unable to evaluate, the request is **denied** (fail-closed), not allowed.
5. **F014:** an agent call to an `admin_only` action is denied when the caller isn't admin (workspace having
   *an* admin no longer flips it true); allowed only with the explicit default-off "agents inherit admin" policy on.
6. **F042/F043:** an **empty-permission** key is denied on **both** planes; a **super_admin** passes all seven
   previously-`admin`-only routers.
7. **Act-vs-ask (Balanced):** a destructive/external/over-budget action routes to **ask** (emits a PRD-163
   card/row and does not execute until approved); a read/low-risk-internal action executes without asking.
8. **Deny > ask > allow:** when handlers disagree, deny wins over ask wins over allow.

**Wave bar:** every tool call on every surface (chat, recipe, board, heartbeat, webhook) passes through one
`PolicyGate`; guardrails are universal, not platform-tool-specific; Auto acts within the Balanced bounds without
babysitting.

---

## 7. Risks & rollback

- **Highest-risk wave — it touches every execution path.** Mitigation: build behind `AUTOMATOS_POLICY_PLANE`
  (default off in this PR), characterization tests first, roll out per-surface.
- **Do NOT implement as shell hooks** — Claude Code's local model doesn't fit a SaaS backend and the harness
  docs themselves warn Bash-string matching is fragile. In-process typed handlers with tenant scope only.
- **Sequencing:** Step 2 (chokepoint) ships and green **before** Step 4 (plane) so nothing new enters
  ungoverned while more surfaces join.
- **Rollback:** the flag disables the plane and falls back to today's per-router gates; each finding-fix is an
  independent commit.

---

## 8. References

- Review §9.3 (buildable target — the loop + policy mechanics), §9.4 (primitive replacements), §9.5 steps 2 & 4
  (migration order), §12.4 (Ring 2 — the policy plane), §13 Wave 4, §14 (act-vs-ask, LOCKED Balanced)
- Findings F085/F060/F086/F059/F040/F014/F042/F043 — `reports/PLATFORM_OS_REVIEW_2026-07-01.md`
- Reuses: PRD-163 approval cards, PRD-168 actor identity, the DB price registry, the per-agent ACL
- CLAUDE.md §4 (no shims), §5 (delete what you supersede), §12 (no unilateral descope)
