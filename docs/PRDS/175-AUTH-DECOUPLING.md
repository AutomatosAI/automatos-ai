# PRD-175 — Auth Decoupling (open-core) (Wave 5)

**Status:** Draft v1 — pending approval
**Type:** Architecture (open-core / auth)
**Priority:** P0 — unblocks Wave 6 (deployability) and the open-core thesis
**Owner:** Gerard Kavanagh
**Author:** Gerard Kavanagh + Claude (Opus 4.8)
**Date:** 2026-07-02
**Phase:** B — Policy plane & deployability · **Size:** high risk but a one-function seam · **Risk:** high (auth touches every request), contained by an existing seam
**Depends on:** none hard — helps W6 (deployability & reliability baseline)
**Parent:** [PLATFORM-OS-ROADMAP.md](./PLATFORM-OS-ROADMAP.md)
**Source:** review §4/§5 (F008, F075), §13 Wave 5; the absent PRD-150 (open-core / auth-decoupling)
**Findings register pinned to:** `37fdecc4e` — re-confirm each `file:line` on current `main` before editing
**Findings in scope:** F008, F075

---

## Operating Principle

> **One core, two editions, one seam.** The model of *who is calling* is chosen once, at the edition
> boundary, and every surface reads that one answer. In `saas` the boundary is Clerk; in `local` it is a
> single auto-authenticated local user in a single local workspace — **no login, no external SaaS, no
> Clerk env.** We do not fork the auth system: we mount the existing Clerk path only when the edition asks
> for it, and fall through to the identity the backend *already has* (`hybrid.py`'s anonymous dev-fallback)
> otherwise. The edition is a flag, not a code path the reader has to reason about at every call site.

---

## 1. Purpose

Today Automatos **cannot be run from a fresh clone by anyone without a Clerk tenant.** The frontend mounts
`ClerkProvider` **unconditionally** (`providers.tsx:33`) and `clerkMiddleware` calls `auth.protect()` on every
non-public route (`middleware.ts:14`), with **zero edition flag anywhere** (F008). Without a Clerk
publishable key the UI renders nothing — sign-in itself is a Clerk surface — so the open-core thesis ("`git
clone && docker compose up` → a working local instance, no login, zero external SaaS", roadmap §2
Deployability bar) is **undeployable from zero**. This is the frontend half of the absent PRD-150 that never
landed; the backend half already exists and is unused.

The backend is in fact ready: `get_request_context_hybrid` (`hybrid.py:653`) already supports Clerk JWT, API
key, **and an anonymous dev-fallback** (`hybrid.py:805-826`) gated by `config.REQUIRE_AUTH`
(`config.py:149`). What is missing is (a) a single, named **edition flag** so the frontend stops mounting
Clerk unconditionally and the backend's local identity is *intentional* rather than an accidental
"auth-not-configured" fallthrough, and (b) moving the one **hardcoded `@automatos.app` admin domain**
(`clerk.py:201`) into `config.py` so the SaaS topology is configuration, not a literal (F075).

**W5 mounts Clerk only in the `saas` edition and serves a no-op local identity in `local` — behind an
`AppAuth` facade over the seam that already exists (`setClerkTokenGetter`), not a new auth system.** It is
high-risk by blast radius (auth is on every request) but small by diff: the review names it "a one-function
seam." It unblocks W6 (which boots that local instance for real) and is the load-bearing half of open-core.

---

## 2. Background

### 2.1 What's working today (reuse, don't reinvent)

- **The token seam already exists.** `apiClient.setClerkTokenGetter(...)` (`api-client.ts:123`) is installed
  by `ClerkApiClientProvider` (`clerk-api-client-provider.tsx:23`). This is the "one-function seam" the review
  names: swapping *which* getter is installed (a real Clerk `getToken`, or a no-op returning `null`) is the
  whole frontend token story. **Do not hand-roll a new auth client** — wrap this.
- **The backend already has a local identity.** `get_request_context_hybrid` (`hybrid.py:653`) already
  resolves Clerk JWT → API key → **anonymous dev-fallback** (`hybrid.py:805-826`), returning a
  `RequestContext` on a resolved workspace with `auth_type="anonymous"` when `config.REQUIRE_AUTH` is false.
  This IS the local session — W5 makes it *deliberate and edition-driven*, not an "auth misconfigured"
  side-effect.
- **Config already holds every Clerk key.** `CLERK_SECRET_KEY`, `CLERK_PUBLISHABLE_KEY`, `CLERK_JWKS_URL`,
  `CLERK_AUDIENCE`, plus `DEFAULT_WORKSPACE_ID` and `REQUIRE_AUTH` all live in `config.py:149,385-389`. The
  edition flag joins them; no new env-reading pattern (CLAUDE.md §4 — no `os.getenv` outside `config.py`).
- **Public routes are already enumerated.** `middleware.ts:3-10` lists the non-protected matcher (`/sign-in`,
  `/sign-up`, webhooks, …). The `local` edition is the degenerate case: *every* route is public.

### 2.2 What's broken / missing

- **F008 — `ClerkProvider` + `auth.protect()` are unconditional; no edition flag.** `providers.tsx:33` wraps
  the entire app in `<ClerkProvider>`; `middleware.ts:12-15` runs `clerkMiddleware` and calls `auth.protect()`
  on all non-public routes. With no Clerk publishable key the app serves nothing and there is **no flag to
  turn Clerk off**. PRD-150's frontend half never landed — this is why open-core is undeployable from zero.
  (Clerk coupling on the frontend spans ~34 files; the mount is the choke point, so W5 gates the *mount*, not
  all 34.)
- **F075 — hardcoded `@automatos.app` admin domain.** `clerk.py:201-205` decides platform-staff by
  `email.lower().endswith("@automatos.app")` — a literal baked into the SaaS defence-in-depth check (part of
  the review's "hardcoded SaaS topology" medium cluster). It must move to `config.py` so a self-hosted/SaaS
  operator sets their own staff domain by configuration.

### 2.3 Why now

The open-core Deployability bar (roadmap §2) is **W5 → W6**: W5 makes the app *not require* Clerk; W6 boots
the fresh-clone local instance and asserts a 200 on health with no external credentials. Until W5 lands, W6
has nothing to boot into — the UI is dark without a Clerk tenant. W5 has **no hard dependency** (the backend
seam is already merged), so it is startable now, and it is the review's Phase-B deployability unlock. It is
also the honest substrate for the open-core *messaging*: we cannot claim "runs locally, no login" until the
flag exists and the acceptance test proves it.

---

## 3. Findings in scope

| ID | Sev | Location (pinned `37fdecc4e`) | Defect | Fix |
|---|---|---|---|---|
| **F008** | High (missing) | `frontend/components/providers.tsx:33`; `frontend/middleware.ts:12-15` | `ClerkProvider` + `auth.protect()` mounted **unconditionally**; **no edition flag** → app serves nothing without a Clerk tenant; PRD-150's frontend half never landed | `AppAuth` facade + `AUTH_EDITION` (local\|saas): mount Clerk + `auth.protect()` **only when `saas`**; a no-op local identity (auto-authenticated single local user/workspace) when `local` |
| **F075** | Medium | `orchestrator/core/auth/clerk.py:201-205` | Hardcoded `@automatos.app` platform-staff domain (hardcoded SaaS topology) | Move the staff domain to `config.py` (`PLATFORM_STAFF_EMAIL_DOMAIN`); read from config, not a literal |

---

## 4. Design & changes

Minimal diff. The seam and the local identity both already exist — W5 adds **one flag, two conditional
mounts (frontend), one config read (backend)**, and deletes the literal it supersedes (CLAUDE.md §5). No
backward-compat shim, no `_legacy` provider (CLAUDE.md §4).

### 4.1 The edition flag — `AUTH_EDITION` (one flag, one source of truth)

- **Backend:** add `AUTH_EDITION` to `config.py` beside the API-security block (`config.py:149`), values
  `local | saas`, default **`saas`** (the SaaS default must not change silently for the running product;
  local is opt-in). In `local`, the edition **implies** the local-session posture — `REQUIRE_AUTH` is treated
  as `false` and a `DEFAULT_WORKSPACE_ID` is required — so operators set *one* flag, not three that can
  contradict. (Re-confirm `REQUIRE_AUTH`/`DEFAULT_WORKSPACE_ID` names on `main`.)
- **Frontend:** surface the same edition to the client as `NEXT_PUBLIC_AUTH_EDITION` (build-time public env,
  the existing `NEXT_PUBLIC_*` convention already used for `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`). One small
  `lib/auth-edition.ts` reads it once and exports `authEdition` / `isSaaS` — no scattered `process.env` reads.

### 4.2 `AppAuth` facade — wrap the seam, don't replace it

Introduce a thin **`AppAuth`** boundary on the frontend that owns the three edition-conditional mounts. It is
a facade over the *existing* pieces, not a new auth system:

- **`providers.tsx:33` becomes conditional.** In `saas`, render `<ClerkProvider>` exactly as today (keep the
  Studio `appearance` block unchanged) wrapping `<ClerkApiClientProvider>`. In `local`, render **neither** —
  wrap the same children in a `LocalAuthProvider` that installs a **no-op token getter**
  (`apiClient.setClerkTokenGetter(async () => null)`, reusing the seam at `api-client.ts:123`) so the API
  client sends no bearer and the backend falls to its local identity. Everything below the auth boundary
  (`RoleProvider`, `ThemeProvider`, `WorkspaceProvider`, children) is identical across editions.
- **`middleware.ts` becomes conditional on the edition.** In `saas`, keep `clerkMiddleware` + `auth.protect()`
  and the existing public-route matcher unchanged. In `local`, export a pass-through middleware (no
  `clerkMiddleware`, no `auth.protect()`) so **every** route is served — the `local` edition treats the whole
  app as public. Gate on `NEXT_PUBLIC_AUTH_EDITION` so no Clerk symbol executes under `local`.
- **The Clerk-bound auth routes are `saas`-only.** `app/sign-in`, `app/sign-up` (and any Clerk hook usage on
  the render path) must not be reachable/mounted under `local`; there is no login in `local`. Redirect `local`
  visits of those routes to `/`.
- **`ClerkApiClientProvider` is reused unchanged in `saas`** — it already installs the real Clerk getter via
  the seam. `local` swaps in the no-op getter; same seam, different getter. That is the "one-function seam."

> The facade **must not** duplicate identity logic. It only *chooses which existing provider mounts*. Backend
> identity stays entirely in `hybrid.py`; the frontend never becomes a second source of truth for who the user
> is (CLAUDE.md §2 reuse; roadmap "one core two editions behind a flag").

### 4.3 Backend — make the local session deliberate (reuse `hybrid.py`)

- **No new auth dependency.** `get_request_context_hybrid` already yields an anonymous `RequestContext` on a
  resolved workspace when `REQUIRE_AUTH` is false (`hybrid.py:805-826`). Under `AUTH_EDITION=local`, this is
  the local session. The only change is to make it *edition-driven*: when `AUTH_EDITION=local`, `REQUIRE_AUTH`
  is forced false and the fallback resolves the single local user/workspace via `DEFAULT_WORKSPACE_ID`
  (existing `config.py:389` + `hybrid.py:817`). No `saas` behaviour changes — `saas` keeps `REQUIRE_AUTH`
  secure-by-default and the full Clerk→API-key→(fail-closed) chain.
- **Boot guard.** On startup, if `AUTH_EDITION=saas`, require the Clerk env (`CLERK_JWKS_URL` /
  `CLERK_SECRET_KEY`) to be present and fail fast with a clear message if not (a `saas` boot with no Clerk is a
  misconfiguration, not a silent anonymous downgrade). If `AUTH_EDITION=local`, require `DEFAULT_WORKSPACE_ID`
  (W6 seeds it). This keeps the two editions from silently blending into today's accidental fallthrough.

### 4.4 F075 — admin domain to config

- Add `PLATFORM_STAFF_EMAIL_DOMAIN` (default `automatos.app`) to `config.py` in the API-security block. At
  `clerk.py:201`, replace the literal
  `email.lower().endswith("@automatos.app")` with a check against
  `config.PLATFORM_STAFF_EMAIL_DOMAIN` (and update the paired warning text at `clerk.py:205`). Confirm no
  other `@automatos.app` staff-gate literal exists (grep found only `clerk.py:201-205`; the
  `agent@automatos.app` git-identity default in `workspace_manager.py:218` is unrelated and out of scope). In
  `local` this check never runs (no Clerk JWTs), so the local admin story is the local identity itself.

### 4.5 Prior context — this is the absent PRD-150

PRD-150 (open-core / auth-decoupling) planned: OSS-core / private-SaaS; make Clerk **pluggable behind an
`AuthProvider`** so OSS runs local no-login; the graph found a **3-layer Clerk coupling** — *mechanism*
(the token getter + middleware), *schema* (Clerk claims → `UserContext`), *~19-file leakage* across the
frontend (current grep: ~34 frontend files reference `@clerk`, ~63 backend `.py` files reference Clerk). Gerard
chose **full package-split + centralize-identity + full-stack**. W5 delivers the *deployability-critical* half
of that: the **mount** is edition-gated (the choke point that makes `local` boot), and backend identity is
centralized on the one `hybrid.py` dependency. W5 does **not** re-plumb all ~34/63 leakage sites or split the
Clerk package out of the frontend bundle — that is the larger PRD-150 package-split, and pulling it into W5
would balloon a one-function-seam wave into a repo-wide refactor. **Whether the full leakage-elimination /
package-split rides in W5 or is its own PRD is an owner decision (§6), not a silent descope (CLAUDE.md §12).**

---

## 5. Test-first acceptance

Write these **failing first**, then implement to green. The wave's DoD (review §13): *the app boots and
serves a local session with `AUTH_EDITION=local` and no Clerk env set.*

1. **Headline (review §13) — backend local session, zero Clerk env.** With `AUTH_EDITION=local`, a
   `DEFAULT_WORKSPACE_ID` seeded, and **no `CLERK_*` env vars set**, the backend boots and a request with **no
   bearer token** resolves to an authenticated local `RequestContext` (single local user, the local
   workspace) — not a 401. Assert `auth_type` is the anonymous/local type and the workspace is the configured
   default. This is the exact gap that blocks open-core.
2. **F008 frontend — no Clerk mount under `local`.** With `NEXT_PUBLIC_AUTH_EDITION=local` and no publishable
   key, the app **renders** (children mount) and **`ClerkProvider` is not in the tree** (the `LocalAuthProvider`
   is), and the installed api-client token getter returns `null`. A test on `providers.tsx` asserts the
   conditional picks the local branch; a middleware test asserts a protected path is served (no
   `auth.protect()` redirect) under `local`.
3. **SaaS edition unchanged.** With `AUTH_EDITION=saas` and Clerk env present, `ClerkProvider` **is** mounted,
   `clerkMiddleware`/`auth.protect()` guards a non-public route (unauthenticated → redirect/401), and the real
   Clerk token getter is installed. A `saas` boot with **missing** Clerk env **fails fast** (boot guard, §4.3),
   not a silent anonymous downgrade.
4. **F075 — admin domain from config.** `clerk.py`'s staff check reads `config.PLATFORM_STAFF_EMAIL_DOMAIN`:
   a Clerk `admin`-metadata claim on an email **matching** the configured domain keeps `admin`; a
   **non-matching** email is demoted to `user`; changing the config value changes the accepted domain (no
   `@automatos.app` literal remains — assert by grep in the test/CI note).

**Wave bar:** `AUTH_EDITION=local` + no Clerk env → backend serves an authenticated local session **and** the
frontend renders with no Clerk mount; `AUTH_EDITION=saas` + Clerk env → Clerk mounts and protects routes
exactly as today; the admin domain is read from config on both. (The fresh-clone `docker compose up` → 200-on-
health smoke test is **W6**, which consumes this flag.)

---

## 6. Risks & rollback

- **Blast radius is auth-on-every-request, but the diff is a seam.** Mitigation: the `saas` path is
  byte-for-byte the current behaviour (same `ClerkProvider`, same `middleware`, same `hybrid.py` chain); only
  the `local` branch is new, and `local` is opt-in (default `saas`). Ship the flag defaulting to `saas` so
  nothing changes for the running product until an operator sets `local`.
- **Silent-downgrade risk (the one real danger):** a `saas` deploy that loses its Clerk env must **not** fall
  through to the anonymous local identity and serve tenant data unauthenticated. The §4.3 boot guard (fail
  fast when `saas` + no Clerk) is the mitigation and is a required test (§5.3). This is why the edition is
  *explicit*, not inferred from "is Clerk configured".
- **Scope-creep risk:** the full PRD-150 leakage-elimination / Clerk-package-split (~34 fe / ~63 be sites) is
  **not** in W5 by design — W5 gates the mount and centralizes backend identity. Surfaced as an owner decision
  (below), not deferred unilaterally (CLAUDE.md §12).
- **Rollback:** the flag disables the new path — set `AUTH_EDITION=saas` (the default) and the app is exactly
  today's product. Each change is an independent commit: (a) `config.py` flag + staff domain, (b) `clerk.py`
  config read, (c) frontend `AppAuth`/middleware conditional, (d) backend edition→REQUIRE_AUTH wiring + boot
  guard. The headline local-session test (§5.1) stays green as the permanent open-core regression guard.

**Owner decision surfaced (§6, not descoped):** does W5 carry **only** the deployability-critical mount-gate +
admin-domain move (this PRD as written), or also the **full PRD-150 leakage-elimination / Clerk-package-split**
(the ~34-fe/~63-be decoupling so `local` ships no Clerk code at all)? Recommendation: **ship the mount-gate now
(unblocks W6), do the package-split as its own PRD** — but that is Gerard's call.

---

## 7. References

- Review §4/§5 — F008 (Clerk mounted unconditionally, no edition flag), F075 (hardcoded `@automatos.app`):
  `reports/PLATFORM_OS_REVIEW_2026-07-01.md`
- Review §13 Wave 5 (acceptance: app boots + serves a local session with `AUTH_EDITION=local`, no Clerk env)
- Roadmap §2 Deployability bar (one core two editions behind a flag; local no-login, zero external SaaS) →
  W5 unblocks W6 · [PLATFORM-OS-ROADMAP.md](./PLATFORM-OS-ROADMAP.md)
- Absent **PRD-150** (open-core / auth-decoupling): 3-layer Clerk coupling (mechanism / schema / ~19-file
  leakage); Gerard chose full pkg-split + centralize-identity + full-stack
- Reuses: `setClerkTokenGetter` seam (`api-client.ts:123`, `clerk-api-client-provider.tsx:23`); `hybrid.py`
  anonymous dev-fallback (`hybrid.py:805-826`); existing Clerk config keys (`config.py:149,385-389`)
- CLAUDE.md §4 (no backward-compat shims; no `os.getenv` outside `config.py`), §5 (delete the literal you
  supersede), §12 (no unilateral descope)
