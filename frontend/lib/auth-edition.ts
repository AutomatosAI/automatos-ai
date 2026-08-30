/**
 * PRD-175 (F008) — the open-core edition, read once for the client.
 *
 * One core, two editions, one seam. This is the frontend mirror of the backend
 * `config.AUTH_EDITION`, surfaced as the build-time public env
 * `NEXT_PUBLIC_AUTH_EDITION` (the existing `NEXT_PUBLIC_*` convention already
 * used for `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`).
 *
 *   saas  → Clerk is the identity boundary (the running product; the default).
 *   local → no login, no Clerk mount, no external SaaS — every route is public
 *           and the API client sends no bearer, so the backend falls to its
 *           single local identity (`hybrid.py` anonymous dev-fallback).
 *
 * An unknown value falls back to `saas` so a typo never silently un-guards the
 * app. Nothing else in the frontend should read `process.env.NEXT_PUBLIC_AUTH_EDITION`
 * directly — import `isSaaS` / `isLocal` / `authEdition` from here.
 *
 * PRD-233 S7 — this file is also the ONLY place that says which surfaces exist
 * in the hosted edition but not locally (`SAAS_ONLY_ROUTES` below). Nav sources
 * and route pages consult that list through `isRouteAvailableInEdition` /
 * `filterNavForEdition`; in `saas` every helper is the identity, so the running
 * product renders exactly as before.
 */
export type AuthEdition = 'local' | 'saas'

function resolveEdition(): AuthEdition {
  const raw = (process.env.NEXT_PUBLIC_AUTH_EDITION || '').trim().toLowerCase()
  return raw === 'local' ? 'local' : 'saas'
}

export const authEdition: AuthEdition = resolveEdition()
export const isSaaS: boolean = authEdition === 'saas'
export const isLocal: boolean = authEdition === 'local'

/**
 * PRD-233 S7 — the hosted-edition surfaces, as route prefixes (owner decision
 * recorded 2026-08-29). The local edition has no accounts, teams or plans:
 *
 *   /admin/workspaces   Workspace Admin — cross-tenant platform administration
 *   /team               Team — members, roles, invitations
 *   /accept-invitation  invitation acceptance (Clerk-bound)
 *   /sign-in, /sign-up, /reset-password, /sso-callback
 *                       identity flows (Clerk-bound; sign-in/up already send
 *                       visitors home in local — PRD-175)
 *
 * Why a list and not the role: the local operator is deliberately `super_admin`
 * (role-context.tsx; hybrid.py's anonymous lane), so a role gate would either
 * show every hosted-only surface on the operator's machine or demote the
 * operator. Hiding is by this list ONLY — nothing else decides. Plan/trial
 * pills are not routes: they read the trial ledger, which a local workspace
 * never has.
 *
 * Matching is on a path-segment boundary (`/team` covers `/team/x`, never
 * `/teams`); query and hash are ignored.
 */
export const SAAS_ONLY_ROUTES: readonly string[] = [
  '/admin/workspaces',
  '/team',
  '/accept-invitation',
  '/sign-in',
  '/sign-up',
  '/reset-password',
  '/sso-callback',
  '/admin/plugins',
  '/dev/reset-onboarding',
]

function routePath(href: string): string {
  const cut = href.search(/[?#]/)
  return cut === -1 ? href : href.slice(0, cut)
}

function isUnderPrefix(path: string, prefix: string): boolean {
  return path === prefix || path.startsWith(`${prefix}/`)
}

/**
 * Does `href` exist in this edition? Always true in `saas`; false in `local`
 * for anything under SAAS_ONLY_ROUTES.
 */
export function isRouteAvailableInEdition(href: string): boolean {
  if (isSaaS) return true
  const path = routePath(href)
  return !SAAS_ONLY_ROUTES.some((prefix) => isUnderPrefix(path, prefix))
}

/**
 * The nav items of this edition — a new array holding only the entries whose
 * `href` exists here, in the given order. Identity in `saas`.
 */
export function filterNavForEdition<T extends { href: string }>(items: readonly T[]): T[] {
  return items.filter((item) => isRouteAvailableInEdition(item.href))
}

/**
 * The plan-tier exposure (PRD-222 US-024) as nav gating must honour it here.
 *
 *   saas  → the workspace's exposure, untouched.
 *   local → undefined. There is no plan locally: the entrypoint seeds the local
 *           workspace with `plan = NULL`, and `GET /api/workspaces/current`
 *           derives the `basic` profile from the missing plan
 *           (`exposure_for_plan(plan or "basic")` → nav.analytics = false),
 *           which would hide Analytics on the operator's own instance.
 *           nav-exposure fails open on an unknown exposure, so the only thing
 *           hidden locally is SAAS_ONLY_ROUTES.
 */
export function navExposureForEdition<T>(exposure: T | null | undefined): T | null | undefined {
  return isSaaS ? exposure : undefined
}
