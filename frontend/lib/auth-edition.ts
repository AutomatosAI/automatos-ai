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
 * directly — import `isSaaS` / `isLocal` / `isEnterprise` / `authEdition` from here.
 * PRD-277 US-501: the read itself lives in lib/edition-name.ts, which also says why.
 *
 *   enterprise → people sign in with the organisation's identity provider (PRD-277): no
 *                Clerk mount; the API's session cookie carries who they are.
 *
 * PRD-233 S7 — this file is also the ONLY place that says which surfaces exist
 * in the hosted edition but not locally (`SAAS_ONLY_ROUTES` below). Nav sources
 * and route pages consult that list through `isRouteAvailableInEdition` /
 * `filterNavForEdition`; in `saas` every helper is the identity, so the running
 * product renders exactly as before.
 */
import { isEnterprise, resolveEdition, type AuthEdition } from './edition-name'

export { isEnterprise, resolveEdition }
export type { AuthEdition }

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
 * operator. Hiding is by this list ONLY — nothing else decides. The credit
 * balance chip is not a route: it is mounted for `saas` only and reads the
 * billing summary, which the local edition never serves (nothing is metered).
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

/**
 * PRD-277 US-502, US-504 — the hosted surfaces enterprise has too: its own sign-in page (one
 * button, the organisation's identity provider) and the Team page (members, people waiting for
 * a workspace; no invitations). Everything else in SAAS_ONLY_ROUTES stays hidden there.
 */
export const ENTERPRISE_ROUTES: readonly string[] = ['/team', '/sign-in']

function routePath(href: string): string {
  const cut = href.search(/[?#]/)
  return cut === -1 ? href : href.slice(0, cut)
}

function isUnderPrefix(path: string, prefix: string): boolean {
  return path === prefix || path.startsWith(`${prefix}/`)
}

/**
 * Does `href` exist in this edition? Always true in `saas`; false in `local`
 * for anything under SAAS_ONLY_ROUTES; in `enterprise`, false for those except
 * ENTERPRISE_ROUTES.
 */
export function isRouteAvailableInEdition(href: string): boolean {
  if (isSaaS) return true
  const path = routePath(href)
  if (isEnterprise && ENTERPRISE_ROUTES.some((prefix) => isUnderPrefix(path, prefix))) return true
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
