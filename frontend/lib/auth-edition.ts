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
 */
export type AuthEdition = 'local' | 'saas'

function resolveEdition(): AuthEdition {
  const raw = (process.env.NEXT_PUBLIC_AUTH_EDITION || '').trim().toLowerCase()
  return raw === 'local' ? 'local' : 'saas'
}

export const authEdition: AuthEdition = resolveEdition()
export const isSaaS: boolean = authEdition === 'saas'
export const isLocal: boolean = authEdition === 'local'
