/**
 * PRD-277 US-501 — the build's edition, read from `NEXT_PUBLIC_AUTH_EDITION`. The one place that
 * reads it; lib/auth-edition.ts is the seam every component imports (`isSaaS`, `isLocal`,
 * `isEnterprise`, `authEdition`) and re-exports what is here.
 *
 *   saas       → Clerk is the identity boundary (the running product; the default).
 *   local      → no login: one operator on their own machine.
 *   enterprise → people sign in with the organisation's identity provider; the API issues the
 *                session and keeps it in a cookie scripts cannot read (PRD-277 §5.3).
 *
 * An unknown value falls back to `saas`, so a typo never silently un-guards the app.
 *
 * Why its own module: `isEnterprise` is read by modules that many tests load under a mock of
 * lib/auth-edition holding only the names they need. Those modules import `isEnterprise` from
 * here, so such a mock leaves it the build's real value instead of an undefined export.
 */
export type AuthEdition = 'local' | 'saas' | 'enterprise'

/** True in the enterprise edition: its own read of the variable, compared once. */
function namesEnterprise(): boolean {
  const raw = (process.env.NEXT_PUBLIC_AUTH_EDITION || '').trim().toLowerCase()
  return raw === 'enterprise'
}

/**
 * The edition. The `'local'` comparison keeps main's exact shape on purpose: `raw` read once and
 * used once, so the minifier inlines it and a hosted build compiles to
 * `"local"==="saas".trim().toLowerCase()?"local":"saas"`, the text the CI job "Prod images built
 * with default args carry production behaviour" looks for (lib/__tests__/edition-name-shape.test.ts).
 */
export function resolveEdition(): AuthEdition {
  if (namesEnterprise()) return 'enterprise'
  const raw = (process.env.NEXT_PUBLIC_AUTH_EDITION || '').trim().toLowerCase()
  return raw === 'local' ? 'local' : 'saas'
}

/** True in the enterprise edition (a build-time constant). */
export const isEnterprise: boolean = namesEnterprise()
