/**
 * PRD-279 US-402 — the enterprise image's Content-Security-Policy, with the media origins the
 * running server was given.
 *
 * Next evaluates next.config.js's `headers()` in `next build`, so a policy written there is fixed
 * in the image. In the enterprise edition the middleware (session-middleware.ts) sets the policy
 * on every response it runs for instead, built by the same function next.config.js uses
 * (csp-policy.mjs), with the inputs an enterprise build has: no Clerk Frontend API host (the
 * edition mounts no Clerk), no absolute API origin (the API is the page's own origin, PRD-277
 * §5.3; enterprise-build.js refuses one), no payment sources (no billing, PRD-277 FR-24), and the
 * media origins read from MEDIA_ORIGINS. With no media origins it is exactly the policy the build
 * put on the paths the middleware never runs for (enterprise-build.js), so it is never looser
 * than the build-time policy, only wider by the origins the operator named.
 *
 * MEDIA_ORIGINS is read on each response, as the Edge runtime gives the server's environment.
 * start-up.ts stops the server before it serves anything when the value is bad; should one get
 * here anyway, the policy names no media origin at all (the strictest answer) and says so in the
 * server log, without the value.
 */
import { contentSecurityPolicy } from '@/csp-policy.mjs'

import { parseMediaOrigins } from './media-origins'

export const CSP_HEADER = 'Content-Security-Policy'

/** The enterprise policy with `mediaOrigins` in img-src and media-src. */
export function enterprisePolicy(mediaOrigins: readonly string[]): string {
  return contentSecurityPolicy({ mediaSources: mediaOrigins.join(' ') })
}

/** The policy for the running server: its MEDIA_ORIGINS, or none when that is not a list of origins. */
export function runningPolicy(raw: string | undefined = process.env.MEDIA_ORIGINS): string {
  const parsed = parseMediaOrigins(raw)
  if (parsed.ok) return enterprisePolicy(parsed.origins)
  console.error(`[content-security-policy] ${parsed.error} The policy names no media origin.`)
  return enterprisePolicy([])
}

/** `response`, carrying the running server's policy (the header is set on the response the middleware returns). */
export function withRunningPolicy<T extends Response>(response: T, raw: string | undefined = process.env.MEDIA_ORIGINS): T {
  response.headers.set(CSP_HEADER, runningPolicy(raw))
  return response
}
