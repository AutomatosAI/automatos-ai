/**
 * PRD-279 US-402 — the enterprise image's Content-Security-Policy, with the media origins the
 * running server was given.
 *
 * Next evaluates next.config.js's `headers()` in `next build`, so a policy written there is fixed
 * in the image. In the enterprise edition the middleware (session-middleware.ts) sets the policy
 * on every response it runs for instead, built by the function next.config.js uses for this
 * edition (enterpriseContentSecurityPolicy, csp-policy.mjs: no Clerk, Automatos or payment host,
 * PRD-277 D-12), with the media origins read from MEDIA_ORIGINS and, while ENTERPRISE_CALLS_CLOUD_SYNC
 * is on (cloud-sync-setting.ts), Automatos Cloud's screenshot path in img-src. With no media origin
 * and Cloud sync off it is exactly the policy the build put on the paths the middleware never runs
 * for (enterprise-build.js), so it is never looser than the build-time policy, only wider by what
 * the operator named or switched on.
 *
 * Both settings are read on each response, as the Edge runtime gives the server's environment, but
 * parsed once per distinct pair of values: the policy for the last pair seen is kept, and built
 * again only when either differs. start-up.ts stops the server before it serves anything when a
 * value is bad; should one get here anyway, the policy names no media origin, or no Cloud
 * screenshot, for it (the strictest answer) and the server log says so once per setting and
 * process, without the value.
 */
import { enterpriseContentSecurityPolicy } from '@/csp-policy.mjs'

import { CLOUD_SYNC_SETTING, parseCloudSync } from './cloud-sync-setting'
import { MEDIA_ORIGINS_SETTING, parseMediaOrigins } from './media-origins'

export const CSP_HEADER = 'Content-Security-Policy'

/** The enterprise policy with `mediaOrigins` in img-src and media-src, and Cloud's screenshots while `cloudSync` is on. */
export function enterprisePolicy(mediaOrigins: readonly string[], cloudSync = false): string {
  return enterpriseContentSecurityPolicy({ mediaSources: mediaOrigins.join(' '), cloudImages: cloudSync })
}

/** The last values seen and their policy (a cache), and the settings whose bad value has been logged. */
let lastPolicy: { readonly raw: string; readonly cloudSync: string; readonly policy: string } | null = null
const refusalsLogged = new Set<string>()

function logRefusalOnce(setting: string, message: string): void {
  if (refusalsLogged.has(setting)) return
  refusalsLogged.add(setting)
  console.error(`[content-security-policy] ${message}`)
}

function mediaOriginsFor(raw: string): readonly string[] {
  const parsed = parseMediaOrigins(raw)
  if (parsed.ok) return parsed.origins
  logRefusalOnce(MEDIA_ORIGINS_SETTING, `${parsed.error} The policy names no media origin.`)
  return []
}

function cloudSyncFor(raw: string): boolean {
  const parsed = parseCloudSync(raw)
  if (parsed.ok) return parsed.on
  logRefusalOnce(CLOUD_SYNC_SETTING, `${parsed.error} The policy names no Automatos Cloud screenshot.`)
  return false
}

/**
 * The policy for the running server: its MEDIA_ORIGINS, or none when that is not a list of origins,
 * and Automatos Cloud's screenshots only while its ENTERPRISE_CALLS_CLOUD_SYNC is on (none for a bad value).
 */
export function runningPolicy(
  raw: string | undefined = process.env.MEDIA_ORIGINS,
  cloudSync: string | undefined = process.env.ENTERPRISE_CALLS_CLOUD_SYNC,
): string {
  const value = raw ?? ''
  const cloud = cloudSync ?? ''
  if (lastPolicy === null || lastPolicy.raw !== value || lastPolicy.cloudSync !== cloud) {
    lastPolicy = { raw: value, cloudSync: cloud, policy: enterprisePolicy(mediaOriginsFor(value), cloudSyncFor(cloud)) }
  }
  return lastPolicy.policy
}

/** `response`, carrying the running server's policy (the header is set on the response the middleware returns). */
export function withRunningPolicy<T extends Response>(
  response: T,
  raw: string | undefined = process.env.MEDIA_ORIGINS,
  cloudSync: string | undefined = process.env.ENTERPRISE_CALLS_CLOUD_SYNC,
): T {
  response.headers.set(CSP_HEADER, runningPolicy(raw, cloudSync))
  return response
}
