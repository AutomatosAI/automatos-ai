/**
 * PRD-277 US-505, PRD-279 US-402 — what next.config.js does differently when it builds the
 * `enterprise` edition, so that one image serves any install's address.
 *
 * 1. No build-time address. The web app and the API answer on one host (`/` and `/api`, PRD-277
 *    §5.3): the session cookie is `__Host-` and same-origin, the cross-site header wrapper acts only
 *    on same-origin requests (lib/enterprise/session-fetch.ts), and the browser's API calls are
 *    relative. An absolute API or WebSocket address baked into an enterprise build would send those
 *    calls to another host, where neither the cookie nor the header goes: a half-working mix. So the
 *    build refuses one (`refuseBuildTimeAddresses`), naming the variable, never its value.
 *
 * 2. The Content-Security-Policy from the running server. `headers()` is evaluated by `next build`,
 *    so its policy is fixed in the image. In enterprise the policy of every response the middleware
 *    runs for is set by the middleware instead, with the media origins the server was started with
 *    (lib/enterprise/content-security-policy.ts). `enterpriseHeaderRules` keeps every other
 *    security header on every path, and the build-time policy only on the paths the middleware never
 *    runs for (middleware.ts `config.matcher`): `_next` and a path with an extension outside `/api`
 *    and `/trpc`. A response never gets the policy from both, which a browser would enforce as the
 *    stricter of the two, and every document gets it from one.
 *
 * CommonJS, like csp-sources.js: next.config.js requires it, and so does its test.
 */
'use strict'

const CSP_HEADER = 'Content-Security-Policy'
const ENTERPRISE_EDITION = 'enterprise'

/** The build arguments that would fix an address in the image. */
const BUILD_TIME_ADDRESSES = ['NEXT_PUBLIC_API_URL', 'NEXT_PUBLIC_WS_URL']

/**
 * The paths middleware.ts's matcher never runs for, as next.config `headers()` sources (the same
 * path-to-regexp syntax): its first pattern skips `_next` and a path whose last segment has an
 * extension, and its third runs again for anything under `api` or `trpc`.
 */
const PATHS_WITHOUT_MIDDLEWARE = Object.freeze(['/:path(_next.*)', '/:path((?!api|trpc).+\\.\\w+)'])

/** True when NEXT_PUBLIC_AUTH_EDITION names enterprise, read as lib/edition-name.ts reads it. */
function isEnterpriseBuild(edition) {
  return String(edition || '').trim().toLowerCase() === ENTERPRISE_EDITION
}

/** Throws, naming each variable, when an enterprise build is given an address of its own. */
function refuseBuildTimeAddresses(env) {
  const given = BUILD_TIME_ADDRESSES.filter((name) => String(env[name] || '').trim() !== '')
  if (given.length === 0) return
  throw new Error(
    `The enterprise web app calls its API and WebSocket on the page's own address, so ${given.join(' and ')} ` +
      'must be left empty when it is built (PRD-277 US-505).',
  )
}

/** `rule`'s headers on every path without the policy, and the policy alone on the paths the middleware skips. */
function enterpriseHeaderRules(rule) {
  const policy = rule.headers.filter((header) => header.key === CSP_HEADER)
  const others = rule.headers.filter((header) => header.key !== CSP_HEADER)
  return [{ ...rule, headers: others }, ...PATHS_WITHOUT_MIDDLEWARE.map((source) => ({ source, headers: policy }))]
}

module.exports = {
  BUILD_TIME_ADDRESSES,
  PATHS_WITHOUT_MIDDLEWARE,
  isEnterpriseBuild,
  refuseBuildTimeAddresses,
  enterpriseHeaderRules,
}
