/**
 * The web app's Content-Security-Policy, built in ONE place for two readers:
 *
 *   - next.config.js, at build time, for every edition's `headers()` (`require`d: Node 24 loads an
 *     ES module from CommonJS);
 *   - the enterprise middleware, on each response, with the media origins the running server was
 *     given (PRD-279 US-402, lib/enterprise/content-security-policy.ts).
 *
 * An ES module with no Node globals on purpose: the middleware runs on the Edge runtime, and the
 * test suite imports it from TypeScript. The build-time inputs that need Node (the Clerk key's
 * Frontend API host, decoded with Buffer) stay in csp-sources.js and arrive here as plain strings.
 *
 * PRD-277 / PRD-279 moved the directive list here from next.config.js unchanged, with the
 * constants it names (they were in csp-sources.js, which re-exports them): for the same inputs the
 * policy is the same string it was. The enterprise edition builds its own list
 * (enterpriseContentSecurityPolicy, PRD-277 D-12), at build time and in the middleware alike.
 */

/**
 * Clerk production readiness (19 Oct 2026). Clerk's CSP guide
 * (clerk.com/docs/guides/secure/best-practices/csp-headers) needs its abuse and fraud protection
 * hosts in script-src and frame-src, and in connect-src with any port (they are not served on 443
 * only).
 */
export const CLERK_PROTECT_SOURCE = 'https://*.protect.clerk.com'
export const CLERK_PROTECT_CONNECT_SOURCE = `${CLERK_PROTECT_SOURCE}:*`

/** A bare scheme://host[:port] origin, optionally with a leading `*.` (csp-sources.js's media origins). */
export const ORIGIN = /^https?:\/\/(\*\.)?[a-z0-9-]+(\.[a-z0-9-]+)*(:\d{1,5})?$/i

/**
 * PRD-270 K15: a Cloud listing's screenshots load from the catalogue files route of Automatos Cloud
 * (lib/marketplace/cloud-files.ts names the same origin and path, AUTOMATOS_CLOUD_ORIGIN and
 * CLOUD_FILES_PATH), or of the API this build talks to (the hosted edition's own catalogue), straight
 * from the browser; the route serves the bytes itself, never a redirect. img-src names that path on
 * both origins, not the whole origin; an API origin that is not a bare origin, or is Automatos Cloud's
 * own, adds nothing.
 */
export const AUTOMATOS_CLOUD_ORIGIN = 'https://api.automatos.app'
export const CLOUD_FILES_PATH = '/api/marketplace/catalog/files/'

/** img-src's sources for Cloud listings' screenshots, space-separated: Automatos Cloud's files route, then the build's API's. */
export function cloudImageSources(apiOrigin) {
  const api = String(apiOrigin || '')
  const origins = ORIGIN.test(api) && api !== AUTOMATOS_CLOUD_ORIGIN ? [AUTOMATOS_CLOUD_ORIGIN, api] : [AUTOMATOS_CLOUD_ORIGIN]
  return origins.map((origin) => `${origin}${CLOUD_FILES_PATH}`).join(' ')
}

/** No payment sources: the editions without billing (csp-sources.js stripeCspFor). */
const NO_PAYMENT_SOURCES = Object.freeze({ script: '', frame: '', connect: '', img: '' })

/** One CSP directive: the given parts, empty ones dropped, space-separated. */
const directive = (...parts) => parts.filter(Boolean).join(' ')

/**
 * The policy, as one header value.
 *
 * @param {object} inputs
 * @param {string} [inputs.clerkFrontendApi] the Clerk Frontend API origin the build's key names, or ''
 * @param {string} [inputs.apiOrigin] the build's absolute API origin, or '' (relative API)
 * @param {{script: string, frame: string, connect: string, img: string}} [inputs.stripe] Stripe's sources, or none
 * @param {string} [inputs.mediaSources] the media origins, space-separated, or ''
 * @returns {string}
 */
export function contentSecurityPolicy({ clerkFrontendApi = '', apiOrigin = '', stripe = NO_PAYMENT_SOURCES, mediaSources = '' } = {}) {
  return [
    "default-src 'self'",
    directive(
      "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://*.clerk.accounts.dev",
      clerkFrontendApi,
      CLERK_PROTECT_SOURCE,
      'https://challenges.cloudflare.com https://cdn.jsdelivr.net',
      stripe.script,
    ),
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
    directive(
      "img-src 'self' data: blob: https://*.clerk.accounts.dev https://img.clerk.com https://*.googleusercontent.com https://logos.composio.dev",
      cloudImageSources(apiOrigin),
      stripe.img,
      mediaSources,
    ),
    `media-src 'self' blob: ${mediaSources}`.trim(),
    "font-src 'self' data: https://cdn.jsdelivr.net",
    // The API origin is derived from NEXT_PUBLIC_API_URL so the local edition
    // (http://localhost:8000) is allowed too — without it the browser refuses
    // every API call ("Failed to fetch") before it leaves the page. In SaaS the
    // value is https://api.automatos.app, already covered by the wildcard. In
    // enterprise the API is the page's own origin ('self').
    directive(
      `connect-src 'self' ${apiOrigin} https://*.automatos.app https://*.clerk.accounts.dev`,
      clerkFrontendApi,
      CLERK_PROTECT_CONNECT_SOURCE,
      'https://api.clerk.com https://cdn.jsdelivr.net',
      stripe.connect,
      'wss: ws:',
    ),
    // F352 (night 10b): a PDF preview (Deliverables side panel, Explorer, chat file card)
    // is fetched with the Bearer header and shown as a blob: URL in an <iframe>
    // (useAuthenticatedBlobUrl). 'self' never matches blob:, so frame-src names it. The
    // blob document inherits this policy, and the browser's PDF viewer is a plugin, so
    // object-src allows blob: too. Only this page's own script can mint a blob: URL.
    directive(
      "frame-src 'self' blob: https://*.clerk.accounts.dev",
      CLERK_PROTECT_SOURCE,
      'https://challenges.cloudflare.com',
      stripe.frame,
    ),
    "worker-src 'self' blob:",
    "object-src blob:",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join('; ')
}

/**
 * PRD-277 D-12 (#1605): the enterprise edition's policy, naming only the sources that edition uses.
 * It mounts no Clerk, so it names none of Clerk's hosts, nor Cloudflare's challenge, which Clerk's
 * bot protection loads. Its API is the page's own origin ('self'), so it names no Automatos origin
 * in connect-src. It has no billing, so it names no Stripe host. A Cloud listing's screenshots load
 * from Automatos Cloud's files route (the path cloudImageSources names, not the origin) only while
 * `cloudImages` is true, that is, while the install's operator has switched Cloud sync on. The other
 * sources are the shared policy's: jsDelivr (the code editor loads from it), Composio's app logos,
 * Google-hosted profile pictures, and the WebSocket schemes.
 *
 * @param {object} inputs
 * @param {string} [inputs.mediaSources] the media origins, space-separated, or ''
 * @param {boolean} [inputs.cloudImages] whether Cloud sync is switched on
 * @returns {string}
 */
export function enterpriseContentSecurityPolicy({ mediaSources = '', cloudImages = false } = {}) {
  return [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
    directive(
      "img-src 'self' data: blob: https://*.googleusercontent.com https://logos.composio.dev",
      cloudImages ? cloudImageSources('') : '',
      mediaSources,
    ),
    `media-src 'self' blob: ${mediaSources}`.trim(),
    "font-src 'self' data: https://cdn.jsdelivr.net",
    "connect-src 'self' https://cdn.jsdelivr.net wss: ws:",
    "frame-src 'self' blob:",
    "worker-src 'self' blob:",
    "object-src blob:",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join('; ')
}
