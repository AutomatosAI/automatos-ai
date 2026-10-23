/**
 * Build-time sources for the Content-Security-Policy in next.config.js: the media origins (F237,
 * below), Clerk's hosts (clerkFrontendApiOrigin and the protection hosts, further down) and
 * Stripe's (PRD-265 W2, at the end).
 *
 * F237 (PRD-251B): the origins the browser may load media from. Socials thumbnails and a post's media are presigned links to object storage
 * (a MinIO host locally, AWS S3 in SaaS), so img-src and media-src must allow its public origin.
 *
 * NEXT_PUBLIC_MEDIA_ORIGINS is read when the app is BUILT (next.config.js headers are baked into
 * the standalone build): one or more origins separated by commas or spaces, such as
 * "http://localhost:9000" (the local edition's MinIO, docker-compose.yml) or
 * "https://*.amazonaws.com" (the Dockerfile's default: the hosted product). Anything that is not a
 * bare scheme://host[:port] origin is left out, so a stray path or keyword never widens the policy.
 *
 * CommonJS on purpose: next.config.js requires it, and so does its test.
 */
'use strict'

const ORIGIN = /^https?:\/\/(\*\.)?[a-z0-9-]+(\.[a-z0-9-]+)*(:\d{1,5})?$/i

function mediaOrigins(raw) {
  return String(raw || '')
    .split(/[\s,]+/)
    .map((token) => token.trim().replace(/\/+$/, ''))
    .filter((token) => ORIGIN.test(token))
}

/**
 * Clerk production readiness (19 Oct 2026). A development key (pk_test_) loads clerk-js from
 * *.clerk.accounts.dev; a production key (pk_live_) loads it from the instance's own Frontend API
 * host, e.g. clerk.automatos.app, which the policy must name. Clerk's CSP guide
 * (clerk.com/docs/guides/secure/best-practices/csp-headers) also needs its abuse and fraud
 * protection hosts in script-src and frame-src, and in connect-src with any port (they are not
 * served on 443 only).
 */
const CLERK_PROTECT_SOURCE = 'https://*.protect.clerk.com'
const CLERK_PROTECT_CONNECT_SOURCE = `${CLERK_PROTECT_SOURCE}:*`

// A publishable key is pk_test_ or pk_live_, then base64 of the Frontend API host and a '$'.
const PUBLISHABLE_KEY = /^pk_(?:test|live)_([A-Za-z0-9+/=_-]+)$/
const FRONTEND_API_HOST = /^[a-z0-9-]+(\.[a-z0-9-]+)+$/i

/**
 * The Frontend API origin a Clerk publishable key points at ("https://clerk.automatos.app"), or ''
 * when there is no key or it does not decode to a bare host: a malformed key never widens the policy.
 * NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY is baked in at build time, so the policy follows the key the
 * build ships with, development or production, with no second setting to keep in step.
 */
function clerkFrontendApiOrigin(publishableKey) {
  const match = PUBLISHABLE_KEY.exec(String(publishableKey || '').trim())
  if (!match) return ''
  const decoded = Buffer.from(match[1], 'base64').toString('utf8')
  const host = decoded.endsWith('$') ? decoded.slice(0, -1) : ''
  return FRONTEND_API_HOST.test(host) ? `https://${host}` : ''
}

/**
 * PRD-265 W2: payments happen inside Settings → Billing. Stripe's card and address fields are
 * iframes in our own form, loaded by Stripe.js; Link (Stripe's saved-card wallet) adds its own
 * frames, calls and card art; and a bank's 3-D Secure check is framed from hooks.stripe.com.
 * These are the sources Stripe's CSP guide lists (docs.stripe.com/security/guide#content-security-policy),
 * maps.googleapis.com being the address field's autocomplete. Fixed hosts, no wildcard beyond
 * Stripe's and Link's own subdomains.
 */
const LINK_SOURCES = ['https://link.com', 'https://*.link.com']
const STRIPE_SCRIPT_SOURCES = ['https://js.stripe.com', 'https://*.js.stripe.com', 'https://maps.googleapis.com']
const STRIPE_FRAME_SOURCES = ['https://js.stripe.com', 'https://*.js.stripe.com', 'https://hooks.stripe.com', ...LINK_SOURCES]
const STRIPE_CONNECT_SOURCES = ['https://api.stripe.com', 'https://maps.googleapis.com', ...LINK_SOURCES]
const STRIPE_IMG_SOURCES = ['https://*.link.com']

/** Stripe's sources for each directive, space-separated, for next.config.js. */
const STRIPE_CSP = Object.freeze({
  script: STRIPE_SCRIPT_SOURCES.join(' '),
  frame: STRIPE_FRAME_SOURCES.join(' '),
  connect: STRIPE_CONNECT_SOURCES.join(' '),
  img: STRIPE_IMG_SOURCES.join(' '),
})
const NO_STRIPE = Object.freeze({ script: '', frame: '', connect: '', img: '' })

/**
 * Stripe's sources for the build's edition (NEXT_PUBLIC_AUTH_EDITION, read the way
 * lib/auth-edition.ts reads it): the local edition has no billing, so its policy names no
 * Stripe host at all; anything else is the hosted edition.
 */
function stripeCspFor(edition) {
  return String(edition || '').trim().toLowerCase() === 'local' ? NO_STRIPE : STRIPE_CSP
}

module.exports = { mediaOrigins, clerkFrontendApiOrigin, CLERK_PROTECT_SOURCE, CLERK_PROTECT_CONNECT_SOURCE, STRIPE_CSP, stripeCspFor }
