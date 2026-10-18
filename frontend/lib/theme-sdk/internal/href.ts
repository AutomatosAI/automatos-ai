/**
 * PRD-261 A-10a — which hrefs the SDK's navigation accepts. A theme navigates inside the app
 * by path, or links out over https; anything else (another scheme, a protocol-relative URL, a
 * backslash path a browser may read as another host) is refused before it reaches a link.
 */

const INTERNAL_HREF = /^\/(?![/\\])/
const EXTERNAL_HREF = /^https:\/\/[^/\\\s]/i

/** A path inside the app: '/chat', '/missions/7?tab=log'. */
export function isInternalHref(href: unknown): href is string {
  return typeof href === 'string' && INTERNAL_HREF.test(href)
}

/** An https link out of the app: 'https://docs.automatos.app'. */
export function isExternalHref(href: unknown): href is string {
  return typeof href === 'string' && EXTERNAL_HREF.test(href)
}

/** Throws a TypeError for an href the SDK does not navigate to. */
export function assertInternalHref(href: unknown): asserts href is string {
  if (!isInternalHref(href)) {
    throw new TypeError(`Automatos theme SDK: "${String(href)}" is not a path inside the app.`)
  }
}
