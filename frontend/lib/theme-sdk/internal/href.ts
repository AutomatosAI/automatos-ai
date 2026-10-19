/**
 * PRD-261 A-10a — which hrefs the SDK's navigation accepts. A theme navigates inside the app
 * by path, or links out over https; anything else (another scheme, a protocol-relative URL, a
 * backslash path a browser may read as another host) is refused before it reaches a link.
 *
 * Browsers strip ASCII tab, LF and CR from a URL before parsing it, so '/\t/evil.com' resolves
 * as '//evil.com'. The whole href is therefore scanned: a C0 control, DEL or any whitespace
 * anywhere refuses it, and a path inside the app also refuses a backslash anywhere.
 */

const INTERNAL_HREF = /^\/(?![/\\])/
const EXTERNAL_HREF = /^https:\/\/[^/\\\s]/i
const WHITESPACE = /\s/
const BACKSLASH = '\\'
/** The last C0 control code point (U+0000..U+001F). */
const C0_CONTROL_MAX = 0x1f
/** DEL (U+007F). */
const DEL_CODE_POINT = 0x7f

function isUnsafeChar(char: string, refuseBackslash: boolean): boolean {
  const code = char.charCodeAt(0)
  if (code <= C0_CONTROL_MAX || code === DEL_CODE_POINT) return true
  if (refuseBackslash && char === BACKSLASH) return true
  return WHITESPACE.test(char)
}

function hasUnsafeChar(href: string, refuseBackslash: boolean): boolean {
  for (const char of href) {
    if (isUnsafeChar(char, refuseBackslash)) return true
  }
  return false
}

/** A path inside the app: '/chat', '/missions/7?tab=log'. */
export function isInternalHref(href: unknown): href is string {
  return typeof href === 'string' && INTERNAL_HREF.test(href) && !hasUnsafeChar(href, true)
}

/** An https link out of the app: 'https://docs.automatos.app'. */
export function isExternalHref(href: unknown): href is string {
  return typeof href === 'string' && EXTERNAL_HREF.test(href) && !hasUnsafeChar(href, false)
}

/** Throws a TypeError for an href the SDK does not navigate to. */
export function assertInternalHref(href: unknown): asserts href is string {
  if (!isInternalHref(href)) {
    throw new TypeError(`Automatos theme SDK: "${String(href)}" is not a path inside the app.`)
  }
}
