import { writeCookie } from '@/lib/cookies'
import { UI_STYLE_COOKIE_MAX_AGE_S } from '@/lib/ui-style'

import { parseThemeTone } from '../boot-sanitize'
import type { PrepaintToneRule, ThemeTone } from '../boot-types'

/**
 * PRD-261 A-19d2 (plan §2.2.3, §2.2.4) — the tone an installed theme's look sets, and its cookie.
 * The theme's prepaint table may carry a tone rule (`tone: {knob, values}`, A-19c): the tone each
 * choice of one choice knob gives a look. Under a theme whose tone follows its look ('from-look'),
 * the generic look runtime (./provider.tsx) writes the tone of the look it paints to
 * `automatos-theme-tone` and hands it to ThemeProvider's forcedTheme; the root layout's boot reads
 * the cookie on the next request (lib/theme-host/boot.ts), so the first paint is already that tone.
 * Leaving the theme deletes it. The host writes it through the one cookie writer (lib/cookies.ts),
 * with the style cookie's lifetime, so the tone never outlives the style; a theme never writes
 * cookies (docs/theme-host/sdk-surface-map.md).
 *
 * One cookie serves the whole browser, and a switch from one installed theme to another reloads
 * without leaving the first, so the value names its theme: `<tone>.<mark>`, the mark a short hash of
 * the theme's id (never the id itself, so the value is bounded and URL-safe). The boot accepts it only
 * for the theme whose mark it carries; another theme's, the old bare tone or anything else is no
 * tone, and that theme's first load is in the user's own tone, as with no cookie.
 */

export const THEME_TONE_COOKIE = 'automatos-theme-tone'

/** Max-Age that deletes a cookie. */
const COOKIE_EXPIRED_S = 0

/** Between the tone and its theme's mark in the cookie's value. */
const MARK_SEPARATOR = '.'

/** FNV-1a, 32 bits: the theme's mark. Not a secret, only a short name for its id. */
const FNV_OFFSET_BASIS = 0x811c9dc5
const FNV_PRIME = 0x01000193
const MARK_RADIX = 16
const MARK_LENGTH = 8

/** The mark of a theme's id in the cookie: eight hex digits. */
export function themeToneMark(themeId: string): string {
  let hash = FNV_OFFSET_BASIS
  for (let index = 0; index < themeId.length; index += 1) {
    hash = Math.imul(hash ^ themeId.charCodeAt(index), FNV_PRIME)
  }
  return (hash >>> 0).toString(MARK_RADIX).padStart(MARK_LENGTH, '0')
}

/** The cookie's value for `themeId`'s look tone. */
export function themeToneCookieValue(themeId: string, tone: ThemeTone): string {
  return `${tone}${MARK_SEPARATOR}${themeToneMark(themeId)}`
}

/** The cookie, strictly, for `themeId`: its own `<'light' | 'dark'>.<mark>`; anything else (another theme's, the old bare tone) none. */
export function parseThemeToneCookie(value: string | null | undefined, themeId: string): ThemeTone | null {
  const tone = typeof value === 'string' ? parseThemeTone(value.split(MARK_SEPARATOR)[0]) : null
  // The whole value must be the theme's own: the tone alone, or another theme's mark, is none.
  return tone !== null && value === themeToneCookieValue(themeId, tone) ? tone : null
}

/** The tone the rule gives the look (its knob's choice); null with no rule or a choice it doesn't map. */
export function toneOfLook(rule: PrepaintToneRule | undefined, look: Readonly<Record<string, unknown>>): ThemeTone | null {
  if (rule === undefined) return null
  const choice = look[rule.knob]
  if (typeof choice !== 'string' || !Object.prototype.hasOwnProperty.call(rule.values, choice)) return null
  return parseThemeTone(rule.values[choice])
}

/** Keeps `themeId`'s look tone for its next load's boot. */
export function writeThemeToneCookie(themeId: string, tone: ThemeTone): void {
  writeCookie(THEME_TONE_COOKIE, themeToneCookieValue(themeId, tone), UI_STYLE_COOKIE_MAX_AGE_S)
}

function hasThemeToneCookie(): boolean {
  return document.cookie.split('; ').some((entry) => entry.startsWith(`${THEME_TONE_COOKIE}=`))
}

/** Deletes the cookie; a no-op when there is none, so Classic and Studio loads write nothing. */
export function clearThemeToneCookie(): void {
  if (hasThemeToneCookie()) writeCookie(THEME_TONE_COOKIE, '', COOKIE_EXPIRED_S)
}
