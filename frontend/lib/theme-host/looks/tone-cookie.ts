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
 */

export const THEME_TONE_COOKIE = 'automatos-theme-tone'

/** Max-Age that deletes a cookie. */
const COOKIE_EXPIRED_S = 0

/** The cookie, strictly: 'light' or 'dark', anything else none. */
export function parseThemeToneCookie(value: string | null | undefined): ThemeTone | null {
  return parseThemeTone(value)
}

/** The tone the rule gives the look (its knob's choice); null with no rule or a choice it doesn't map. */
export function toneOfLook(rule: PrepaintToneRule | undefined, look: Readonly<Record<string, unknown>>): ThemeTone | null {
  if (rule === undefined) return null
  const choice = look[rule.knob]
  if (typeof choice !== 'string' || !Object.prototype.hasOwnProperty.call(rule.values, choice)) return null
  return parseThemeTone(rule.values[choice])
}

export function writeThemeToneCookie(tone: ThemeTone): void {
  writeCookie(THEME_TONE_COOKIE, tone, UI_STYLE_COOKIE_MAX_AGE_S)
}

function hasThemeToneCookie(): boolean {
  return document.cookie.split('; ').some((entry) => entry.startsWith(`${THEME_TONE_COOKIE}=`))
}

/** Deletes the cookie; a no-op when there is none, so Classic and Studio loads write nothing. */
export function clearThemeToneCookie(): void {
  if (hasThemeToneCookie()) writeCookie(THEME_TONE_COOKIE, '', COOKIE_EXPIRED_S)
}
