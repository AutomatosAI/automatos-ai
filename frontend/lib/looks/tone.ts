import type { GroundKey } from './catalogue'
import { isDarkGround } from './engine'

/**
 * PRD-260 K4 — under Play the look's ground sets the tone: Paper forces
 * Light, Graphite, Obsidian and Night force Dark, through next-themes'
 * forcedTheme (components/providers.tsx), which never overwrites the user's
 * stored tone. The tone rides in its own cookie, which builtin Play's boot
 * names (lib/theme-host/builtin/play-boot.ts): the root layout reads it into
 * the boot's tone under Play only (PRD-261 A-03), so forcedTheme starts on the
 * server and the first paint is already that tone.
 *
 * PRD-261 A-14d: Play reaches the core only through the SDK, which has no
 * cookie writer, so Play writes its tone cookie itself, with the flags of the
 * core's cookie writer (lib/cookies.ts) and the style cookie's lifetime.
 * lib/__tests__/cookies.test.ts holds the style and tone cookies' flags equal.
 */

export const PLAY_TONES = ['dark', 'light'] as const
export type PlayTone = (typeof PLAY_TONES)[number]

export const PLAY_TONE_COOKIE = 'automatos-play-tone'
/** A year: the style cookie's lifetime (UI_STYLE_COOKIE_MAX_AGE_S in lib/ui-style.ts), so the tone never outlives it. */
const PLAY_TONE_COOKIE_MAX_AGE_S = 60 * 60 * 24 * 365
/** Max-Age that deletes a cookie. */
const COOKIE_EXPIRED_S = 0
/** The page's protocol when the cookie must be Secure. */
const SECURE_PROTOCOL = 'https:'

/** The tone cookie for the whole site, Lax, Secure when the page is on https. A Max-Age of 0 deletes it. */
function writeToneCookie(value: string, maxAgeS: number): void {
  const secure = window.location.protocol === SECURE_PROTOCOL ? '; Secure' : ''
  document.cookie = `${PLAY_TONE_COOKIE}=${value}; Path=/; Max-Age=${maxAgeS}; SameSite=Lax${secure}`
}

export function toneOfGround(key: GroundKey): PlayTone {
  return isDarkGround(key) ? 'dark' : 'light'
}

/** The tone cookie, strictly: 'dark' or 'light', anything else → none. */
export function parsePlayTone(value: string | null | undefined): PlayTone | undefined {
  return PLAY_TONES.find((tone) => tone === value)
}

export function writePlayToneCookie(tone: PlayTone): void {
  writeToneCookie(tone, PLAY_TONE_COOKIE_MAX_AGE_S)
}

function hasPlayToneCookie(): boolean {
  return document.cookie.split('; ').some((entry) => entry.startsWith(`${PLAY_TONE_COOKIE}=`))
}

/** Deletes the tone cookie; a no-op when there is none, so Classic and Studio loads write nothing. */
export function clearPlayToneCookie(): void {
  if (hasPlayToneCookie()) writeToneCookie('', COOKIE_EXPIRED_S)
}
