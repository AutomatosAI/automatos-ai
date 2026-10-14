import { UI_STYLE_COOKIE_MAX_AGE_S, type UiStyle } from '@/lib/ui-style'

import type { GroundKey } from './catalogue'
import { isDarkGround } from './css'

/**
 * PRD-260 K4 — under Play the look's ground sets the tone: Paper forces
 * Light, Graphite, Obsidian and Night force Dark, through next-themes'
 * forcedTheme (components/providers.tsx), which never overwrites the user's
 * stored tone. The tone rides in its own cookie so app/layout.tsx starts
 * forcedTheme on the server and the first paint is already that tone.
 */

export const PLAY_TONES = ['dark', 'light'] as const
export type PlayTone = (typeof PLAY_TONES)[number]

export const PLAY_TONE_COOKIE = 'automatos-play-tone'
/** Max-Age that deletes a cookie. */
const COOKIE_EXPIRED_S = 0

export function toneOfGround(key: GroundKey): PlayTone {
  return isDarkGround(key) ? 'dark' : 'light'
}

/** The tone cookie, strictly: 'dark' or 'light', anything else → none. */
export function parsePlayTone(value: string | null | undefined): PlayTone | undefined {
  return PLAY_TONES.find((tone) => tone === value)
}

/** The tone the server forces: the cookie's, and only while the style is play. */
export function playToneFor(style: UiStyle, toneCookie: string | null | undefined): PlayTone | undefined {
  return style === 'play' ? parsePlayTone(toneCookie) : undefined
}

function writeToneCookie(value: string, maxAgeS: number): void {
  const secure = window.location.protocol === 'https:' ? '; Secure' : ''
  document.cookie = `${PLAY_TONE_COOKIE}=${value}; Path=/; Max-Age=${maxAgeS}; SameSite=Lax${secure}`
}

export function writePlayToneCookie(tone: PlayTone): void {
  writeToneCookie(tone, UI_STYLE_COOKIE_MAX_AGE_S)
}

function hasPlayToneCookie(): boolean {
  return document.cookie.split('; ').some((entry) => entry.startsWith(`${PLAY_TONE_COOKIE}=`))
}

/** Deletes the tone cookie; a no-op when there is none, so Classic and Studio loads write nothing. */
export function clearPlayToneCookie(): void {
  if (hasPlayToneCookie()) writeToneCookie('', COOKIE_EXPIRED_S)
}
