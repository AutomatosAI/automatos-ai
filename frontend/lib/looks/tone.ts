import { writeCookie } from '@/lib/cookies'
import { parseThemeTone } from '@/lib/theme-host/boot-sanitize'
import { UI_STYLE_COOKIE_MAX_AGE_S } from '@/lib/ui-style'

import type { GroundKey } from './catalogue'
import { isDarkGround } from './css'

/**
 * PRD-260 K4 — under Play the look's ground sets the tone: Paper forces
 * Light, Graphite, Obsidian and Night force Dark, through next-themes'
 * forcedTheme (components/providers.tsx), which never overwrites the user's
 * stored tone. The tone rides in its own cookie, which builtin Play's boot
 * names (lib/theme-host/builtin/play-boot.ts): the root layout reads it into
 * the boot's tone under Play only (PRD-261 A-03), so forcedTheme starts on the
 * server and the first paint is already that tone.
 */

export const PLAY_TONES = ['dark', 'light'] as const
export type PlayTone = (typeof PLAY_TONES)[number]

export const PLAY_TONE_COOKIE = 'automatos-play-tone'
/** Max-Age that deletes a cookie. */
const COOKIE_EXPIRED_S = 0

export function toneOfGround(key: GroundKey): PlayTone {
  return isDarkGround(key) ? 'dark' : 'light'
}

/** The tone cookie, strictly: 'dark' or 'light', anything else → none (the theme host's one tone parser). */
export function parsePlayTone(value: string | null | undefined): PlayTone | undefined {
  return parseThemeTone(value) ?? undefined
}

export function writePlayToneCookie(tone: PlayTone): void {
  writeCookie(PLAY_TONE_COOKIE, tone, UI_STYLE_COOKIE_MAX_AGE_S)
}

function hasPlayToneCookie(): boolean {
  return document.cookie.split('; ').some((entry) => entry.startsWith(`${PLAY_TONE_COOKIE}=`))
}

/** Deletes the tone cookie; a no-op when there is none, so Classic and Studio loads write nothing. */
export function clearPlayToneCookie(): void {
  if (hasPlayToneCookie()) writeCookie(PLAY_TONE_COOKIE, '', COOKIE_EXPIRED_S)
}
