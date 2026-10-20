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
 * PRD-261 A-14d: the host writes and deletes the cookie
 * (lib/theme-host/builtin/play-tone-cookie.ts, through the one cookie writer,
 * lib/cookies.ts): a theme never writes cookies (docs/theme-host/sdk-surface-map.md).
 * This file holds only the tones, the cookie's name and its parser.
 */

export const PLAY_TONES = ['dark', 'light'] as const
export type PlayTone = (typeof PLAY_TONES)[number]

export const PLAY_TONE_COOKIE = 'automatos-play-tone'

export function toneOfGround(key: GroundKey): PlayTone {
  return isDarkGround(key) ? 'dark' : 'light'
}

/** The tone cookie, strictly: 'dark' or 'light', anything else → none. */
export function parsePlayTone(value: string | null | undefined): PlayTone | undefined {
  return PLAY_TONES.find((tone) => tone === value)
}
