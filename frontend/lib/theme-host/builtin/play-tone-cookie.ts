import { writeCookie } from '@/lib/cookies'
import { PLAY_TONE_COOKIE, type PlayTone } from '@/lib/looks/tone'
import { UI_STYLE_COOKIE_MAX_AGE_S } from '@/lib/ui-style'

/**
 * PRD-260 K4, PRD-261 A-14d — builtin Play's tone cookie, written by the host. Under Play the
 * look's ground sets the tone (lib/looks/tone.ts toneOfGround), and the host's look runtime
 * (./play-look-provider.tsx, components/theme-host/theme-look-runtime.tsx) keeps it in its cookie
 * so the root layout starts the next request on that tone (./play-boot.ts names the cookie). The
 * host owns the cookie (docs/theme-host/sdk-surface-map.md: `@/lib/cookies` and `@/lib/ui-style`
 * are host-only; a theme never writes cookies), so Play's own lib/looks/tone.ts holds only the
 * tones, the cookie's name and its parser. Written through the one cookie writer, lib/cookies.ts
 * (P260-FX-27, M88), with the style cookie's lifetime, so the tone never outlives the style.
 */

/** Max-Age that deletes a cookie. */
const COOKIE_EXPIRED_S = 0

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
