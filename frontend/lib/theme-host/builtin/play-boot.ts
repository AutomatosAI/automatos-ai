/**
 * PRD-261 A-03 (PA-7) — builtin Play's first-paint data, and nothing else: data only and
 * server-safe (no stylesheet, no client code), so the root layout can read it on every request.
 * Play's stylesheet lives in play-client.ts, reached only through a dynamic import().
 *
 * The prepaint table is the look catalogue's (lib/looks/prepaint.ts, the glow as a numeric rule);
 * the look storage keys are where Play keeps the active look per account (lib/looks/storage.ts,
 * F5); the guard is the look schema and interface; the tone cookie is the one LookProvider writes
 * the look's tone to (PRD-260 K4), read under Play's 'from-look' tone policy.
 */
import { lookPrepaintTable } from '@/lib/looks/prepaint'
import { scopedKey } from '@/lib/looks/scope'
import { ACTIVE_LOOK_KEY, LOOK_SCOPE_KEY } from '@/lib/looks/storage'
import { PLAY_TONE_COOKIE } from '@/lib/looks/tone'
import { LOOK_INTERFACE, LOOK_SCHEMA } from '@/lib/looks/types'

import type { BuiltinThemeBootData } from '../boot-types'

/** The scope part is appended by the boot script: scopedKey(ACTIVE_LOOK_KEY, scope). */
const NO_SCOPE = ''

export const BUILTIN_PLAY_BOOT: BuiltinThemeBootData = Object.freeze({
  prepaint: lookPrepaintTable(),
  lookStorage: Object.freeze({
    pointerKey: LOOK_SCOPE_KEY,
    scopedPrefix: scopedKey(ACTIVE_LOOK_KEY, NO_SCOPE),
    fallbackKey: ACTIVE_LOOK_KEY,
  }),
  lookGuard: Object.freeze({ schema: LOOK_SCHEMA, iface: LOOK_INTERFACE }),
  toneCookie: PLAY_TONE_COOKIE,
})
