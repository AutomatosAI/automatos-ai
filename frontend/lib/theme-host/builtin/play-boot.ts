/**
 * PRD-261 A-03 (PA-7) — builtin Play's first-paint data, and nothing else: data only and
 * server-safe (no stylesheet, no client code), so the root layout can read it on every request.
 * Play's stylesheet lives in play-client.ts, reached only through a dynamic import().
 *
 * The prepaint table is the look catalogue's (play-prepaint.ts, the glow as a numeric rule);
 * the look storage keys are where Play keeps the active look per account (F5): its legacy keys
 * (play-look-keys.ts, A-09b), the same ones the look store reads and writes; the guard is the
 * look schema and interface; the tone cookie is the one LookProvider writes the look's tone to
 * (PRD-260 K4), read under Play's 'from-look' tone policy.
 */
import { bootLookStorage } from '@/lib/looks/storage'
import { PLAY_TONE_COOKIE } from '@/lib/looks/tone'
import { LOOK_INTERFACE, LOOK_SCHEMA } from '@/lib/looks/types'

import type { BuiltinThemeBootData } from '../boot-types'
import { BUILTIN_PLAY_LOOK_KEYS } from './play-look-keys'
import { lookPrepaintTable } from './play-prepaint'

export const BUILTIN_PLAY_BOOT: BuiltinThemeBootData = Object.freeze({
  prepaint: lookPrepaintTable(),
  // The boot script appends the scope the pointer names: scopedPrefix + scope.
  lookStorage: bootLookStorage(BUILTIN_PLAY_LOOK_KEYS),
  lookGuard: Object.freeze({ schema: LOOK_SCHEMA, iface: LOOK_INTERFACE }),
  toneCookie: PLAY_TONE_COOKIE,
})
