import { lookToCssVars } from '@/lib/looks/css'
import type { Look } from '@/lib/looks/types'

import type { MascotName } from './types'

/**
 * PRD-260 D6 — the light ring under Auto: which look colour it takes in each
 * state, ported from Automatos Play's lib/mascot/ring.ts (the Ember rule). His
 * light when calm, the accent when busy, the fixed error coral for oops, a dusk
 * made from the ground while he sleeps or is lost, and the ground's secondary
 * text while loading. components/mascot/auto-mascot.module.css draws the same
 * mapping from data-ring, through app/play-theme.css's --pl-color-* tokens;
 * lib/mascot/__tests__/ring.test.ts holds the two together.
 */

/** light: calm; accent: busy; error: oops; dusk: asleep or lost; quiet: loading. */
export type RingTone = 'light' | 'accent' | 'error' | 'dusk' | 'quiet'

const RING_TONES: Readonly<Record<MascotName, RingTone>> = {
  hello: 'light',
  idle: 'light',
  listening: 'light',
  celebrate: 'light',
  thinking: 'accent',
  working: 'accent',
  searching: 'accent',
  reading: 'accent',
  oops: 'error',
  sleeping: 'dusk',
  lost: 'dusk',
  loading: 'quiet',
}

/** The look variable (lib/looks/css.ts) each tone's ring is drawn in. */
export const RING_VARS: Readonly<Record<RingTone, string>> = {
  light: '--pl-auto-light',
  accent: '--pl-accent',
  error: '--pl-error',
  dusk: '--pl-auto-dusk',
  quiet: '--pl-ground-text-2',
}

export function ringTone(state: MascotName): RingTone {
  return RING_TONES[state]
}

/** The ring's #rrggbb colour for a state under a look. */
export function ringColour(state: MascotName, look: Look): string {
  const name = RING_VARS[ringTone(state)]
  const colour = lookToCssVars(look)[name]
  if (colour === undefined) throw new Error(`A look sets no ${name}`)
  return colour
}
