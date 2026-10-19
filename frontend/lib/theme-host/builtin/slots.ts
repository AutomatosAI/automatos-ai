/**
 * PRD-261 A-04b — each builtin theme's registrations (its frame, slots, pages and routes), by the
 * theme's ref. The theme slots (components/theme-host/theme-slot-registrations.tsx) read them by
 * the active theme's ref and never name a builtin theme; an installed theme's come from the theme
 * host instead. Adding or removing a builtin theme edits only this folder (as ./clients.ts).
 */
import type { ThemeEntry } from '../registry-types'
import type { ThemeRegistrations } from '../types'
import { BUILTIN_PLAY_REF } from './play-entry'
import { BUILTIN_PLAY_REGISTRATIONS } from './play-slots'

export const BUILTIN_THEME_REGISTRATIONS: ReadonlyMap<string, ThemeRegistrations> = new Map<string, ThemeRegistrations>([
  [BUILTIN_PLAY_REF, BUILTIN_PLAY_REGISTRATIONS],
])

/** What the active theme registered when it is a builtin one; null for Classic, Studio and an installed theme. */
export function builtinRegistrationsOf(entry: ThemeEntry | null): ThemeRegistrations | null {
  if (entry === null || entry.kind !== 'builtin') return null
  return BUILTIN_THEME_REGISTRATIONS.get(entry.ref) ?? null
}
