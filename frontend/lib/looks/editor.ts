import { findPreset, isPresetId, PRESETS } from './presets'
import { GLOW_MAX, GLOW_MIN, LOOK_TEXT_MAX, type Look } from './types'

/**
 * PRD-260 — the Looks page's rules, pure: which look the active one started
 * from, whether it has been edited since, and what 'Save as new look' makes.
 * No React, no storage.
 */

/** The fields the editor changes. Everything else (schema, id, name, sub) stays as picked. */
export const EDITABLE_FIELDS = ['ground', 'accent', 'font', 'shell', 'light', 'glow'] as const satisfies readonly (keyof Look)[]

export type EditableField = (typeof EDITABLE_FIELDS)[number]
export type LookValues = Pick<Look, EditableField>

/** What a freshly saved look says under its name (the Looks prototype). */
export const SAVED_SUB = 'Saved just now'
export const EDITED_MARK = 'edited'

/** The preset or saved look the active look was picked from, as it was saved. */
export function baseLook(look: Look, saved: readonly Look[]): Look | undefined {
  return findPreset(look.id) ?? saved.find((entry) => entry.id === look.id)
}

export function sameValues(a: Look, b: Look): boolean {
  return EDITABLE_FIELDS.every((field) => a[field] === b[field])
}

/** True when the active look's values differ from the look it was picked from. */
export function isEdited(look: Look, saved: readonly Look[]): boolean {
  const base = baseLook(look, saved)
  return base !== undefined && !sameValues(base, look)
}

/** The line beside the heading: 'Ember', or 'Ember · edited'. */
export function statusLine(look: Look, saved: readonly Look[]): string {
  return isEdited(look, saved) ? `${look.name} · ${EDITED_MARK}` : look.name
}

/** A copy of the look with some values changed. Never mutates `look`. */
export function withValues(look: Look, values: Partial<LookValues>): Look {
  return { ...look, ...values }
}

/**
 * A slider reading as a whole glow from GLOW_MIN to `max`: the Glow slider
 * passes the look's safeGlow() (lib/looks/css.ts), the cap glowVars() paints
 * at; never past GLOW_MAX.
 */
export function clampGlow(value: number, max: number = GLOW_MAX): number {
  if (!Number.isFinite(value)) return GLOW_MIN
  return Math.min(max, GLOW_MAX, Math.max(GLOW_MIN, Math.round(value)))
}

/** 'Off' at 0, else '24%'. */
export function glowLabel(glow: number): string {
  return glow === GLOW_MIN ? 'Off' : `${glow}%`
}

/** 'My look N' with the lowest N from the number of saved looks up that no saved look is called. */
export function newLookName(saved: readonly Look[]): string {
  const names = new Set(saved.map((entry) => entry.name))
  let n = saved.length + 1
  while (names.has(`My look ${n}`)) n += 1
  return `My look ${n}`
}

/** 'look-N' with the lowest N that no preset or saved look uses (a valid LOOK_ID_PATTERN id). */
export function freshLookId(saved: readonly Look[]): string {
  const ids = new Set([...PRESETS, ...saved].map((entry) => entry.id))
  let n = 1
  while (ids.has(`look-${n}`)) n += 1
  return `look-${n}`
}

/** The active look's values under a new id and name, ready for saveLook. */
export function asNewLook(look: Look, saved: readonly Look[]): Look {
  return { ...look, id: freshLookId(saved), name: newLookName(saved), sub: SAVED_SUB }
}

/** True when the user can rename or delete it: one of their saved looks, never a preset. */
export function isSavedLook(look: Look, saved: readonly Look[]): boolean {
  return !isPresetId(look.id) && saved.some((entry) => entry.id === look.id)
}

const COPY_WORD = 'copy'
/** Room for ' copy 99' after the clipped name, so the copy's name stays within LOOK_TEXT_MAX. */
const COPY_ROOM = 8

/** 'Ember copy', then 'Ember copy 2'... the first no preset or saved look is called. */
export function copyName(name: string, saved: readonly Look[]): string {
  const taken = new Set([...PRESETS, ...saved].map((entry) => entry.name))
  const base = Array.from(name).slice(0, LOOK_TEXT_MAX - COPY_ROOM).join('').trim()
  let candidate = `${base} ${COPY_WORD}`
  for (let n = 2; taken.has(candidate); n += 1) candidate = `${base} ${COPY_WORD} ${n}`
  return candidate
}

/** A copy of the look as it was picked, under a fresh id and '<name> copy'. */
export function asDuplicate(look: Look, saved: readonly Look[]): Look {
  return { ...look, id: freshLookId(saved), name: copyName(look.name, saved) }
}
