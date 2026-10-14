import { isPresetId } from './presets'
import type { Look } from './types'
import { validateLook } from './validate'

/**
 * PRD-260 — looks are kept per browser in localStorage, which may be missing,
 * blocked or full: then Play runs in Ember and nothing is kept. Every look read
 * back goes through validateLook().
 */

/** The user's saved looks, a JSON array of Look objects. */
export const SAVED_LOOKS_KEY = 'automatos-play.looks'
/** The active look: its id plus its values, so an unsaved edit survives a reload. */
export const ACTIVE_LOOK_KEY = 'automatos-play.look'

export type StorageSource = () => Storage

export const browserStorage: StorageSource = () => window.localStorage

export interface StoredLooks {
  /** null when nothing valid is stored: the caller uses DEFAULT_LOOK. */
  readonly active: Look | null
  readonly saved: readonly Look[]
}

const NOTHING_STORED: StoredLooks = { active: null, saved: [] }

type Parsed = { readonly ok: true; readonly value: unknown } | { readonly ok: false }

function parseJson(raw: string, what: string): Parsed {
  try {
    return { ok: true, value: JSON.parse(raw) }
  } catch (error) {
    console.warn(`[looks] ${what} in localStorage isn't JSON, so it's ignored:`, error)
    return { ok: false }
  }
}

export function parseActiveLook(raw: string | null): Look | null {
  if (raw === null) return null
  const parsed = parseJson(raw, 'the active Look')
  if (!parsed.ok) return null
  const result = validateLook(parsed.value)
  if (result.ok) return result.look
  console.warn("[looks] the active Look in localStorage isn't valid, so Play uses Ember:", result.errors)
  return null
}

function keepLook(kept: readonly Look[], entry: unknown): readonly Look[] {
  const result = validateLook(entry)
  if (!result.ok) {
    console.warn("[looks] skipped a saved Look that isn't valid:", result.errors)
    return kept
  }
  const { id } = result.look
  if (isPresetId(id) || kept.some((look) => look.id === id)) {
    console.warn(`[looks] skipped a saved Look whose id "${id}" is already taken`)
    return kept
  }
  return [...kept, result.look]
}

/** The valid, uniquely named saved looks; anything else is skipped with a warning. */
export function parseSavedLooks(raw: string | null): readonly Look[] {
  if (raw === null) return []
  const parsed = parseJson(raw, 'the saved Looks')
  if (!parsed.ok) return []
  if (!Array.isArray(parsed.value)) {
    console.warn("[looks] the saved Looks in localStorage aren't a list, so they're ignored")
    return []
  }
  const entries: readonly unknown[] = parsed.value
  return entries.reduce<readonly Look[]>(keepLook, [])
}

export function readStoredLooks(storage: StorageSource): StoredLooks {
  try {
    const store = storage()
    return {
      active: parseActiveLook(store.getItem(ACTIVE_LOOK_KEY)),
      saved: parseSavedLooks(store.getItem(SAVED_LOOKS_KEY)),
    }
  } catch (error) {
    console.warn("[looks] can't read localStorage, so Play uses Ember and Looks aren't kept:", error)
    return NOTHING_STORED
  }
}

export function writeActiveLook(storage: StorageSource, look: Look): void {
  try {
    storage().setItem(ACTIVE_LOOK_KEY, JSON.stringify(look))
  } catch (error) {
    console.warn("[looks] can't save the active Look to localStorage; it lasts until the page closes", error)
  }
}

export function writeSavedLooks(storage: StorageSource, looks: readonly Look[]): void {
  try {
    storage().setItem(SAVED_LOOKS_KEY, JSON.stringify(looks))
  } catch (error) {
    console.warn("[looks] can't save the saved Looks to localStorage; they last until the page closes", error)
  }
}
