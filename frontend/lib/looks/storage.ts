import { isPresetId } from './presets'
import { moveUnscopedOnce, scopedKey } from './scope'
import type { Look } from './types'
import { validateLook } from './validate'

/**
 * PRD-260 — looks are kept in localStorage, per signed-in user and workspace
 * (F5: each key is scoped, lib/looks/scope.ts). localStorage may be missing,
 * blocked or full: then Play runs in Ember, nothing is kept, and the read and
 * write helpers say so (StoredLooks.persisted, the writes' boolean) so the
 * Looks page can tell the user. Every look read back goes through
 * validateLook().
 */

/** The user's saved looks, a JSON array of Look objects: this base plus the scope. Unscoped, the key before F5. */
export const SAVED_LOOKS_KEY = 'automatos-play.looks'
/** The active look: its id plus its values, so an unsaved edit survives a reload. Scoped as SAVED_LOOKS_KEY. */
export const ACTIVE_LOOK_KEY = 'automatos-play.look'
/** The scope whose active look the pre-paint script paints: it runs before anyone is signed in. */
export const LOOK_SCOPE_KEY = 'automatos-play.look.current'
/** The scope the unscoped looks moved to: they move once, to the first account that reads them. */
export const UNSCOPED_LOOKS_MOVED_KEY = 'automatos-play.looks.moved-to'

export type StorageSource = () => Storage

export const browserStorage: StorageSource = () => window.localStorage

export interface LookKeys {
  readonly saved: string
  readonly active: string
}

export function lookKeys(scope: string): LookKeys {
  return { saved: scopedKey(SAVED_LOOKS_KEY, scope), active: scopedKey(ACTIVE_LOOK_KEY, scope) }
}

export interface StoredLooks {
  /** The scope they were read for. */
  readonly scope: string
  /** null when nothing valid is stored: the caller uses DEFAULT_LOOK. */
  readonly active: Look | null
  readonly saved: readonly Look[]
  /** False when storage can't be read: nothing the user does is kept past this tab. */
  readonly persisted: boolean
}

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

export function readStoredLooks(storage: StorageSource, scope: string): StoredLooks {
  const keys = lookKeys(scope)
  try {
    const store = storage()
    return {
      scope,
      active: parseActiveLook(store.getItem(keys.active)),
      saved: parseSavedLooks(store.getItem(keys.saved)),
      persisted: true,
    }
  } catch (error) {
    console.warn("[looks] can't read localStorage, so Play uses Ember and Looks aren't kept:", error)
    return { scope, active: null, saved: [], persisted: false }
  }
}

/** The first read for a scope: what this browser kept before F5 moves to it once (never deleted), then readStoredLooks. */
export function loadStoredLooks(storage: StorageSource, scope: string): StoredLooks {
  const keys = lookKeys(scope)
  try {
    moveUnscopedOnce(storage(), UNSCOPED_LOOKS_MOVED_KEY, scope, [
      { from: SAVED_LOOKS_KEY, to: keys.saved },
      { from: ACTIVE_LOOK_KEY, to: keys.active },
    ])
  } catch (error) {
    console.warn("[looks] can't move this browser's earlier Looks to the signed-in account:", error)
  }
  return readStoredLooks(storage, scope)
}

/** True when it was kept; false (with a warning) when storage refused it. */
function write(storage: StorageSource, key: string, value: unknown, what: string): boolean {
  try {
    storage().setItem(key, JSON.stringify(value))
    return true
  } catch (error) {
    console.warn(`[looks] can't save ${what} to localStorage; it lasts until the page closes`, error)
    return false
  }
}

export function writeActiveLook(storage: StorageSource, scope: string, look: Look): boolean {
  return write(storage, lookKeys(scope).active, look, 'the active Look')
}

export function writeSavedLooks(storage: StorageSource, scope: string, looks: readonly Look[]): boolean {
  return write(storage, lookKeys(scope).saved, looks, 'the saved Looks')
}

/** Points the pre-paint script at this scope's active look (lib/looks/prepaint.ts). */
export function rememberLookScope(storage: StorageSource, scope: string): boolean {
  try {
    const store = storage()
    if (store.getItem(LOOK_SCOPE_KEY) !== scope) store.setItem(LOOK_SCOPE_KEY, scope)
    return true
  } catch (error) {
    console.warn("[looks] can't note the signed-in account's Look for the next page load:", error)
    return false
  }
}
