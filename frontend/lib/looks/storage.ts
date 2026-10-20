import type { ThemeLookStorage } from '@/lib/theme-host/boot-types'
import { BUILTIN_PLAY_LOOK_KEYS } from '@/lib/theme-host/builtin/play-look-keys'

import { validateLook } from './engine'
import { DEFAULT_LOOK, isPresetId } from './presets'
import { moveUnscopedOnce, scopedKey } from './scope'
import type { Look } from './types'

/**
 * PRD-260, PRD-261 A-09b — this browser's copy of the looks, per signed-in
 * user and workspace (F5: each key is scoped, lib/looks/scope.ts) and per
 * theme: the keys come from the active theme (LookKeyBases). An installed
 * theme's are `automatos-theme.<themeId>.*` (installedLookKeys); builtin
 * Play's are its PRD-260 keys (lib/theme-host/builtin/play-look-keys.ts), the
 * default here, so no existing look moves before A-18. The server keeps the
 * looks (GET/PUT /api/themes/looks/{themeId}); this copy is its write-through
 * cache (components/looks/use-look-store.ts), and what Play runs on when the
 * server can't be reached. localStorage may be missing, blocked or full: then
 * Play runs in Ember, nothing is kept, and the read and write helpers say so
 * (StoredLooks.persisted, the writes' boolean) so the Looks page can tell the
 * user. Every look read back, from here or from the server, goes through
 * validateLook().
 */

/** Where one theme keeps its looks in this browser; each key gets the scope appended. */
export interface LookKeyBases {
  /** The saved looks, a JSON array of Look objects. */
  readonly saved: string
  /** The active look: its id plus its values, so an unsaved edit survives a reload. */
  readonly active: string
  /** The scope whose active look the boot script paints: it runs before anyone is signed in. */
  readonly pointer: string
  /** The scope the unscoped (pre-F5) looks moved to; null for a theme that never had unscoped keys. */
  readonly moved: string | null
}

/** A theme as the look store sees it: the id the server keeps its looks under, and its keys here. */
export interface LookTheme {
  readonly themeId: string
  readonly keys: LookKeyBases
}

/** The prefix of an installed theme's keys: `automatos-theme.<themeId>.looks` and so on. */
export const INSTALLED_LOOK_KEY_PREFIX = 'automatos-theme.'

/** An installed theme's keys. Its looks were never unscoped, so nothing moves. */
export function installedLookKeys(themeId: string): LookKeyBases {
  const base = `${INSTALLED_LOOK_KEY_PREFIX}${themeId}`
  return Object.freeze({ saved: `${base}.looks`, active: `${base}.look`, pointer: `${base}.look.current`, moved: null })
}

export function installedLookTheme(themeId: string): LookTheme {
  return Object.freeze({ themeId, keys: installedLookKeys(themeId) })
}

/** Builtin Play's keys (PRD-260), the names its callers and tests know them by, until A-19. */
export const SAVED_LOOKS_KEY = BUILTIN_PLAY_LOOK_KEYS.saved
export const ACTIVE_LOOK_KEY = BUILTIN_PLAY_LOOK_KEYS.active
export const LOOK_SCOPE_KEY = BUILTIN_PLAY_LOOK_KEYS.pointer
export const UNSCOPED_LOOKS_MOVED_KEY = BUILTIN_PLAY_LOOK_KEYS.moved as string

/**
 * Two marks per scope, beside the saved looks (`<saved>.<mark>:<scope>`): `pending`, this copy
 * changed here and the server hasn't kept it yet (it is sent on the next load); `synced`, this
 * browser has met the server's copy once (its own looks have joined it, lib/looks/store.ts
 * mergeLooks), so from now on the server's copy is the truth.
 */
export type LookMark = 'pending' | 'synced'

export type StorageSource = () => Storage

export const browserStorage: StorageSource = () => window.localStorage

export interface LookKeys {
  readonly saved: string
  readonly active: string
}

export function lookKeys(scope: string, bases: LookKeyBases = BUILTIN_PLAY_LOOK_KEYS): LookKeys {
  return { saved: scopedKey(bases.saved, scope), active: scopedKey(bases.active, scope) }
}

export function lookMarkKey(scope: string, bases: LookKeyBases, mark: LookMark): string {
  return scopedKey(`${bases.saved}.${mark}`, scope)
}

/** What the boot script reads to paint the cached active look before first paint (lib/theme-host/boot.ts). */
export function bootLookStorage(bases: LookKeyBases): ThemeLookStorage {
  return Object.freeze({ pointerKey: bases.pointer, scopedPrefix: scopedKey(bases.active, ''), fallbackKey: bases.active })
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

/** A look from storage or the server, checked; null (with a warning) when it isn't one. */
export function activeLookOf(value: unknown, where = 'localStorage'): Look | null {
  const result = validateLook(value)
  if (result.ok) return result.look
  console.warn(`[looks] the active Look in ${where} isn't valid, so Play uses Ember:`, result.errors)
  return null
}

export function parseActiveLook(raw: string | null): Look | null {
  if (raw === null) return null
  const parsed = parseJson(raw, 'the active Look')
  return parsed.ok ? activeLookOf(parsed.value) : null
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

/** The valid, uniquely named looks of a list from storage or the server; anything else is skipped with a warning. */
export function savedLooksOf(value: unknown, where = 'localStorage'): readonly Look[] {
  if (!Array.isArray(value)) {
    console.warn(`[looks] the saved Looks in ${where} aren't a list, so they're ignored`)
    return []
  }
  const entries: readonly unknown[] = value
  return entries.reduce<readonly Look[]>(keepLook, [])
}

/** The valid, uniquely named saved looks; anything else is skipped with a warning. */
export function parseSavedLooks(raw: string | null): readonly Look[] {
  if (raw === null) return []
  const parsed = parseJson(raw, 'the saved Looks')
  return parsed.ok ? savedLooksOf(parsed.value) : []
}

export function readStoredLooks(
  storage: StorageSource,
  scope: string,
  bases: LookKeyBases = BUILTIN_PLAY_LOOK_KEYS,
): StoredLooks {
  const keys = lookKeys(scope, bases)
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
export function loadStoredLooks(
  storage: StorageSource,
  scope: string,
  bases: LookKeyBases = BUILTIN_PLAY_LOOK_KEYS,
): StoredLooks {
  const keys = lookKeys(scope, bases)
  try {
    if (bases.moved !== null) {
      moveUnscopedOnce(storage(), bases.moved, scope, [
        { from: bases.saved, to: keys.saved },
        { from: bases.active, to: keys.active },
      ])
    }
  } catch (error) {
    console.warn("[looks] can't move this browser's earlier Looks to the signed-in account:", error)
  }
  return readStoredLooks(storage, scope, bases)
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

export function writeActiveLook(
  storage: StorageSource,
  scope: string,
  look: Look,
  bases: LookKeyBases = BUILTIN_PLAY_LOOK_KEYS,
): boolean {
  return write(storage, lookKeys(scope, bases).active, look, 'the active Look')
}

export function writeSavedLooks(
  storage: StorageSource,
  scope: string,
  looks: readonly Look[],
  bases: LookKeyBases = BUILTIN_PLAY_LOOK_KEYS,
): boolean {
  return write(storage, lookKeys(scope, bases).saved, looks, 'the saved Looks')
}

/** True when the mark is set for this scope. Unreadable storage: false. */
export function hasLookMark(storage: StorageSource, scope: string, bases: LookKeyBases, mark: LookMark): boolean {
  try {
    return storage().getItem(lookMarkKey(scope, bases, mark)) !== null
  } catch {
    return false
  }
}

/** Sets (true) or clears (false) the mark; false, with a warning, when storage refused. */
export function setLookMark(storage: StorageSource, scope: string, bases: LookKeyBases, mark: LookMark, on: boolean): boolean {
  try {
    const store = storage()
    const key = lookMarkKey(scope, bases, mark)
    if (on) store.setItem(key, '1')
    else store.removeItem(key)
    return true
  } catch (error) {
    console.warn(`[looks] can't note the '${mark}' mark in localStorage:`, error)
    return false
  }
}

/**
 * The look the pre-paint script paints (lib/looks/prepaint.ts): the active
 * look of the scope the pointer key names, else the one this browser kept
 * before F5, else the default. It only reads: nothing moves, nothing is
 * written. The LookProvider paints it under Play until the account's own
 * looks are read, so Play always has its variables, also when the account or
 * its workspace isn't known yet and the style was switched to Play after the
 * page loaded (no pre-paint ran).
 */
export function readPrepaintLook(storage: StorageSource, bases: LookKeyBases = BUILTIN_PLAY_LOOK_KEYS): Look {
  try {
    const store = storage()
    const pointer = store.getItem(bases.pointer)
    return parseActiveLook(store.getItem(pointer ? lookKeys(pointer, bases).active : bases.active)) ?? DEFAULT_LOOK
  } catch (error) {
    console.warn("[looks] can't read localStorage, so Play paints Ember until the account's Looks are read:", error)
    return DEFAULT_LOOK
  }
}

/** Points the pre-paint script at this scope's active look (lib/looks/prepaint.ts). */
export function rememberLookScope(
  storage: StorageSource,
  scope: string,
  bases: LookKeyBases = BUILTIN_PLAY_LOOK_KEYS,
): boolean {
  try {
    const store = storage()
    if (store.getItem(bases.pointer) !== scope) store.setItem(bases.pointer, scope)
    return true
  } catch (error) {
    console.warn("[looks] can't note the signed-in account's Look for the next page load:", error)
    return false
  }
}
