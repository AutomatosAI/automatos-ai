import { moveUnscopedOnce, scopedKey } from '@/lib/look-kit/scope'

import type { ThemeLookStorage } from '../boot-types'
import { checkLook, isPresetId, type ThemeLook, type ThemeLookRules } from './spec'

/**
 * PRD-261 A-19d1 (plan §2.2.4) — this browser's copy of one theme's looks, per signed-in user and
 * workspace (PRD-260 F5: each key carries the scope, lib/look-kit/scope.ts). An installed theme keeps
 * them under `automatos-theme.<themeId>.*` (themeLookKeys), the keys its boot names to the boot script
 * (orchestrator serve.py look_storage; orchestrator/tests/fixtures/themes/look-storage.json holds the
 * two equal). The server keeps the looks (GET/PUT /api/themes/looks/{themeId}); this copy is its
 * write-through cache (./server.ts, ./store.ts) and what the theme runs on when the server can't be
 * reached.
 *
 * localStorage may be missing, blocked or full: then the theme runs on its default look, nothing is
 * kept, and the read and write helpers say so (StoredLooks.persisted, the writes' boolean) so the
 * look screens can tell the user. Every look read back, from here or from the server, goes through
 * the theme's checks (./spec.ts checkLook), so only a look of this theme's format that its table can
 * paint is ever taken.
 *
 * A key set may name a marker (`moved`): what a browser kept under the unscoped keys then moves to
 * the first scope that reads it, once (lib/look-kit/scope.ts moveUnscopedOnce), never to a second
 * account, and the unscoped keys are never deleted. An installed theme's looks were never unscoped,
 * so its keys name none and nothing moves.
 */

/** Where one theme keeps its looks in this browser; each key but `pointer` and `moved` gets the scope appended. */
export interface LookKeyBases {
  /** The saved looks, a JSON array of look objects. */
  readonly saved: string
  /** The active look: its id plus its values, so an unsaved edit survives a reload. */
  readonly active: string
  /** The scope whose active look the boot script paints: it runs before anyone is signed in. */
  readonly pointer: string
  /** The scope the unscoped looks moved to; null for a theme that never had unscoped keys. */
  readonly moved: string | null
}

/** A theme as the look store sees it: the id the server keeps its looks under, its keys here and its rules. */
export interface LookTheme {
  readonly themeId: string
  readonly keys: LookKeyBases
  readonly rules: ThemeLookRules
}

/** The prefix of an installed theme's keys: `automatos-theme.<themeId>.looks` and so on. */
export const INSTALLED_LOOK_KEY_PREFIX = 'automatos-theme.'
const SAVED_KEY_SUFFIX = '.looks'
const ACTIVE_KEY_SUFFIX = '.look'
const POINTER_KEY_SUFFIX = '.current'
/** Where a read or write happened, as a warning names it. */
const LOCAL_STORAGE = 'localStorage'

/** An installed theme's keys. Its looks were never unscoped, so nothing moves. */
export function themeLookKeys(themeId: string): LookKeyBases {
  const base = `${INSTALLED_LOOK_KEY_PREFIX}${themeId}`
  const active = `${base}${ACTIVE_KEY_SUFFIX}`
  return Object.freeze({ saved: `${base}${SAVED_KEY_SUFFIX}`, active, pointer: `${active}${POINTER_KEY_SUFFIX}`, moved: null })
}

/** The theme with its id, its rules and (unless given) its installed keys. */
export function lookTheme(themeId: string, rules: ThemeLookRules, keys: LookKeyBases = themeLookKeys(themeId)): LookTheme {
  return Object.freeze({ themeId, keys, rules })
}

/**
 * Two marks per scope, beside the saved looks (`<saved>.<mark>:<scope>`): `pending`, this copy changed
 * here and the server hasn't kept it yet (it is sent on the next load); `synced`, this browser has met
 * the server's copy once (its own looks have joined it, ./state.ts mergeLooks), so from now on the
 * server's copy is the truth.
 */
export type LookMark = 'pending' | 'synced'

export type StorageSource = () => Storage

export const browserStorage: StorageSource = () => window.localStorage

export interface LookKeys {
  readonly saved: string
  readonly active: string
}

export function lookKeys(scope: string, bases: LookKeyBases): LookKeys {
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
  /** null when nothing valid is stored: the caller uses the theme's default look. */
  readonly active: ThemeLook | null
  readonly saved: readonly ThemeLook[]
  /** False when storage can't be read: nothing the user does is kept past this tab. */
  readonly persisted: boolean
}

type Parsed = { readonly ok: true; readonly value: unknown } | { readonly ok: false }

function parseJson(raw: string, what: string): Parsed {
  try {
    return { ok: true, value: JSON.parse(raw) }
  } catch (error) {
    console.warn(`[looks] ${what} in ${LOCAL_STORAGE} isn't JSON, so it's ignored:`, error)
    return { ok: false }
  }
}

/** A look from storage or the server, checked; null (with a warning) when it isn't one of the theme's. */
export function activeLookOf(rules: ThemeLookRules, value: unknown, where = LOCAL_STORAGE): ThemeLook | null {
  const result = checkLook(rules, value)
  if (result.ok) return result.look
  console.warn(`[looks] the active Look in ${where} isn't valid, so the theme uses its default look:`, result.errors)
  return null
}

export function parseActiveLook(rules: ThemeLookRules, raw: string | null): ThemeLook | null {
  if (raw === null) return null
  const parsed = parseJson(raw, 'the active Look')
  return parsed.ok ? activeLookOf(rules, parsed.value) : null
}

function keepLook(rules: ThemeLookRules, kept: readonly ThemeLook[], entry: unknown): readonly ThemeLook[] {
  const result = checkLook(rules, entry)
  if (!result.ok) {
    console.warn("[looks] skipped a saved Look that isn't valid:", result.errors)
    return kept
  }
  const { id } = result.look
  if (isPresetId(rules, id) || kept.some((look) => look.id === id)) {
    console.warn(`[looks] skipped a saved Look whose id "${id}" is already taken`)
    return kept
  }
  return [...kept, result.look]
}

/** The valid, uniquely named looks of a list from storage or the server; anything else is skipped with a warning. */
export function savedLooksOf(rules: ThemeLookRules, value: unknown, where = LOCAL_STORAGE): readonly ThemeLook[] {
  if (!Array.isArray(value)) {
    console.warn(`[looks] the saved Looks in ${where} aren't a list, so they're ignored`)
    return []
  }
  const entries: readonly unknown[] = value
  return entries.reduce<readonly ThemeLook[]>((kept, entry) => keepLook(rules, kept, entry), [])
}

export function parseSavedLooks(rules: ThemeLookRules, raw: string | null): readonly ThemeLook[] {
  if (raw === null) return []
  const parsed = parseJson(raw, 'the saved Looks')
  return parsed.ok ? savedLooksOf(rules, parsed.value) : []
}

export function readStoredLooks(storage: StorageSource, scope: string, theme: LookTheme): StoredLooks {
  const keys = lookKeys(scope, theme.keys)
  try {
    const store = storage()
    return {
      scope,
      active: parseActiveLook(theme.rules, store.getItem(keys.active)),
      saved: parseSavedLooks(theme.rules, store.getItem(keys.saved)),
      persisted: true,
    }
  } catch (error) {
    console.warn(`[looks] can't read ${LOCAL_STORAGE}, so the theme uses its default look and Looks aren't kept:`, error)
    return { scope, active: null, saved: [], persisted: false }
  }
}

/** The first read for a scope: what the unscoped keys held moves to it once (never deleted), then readStoredLooks. */
export function loadStoredLooks(storage: StorageSource, scope: string, theme: LookTheme): StoredLooks {
  const bases = theme.keys
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
  return readStoredLooks(storage, scope, theme)
}

/** True when it was kept; false (with a warning) when storage refused it. */
function write(storage: StorageSource, key: string, value: unknown, what: string): boolean {
  try {
    storage().setItem(key, JSON.stringify(value))
    return true
  } catch (error) {
    console.warn(`[looks] can't save ${what} to ${LOCAL_STORAGE}; it lasts until the page closes`, error)
    return false
  }
}

export function writeActiveLook(storage: StorageSource, scope: string, bases: LookKeyBases, look: ThemeLook): boolean {
  return write(storage, lookKeys(scope, bases).active, look, 'the active Look')
}

export function writeSavedLooks(storage: StorageSource, scope: string, bases: LookKeyBases, looks: readonly ThemeLook[]): boolean {
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
    console.warn(`[looks] can't note the '${mark}' mark in ${LOCAL_STORAGE}:`, error)
    return false
  }
}

/**
 * The look the boot script paints: the active look of the scope the pointer key names, else the one
 * under the unscoped key, else the theme's default. It only reads: nothing moves, nothing is written.
 * The runtime paints it until the account's own looks are read (A-19d2), so the theme always has its
 * variables, also when the account or its workspace isn't known yet and the theme was switched to
 * after the page loaded (no first paint of its own ran).
 */
export function readPrepaintLook(storage: StorageSource, theme: LookTheme): ThemeLook {
  const bases = theme.keys
  try {
    const store = storage()
    const pointer = store.getItem(bases.pointer)
    const raw = store.getItem(pointer ? lookKeys(pointer, bases).active : bases.active)
    return parseActiveLook(theme.rules, raw) ?? theme.rules.defaultLook
  } catch (error) {
    console.warn(`[looks] can't read ${LOCAL_STORAGE}, so the theme paints its default look until the account's Looks are read:`, error)
    return theme.rules.defaultLook
  }
}

/** Points the boot script at this scope's active look for the next page load. */
export function rememberLookScope(storage: StorageSource, scope: string, bases: LookKeyBases): boolean {
  try {
    const store = storage()
    if (store.getItem(bases.pointer) !== scope) store.setItem(bases.pointer, scope)
    return true
  } catch (error) {
    console.warn("[looks] can't note the signed-in account's Look for the next page load:", error)
    return false
  }
}
