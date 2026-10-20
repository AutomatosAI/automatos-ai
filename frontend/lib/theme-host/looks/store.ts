import { type Dispatch, useCallback, useEffect, useMemo, useReducer, useRef } from 'react'

import { apiLookServer, type LookServer, useLookServerSync } from './server'
import { checkLook, isPresetId, type ThemeLook, type ThemeLookCheck } from './spec'
import { initialLookState, type LookAction, type LookState, lookReducer } from './state'
import {
  loadStoredLooks,
  lookKeys,
  readStoredLooks,
  writeActiveLook,
  writeSavedLooks,
  type LookTheme,
  type StorageSource,
} from './storage'

/**
 * PRD-261 A-19d1 (plan §2.2.4) — the look state of one theme for the signed-in user and workspace
 * (PRD-260 F5: `accountScope`, null until both are known). It is read from this browser's copy after
 * mount, so the page paints at once, then from the server (GET /api/themes/looks/{themeId}, ./server.ts),
 * which is the truth; a change is written to the copy at once and sent to the server (PUT). Other
 * tabs follow through the copy (M25). Every look handed in passes the theme's checks first
 * (./spec.ts checkLook: its knobs and its prepaint table), and the theme's presets are never saved
 * over, renamed or deleted.
 *
 * `theme` null (no theme with looks runs) keeps the store inert: nothing is read, written or sent,
 * there is no look and every action refuses, so the host can keep the store mounted in one tree
 * shape (M69); switching it off forgets what was read, so switching back reads again. Until this
 * theme's looks are read, the store answers the theme's default look and `loaded: false`: another
 * theme's looks are never handed out.
 */
export interface LookStore {
  /** The active look, possibly edited since it was picked; null when no theme with looks runs. */
  readonly look: ThemeLook | null
  /** The user's own looks (the presets ship in the theme's boot). */
  readonly saved: readonly ThemeLook[]
  /** False until storage has been read; the boot's first paint covers that moment. */
  readonly loaded: boolean
  /** False once storage, or the server, refused a write: the look screens say it isn't kept (M98). */
  readonly persisted: boolean
  /** The scope the looks belong to; null until storage has been read. */
  readonly scope: string | null
  /** Makes the look the active one, once it passes the theme's checks. */
  setLook(input: unknown): ThemeLookCheck
  /** Adds the look to the saved looks, or replaces the one with its id. Presets can't be saved over. */
  saveLook(input: unknown): ThemeLookCheck
  /** Deleting the active look goes back to the theme's default. A preset can't be deleted. */
  deleteLook(id: string): void
  /** Renames a saved look. A preset can't be renamed. */
  renameLook(id: string, name: string): ThemeLookCheck
}

export interface LookStoreOptions {
  /** Where the looks are kept besides this browser; null keeps them here only. Default: the API. */
  readonly server?: LookServer | null
}

function refusal(error: string): ThemeLookCheck {
  return Object.freeze({ ok: false as const, errors: Object.freeze([error]) })
}

export const LOOKS_NOT_AVAILABLE = refusal("Looks aren't available here.")
export const PRESET_REFUSED = refusal("A preset can't be changed. Save it as a new look instead.")
export const NOT_SAVED = refusal("There's no saved look with that id.")

const NO_SAVED: readonly ThemeLook[] = Object.freeze([])

/**
 * Another tab wrote this scope's looks (or cleared storage, key null): take what it stored. The
 * 'synced' action marks nothing for writing, and setting an identical value fires no event, so tabs
 * can't ping-pong.
 */
function useStorageSync(storage: StorageSource, scope: string | null, theme: LookTheme | null, dispatch: Dispatch<LookAction>): void {
  useEffect(() => {
    if (scope === null || theme === null) return undefined
    const keys = lookKeys(scope, theme.keys)
    const onStorage = (event: StorageEvent) => {
      if (event.key !== null && event.key !== keys.saved && event.key !== keys.active) return
      dispatch({ type: 'synced', themeId: theme.themeId, rules: theme.rules, ...readStoredLooks(storage, scope, theme) })
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [storage, scope, theme, dispatch])
}

function sameTheme(a: LookTheme | null, b: LookTheme | null): boolean {
  if (a === null || b === null) return a === b
  return a.themeId === b.themeId && a.rules === b.rules && JSON.stringify(a.keys) === JSON.stringify(b.keys)
}

/**
 * The theme the looks are for, one object per theme: a caller that builds it on every render
 * (lookTheme(id, rules)) never makes the store read it, or the server, again. The rules are compared
 * by reference: the caller keeps one rules object per boot.
 */
function useKeptTheme(given: LookTheme | null): LookTheme | null {
  const kept = useRef(given)
  if (!sameTheme(kept.current, given)) kept.current = given
  return kept.current
}

/** Each part the user changed is written to this browser's copy, under the keys it was read with. */
function useCopyWrites(storage: StorageSource, theme: LookTheme | null, state: LookState, dispatch: Dispatch<LookAction>): void {
  const { look, saved, lookChanged, savedChanged, scope } = state
  const keys = theme !== null && state.themeId === theme.themeId ? theme.keys : null

  useEffect(() => {
    if (!lookChanged || keys === null || scope === null || look === null) return
    dispatch({ type: 'written', ok: writeActiveLook(storage, scope, keys, look) })
  }, [storage, scope, keys, look, lookChanged, dispatch])

  useEffect(() => {
    if (!savedChanged || keys === null || scope === null) return
    dispatch({ type: 'written', ok: writeSavedLooks(storage, scope, keys, saved) })
  }, [storage, scope, keys, saved, savedChanged, dispatch])
}

/** The actions, each behind the theme's checks; with no theme every one refuses. */
function useLookActions(theme: LookTheme | null, saved: readonly ThemeLook[], dispatch: Dispatch<LookAction>) {
  const setLook = useCallback((input: unknown): ThemeLookCheck => {
    if (theme === null) return LOOKS_NOT_AVAILABLE
    const result = checkLook(theme.rules, input)
    if (result.ok) dispatch({ type: 'set', look: result.look })
    return result
  }, [theme, dispatch])

  const saveLook = useCallback((input: unknown): ThemeLookCheck => {
    if (theme === null) return LOOKS_NOT_AVAILABLE
    const id = (input as { id?: unknown } | null)?.id
    if (typeof id === 'string' && isPresetId(theme.rules, id)) return PRESET_REFUSED
    const result = checkLook(theme.rules, input)
    if (result.ok) dispatch({ type: 'save', look: result.look })
    return result
  }, [theme, dispatch])

  const deleteLook = useCallback((id: string) => dispatch({ type: 'delete', id }), [dispatch])

  const renameLook = useCallback((id: string, name: string): ThemeLookCheck => {
    if (theme === null) return LOOKS_NOT_AVAILABLE
    if (isPresetId(theme.rules, id)) return PRESET_REFUSED
    const existing = saved.find((entry) => entry.id === id)
    if (!existing) return NOT_SAVED
    const result = checkLook(theme.rules, { ...existing, name })
    if (result.ok) dispatch({ type: 'rename', id, name: result.look.name })
    return result
  }, [theme, saved, dispatch])

  return useMemo(() => ({ setLook, saveLook, deleteLook, renameLook }), [setLook, saveLook, deleteLook, renameLook])
}

export function useLookStore(
  storage: StorageSource,
  accountScope: string | null,
  given: LookTheme | null,
  options: LookStoreOptions = {},
): LookStore {
  const theme = useKeptTheme(given)
  const [state, dispatch] = useReducer(lookReducer, theme?.rules ?? null, initialLookState)
  const server = options.server === undefined ? apiLookServer : options.server
  const here = theme !== null && state.loaded && state.themeId === theme.themeId

  useEffect(() => {
    if (theme === null) dispatch({ type: 'reset' })
    if (accountScope === null || theme === null) return
    dispatch({ type: 'loaded', themeId: theme.themeId, rules: theme.rules, ...loadStoredLooks(storage, accountScope, theme) })
  }, [storage, accountScope, theme])
  useStorageSync(storage, state.scope, theme, dispatch)
  useCopyWrites(storage, theme, state, dispatch)
  useLookServerSync({ storage, theme, server, enabled: here }, state, dispatch)

  const saved = here ? state.saved : NO_SAVED
  const actions = useLookActions(theme, saved, dispatch)
  const look = here ? state.look : (theme?.rules.defaultLook ?? null)
  const { persisted } = state
  const scope = here ? state.scope : null
  return useMemo(() => ({ look, saved, loaded: here, persisted, scope, ...actions }), [look, saved, here, persisted, scope, actions])
}
