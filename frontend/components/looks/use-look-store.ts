import { type Dispatch, useCallback, useEffect, useMemo, useReducer } from 'react'

import { isPresetId } from '@/lib/looks/presets'
import {
  loadStoredLooks,
  lookKeys,
  readStoredLooks,
  writeActiveLook,
  writeSavedLooks,
  type StorageSource,
} from '@/lib/looks/storage'
import { INITIAL_LOOK_STATE, type LookAction, lookReducer } from '@/lib/looks/store'
import type { Look, ValidationResult } from '@/lib/looks/types'
import { validateLook } from '@/lib/looks/validate'

/**
 * PRD-260 — the look state, read from storage after mount for the signed-in
 * user and workspace (F5: `scope`, null until both are known), written back
 * only when the user changes it, and kept in step with other tabs (M25).
 */
export interface LookStore {
  /** The active look, possibly edited since it was picked. */
  readonly look: Look
  /** The user's own looks (presets are in lib/looks/presets.ts). */
  readonly saved: readonly Look[]
  /** False until storage has been read; the pre-paint script covers that first moment. */
  readonly loaded: boolean
  /** False once storage refused a read or a write: changes last until the tab closes (M98). */
  readonly persisted: boolean
  /** The scope the looks belong to; null until storage has been read. */
  readonly scope: string | null
  setLook(look: Look): void
  /** Adds the look to the saved looks, or replaces the one with its id. Presets can't be saved over. */
  saveLook(look: Look): ValidationResult
  /** Deleting the active look switches Play back to the default. */
  deleteLook(id: string): void
  renameLook(id: string, name: string): ValidationResult
}

const PRESET_REFUSED: ValidationResult = {
  ok: false,
  errors: ["A preset can't be changed. Save it as a new look instead."],
}

const NOT_SAVED: ValidationResult = { ok: false, errors: ["There's no saved look with that id."] }

/**
 * Another tab wrote this scope's looks (or cleared storage, key null): take
 * what it stored. The 'synced' action marks nothing for writing, and setting
 * an identical value fires no event, so tabs can't ping-pong.
 */
function useStorageSync(storage: StorageSource, scope: string | null, dispatch: Dispatch<LookAction>): void {
  useEffect(() => {
    if (scope === null) return undefined
    const keys = lookKeys(scope)
    const onStorage = (event: StorageEvent) => {
      if (event.key !== null && event.key !== keys.saved && event.key !== keys.active) return
      dispatch({ type: 'synced', ...readStoredLooks(storage, scope) })
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [storage, scope, dispatch])
}

export function useLookStore(storage: StorageSource, accountScope: string | null): LookStore {
  const [state, dispatch] = useReducer(lookReducer, INITIAL_LOOK_STATE)
  const { look, saved, loaded, persisted, scope, lookChanged, savedChanged } = state

  useEffect(() => {
    if (accountScope !== null) dispatch({ type: 'loaded', ...loadStoredLooks(storage, accountScope) })
  }, [storage, accountScope])
  useStorageSync(storage, scope, dispatch)

  useEffect(() => {
    if (lookChanged && scope !== null) dispatch({ type: 'written', ok: writeActiveLook(storage, scope, look) })
  }, [storage, scope, look, lookChanged])

  useEffect(() => {
    if (savedChanged && scope !== null) dispatch({ type: 'written', ok: writeSavedLooks(storage, scope, saved) })
  }, [storage, scope, saved, savedChanged])

  const setLook = useCallback((next: Look) => dispatch({ type: 'set', look: next }), [])
  const deleteLook = useCallback((id: string) => dispatch({ type: 'delete', id }), [])

  const saveLook = useCallback((next: Look): ValidationResult => {
    if (isPresetId(next.id)) return PRESET_REFUSED
    const result = validateLook(next)
    if (result.ok) dispatch({ type: 'save', look: result.look })
    return result
  }, [])

  const renameLook = useCallback(
    (id: string, name: string): ValidationResult => {
      const existing = saved.find((entry) => entry.id === id)
      if (!existing) return NOT_SAVED
      const result = validateLook({ ...existing, name })
      if (result.ok) dispatch({ type: 'rename', id, name: result.look.name })
      return result
    },
    [saved],
  )

  return useMemo(
    () => ({ look, saved, loaded, persisted, scope, setLook, saveLook, deleteLook, renameLook }),
    [look, saved, loaded, persisted, scope, setLook, saveLook, deleteLook, renameLook],
  )
}
