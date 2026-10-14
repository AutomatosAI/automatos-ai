import { useCallback, useEffect, useMemo, useReducer } from 'react'

import { isPresetId } from '@/lib/looks/presets'
import {
  readStoredLooks,
  writeActiveLook,
  writeSavedLooks,
  type StorageSource,
} from '@/lib/looks/storage'
import { INITIAL_LOOK_STATE, lookReducer } from '@/lib/looks/store'
import type { Look, ValidationResult } from '@/lib/looks/types'
import { validateLook } from '@/lib/looks/validate'

/** PRD-260 — the look state, read from storage after mount and written back only when the user changes it. */
export interface LookStore {
  /** The active look, possibly edited since it was picked. */
  readonly look: Look
  /** The user's own looks (presets are in lib/looks/presets.ts). */
  readonly saved: readonly Look[]
  /** False until storage has been read; the pre-paint script covers that first moment. */
  readonly loaded: boolean
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

export function useLookStore(storage: StorageSource): LookStore {
  const [state, dispatch] = useReducer(lookReducer, INITIAL_LOOK_STATE)
  const { look, saved, loaded, lookChanged, savedChanged } = state

  useEffect(() => {
    dispatch({ type: 'loaded', ...readStoredLooks(storage) })
  }, [storage])

  useEffect(() => {
    if (lookChanged) writeActiveLook(storage, look)
  }, [storage, look, lookChanged])

  useEffect(() => {
    if (savedChanged) writeSavedLooks(storage, saved)
  }, [storage, saved, savedChanged])

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
    () => ({ look, saved, loaded, setLook, saveLook, deleteLook, renameLook }),
    [look, saved, loaded, setLook, saveLook, deleteLook, renameLook],
  )
}
