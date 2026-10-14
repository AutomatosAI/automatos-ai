import { DEFAULT_LOOK, isPresetId } from './presets'
import type { Look } from './types'

/**
 * PRD-260 — what the LookProvider holds, as a pure reducer. `lookChanged` /
 * `savedChanged` say whether the user changed that part since it was read from
 * storage: only then is it written back, so a read never rewrites what's stored.
 */

export interface LookState {
  readonly look: Look
  readonly saved: readonly Look[]
  /** False until storage has been read (after hydration). */
  readonly loaded: boolean
  readonly lookChanged: boolean
  readonly savedChanged: boolean
}

export type LookAction =
  | { readonly type: 'loaded'; readonly active: Look | null; readonly saved: readonly Look[] }
  | { readonly type: 'set'; readonly look: Look }
  /** Adds a saved look, or replaces the saved look with the same id. Never a preset. */
  | { readonly type: 'save'; readonly look: Look }
  | { readonly type: 'delete'; readonly id: string }
  | { readonly type: 'rename'; readonly id: string; readonly name: string }

export const INITIAL_LOOK_STATE: LookState = {
  look: DEFAULT_LOOK,
  saved: [],
  loaded: false,
  lookChanged: false,
  savedChanged: false,
}

function upsert(saved: readonly Look[], look: Look): readonly Look[] {
  return saved.some((entry) => entry.id === look.id)
    ? saved.map((entry) => (entry.id === look.id ? look : entry))
    : [...saved, look]
}

function deleteSaved(state: LookState, id: string): LookState {
  if (!state.saved.some((entry) => entry.id === id)) return state
  const saved = state.saved.filter((entry) => entry.id !== id)
  // The active look was the one deleted: Play goes back to the default.
  if (state.look.id !== id) return { ...state, saved, savedChanged: true }
  return { ...state, saved, look: DEFAULT_LOOK, savedChanged: true, lookChanged: true }
}

function renameSaved(state: LookState, id: string, name: string): LookState {
  if (!state.saved.some((entry) => entry.id === id)) return state
  const saved = state.saved.map((entry) => (entry.id === id ? { ...entry, name } : entry))
  if (state.look.id !== id) return { ...state, saved, savedChanged: true }
  return { ...state, saved, look: { ...state.look, name }, savedChanged: true, lookChanged: true }
}

export function lookReducer(state: LookState, action: LookAction): LookState {
  switch (action.type) {
    case 'loaded':
      // A change made before the read (unlikely: it runs on mount) wins over what's stored.
      return {
        ...state,
        look: state.lookChanged ? state.look : (action.active ?? DEFAULT_LOOK),
        saved: state.savedChanged ? state.saved : action.saved,
        loaded: true,
      }
    case 'set':
      return { ...state, look: action.look, lookChanged: true }
    case 'save':
      if (isPresetId(action.look.id)) return state
      return { ...state, saved: upsert(state.saved, action.look), savedChanged: true }
    case 'delete':
      return deleteSaved(state, action.id)
    case 'rename':
      return renameSaved(state, action.id, action.name)
  }
}
