import { DEFAULT_LOOK, isPresetId } from './presets'
import type { StoredLooks } from './storage'
import type { Look } from './types'

/**
 * PRD-260 — what the LookProvider holds, as a pure reducer. `lookChanged` /
 * `savedChanged` say whether the user changed that part since it was read from
 * storage: only then is it written back, so a read never rewrites what's stored.
 * Looks belong to one scope, the signed-in user and workspace (F5,
 * lib/looks/scope.ts): `loaded` for another scope, or `synced` from another
 * tab, takes what that storage holds whole and marks nothing for writing, so
 * one account's looks never land under another's keys and two tabs can't
 * ping-pong (M25).
 */

export interface LookState {
  readonly look: Look
  readonly saved: readonly Look[]
  /** False until storage has been read (after hydration, once the account is known). */
  readonly loaded: boolean
  /** The scope the looks were read for, and are written to; null until then. */
  readonly scope: string | null
  /** False once storage refused a read or a write: what the user does lasts only as long as this tab (M98). */
  readonly persisted: boolean
  readonly lookChanged: boolean
  readonly savedChanged: boolean
}

export type LookAction =
  /** Storage read for a scope (a first read, or another account or workspace). */
  | ({ readonly type: 'loaded' } & StoredLooks)
  /** Another tab changed this scope's looks: take them, write nothing back. */
  | ({ readonly type: 'synced' } & StoredLooks)
  /** What a write to storage returned. */
  | { readonly type: 'written'; readonly ok: boolean }
  | { readonly type: 'set'; readonly look: Look }
  /** Adds a saved look, or replaces the saved look with the same id. Never a preset. */
  | { readonly type: 'save'; readonly look: Look }
  | { readonly type: 'delete'; readonly id: string }
  | { readonly type: 'rename'; readonly id: string; readonly name: string }

export const INITIAL_LOOK_STATE: LookState = {
  look: DEFAULT_LOOK,
  saved: [],
  loaded: false,
  scope: null,
  persisted: true,
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

/** The first read. A change made before it (unlikely: it runs on mount) wins over what's stored, and is written. */
function firstRead(state: LookState, read: StoredLooks): LookState {
  return {
    ...state,
    look: state.lookChanged ? state.look : (read.active ?? DEFAULT_LOOK),
    saved: state.savedChanged ? state.saved : read.saved,
    loaded: true,
    scope: read.scope,
    persisted: read.persisted,
  }
}

/** Exactly what storage holds for the scope, nothing marked for writing. */
function replaced(state: LookState, read: StoredLooks): LookState {
  return {
    ...state,
    look: read.active ?? DEFAULT_LOOK,
    saved: read.saved,
    loaded: true,
    scope: read.scope,
    persisted: read.persisted,
    lookChanged: false,
    savedChanged: false,
  }
}

function stored(state: LookState, action: Extract<LookAction, { type: 'loaded' | 'synced' }>): LookState {
  const { type, ...read } = action
  if (type === 'synced') return read.scope === state.scope ? replaced(state, read) : state
  if (!state.loaded) return firstRead(state, read)
  // The same scope read again (a workspace refresh) keeps what this tab holds, unsaved edits included.
  return read.scope === state.scope ? state : replaced(state, read)
}

export function lookReducer(state: LookState, action: LookAction): LookState {
  switch (action.type) {
    case 'loaded':
    case 'synced':
      return stored(state, action)
    case 'written':
      return action.ok || !state.persisted ? state : { ...state, persisted: false }
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
