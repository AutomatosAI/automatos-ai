import { freshLookId } from '@/lib/looks/editor'
import { DEFAULT_LOOK, isPresetId } from '@/lib/looks/presets'
import type { Look } from '@/lib/looks/types'

import type { StoredLooks } from './play-look-storage'

/**
 * PRD-260 — what the LookProvider holds, as a pure reducer. `lookChanged` /
 * `savedChanged` say whether the user changed that part since it was read from
 * storage: only then is it written back, so a read never rewrites what's stored.
 * Looks belong to one scope, the signed-in user and workspace (F5,
 * lib/looks/scope.ts): `loaded` for another scope, or `synced` from another
 * tab, takes what that storage holds whole and marks nothing for writing, so
 * one account's looks never land under another's keys and two tabs can't
 * ping-pong (M25).
 * PRD-261 A-09b: the looks also belong to one theme (`themeId`, the id the
 * server keeps them under), and `served` takes the server's copy: the server
 * is the truth, but never over a change the user made since the last read,
 * which the store sends to the server instead (lib/theme-host/builtin/play-look-store.ts).
 */

export interface LookState {
  readonly look: Look
  readonly saved: readonly Look[]
  /** False until storage has been read (after hydration, once the account is known). */
  readonly loaded: boolean
  /** The scope the looks were read for, and are written to; null until then. */
  readonly scope: string | null
  /** The theme they were read for (PRD-261 A-09b); undefined until then, or when the read named none. */
  readonly themeId?: string
  /** False once storage refused a read or a write: what the user does lasts only as long as this tab (M98). */
  readonly persisted: boolean
  readonly lookChanged: boolean
  readonly savedChanged: boolean
}

/** Which theme a read was for: another theme's looks are another set, as another scope's are. */
interface ThemeOfRead {
  readonly themeId?: string
}

export type LookAction =
  /** Storage read for a scope (a first read, or another account or workspace). */
  | ({ readonly type: 'loaded' } & StoredLooks & ThemeOfRead)
  /** Another tab changed this scope's looks: take them, write nothing back. */
  | ({ readonly type: 'synced' } & StoredLooks & ThemeOfRead)
  /** The server's looks for this scope and theme (GET /api/themes/looks/{themeId}): the truth, after the cache. */
  | ({ readonly type: 'served'; readonly scope: string; readonly active: Look | null; readonly saved: readonly Look[] } & ThemeOfRead)
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
function firstRead(state: LookState, read: StoredLooks & ThemeOfRead): LookState {
  return {
    ...state,
    look: state.lookChanged ? state.look : (read.active ?? DEFAULT_LOOK),
    saved: state.savedChanged ? state.saved : read.saved,
    loaded: true,
    scope: read.scope,
    themeId: read.themeId,
    persisted: read.persisted,
  }
}

/** Exactly what storage holds for the scope, nothing marked for writing. */
function replaced(state: LookState, read: StoredLooks & ThemeOfRead): LookState {
  return {
    ...state,
    look: read.active ?? DEFAULT_LOOK,
    saved: read.saved,
    loaded: true,
    scope: read.scope,
    themeId: read.themeId,
    persisted: read.persisted,
    lookChanged: false,
    savedChanged: false,
  }
}

function sameSet(state: LookState, read: ThemeOfRead & { readonly scope: string }): boolean {
  return read.scope === state.scope && read.themeId === state.themeId
}

function stored(state: LookState, action: Extract<LookAction, { type: 'loaded' | 'synced' }>): LookState {
  const { type, ...read } = action
  if (type === 'synced') return sameSet(state, read) ? replaced(state, read) : state
  if (!state.loaded) return firstRead(state, read)
  // The same scope read again (a workspace refresh) keeps what this tab holds, unsaved edits included.
  return sameSet(state, read) ? state : replaced(state, read)
}

function sameValue(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b)
}

/**
 * The server's copy, for the set this tab holds: each part the user hasn't changed since the read
 * takes the server's value (the same value keeps its object, so nothing repaints); a part the user
 * changed stays, and is what the store sends next. A server with no active look leaves the look.
 */
function served(state: LookState, action: Extract<LookAction, { type: 'served' }>): LookState {
  if (!state.loaded || !sameSet(state, action)) return state
  const serverLook = action.active ?? state.look
  const look = state.lookChanged || sameValue(serverLook, state.look) ? state.look : serverLook
  const saved = state.savedChanged || sameValue(action.saved, state.saved) ? state.saved : action.saved
  return look === state.look && saved === state.saved ? state : { ...state, look, saved }
}

export function lookReducer(state: LookState, action: LookAction): LookState {
  switch (action.type) {
    case 'loaded':
    case 'synced':
      return stored(state, action)
    case 'served':
      return served(state, action)
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

/** Two sets of looks, saved and active. */
export interface LookSet {
  readonly active: Look | null
  readonly saved: readonly Look[]
}

function addLocal(merged: readonly Look[], look: Look): readonly Look[] {
  if (merged.some((entry) => sameValue(entry, look))) return merged
  // The same id for another look (each browser numbered its own): it keeps its values under a new id.
  const taken = merged.some((entry) => entry.id === look.id)
  return [...merged, taken ? { ...look, id: freshLookId(merged) } : look]
}

/**
 * PRD-261 A-09b — the first time this browser meets the server's copy (or the server holds none),
 * the looks only this browser kept join it, so moving to the server loses none: the server's
 * saved looks first, then each local one the server lacks; the server's active look, else this
 * browser's. After that the server's copy is the truth (a delete elsewhere stays deleted).
 */
export function mergeLooks(server: LookSet, local: LookSet): LookSet {
  return { active: server.active ?? local.active, saved: local.saved.reduce(addLocal, server.saved) }
}

export function sameLookSet(a: LookSet, b: LookSet): boolean {
  return sameValue(a.active, b.active) && sameValue(a.saved, b.saved)
}

/** Sends a store's saves to the server (createLookOutbox). */
export interface LookOutbox {
  /** Queues a save of the set under `key` (its theme and scope); `send` runs once the saves before it have their answers. */
  queue(key: string, send: () => Promise<unknown>): void
}

/**
 * PRD-261 A-09b — one save at a time, in the order they were queued: two in flight can be handled
 * in either order (the backend runs each in its own thread, and a PUT replaces the whole set), so
 * an older one landing last would put back what a newer one replaced, with the newer one's answer
 * already taken as kept. A save still waiting when a newer one is queued under the same key is
 * dropped: the newer one carries the whole set.
 */
export function createLookOutbox(): LookOutbox {
  let tail: Promise<unknown> = Promise.resolve()
  let queued = 0
  /** Saves queued and not yet answered: at 0 a new save has nothing before it, so it starts at once. */
  let unanswered = 0
  const newest = new Map<string, number>()
  return {
    queue(key, send) {
      queued += 1
      const ticket = queued
      newest.set(key, ticket)
      const run = () => (newest.get(key) === ticket ? send() : undefined)
      // Started now when idle (a newer save queued in the same tick then waits behind it, never
      // drops it); otherwise once the saves before it have their answers. A throwing send rejects.
      const started = unanswered === 0 ? new Promise<unknown>((resolve) => resolve(run())) : tail.then(run)
      unanswered += 1
      tail = started
        .catch((error: unknown) => console.warn("[looks] a save's answer couldn't be handled:", error))
        .finally(() => {
          unanswered -= 1
        })
    },
  }
}
