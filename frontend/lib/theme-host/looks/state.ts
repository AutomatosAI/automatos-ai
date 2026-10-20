import { freshLookId, isPresetId, type ThemeLook, type ThemeLookRules } from './spec'
import type { StoredLooks } from './storage'

/**
 * PRD-261 A-19d1 (plan §2.2.4) — what the generic look runtime holds, as a pure reducer.
 * `lookChanged` / `savedChanged` say whether the user changed that part since it was read: only
 * then is it written back, so a read never rewrites what's stored. The looks belong to one scope,
 * the signed-in user and workspace (PRD-260 F5, lib/look-kit/scope.ts), and to one theme (`themeId`,
 * the id the server keeps them under): `loaded` for another scope or theme, or `synced` from another
 * tab, takes what that storage holds whole and marks nothing for writing, so one account's or one
 * theme's looks never land under another's keys and two tabs can't ping-pong (M25). `served` takes
 * the server's copy: the truth, but never over a change the user made since the read, which the
 * store sends to the server instead (./server.ts).
 *
 * The theme's rules ride the read (`rules`): its default look, for a scope with no active look and
 * after the active look is deleted, and its presets, which are never saved over.
 */

/** What the reducer needs of a theme's rules. */
export type LookStateRules = Pick<ThemeLookRules, 'defaultLook' | 'presets'>

export interface LookState {
  /** The active look; null only before any theme's looks were read, when none was given. */
  readonly look: ThemeLook | null
  readonly saved: readonly ThemeLook[]
  /** False until storage has been read (after hydration, once the account is known). */
  readonly loaded: boolean
  /** The scope the looks were read for, and are written to; null until then. */
  readonly scope: string | null
  /** The theme they were read for; null until then. */
  readonly themeId: string | null
  /** That theme's rules; null until then, when none was given. */
  readonly rules: LookStateRules | null
  /** False once storage refused a read or a write: what the user does lasts only as long as this tab (M98). */
  readonly persisted: boolean
  readonly lookChanged: boolean
  readonly savedChanged: boolean
}

/** Which theme a read was for, and its rules: another theme's looks are another set, as another scope's are. */
interface ThemeOfRead {
  readonly themeId: string
  readonly rules: LookStateRules
}

export type LookAction =
  /** Storage read for a scope and theme (a first read, or another account, workspace or theme). */
  | ({ readonly type: 'loaded' } & StoredLooks & ThemeOfRead)
  /** Another tab changed this scope's looks: take them, write nothing back. */
  | ({ readonly type: 'synced' } & StoredLooks & ThemeOfRead)
  /** The server's looks for this scope and theme (GET /api/themes/looks/{themeId}): the truth, after the cache. */
  | {
    readonly type: 'served'
    readonly themeId: string
    readonly scope: string
    readonly active: ThemeLook | null
    readonly saved: readonly ThemeLook[]
  }
  /** No theme with looks runs any more: forget what was read, so the next read is a first read. */
  | { readonly type: 'reset' }
  /** What a write to storage returned. */
  | { readonly type: 'written'; readonly ok: boolean }
  | { readonly type: 'set'; readonly look: ThemeLook }
  /** Adds a saved look, or replaces the saved look with the same id. Never a preset. */
  | { readonly type: 'save'; readonly look: ThemeLook }
  | { readonly type: 'delete'; readonly id: string }
  | { readonly type: 'rename'; readonly id: string; readonly name: string }

/** Nothing read yet: the theme's default look when its rules are given, nothing saved, nothing to write. */
export function initialLookState(rules: LookStateRules | null = null): LookState {
  return {
    look: rules === null ? null : rules.defaultLook,
    saved: [],
    loaded: false,
    scope: null,
    themeId: null,
    rules,
    persisted: true,
    lookChanged: false,
    savedChanged: false,
  }
}

function upsert(saved: readonly ThemeLook[], look: ThemeLook): readonly ThemeLook[] {
  return saved.some((entry) => entry.id === look.id)
    ? saved.map((entry) => (entry.id === look.id ? look : entry))
    : [...saved, look]
}

function deleteSaved(state: LookState, id: string): LookState {
  if (!state.saved.some((entry) => entry.id === id)) return state
  const saved = state.saved.filter((entry) => entry.id !== id)
  if (state.look?.id !== id) return { ...state, saved, savedChanged: true }
  // The active look was the one deleted: the theme goes back to its default.
  return { ...state, saved, look: state.rules?.defaultLook ?? null, savedChanged: true, lookChanged: true }
}

function renameSaved(state: LookState, id: string, name: string): LookState {
  if (!state.saved.some((entry) => entry.id === id)) return state
  const saved = state.saved.map((entry) => (entry.id === id ? { ...entry, name } : entry))
  if (state.look === null || state.look.id !== id) return { ...state, saved, savedChanged: true }
  return { ...state, saved, look: { ...state.look, name }, savedChanged: true, lookChanged: true }
}

type Read = StoredLooks & ThemeOfRead

/** The first read. A change made before it (unlikely: it runs on mount) wins over what's stored, and is written. */
function firstRead(state: LookState, read: Read): LookState {
  return {
    ...state,
    look: state.lookChanged ? state.look : (read.active ?? read.rules.defaultLook),
    saved: state.savedChanged ? state.saved : read.saved,
    loaded: true,
    scope: read.scope,
    themeId: read.themeId,
    rules: read.rules,
    persisted: read.persisted,
  }
}

/** Exactly what storage holds for the scope and theme, nothing marked for writing. */
function replaced(state: LookState, read: Read): LookState {
  return {
    ...state,
    look: read.active ?? read.rules.defaultLook,
    saved: read.saved,
    loaded: true,
    scope: read.scope,
    themeId: read.themeId,
    rules: read.rules,
    persisted: read.persisted,
    lookChanged: false,
    savedChanged: false,
  }
}

function sameSet(state: LookState, read: { readonly themeId: string; readonly scope: string }): boolean {
  return read.scope === state.scope && read.themeId === state.themeId
}

function stored(state: LookState, action: Extract<LookAction, { type: 'loaded' | 'synced' }>): LookState {
  const { type, ...read } = action
  if (type === 'synced') return state.loaded && sameSet(state, read) ? replaced(state, read) : state
  if (!state.loaded) return firstRead(state, read)
  // The same scope and theme read again (a workspace refresh) keeps what this tab holds, unsaved edits included.
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

function saveLook(state: LookState, look: ThemeLook): LookState {
  if (state.rules !== null && isPresetId(state.rules, look.id)) return state
  return { ...state, saved: upsert(state.saved, look), savedChanged: true }
}

export function lookReducer(state: LookState, action: LookAction): LookState {
  switch (action.type) {
    case 'loaded':
    case 'synced':
      return stored(state, action)
    case 'served':
      return served(state, action)
    case 'reset':
      return state.loaded ? initialLookState(state.rules) : state
    case 'written':
      return action.ok || !state.persisted ? state : { ...state, persisted: false }
    case 'set':
      return { ...state, look: action.look, lookChanged: true }
    case 'save':
      return saveLook(state, action.look)
    case 'delete':
      return deleteSaved(state, action.id)
    case 'rename':
      return renameSaved(state, action.id, action.name)
  }
}

/** Two sets of looks, saved and active. */
export interface LookSet {
  readonly active: ThemeLook | null
  readonly saved: readonly ThemeLook[]
}

/**
 * The first time this browser meets the server's copy (or the server holds none), the looks only
 * this browser kept join it, so moving to the server loses none: the server's saved looks first,
 * then each local one the server lacks (one whose id the server uses for another look keeps its
 * values under a fresh id); the server's active look, else this browser's. After that the server's
 * copy is the truth (a delete elsewhere stays deleted).
 */
export function mergeLooks(server: LookSet, local: LookSet, rules: Pick<ThemeLookRules, 'presets'>): LookSet {
  const saved = local.saved.reduce<readonly ThemeLook[]>((merged, look) => {
    if (merged.some((entry) => sameValue(entry, look))) return merged
    const taken = merged.some((entry) => entry.id === look.id)
    return [...merged, taken ? { ...look, id: freshLookId(rules, merged) } : look]
  }, server.saved)
  return { active: server.active ?? local.active, saved }
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
 * One save at a time, in the order they were queued: two in flight can be handled in either order
 * (the backend runs each in its own thread, and a PUT replaces the whole set), so an older one
 * landing last would put back what a newer one replaced, with the newer one's answer already taken
 * as kept. A save still waiting when a newer one is queued under the same key is dropped: the newer
 * one carries the whole set.
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
