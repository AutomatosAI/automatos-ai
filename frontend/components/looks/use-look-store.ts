import { type Dispatch, type MutableRefObject, useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'

import { useUiStyleOptional } from '@/contexts/ui-style-context'
import { apiClient } from '@/lib/api-client'
import { isPresetId } from '@/lib/looks/presets'
import { SCOPE_SEPARATOR } from '@/lib/looks/scope'
import {
  activeLookOf,
  hasLookMark,
  loadStoredLooks,
  lookKeys,
  readStoredLooks,
  savedLooksOf,
  setLookMark,
  writeActiveLook,
  writeSavedLooks,
  type LookTheme,
  type StorageSource,
} from '@/lib/looks/storage'
import {
  createLookOutbox,
  INITIAL_LOOK_STATE,
  type LookAction,
  type LookOutbox,
  type LookSet,
  type LookState,
  lookReducer,
  mergeLooks,
  sameLookSet,
} from '@/lib/looks/store'
import type { Look, ValidationResult } from '@/lib/looks/types'
import { validateLook } from '@/lib/looks/validate'
import { BUILTIN_PLAY_REF } from '@/lib/theme-host/builtin/play-entry'
import { BUILTIN_PLAY_LOOK_THEME } from '@/lib/theme-host/builtin/play-look-keys'

/**
 * PRD-260, PRD-261 A-09b — the look state of one theme, for the signed-in user
 * and workspace (F5: `scope`, null until both are known). It is read from this
 * browser's copy after mount, so the page paints at once, then from the server
 * (GET /api/themes/looks/{themeId}), which is the truth; a change is written
 * to the copy at once and sent to the server (PUT, useLookServerSync). Other
 * tabs follow through the copy (M25). The theme gives the keys
 * (lib/looks/storage.ts): an installed theme's are `automatos-theme.<id>.*`;
 * with none given, the looks are builtin Play's under its PRD-260 keys, kept
 * in step with the server while Play is the style.
 */
export interface LookStore {
  /** The active look, possibly edited since it was picked. */
  readonly look: Look
  /** The user's own looks (presets are in lib/looks/presets.ts). */
  readonly saved: readonly Look[]
  /** False until storage has been read; the pre-paint script covers that first moment. */
  readonly loaded: boolean
  /** False once storage, or the server, refused a write: the Looks page says it isn't kept (M98). */
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

export interface LookStoreOptions {
  /** Whose looks: an installed theme's (installedLookTheme), always the active one. Default: builtin Play's. */
  readonly theme?: LookTheme
  /** Where the looks are kept besides this browser; null keeps them here only. Default: the API. */
  readonly server?: LookServer | null
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
function useStorageSync(
  storage: StorageSource,
  scope: string | null,
  theme: LookTheme,
  dispatch: Dispatch<LookAction>,
): void {
  const { themeId, keys: bases } = theme
  useEffect(() => {
    if (scope === null) return undefined
    const keys = lookKeys(scope, bases)
    const onStorage = (event: StorageEvent) => {
      if (event.key !== null && event.key !== keys.saved && event.key !== keys.active) return
      dispatch({ type: 'synced', themeId, ...readStoredLooks(storage, scope, bases) })
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [storage, scope, themeId, bases, dispatch])
}

function sameTheme(a: LookTheme, b: LookTheme): boolean {
  return a.themeId === b.themeId && JSON.stringify(a.keys) === JSON.stringify(b.keys)
}

/**
 * The theme the looks are for, and whether it is the style now (only then does the server hear of
 * them). The same theme keeps one object, so a caller that builds it on every render
 * (installedLookTheme(id)) never makes the store read it, or the server, again.
 */
function useLookTheme(given: LookTheme | undefined): { readonly theme: LookTheme; readonly active: boolean } {
  const activeRef = useUiStyleOptional()?.activeTheme?.ref
  const wanted = given ?? BUILTIN_PLAY_LOOK_THEME
  const kept = useRef(wanted)
  if (!sameTheme(kept.current, wanted)) kept.current = wanted
  return { theme: kept.current, active: given !== undefined || activeRef === BUILTIN_PLAY_REF }
}

interface CopyWrites {
  readonly storage: StorageSource
  readonly theme: LookTheme
  /** The scope and theme the state was read for: a write goes only there. */
  readonly scope: string | null
  readonly readFor: string | undefined
}

/** Each part the user changed is written to this browser's copy, under the keys it was read with. */
function useCopyWrites(
  { storage, theme, scope, readFor }: CopyWrites,
  state: LookState,
  dispatch: Dispatch<LookAction>,
): void {
  const { look, saved, lookChanged, savedChanged } = state
  const { themeId, keys } = theme
  const here = readFor === themeId

  useEffect(() => {
    if (!lookChanged || !here || scope === null) return
    dispatch({ type: 'written', ok: writeActiveLook(storage, scope, look, keys) })
  }, [storage, scope, here, keys, look, lookChanged, dispatch])

  useEffect(() => {
    if (!savedChanged || !here || scope === null) return
    dispatch({ type: 'written', ok: writeSavedLooks(storage, scope, saved, keys) })
  }, [storage, scope, here, keys, saved, savedChanged, dispatch])
}

// ---------------------------------------------------------------------------
// The server: GET/PUT /api/themes/looks/{themeId} (orchestrator/api/themes_looks.py)
// ---------------------------------------------------------------------------

/** What a save sends: both keys, every time (a save replaces the whole list). */
export interface LookServerBody {
  readonly looks: Record<string, unknown>[]
  readonly active: Record<string, unknown> | null
}

export interface LookServer {
  read(themeId: string): Promise<unknown>
  write(themeId: string, body: LookServerBody): Promise<unknown>
}

/** The backend, through apiClient's named theme methods (so the route-contract check sees them). */
export const apiLookServer: LookServer = {
  read: (themeId) => apiClient.getThemeLooks(themeId),
  write: (themeId, body) => apiClient.putThemeLooks(themeId, body),
}

/** How long after the last change a save is sent: one drag of the glow is many changes. */
export const LOOK_SEND_DELAY_MS = 600

/**
 * Answers that mean the server will never keep this save as sent: the user is told, and it isn't sent again.
 * 403 is one: no person behind the caller, or no members:read (orchestrator/api/themes_looks.py); the looks
 * routes ask for no step-up, so a 403 here is never a reverification to retry. 401 (signed out) carries no
 * status from apiClient and stays pending, sent on the next load.
 */
const REFUSED_STATUSES: readonly number[] = [400, 403, 404, 409, 413, 422]

/** The server keeps looks per workspace, a uuid: a scope without one (a test, no workspace) has nowhere to send them. */
const WORKSPACE_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

export function scopeHasWorkspace(scope: string): boolean {
  const end = scope.indexOf(SCOPE_SEPARATOR)
  return end > 0 && WORKSPACE_ID_PATTERN.test(scope.slice(0, end))
}

export function lookBody({ saved, active }: LookSet): LookServerBody {
  return { looks: saved.map((look) => ({ ...look })), active: active === null ? null : { ...active } }
}

/** The server's answer, every look checked as one from storage is; null when it isn't a looks answer. */
export function servedLooksOf(answer: unknown): (LookSet & { readonly empty: boolean }) | null {
  if (typeof answer !== 'object' || answer === null) return null
  const { looks, active, dropped } = answer as Record<string, unknown>
  if (!Array.isArray(looks)) return null
  if (typeof dropped === 'number' && dropped > 0) {
    console.warn(`[looks] the server left out ${dropped} look(s) this theme's version no longer accepts`)
  }
  const none = active === null || active === undefined
  return {
    active: none ? null : activeLookOf(active, 'the server'),
    saved: savedLooksOf(looks, 'the server'),
    empty: looks.length === 0 && none,
  }
}

export function isRefusal(error: unknown): boolean {
  const status = (error as { status?: unknown } | null)?.status
  return typeof status === 'number' && REFUSED_STATUSES.includes(status)
}

interface SyncContext {
  readonly storage: StorageSource
  readonly theme: LookTheme
  readonly scope: string
  readonly server: LookServer
  readonly dispatch: Dispatch<LookAction>
  /** Counts the changes made here: a save clears the pending mark only when none came after it. */
  readonly changes: MutableRefObject<number>
  /** The store's saves, one at a time: an older one never lands after a newer one. */
  readonly outbox: LookOutbox
}

/**
 * Sends the set. Kept: the pending mark goes, unless a change came after it was queued. Not reached
 * (offline, asleep, signed out): the change stays in this browser, marked, and is sent on the next
 * load. Refused: the user is told it isn't kept (the Looks page's notice, M98) and it isn't sent again.
 */
function writeLooks(ctx: SyncContext, set: LookSet, sentAfter: number): Promise<void> {
  const { storage, theme, scope, server, dispatch, changes } = ctx
  return server.write(theme.themeId, lookBody(set)).then(
    () => {
      if (changes.current === sentAfter) setLookMark(storage, scope, theme.keys, 'pending', false)
    },
    (error: unknown) => {
      console.warn("[looks] the server didn't keep these Looks; this browser still has them:", error)
      if (!isRefusal(error)) return
      setLookMark(storage, scope, theme.keys, 'pending', false)
      dispatch({ type: 'written', ok: false })
    },
  )
}

/** Queues the set: it is sent once the saves before it have their answers, so none lands out of order. */
function sendLooks(ctx: SyncContext, set: LookSet): void {
  const sentAfter = ctx.changes.current
  ctx.outbox.queue(JSON.stringify([ctx.theme.themeId, ctx.scope]), () => writeLooks(ctx, set, sentAfter))
}

/** The server's copy into this browser's and the state; on a first meeting, this browser's own looks join it first. */
function takeServed(ctx: SyncContext, answer: unknown): void {
  const { storage, theme, scope, dispatch } = ctx
  const served = servedLooksOf(answer)
  if (served === null) {
    console.warn("[looks] the server's answer isn't a set of looks, so this browser's copy stands:", answer)
    return
  }
  const changedHere = hasLookMark(storage, scope, theme.keys, 'pending')
  const local = readStoredLooks(storage, scope, theme.keys)
  const firstMeeting = served.empty || !hasLookMark(storage, scope, theme.keys, 'synced')
  const truth = firstMeeting && !changedHere ? mergeLooks(served, local) : served
  if (!changedHere) {
    const kept = (truth.active === null || writeActiveLook(storage, scope, truth.active, theme.keys))
      && writeSavedLooks(storage, scope, truth.saved, theme.keys)
    dispatch({ type: 'written', ok: kept })
  }
  setLookMark(storage, scope, theme.keys, 'synced', true)
  dispatch({ type: 'served', scope, themeId: theme.themeId, active: truth.active, saved: truth.saved })
  if (!changedHere && !sameLookSet(truth, served)) {
    setLookMark(storage, scope, theme.keys, 'pending', true)
    sendLooks(ctx, truth)
  }
}

/**
 * A change made here that the server hasn't kept yet (a save that didn't reach it): sent now, and
 * true. A mark over an empty copy (storage cleared under it) sends nothing, so it never empties
 * the server's looks: the mark goes and the server's copy is read instead.
 */
function sendPending(ctx: SyncContext): boolean {
  const { storage, theme, scope } = ctx
  if (!hasLookMark(storage, scope, theme.keys, 'pending')) return false
  const local = readStoredLooks(storage, scope, theme.keys)
  if (local.active === null && local.saved.length === 0) {
    setLookMark(storage, scope, theme.keys, 'pending', false)
    return false
  }
  sendLooks(ctx, { active: local.active, saved: local.saved })
  return true
}

/** Once per scope and theme, after this browser's copy is read: send what is pending, else read the server's. */
function useServerRead(ctx: SyncContext | null): void {
  useEffect(() => {
    if (ctx === null || sendPending(ctx)) return undefined
    const sync: SyncContext = ctx
    let live = true
    sync.server.read(sync.theme.themeId).then(
      (answer) => {
        if (live) takeServed(sync, answer)
      },
      (error: unknown) => {
        console.warn("[looks] couldn't read the Looks from the server; this browser's copy stands:", error)
      },
    )
    return () => {
      live = false
    }
  }, [ctx])
}

/** Every change made here is marked pending at once and sent once the changes pause. */
function useServerWrite(ctx: SyncContext | null, state: LookState): void {
  const { look, saved, lookChanged, savedChanged } = state
  useEffect(() => {
    if (ctx === null || !(lookChanged || savedChanged)) return undefined
    const sync: SyncContext = ctx
    sync.changes.current += 1
    setLookMark(sync.storage, sync.scope, sync.theme.keys, 'pending', true)
    const timer = setTimeout(() => sendLooks(sync, { active: look, saved }), LOOK_SEND_DELAY_MS)
    return () => clearTimeout(timer)
  }, [ctx, look, saved, lookChanged, savedChanged])
}

interface ServerSync {
  readonly storage: StorageSource
  readonly theme: LookTheme
  readonly server: LookServer | null
  /** The theme is the style, its looks are read here, and the scope names a workspace. */
  readonly enabled: boolean
}

/** Keeps the looks in step with the server: one context per scope and theme, so each is read once. */
function useLookServerSync(
  { storage, theme, server, enabled }: ServerSync,
  state: LookState,
  dispatch: Dispatch<LookAction>,
): void {
  const changes = useRef(0)
  const [outbox] = useState(createLookOutbox)
  const { scope } = state
  const ctx = useMemo<SyncContext | null>(
    () => (enabled && server !== null && scope !== null && scopeHasWorkspace(scope)
      ? { storage, theme, scope, server, dispatch, changes, outbox }
      : null),
    [enabled, server, scope, storage, theme, dispatch, outbox],
  )
  useServerRead(ctx)
  useServerWrite(ctx, state)
}

export function useLookStore(
  storage: StorageSource,
  accountScope: string | null,
  options: LookStoreOptions = {},
): LookStore {
  const [state, dispatch] = useReducer(lookReducer, INITIAL_LOOK_STATE)
  const { look, saved, loaded, persisted, scope } = state
  const { theme, active } = useLookTheme(options.theme)
  const { themeId, keys } = theme
  const server = options.server === undefined ? apiLookServer : options.server

  useEffect(() => {
    if (accountScope !== null) dispatch({ type: 'loaded', themeId, ...loadStoredLooks(storage, accountScope, keys) })
  }, [storage, accountScope, themeId, keys])
  useStorageSync(storage, scope, theme, dispatch)
  useCopyWrites({ storage, theme, scope, readFor: state.themeId }, state, dispatch)
  useLookServerSync({ storage, theme, server, enabled: active && loaded && state.themeId === themeId }, state, dispatch)

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
