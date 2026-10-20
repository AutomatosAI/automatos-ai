import { type Dispatch, type MutableRefObject, useEffect, useMemo, useRef, useState } from 'react'

import { apiClient } from '@/lib/api-client'
import { SCOPE_SEPARATOR } from '@/lib/look-kit/scope'

import type { ThemeLookRules } from './spec'
import { createLookOutbox, type LookAction, type LookOutbox, type LookSet, type LookState, mergeLooks, sameLookSet } from './state'
import {
  activeLookOf,
  hasLookMark,
  readStoredLooks,
  savedLooksOf,
  setLookMark,
  writeActiveLook,
  writeSavedLooks,
  type LookTheme,
  type StorageSource,
} from './storage'

/**
 * PRD-261 A-19d1 (plan §2.2.4) — the server's copy of one theme's looks: GET/PUT
 * /api/themes/looks/{themeId} (orchestrator/api/themes_looks.py), through apiClient's named theme
 * methods so the route-contract check sees them. The server is the truth; this browser's copy
 * (./storage.ts) is read first so the page paints at once, and is the cache when the server can't
 * be reached. A change made here is marked pending at once and sent once the changes pause; one the
 * server never got is sent on the next load instead of reading, and one it refused is not sent
 * again (the look screens say it isn't kept, M98). The first time a browser meets the server's
 * copy, the looks only it kept join the server's (./state.ts mergeLooks).
 */

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

/** How long after the last change a save is sent: one drag of a slider is many changes. */
export const LOOK_SEND_DELAY_MS = 600

/**
 * Answers that mean the server will never keep this save as sent: the user is told, and it isn't sent
 * again. 403 is one: no person behind the caller, or no members:read (themes_looks.py); the looks
 * routes ask for no step-up, so a 403 here is never a reverification to retry. 401 (signed out)
 * carries no status from apiClient and stays pending, sent on the next load.
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
export function servedLooksOf(rules: ThemeLookRules, answer: unknown): (LookSet & { readonly empty: boolean }) | null {
  if (typeof answer !== 'object' || answer === null) return null
  const { looks, active, dropped } = answer as Record<string, unknown>
  if (!Array.isArray(looks)) return null
  if (typeof dropped === 'number' && dropped > 0) {
    console.warn(`[looks] the server left out ${dropped} look(s) this theme's version no longer accepts`)
  }
  const none = active === null || active === undefined
  return {
    active: none ? null : activeLookOf(rules, active, 'the server'),
    saved: savedLooksOf(rules, looks, 'the server'),
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
 * load. Refused: the user is told it isn't kept and it isn't sent again.
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
  const served = servedLooksOf(theme.rules, answer)
  if (served === null) {
    console.warn("[looks] the server's answer isn't a set of looks, so this browser's copy stands:", answer)
    return
  }
  const changedHere = hasLookMark(storage, scope, theme.keys, 'pending')
  const local = readStoredLooks(storage, scope, theme)
  const firstMeeting = served.empty || !hasLookMark(storage, scope, theme.keys, 'synced')
  const truth = firstMeeting && !changedHere ? mergeLooks(served, local, theme.rules) : served
  if (!changedHere) {
    const kept = (truth.active === null || writeActiveLook(storage, scope, theme.keys, truth.active))
      && writeSavedLooks(storage, scope, theme.keys, truth.saved)
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
 * true. A mark over an empty copy (storage cleared under it) sends nothing, so it never empties the
 * server's looks: the mark goes and the server's copy is read instead.
 */
function sendPending(ctx: SyncContext): boolean {
  const { storage, theme, scope } = ctx
  if (!hasLookMark(storage, scope, theme.keys, 'pending')) return false
  const local = readStoredLooks(storage, scope, theme)
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

export interface ServerSync {
  readonly storage: StorageSource
  readonly theme: LookTheme | null
  readonly server: LookServer | null
  /** The state holds this theme's looks, read here. */
  readonly enabled: boolean
}

/** Keeps the looks in step with the server: one context per scope and theme, so each is read once. */
export function useLookServerSync(
  { storage, theme, server, enabled }: ServerSync,
  state: LookState,
  dispatch: Dispatch<LookAction>,
): void {
  const changes = useRef(0)
  const [outbox] = useState(createLookOutbox)
  const { scope } = state
  const ctx = useMemo<SyncContext | null>(
    () => (enabled && theme !== null && server !== null && scope !== null && scopeHasWorkspace(scope)
      ? { storage, theme, scope, server, dispatch, changes, outbox }
      : null),
    [enabled, theme, server, scope, storage, dispatch, outbox],
  )
  useServerRead(ctx)
  useServerWrite(ctx, state)
}
