'use client'

/**
 * PRD-277 US-501, US-502 — EnterpriseSessionProvider's state: ask the API who the session's person
 * is, choose the workspace this tab works in, and hear when the session ends.
 *
 *   - On the sign-in page nothing is asked: it must render with no session.
 *   - `GET /api/v1/auth/me` answers → signed in. The workspace is the stored one when the person
 *     holds it, else their first; an install admin in none opens the install's default one. It is
 *     stored before the app renders, so the app's first requests name it. The kept chat draft is
 *     checked against the person (kept-chat-draft.ts `setDraftOwner`).
 *   - A 401 there → the browser goes to the sign-in page, bringing the person back here afterwards;
 *     on a public page (lib/public-routes.ts, the middleware's own list) it stays where it is.
 *   - Any later 401 from the app's own API (apiClient and the session fetch report every one,
 *     lib/api-unauthorized.ts) → `ended`: the provider shows "your session ended" over the page,
 *     which stays mounted, with the API's reason when it gave one.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import { onUnauthorized } from '@/lib/api-unauthorized'
import { isPublicPath } from '@/lib/public-routes'

import { currentPath, fetchWhoAmI, type WhoAmI } from './auth-api'
import { setDraftOwner } from './kept-chat-draft'
import { sessionEndedReason } from './messages'
import { isInstallAdmin } from './session-context'

export const SIGN_IN_PAGE = '/sign-in'
const WORKSPACE_KEY = 'last_active_workspace'
const UNAUTHORIZED_PREFIX = 'HTTP 401:'

export type SessionLoad = 'loading' | 'signed-in' | 'signed-out' | 'failed'
/** What a page asks of the session: none (sign-in), optional (public) or a signed-in person. */
export type PageGate = 'sign-in' | 'public' | 'protected'

export function isSignInPage(path: string | null): boolean {
  return path === SIGN_IN_PAGE || (path ?? '').startsWith(`${SIGN_IN_PAGE}/`)
}

export function pageGateOf(path: string | null): PageGate {
  if (isSignInPage(path)) return 'sign-in'
  return isPublicPath(path) ? 'public' : 'protected'
}

/** The web app's sign-in page, bringing the person back to `returnTo` (the API's rule decides if it may). */
export function signInPageFor(returnTo: string): string {
  return `${SIGN_IN_PAGE}?return_to=${encodeURIComponent(returnTo)}`
}

/** The workspace this tab works in, from the API's answer and the stored one; null when the person has none. */
export function chooseWorkspace(who: WhoAmI, stored: string | null): string | null {
  if (stored && who.workspaces.some((workspace) => workspace.id === stored)) return stored
  if (who.workspaces.length > 0) return who.workspaces[0].id
  if (isInstallAdmin(who.person.system_role)) return stored || who.default_workspace_id
  return null
}

function storedWorkspace(): string | null {
  try {
    return window.localStorage.getItem(WORKSPACE_KEY)
  } catch (error) {
    console.warn('[session] the stored workspace could not be read', error)
    return null
  }
}

function storeWorkspace(id: string): void {
  try {
    window.localStorage.setItem(WORKSPACE_KEY, id)
  } catch (error) {
    console.warn('[session] the workspace could not be stored', error)
  }
}

function isUnauthorizedError(error: unknown): boolean {
  if (typeof error === 'object' && error !== null && (error as { status?: unknown }).status === 401) return true
  return error instanceof Error && error.message.startsWith(UNAUTHORIZED_PREFIX)
}

export interface SessionState {
  readonly load: SessionLoad
  readonly who: WhoAmI | null
  readonly workspaceId: string | null
  readonly ended: boolean
  /** The `reason` of the 401 that ended the session, when the API gave one the app has words for. */
  readonly endedReason: string | null
  readonly onSignInPage: boolean
  readonly onPublicPage: boolean
  readonly reload: () => void
}

/** Listen for any 401 from the app's own API while mounted. */
function useEndedSession(): { ended: boolean; endedReason: string | null } {
  const [ended, setEnded] = useState({ ended: false, endedReason: null as string | null })
  useEffect(() => onUnauthorized((body) => {
    setEnded((before) => ({ ended: true, endedReason: before.endedReason ?? sessionEndedReason(body) }))
  }), [])
  return ended
}

/**
 * Ask who-am-I on arrival off the sign-in page, and again only when a page that needs a person is
 * opened by someone the API did not know: never on each navigation inside the app.
 */
function useAskOnArrival(gate: PageGate, load: SessionLoad, ask: () => Promise<void>): void {
  const loadRef = useRef(load)
  loadRef.current = load
  const askedRef = useRef(false)
  useEffect(() => {
    if (gate === 'sign-in') {
      askedRef.current = false
      return
    }
    if (!askedRef.current || (gate === 'protected' && loadRef.current === 'signed-out')) {
      askedRef.current = true
      void ask()
    }
    // Keyed on what the page needs only; `ask` is stable for a stable navigate.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gate])
}

export function useSessionState(
  pathname: string | null,
  navigate: (url: string) => void = (url) => window.location.assign(url),
): SessionState {
  const [load, setLoad] = useState<SessionLoad>('loading')
  const [who, setWho] = useState<WhoAmI | null>(null)
  const [workspaceId, setWorkspaceId] = useState<string | null>(null)
  const gate = pageGateOf(pathname)
  const gateRef = useRef(gate)
  gateRef.current = gate

  const reload = useCallback(async () => {
    setLoad('loading')
    try {
      const answer = await fetchWhoAmI()
      const chosen = chooseWorkspace(answer, storedWorkspace())
      if (chosen) storeWorkspace(chosen)
      setDraftOwner(String(answer.person.id))
      setWho(answer)
      setWorkspaceId(chosen)
      setLoad('signed-in')
    } catch (error) {
      if (!isUnauthorizedError(error)) {
        console.error('[session] who-am-I failed', error)
        setLoad('failed')
        return
      }
      setLoad('signed-out')
      // A public page is served signed out: only a page that needs a person goes to sign in.
      if (gateRef.current === 'protected') navigate(signInPageFor(currentPath()))
    }
  }, [navigate])

  useAskOnArrival(gate, load, reload)
  const { ended, endedReason } = useEndedSession()
  return {
    load, who, workspaceId, ended, endedReason,
    onSignInPage: gate === 'sign-in', onPublicPage: gate === 'public',
    reload: () => void reload(),
  }
}
