'use client'

/**
 * PRD-277 US-501, US-502 — EnterpriseSessionProvider's state: ask the API who the session's person
 * is, choose the workspace this tab works in, and hear when the session ends.
 *
 *   - On the sign-in page nothing is asked: it must render with no session.
 *   - `GET /api/v1/auth/me` answers → signed in. The workspace is the stored one when the person
 *     holds it, else their first; an install admin in none opens the install's default one. It is
 *     stored before the app renders, so the app's first requests name it.
 *   - A 401 there → the browser goes to the sign-in page, bringing the person back here afterwards.
 *   - Any later 401 `session_ended` (apiClient reports every 401, lib/api-unauthorized.ts) →
 *     `ended`: the provider shows "your session ended" over the page, which stays mounted.
 */
import { useCallback, useEffect, useState } from 'react'

import { onUnauthorized } from '@/lib/api-unauthorized'

import { currentPath, fetchWhoAmI, type WhoAmI } from './auth-api'
import { isSessionEnded } from './messages'
import { isInstallAdmin } from './session-context'

export const SIGN_IN_PAGE = '/sign-in'
const WORKSPACE_KEY = 'last_active_workspace'
const UNAUTHORIZED_PREFIX = 'HTTP 401:'

export type SessionLoad = 'loading' | 'signed-in' | 'signed-out' | 'failed'

export function isSignInPage(path: string | null): boolean {
  return path === SIGN_IN_PAGE || (path ?? '').startsWith(`${SIGN_IN_PAGE}/`)
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
  readonly onSignInPage: boolean
  readonly reload: () => void
}

export function useSessionState(
  pathname: string | null,
  navigate: (url: string) => void = (url) => window.location.assign(url),
): SessionState {
  const [load, setLoad] = useState<SessionLoad>('loading')
  const [who, setWho] = useState<WhoAmI | null>(null)
  const [workspaceId, setWorkspaceId] = useState<string | null>(null)
  const [ended, setEnded] = useState(false)
  const onSignInPage = isSignInPage(pathname)

  const reload = useCallback(async () => {
    setLoad('loading')
    try {
      const answer = await fetchWhoAmI()
      const chosen = chooseWorkspace(answer, storedWorkspace())
      if (chosen) storeWorkspace(chosen)
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
      navigate(signInPageFor(currentPath()))
    }
  }, [navigate])

  useEffect(() => {
    if (!onSignInPage) void reload()
    // Asked once per arrival off the sign-in page, not on every navigation inside the app.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onSignInPage])

  useEffect(() => onUnauthorized((body) => {
    if (isSessionEnded(body)) setEnded(true)
  }), [])

  return { load, who, workspaceId, ended, onSignInPage, reload: () => void reload() }
}
