'use client'

/**
 * PRD-277 US-501 — what the enterprise web app knows about the signed-in person, and only from the
 * API (`GET /api/v1/auth/me`). The browser never decides who someone is or what role they hold:
 * the context carries the API's answer, and every request is judged again by the API.
 *
 * EnterpriseSessionProvider (components/enterprise) fills it; lib/auth-hooks.ts and the role
 * context read it. Outside enterprise no provider is mounted and the context is null.
 */
import { createContext, useContext } from 'react'

import { signOutAtApi, type SystemRole, type WhoAmI } from './auth-api'

export type EnterpriseSessionStatus = 'loading' | 'signed-in' | 'signed-out'

export interface EnterpriseSession {
  readonly status: EnterpriseSessionStatus
  readonly who: WhoAmI | null
  /** The workspace this tab works in (also stored as `last_active_workspace`). */
  readonly workspaceId: string | null
  readonly signOut: () => Promise<void>
}

export const EnterpriseSessionContext = createContext<EnterpriseSession | null>(null)

/** The session, or null outside the enterprise provider. */
export function useEnterpriseSession(): EnterpriseSession | null {
  return useContext(EnterpriseSessionContext)
}

/** The install role the API stored for the person; `user` until it has answered. */
export function systemRoleOf(session: EnterpriseSession | null): SystemRole {
  return session?.who?.person.system_role ?? 'user'
}

export function isInstallAdmin(role: SystemRole): boolean {
  return role === 'admin' || role === 'super_admin'
}

/** The keys the web app keeps about which workspace a person last used. */
export const WORKSPACE_STORAGE_KEYS: readonly string[] = ['last_active_workspace', 'last_active_org']

function forgetWorkspace(): void {
  try {
    WORKSPACE_STORAGE_KEYS.forEach((key) => window.localStorage.removeItem(key))
  } catch (error) {
    console.warn('[session] the stored workspace could not be cleared', error)
  }
}

/**
 * Sign out: the API ends the session and clears its cookie, then the browser goes to the
 * provider's end-session address when the API gives one, else to the sign-in page. The promise
 * never settles: the page is leaving. A refusal (the session had already ended) still signs out here.
 */
export async function signOutOfEnterprise(navigate: (url: string) => void = (url) => window.location.assign(url)): Promise<void> {
  let redirectTo: string | null = null
  try {
    redirectTo = await signOutAtApi()
  } catch (error) {
    console.warn('[session] the API did not confirm the sign-out', error)
  }
  forgetWorkspace()
  navigate(redirectTo ?? '/sign-in')
  return new Promise<void>(() => {})
}
