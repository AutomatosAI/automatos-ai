'use client'

import type { ReactNode } from 'react'

import { useWorkspaceOptional } from '@/components/workspace-provider'
import { useUser } from '@/lib/auth-hooks'
import { lookScope } from '@/lib/look-kit/scope'

import { LookScopeContext } from './look-scope'

/**
 * PRD-260 F5 — the scope of the signed-in user in the active workspace, for
 * the LookProvider and the shelf below it (useLookScope). Null while Clerk or
 * the workspace is still loading, and while signed out: reading under a
 * provisional scope would show the wrong looks, and the looks this browser
 * kept before F5 would move to it instead of to the account. A workspace that
 * failed to load (the backend asleep) waits too, so a user's looks never fall
 * back to a scope without their workspace; the host asks for it again once the
 * backend answers (./use-workspace-retry.ts, which ThemeLookRuntime calls), and
 * until then the look runtime paints the boot script's look. The local edition
 * is signed in with no user id, so its one user gets lib/look-kit/scope.ts's NO_USER.
 * PRD-261 A-14d: host code (it reads the account and the workspace), so it
 * lives with the host's look runtime (components/theme-host/theme-look-runtime.tsx),
 * out of Play's folders; the context it fills is the host's too (./look-scope.ts,
 * W-A5 seam review). PRD-261 A-19d2 (PB-8a, the W-E3 seam review): the workspace
 * retry left this file for the host's, so it runs for every installed theme the
 * host runs, with looks or without, and only once.
 */

function useAccountScope(enabled: boolean): string | null {
  const workspaceCtx = useWorkspaceOptional()
  const { user, isLoaded, isSignedIn } = useUser()
  const signedIn = enabled && isLoaded && isSignedIn === true
  if (!signedIn) return null
  if (workspaceCtx !== null && (workspaceCtx.isLoading || workspaceCtx.workspace === null)) return null
  return lookScope(workspaceCtx?.workspace?.id, user?.id)
}

export interface AccountLookScopeProps {
  readonly children: ReactNode
  /**
   * PRD-261 A-13c: false while the active theme has no look runtime (Classic, Studio): no scope, so
   * nothing below reads looks. The host keeps this mounted either way, so a style switched in place
   * never changes the tree above the page (M69); it only flips this.
   */
  readonly enabled?: boolean
}

export function AccountLookScope({ children, enabled = true }: AccountLookScopeProps) {
  const scope = useAccountScope(enabled)
  return <LookScopeContext.Provider value={scope}>{children}</LookScopeContext.Provider>
}
