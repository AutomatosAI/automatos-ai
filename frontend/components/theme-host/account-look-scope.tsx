'use client'

import { hashQueryKey, useQueryClient } from '@tanstack/react-query'
import { useEffect, type ReactNode } from 'react'

import { LookScopeContext } from '@/components/looks/look-scope'
import { useWorkspaceOptional } from '@/components/workspace-provider'
import { BACKEND_HEALTH_QUERY_KEY } from '@/hooks/use-backend-health'
import { useUser } from '@/lib/auth-hooks'
import { lookScope } from '@/lib/looks/scope'

/**
 * PRD-260 F5 — the scope of the signed-in user in the active workspace, for
 * the LookProvider and the shelf below it (useLookScope). Null while Clerk or
 * the workspace is still loading, and while signed out: reading under a
 * provisional scope would show the wrong looks, and the looks this browser
 * kept before F5 would move to it instead of to the account. A workspace that
 * failed to load (the backend asleep) waits too, so a user's looks never fall
 * back to a scope without their workspace; it is asked for again once the
 * backend answers (useWorkspaceRetry), and until then the LookProvider paints
 * the pre-paint script's look under Play. The local edition is signed in with
 * no user id, so its one user gets lib/looks/scope.ts's NO_USER.
 * PRD-261 A-14d: host code (it reads the account, the workspace and the
 * query cache), so it lives with the host's look runtime
 * (components/theme-host/theme-look-runtime.tsx), out of Play's folders; Play
 * reads the scope it provides through components/looks/look-scope.ts.
 */

type WorkspaceState = ReturnType<typeof useWorkspaceOptional>

const HEALTH_QUERY_HASH = hashQueryKey(BACKEND_HEALTH_QUERY_KEY)

/** GET /api/workspaces/current failed, and the WorkspaceProvider isn't asking again. */
function workspaceFailed(workspaceCtx: WorkspaceState): boolean {
  return workspaceCtx !== null && !workspaceCtx.isLoading && workspaceCtx.workspace === null && workspaceCtx.error !== null
}

/**
 * The WorkspaceProvider asks for the workspace on sign-in, an organisation
 * change or a route change only, so one that failed while the backend was
 * asleep would stay missing, and the looks unread, for the whole session.
 * While it is missing, the next answer to GET /health (useBackendHealth's
 * query: Play's frame polls it and the asleep screen rechecks it) asks for it
 * again, once per failure; a failed retry waits for the next answer.
 */
function useWorkspaceRetry(failed: boolean, refreshWorkspace: (() => Promise<void>) | undefined): void {
  const queryClient = useQueryClient()
  useEffect(() => {
    if (!failed || refreshWorkspace === undefined) return undefined
    const unsubscribe = queryClient.getQueryCache().subscribe((event) => {
      if (event.type !== 'updated' || event.action.type !== 'success') return
      if (event.query.queryHash !== HEALTH_QUERY_HASH) return
      unsubscribe()
      refreshWorkspace().catch((error: unknown) => {
        console.warn("[looks] couldn't ask for the workspace again after the backend answered:", error)
      })
    })
    return unsubscribe
  }, [queryClient, failed, refreshWorkspace])
}

function useAccountScope(enabled: boolean): string | null {
  const workspaceCtx = useWorkspaceOptional()
  const { user, isLoaded, isSignedIn } = useUser()
  const signedIn = enabled && isLoaded && isSignedIn === true
  useWorkspaceRetry(signedIn && workspaceFailed(workspaceCtx), workspaceCtx?.refreshWorkspace)
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
