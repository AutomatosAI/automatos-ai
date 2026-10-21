'use client'

import { hashQueryKey, useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'

import { useWorkspaceOptional } from '@/components/workspace-provider'
import { BACKEND_HEALTH_QUERY_KEY } from '@/hooks/use-backend-health'
import { useUser } from '@/lib/auth-hooks'

/**
 * PRD-261 A-19d2 (PB-8a, the W-E3 seam review) — the host asks for a workspace that failed to load
 * again once the backend answers. Everything keyed by the workspace waits for it: the looks' scope
 * (./account-look-scope.tsx) and the theme-state scope's workspace key (./theme-state-scope.tsx),
 * which every installed theme's `storage` area reads. The retry used to ride on AccountLookScope,
 * so it ran only while a theme with looks was the style; it is the host's now: ThemeLookRuntime
 * (./theme-look-runtime.tsx) calls it once, whenever the host runs an installed theme or a builtin
 * theme with looks, so there is one subscriber and never a double refetch.
 */

type WorkspaceState = ReturnType<typeof useWorkspaceOptional>

const HEALTH_QUERY_HASH = hashQueryKey(BACKEND_HEALTH_QUERY_KEY)

/** GET /api/workspaces/current failed, and the WorkspaceProvider isn't asking again. */
export function workspaceFailed(workspaceCtx: WorkspaceState): boolean {
  return workspaceCtx !== null && !workspaceCtx.isLoading && workspaceCtx.workspace === null && workspaceCtx.error !== null
}

/**
 * The WorkspaceProvider asks for the workspace on sign-in, an organisation
 * change or a route change only, so one that failed while the backend was
 * asleep would stay missing, and the looks unread, for the whole session.
 * While it is missing, the next answer to GET /health (useBackendHealth's
 * query: a theme's frame polls it and the asleep screen rechecks it) asks for it
 * again, once per failure; a failed retry waits for the next answer.
 */
export function useWorkspaceRetry(failed: boolean, refreshWorkspace: (() => Promise<void>) | undefined): void {
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

/** The host's one retry: while `enabled`, a signed-in user's workspace that failed is asked for again (useWorkspaceRetry). */
export function useHostWorkspaceRetry(enabled: boolean): void {
  const workspaceCtx = useWorkspaceOptional()
  const { isLoaded, isSignedIn } = useUser()
  const signedIn = enabled && isLoaded && isSignedIn === true
  useWorkspaceRetry(signedIn && workspaceFailed(workspaceCtx), workspaceCtx?.refreshWorkspace)
}
