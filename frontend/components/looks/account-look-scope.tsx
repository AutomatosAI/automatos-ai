'use client'

import type { ReactNode } from 'react'

import { useWorkspaceOptional } from '@/components/workspace-provider'
import { useUser } from '@/lib/auth-hooks'
import { lookScope } from '@/lib/looks/scope'

import { LookScopeContext } from './look-scope'

/**
 * PRD-260 F5 — the scope of the signed-in user in the active workspace, for
 * the LookProvider and the shelf below it (useLookScope). Null while Clerk or
 * the workspace is still loading, and while signed out: reading under a
 * provisional scope would show the wrong looks, and the looks this browser
 * kept before F5 would move to it instead of to the account. A workspace that
 * failed to load (the backend asleep) waits too, so a user's looks never fall
 * back to a scope without their workspace. The local edition is signed in
 * with no user id, so its one user gets lib/looks/scope.ts's NO_USER.
 */
function useAccountScope(): string | null {
  const workspaceCtx = useWorkspaceOptional()
  const { user, isLoaded, isSignedIn } = useUser()
  if (!isLoaded || !isSignedIn) return null
  if (workspaceCtx !== null && (workspaceCtx.isLoading || workspaceCtx.workspace === null)) return null
  return lookScope(workspaceCtx?.workspace?.id, user?.id)
}

export function AccountLookScope({ children }: { readonly children: ReactNode }) {
  const scope = useAccountScope()
  return <LookScopeContext.Provider value={scope}>{children}</LookScopeContext.Provider>
}
