'use client'

import { createContext, useContext } from 'react'

import { DEFAULT_LOOK_SCOPE } from '../../lib/looks/scope'

/**
 * PRD-260 F5 — whose looks these are: the signed-in user in the active
 * workspace (lib/looks/scope.ts). The LookProvider keys the looks and the
 * active look by it, and the shelf its seen Deliverables record
 * (hooks/use-celebrate-new.ts). AccountLookScope
 * (components/theme-host/account-look-scope.tsx, mounted inside the
 * WorkspaceProvider by the host's look runtime) provides
 * it; a screen rendered without it (a component test) gets DEFAULT_LOOK_SCOPE.
 * This file holds no auth import, so neither do the LookProvider and the shelf.
 */

export const LookScopeContext = createContext<string | null>(DEFAULT_LOOK_SCOPE)

/** The scope looks are kept under, or null until the account and the workspace are known. */
export function useLookScope(): string | null {
  return useContext(LookScopeContext)
}
