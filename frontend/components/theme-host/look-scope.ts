'use client'

import { createContext, useContext } from 'react'

import { DEFAULT_LOOK_SCOPE } from '@/lib/looks/scope'

/**
 * PRD-260 F5 — whose looks these are: the signed-in user in the active
 * workspace (lib/looks/scope.ts). The LookProvider keys the looks and the
 * active look by it, and the shelf its seen Deliverables record
 * (hooks/use-celebrate-new.ts). AccountLookScope (./account-look-scope.tsx,
 * mounted inside the WorkspaceProvider by the host's look runtime) provides
 * it; a screen rendered without it (a component test) gets DEFAULT_LOOK_SCOPE.
 * This file holds no auth import, so neither do the LookProvider and the shelf.
 *
 * PRD-261 W-A5 (seam review) — host code: the host fills it, so the host owns it.
 * It lived in Play's folders (components/looks) until then, which made the host
 * import Play, and gave an installed copy of Play a context of its own that the
 * host never fills. Play's shelf reads it here until the SDK names the scope
 * (a seam item: lib/__tests__/prd261-port-deliverables.test.ts, PENDING).
 */

export const LookScopeContext = createContext<string | null>(DEFAULT_LOOK_SCOPE)

/** The scope looks are kept under, or null until the account and the workspace are known. */
export function useLookScope(): string | null {
  return useContext(LookScopeContext)
}
