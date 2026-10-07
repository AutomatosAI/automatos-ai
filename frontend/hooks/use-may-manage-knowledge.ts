/**
 * PRE-11 (Gerard, 7 Oct): who sees Add to Knowledge on a ticket card and a report.
 *
 * The server lets a workspace owner or admin add and remove (and the platform
 * super-admin; core/auth/workspace_admin.py). GET /api/workspaces/current answers with
 * the caller's real workspace role, and with "owner" for the super-admin and the local
 * operator, so the button shows only to owner or admin. In the local edition everyone
 * sees it. Until the role is known (or outside the workspace provider) it stays hidden.
 */

import { useWorkspaceOptional } from '@/components/workspace-provider'
import { isLocal } from '@/lib/auth-edition'

/** The workspace roles the server lets add to Knowledge and remove. */
const KNOWLEDGE_ROLES: readonly string[] = ['owner', 'admin']

export function mayManageKnowledge(role: string | null | undefined, local: boolean = isLocal): boolean {
  if (local) return true
  return role != null && KNOWLEDGE_ROLES.includes(role)
}

/** Whether the caller may add a card or a report to Knowledge, as far as the UI knows. */
export function useMayManageKnowledge(): boolean {
  return mayManageKnowledge(useWorkspaceOptional()?.workspace?.role)
}
