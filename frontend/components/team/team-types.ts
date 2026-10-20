/**
 * Shapes the team page reads from `/api/workspaces/{id}/team/members` and
 * `/api/workspaces/{id}/team/invitations`. Moved out of team-management.tsx so the
 * data hook and the list components share one definition.
 */

/** One row of the workspace's member list. */
export interface TeamMember {
    id: number
    user_id: string
    email: string
    name: string | null
    role: 'owner' | 'admin' | 'editor' | 'viewer'
    joined_at: string
    last_active_at: string | null   // last chat or tool run IN THIS WORKSPACE
    // null = not shown to this role (needs audit:view — owner/admin); 0 = nothing happened.
    tool_runs_30d: number | null
    chats_30d: number | null
    invited_by_email: string | null
}

/** An invitation sent but not yet accepted. */
export interface PendingInvitation {
    id: number
    email: string
    role: string
    status: string
    expires_at: string
    created_at: string
}
