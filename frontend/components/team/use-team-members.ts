'use client'

/**
 * The team page's data: the current workspace's members and pending invitations, and the
 * actions on them (revoke an invitation, change a role, remove a member). Each action reloads
 * both lists when it succeeds. Moved out of team-management.tsx unchanged.
 */
import { useState, useEffect } from 'react'
import { apiClient } from '@/lib/api-client'
import { themeErrorMessage, useStepUp } from '@/lib/theme-host/api'
import { useWorkspace } from '@/hooks/use-workspace'
import type { PendingInvitation, TeamMember } from './team-types'

/**
 * PRD-261 A-07f (#1157): a role change needs, in SaaS, a sign-in from the last few minutes
 * (orchestrator/api/team.py `STEP_UP`). Called through `useStepUp`, so a refusal opens Clerk's
 * "confirm it's you" and the change is sent once more; the local edition sends it as is.
 */
function patchMemberRole(workspaceId: string, memberId: number, role: string) {
    return apiClient.patch(`/api/workspaces/${workspaceId}/team/members/${memberId}/role`, { role })
}

/** Members and pending invitations, loaded whenever the workspace id changes. */
function useTeamRoster(workspaceId: string | null) {
    const [members, setMembers] = useState<TeamMember[]>([])
    const [pendingInvites, setPendingInvites] = useState<PendingInvitation[]>([])
    const [loadingMembers, setLoadingMembers] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const fetchMembers = async () => {
        if (!workspaceId) return
        try {
            setLoadingMembers(true)
            const [membersRes, invitesRes] = await Promise.all([
                apiClient.request<TeamMember[]>(`/api/workspaces/${workspaceId}/team/members`),
                apiClient.request<PendingInvitation[]>(`/api/workspaces/${workspaceId}/team/invitations`).catch(() => []),
            ])
            setMembers(membersRes)
            setPendingInvites(invitesRes)
        } catch (err) {
            console.error("Failed to fetch members:", err)
            setError("Failed to load team members.")
        } finally {
            setLoadingMembers(false)
        }
    }

    useEffect(() => {
        if (workspaceId) {
            fetchMembers()
        }
    }, [workspaceId])

    return { members, pendingInvites, loadingMembers, error, fetchMembers }
}

/** Revoke, role change and remove; each asks first where it destroys, then reloads the roster. */
function useMemberActions(workspaceId: string | null, fetchMembers: () => Promise<void>) {
    const changeRole = useStepUp(patchMemberRole)

    const handleRevokeInvite = async (invitationId: number) => {
        if (!confirm("Revoke this pending invitation?")) return
        try {
            await apiClient.request(`/api/workspaces/${workspaceId}/team/invitations/${invitationId}`, {
                method: 'DELETE',
            })
            fetchMembers()
        } catch (err) {
            console.error("Failed to revoke invitation:", err)
            alert("Failed to revoke invitation")
        }
    }

    const handleRoleChange = async (memberId: number, newRole: string) => {
        if (!workspaceId) return
        try {
            await changeRole(workspaceId, memberId, newRole)
            fetchMembers()
        } catch (err) {
            console.error("Failed to update role:", err)
            // The server's sentence, the closed confirmation, or "couldn't confirm it's you".
            alert(themeErrorMessage(err))
        }
    }

    const handleRemoveMember = async (memberId: number) => {
        if (!confirm("Are you sure you want to remove this member?")) return

        try {
            await apiClient.request(`/api/workspaces/${workspaceId}/team/members/${memberId}`, {
                method: 'DELETE'
            })
            fetchMembers()
        } catch (err) {
            console.error("Failed to remove member:", err)
            alert("Failed to remove member")
        }
    }

    return { handleRevokeInvite, handleRoleChange, handleRemoveMember }
}

/**
 * The current workspace's roster and the actions on it. `loading` is true while either the
 * workspace or the roster is still loading.
 */
export function useTeamMembers() {
    const { workspaceId, loading: workspaceLoading } = useWorkspace()
    const roster = useTeamRoster(workspaceId)
    const actions = useMemberActions(workspaceId, roster.fetchMembers)

    // Combined loading state
    const loading = workspaceLoading || roster.loadingMembers

    return { workspaceId, loading, ...roster, ...actions }
}
