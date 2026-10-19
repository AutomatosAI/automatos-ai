'use client'

import { useState } from 'react'
import { Users, UserPlus, RefreshCw } from 'lucide-react'
import { isWithinDays } from '@/lib/format-relative'
import { InviteModal } from './invite-modal'
import { MembersTable } from './members-table'
import { PendingInvitations } from './pending-invitations'
import { ACTIVE_WITHIN_DAYS } from './team-format'
import { useTeamMembers } from './use-team-members'
import { useAuth } from '@/lib/auth-hooks'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { PageHeader, SearchInput } from '@/components/shared'

/**
 * The Team page: header with member count, refresh and invite; the search box; pending
 * invitations; the member list; and the invite modal. Data and actions come from
 * `useTeamMembers` (use-team-members.ts).
 */
export function TeamManagement() {
    const { orgId } = useAuth()
    const {
        workspaceId,
        loading,
        members,
        pendingInvites,
        fetchMembers,
        handleRevokeInvite,
        handleRoleChange,
        handleRemoveMember,
    } = useTeamMembers()
    const [isInviteOpen, setIsInviteOpen] = useState(false)
    const [searchTerm, setSearchTerm] = useState('')

    const filteredMembers = members.filter(m =>
        m.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        m.email.toLowerCase().includes(searchTerm.toLowerCase())
    )

    // The API nulls the activity fields for roles without audit:view; render
    // "—" for those, never a misleading "0 tool runs" or "never".
    const canSeeActivity = members.some((m) => m.tool_runs_30d !== null)

    return (
        <div className="space-y-6">
            {/* Header */}
            <PageHeader
                title="Team"
                titleAccent="Management"
                eyebrow="Workspace · members"
                lede="Who has access to this workspace, what they can do, and who's pending an invite. Roles inherit from workspace to team to individual."
                actions={
                    <>
                        <Badge variant="outline" className="text-brand-primary border-brand-primary/30">
                            <Users className="w-3 h-3 mr-1" />
                            {members.length} Members
                            {canSeeActivity && (
                                <span className="text-muted-foreground font-normal ml-1">
                                    · {members.filter((m) => isWithinDays(m.last_active_at, ACTIVE_WITHIN_DAYS)).length} active this week
                                </span>
                            )}
                        </Badge>

                        <Button
                            onClick={fetchMembers}
                            variant="outline"
                            size="sm"
                            disabled={loading}
                        >
                            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                            Refresh
                        </Button>

                        <Button
                            onClick={() => setIsInviteOpen(true)}
                            className="bg-brand-primary hover:bg-brand-primary/90"
                        >
                            <UserPlus className="w-4 h-4 mr-2" />
                            Invite Member
                        </Button>
                    </>
                }
            />

            {/* Search Bar */}
            <SearchInput
                value={searchTerm}
                onChange={setSearchTerm}
                placeholder="Search members by name or email..."
            />

            {/* Pending invitations */}
            {pendingInvites.length > 0 && (
                <PendingInvitations pendingInvites={pendingInvites} onRevoke={handleRevokeInvite} />
            )}

            {/* Members List */}
            <MembersTable
                members={filteredMembers}
                loading={loading}
                onRoleChange={handleRoleChange}
                onRemove={handleRemoveMember}
            />

            <InviteModal
                isOpen={isInviteOpen}
                onClose={() => setIsInviteOpen(false)}
                onInvite={() => fetchMembers()}
                workspaceId={workspaceId || ""}
            />
        </div>
    )
}
