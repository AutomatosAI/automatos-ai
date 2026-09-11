'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Users, UserPlus, Shield, Trash2, Mail, Search, RefreshCw, Clock, X } from 'lucide-react'
import { apiClient } from '@/lib/api-client'
import { formatAbsoluteDate, formatRelative, isWithinDays } from '@/lib/format-relative'
import { useWorkspace } from '@/hooks/use-workspace'
import { InviteModal } from './invite-modal'
import { useAuth } from '@/lib/auth-hooks'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { PageHeader, SearchInput } from '@/components/shared'

interface TeamMember {
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

/** "Active this week": the pill's count and the row highlight share this window. */
const ACTIVE_WITHIN_DAYS = 7

/** "1 chat" / "4 chats" — the activity cell reads as a sentence, not a tally. */
function plural(n: number, noun: string): string {
    return `${n} ${noun}${n === 1 ? '' : 's'}`
}

/**
 * The display label when `users.name` is empty — the email's local part.
 * Every Clerk-provisioned user has a NULL name today (names were never synced
 * into `users`), so "Unknown User" was the whole roster. This is derived from
 * data the row already carries, not invented.
 */
function displayName(member: Pick<TeamMember, 'name' | 'email'>): string {
    return member.name || member.email.split('@')[0]
}

interface PendingInvitation {
    id: number
    email: string
    role: string
    status: string
    expires_at: string
    created_at: string
}

export function TeamManagement() {
    const { orgId } = useAuth()
    const { workspaceId, loading: workspaceLoading } = useWorkspace()
    const [members, setMembers] = useState<TeamMember[]>([])
    const [pendingInvites, setPendingInvites] = useState<PendingInvitation[]>([])
    const [loadingMembers, setLoadingMembers] = useState(false)
    const [isInviteOpen, setIsInviteOpen] = useState(false)
    const [searchTerm, setSearchTerm] = useState('')
    const [error, setError] = useState<string | null>(null)

    // Combined loading state
    const loading = workspaceLoading || loadingMembers

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

    useEffect(() => {
        if (workspaceId) {
            fetchMembers()
        }
    }, [workspaceId])

    const handleRoleChange = async (memberId: number, newRole: string) => {
        try {
            await apiClient.request(`/api/workspaces/${workspaceId}/team/members/${memberId}/role`, {
                method: 'PATCH',
                body: { role: newRole }
            })
            fetchMembers()
        } catch (err) {
            console.error("Failed to update role:", err)
            alert("Failed to update role")
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

    const filteredMembers = members.filter(m =>
        m.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        m.email.toLowerCase().includes(searchTerm.toLowerCase())
    )

    const getRoleBadgeColor = (role: string) => {
        switch (role) {
            case 'owner': return 'text-purple-400 border-purple-400/30 bg-purple-400/10'
            case 'admin': return 'text-warning border-warning/30 bg-warning/10'
            case 'editor': return 'text-blue-400 border-blue-400/30 bg-blue-400/10'
            case 'viewer': return 'text-green-400 border-green-400/30 bg-green-400/10'
            default: return 'text-muted-foreground border-border bg-secondary/50'
        }
    }

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
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.15 }}
                    className="glass-card rounded-xl border border-border/30 overflow-hidden"
                >
                    <div className="p-4 border-b border-border/30 bg-white/5 flex items-center gap-2">
                        <Clock className="w-4 h-4 text-amber-400" />
                        <h3 className="font-medium text-sm">Pending invitations ({pendingInvites.length})</h3>
                    </div>
                    <div className="divide-y divide-white/5">
                        {pendingInvites.map((inv) => (
                            <div key={inv.id} className="flex items-center justify-between p-4 hover:bg-white/5 transition-colors group">
                                <div className="flex items-center gap-3">
                                    <div className="w-9 h-9 rounded-full bg-amber-400/10 border border-amber-400/30 flex items-center justify-center">
                                        <Mail className="w-4 h-4 text-amber-400" />
                                    </div>
                                    <div>
                                        <div className="font-medium text-foreground">{inv.email}</div>
                                        <div className="text-xs text-muted-foreground">
                                            Invited as <span className="text-foreground">{inv.role}</span> ·
                                            expires {new Date(inv.expires_at).toLocaleDateString()}
                                        </div>
                                    </div>
                                </div>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleRevokeInvite(inv.id)}
                                    className="text-muted-foreground hover:text-red-400 hover:bg-red-400/10"
                                    title="Revoke invitation"
                                >
                                    <X className="w-4 h-4 mr-1" />
                                    Revoke
                                </Button>
                            </div>
                        ))}
                    </div>
                </motion.div>
            )}

            {/* Members List */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="glass-card rounded-xl border border-border/30 overflow-hidden"
            >
                <div className="p-4 pr-14 border-b border-border/30 bg-white/5 hidden md:grid grid-cols-12 gap-4 font-medium text-sm text-muted-foreground">
                    <div className="col-span-4">User</div>
                    <div className="col-span-2">Role</div>
                    <div className="col-span-2">Last active</div>
                    <div className="col-span-2">Activity · 30d</div>
                    <div className="col-span-2">Joined</div>
                </div>

                {loading ? (
                    <div className="p-12 text-center text-muted-foreground flex flex-col items-center gap-2">
                        <RefreshCw className="w-6 h-6 animate-spin opacity-50" />
                        Loading members...
                    </div>
                ) : filteredMembers.length === 0 ? (
                    <div className="p-12 text-center text-muted-foreground">No members found using that search.</div>
                ) : (
                    <div className="divide-y divide-white/5">
                        {filteredMembers.map((member) => (
                            <motion.div
                                key={member.id}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="relative flex flex-col md:grid md:grid-cols-12 gap-2 md:gap-4 p-4 pr-14 items-start md:items-center hover:bg-white/5 transition-colors group"
                            >
                                <div className="md:col-span-4 flex items-center gap-3 w-full md:w-auto min-w-0">
                                    <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-warning/20 to-purple-500/20 flex items-center justify-center border border-white/10 shrink-0">
                                        <span className="font-semibold text-foreground">
                                            {displayName(member).charAt(0).toUpperCase() || '?'}
                                        </span>
                                    </div>
                                    <div className="overflow-hidden">
                                        <div className="font-medium text-foreground truncate">{displayName(member)}</div>
                                        <div className="text-sm text-muted-foreground flex items-center gap-1 truncate">
                                            <Mail className="w-3 h-3 shrink-0" />
                                            {member.email}
                                        </div>
                                    </div>
                                </div>

                                <div className="md:col-span-2 flex items-center gap-2 pl-13 md:pl-0">
                                    <div className="relative inline-block">
                                        <select
                                            value={member.role}
                                            onChange={(e) => handleRoleChange(member.id, e.target.value)}
                                            className={`appearance-none pl-3 pr-8 py-1.5 text-xs font-semibold rounded-full border bg-transparent focus:outline-none focus:ring-1 focus:ring-offset-0 cursor-pointer ${getRoleBadgeColor(member.role)}`}
                                        >
                                            <option value="owner" className="bg-[#1a1a1a] text-purple-400">Owner</option>
                                            <option value="admin" className="bg-[#1a1a1a] text-warning">Admin</option>
                                            <option value="editor" className="bg-[#1a1a1a] text-blue-400">Editor</option>
                                            <option value="viewer" className="bg-[#1a1a1a] text-green-400">Viewer</option>
                                        </select>
                                        {/* Tiny custom arrow since we removed appearance */}
                                        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-current opacity-50">
                                            <svg className="h-3 w-3 fill-current" viewBox="0 0 20 20"><path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" fillRule="evenodd"></path></svg>
                                        </div>
                                    </div>
                                </div>

                                <div
                                    className={`md:col-span-2 text-sm pl-13 md:pl-0 ${
                                        isWithinDays(member.last_active_at, ACTIVE_WITHIN_DAYS) ? 'text-foreground' : 'text-muted-foreground'
                                    }`}
                                    title={
                                        member.tool_runs_30d === null
                                            ? 'Activity is shown to workspace owners and admins'
                                            : member.last_active_at
                                              ? 'Most recent chat or tool run in this workspace'
                                              : 'No chats or tool runs in this workspace yet'
                                    }
                                >
                                    {member.tool_runs_30d === null ? '—' : formatRelative(member.last_active_at)}
                                </div>

                                <div className="md:col-span-2 text-sm text-muted-foreground pl-13 md:pl-0 tabular-nums leading-tight">
                                    {member.tool_runs_30d === null || member.chats_30d === null ? (
                                        '—'
                                    ) : (
                                        <>
                                            <div>{plural(member.tool_runs_30d, 'tool run')}</div>
                                            <div>{plural(member.chats_30d, 'chat')}</div>
                                        </>
                                    )}
                                </div>

                                <div className="md:col-span-2 text-sm text-muted-foreground pl-13 md:pl-0 min-w-0">
                                    <div>{formatAbsoluteDate(member.joined_at)}</div>
                                    {member.invited_by_email && (
                                        <div className="text-xs truncate" title={`Invited by ${member.invited_by_email}`}>
                                            by {member.invited_by_email}
                                        </div>
                                    )}
                                </div>

                                <div className="absolute right-3 top-3 md:top-1/2 md:-translate-y-1/2">
                                    {/* Don't allow deleting owner */}
                                    {member.role !== 'owner' && (
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            onClick={() => handleRemoveMember(member.id)}
                                            className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-red-400 hover:bg-red-400/10 transition-all"
                                            title="Remove member"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </Button>
                                    )}
                                </div>
                            </motion.div>
                        ))}
                    </div>
                )}
            </motion.div>

            <InviteModal
                isOpen={isInviteOpen}
                onClose={() => setIsInviteOpen(false)}
                onInvite={() => fetchMembers()}
                workspaceId={workspaceId || ""}
            />
        </div>
    )
}
