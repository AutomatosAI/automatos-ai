'use client'

import { motion } from 'framer-motion'
import { Trash2, Mail } from 'lucide-react'
import { formatAbsoluteDate, formatRelative, isWithinDays } from '@/lib/format-relative'
import { Button } from '@/components/ui/button'
import { ACTIVE_WITHIN_DAYS, displayName, getRoleBadgeColor, plural } from './team-format'
import type { TeamMember } from './team-types'

interface MemberCellProps {
    member: TeamMember
}

interface MemberRoleCellProps extends MemberCellProps {
    onRoleChange: (memberId: number, newRole: string) => void
}

interface MemberRowProps extends MemberRoleCellProps {
    onRemove: (memberId: number) => void
}

/** The avatar's initial, the display name and the email. */
function MemberIdentityCell({ member }: MemberCellProps) {
    return (
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
    )
}

/** The role as a pill-styled select; picking another role sends the change. */
function MemberRoleCell({ member, onRoleChange }: MemberRoleCellProps) {
    return (
        <div className="md:col-span-2 flex items-center gap-2 pl-13 md:pl-0">
            <div className="relative inline-block">
                <select
                    value={member.role}
                    onChange={(e) => onRoleChange(member.id, e.target.value)}
                    className={`appearance-none pl-3 pr-8 py-1.5 text-xs font-semibold rounded-full border bg-transparent focus:outline-none focus:ring-1 focus:ring-offset-0 cursor-pointer ${getRoleBadgeColor(member.role)}`}
                >
                    {/* The popup is painted by the OS, so the options read the theme's
                        popover surface; the pill above still carries the role colour. */}
                    <option value="owner" className="bg-popover text-popover-foreground">Owner</option>
                    <option value="admin" className="bg-popover text-popover-foreground">Admin</option>
                    <option value="editor" className="bg-popover text-popover-foreground">Editor</option>
                    <option value="viewer" className="bg-popover text-popover-foreground">Viewer</option>
                </select>
                {/* Tiny custom arrow since we removed appearance */}
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-current opacity-50">
                    <svg className="h-3 w-3 fill-current" viewBox="0 0 20 20"><path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" fillRule="evenodd"></path></svg>
                </div>
            </div>
        </div>
    )
}

/** When the member last chatted or ran a tool here; "—" when this role can't see activity. */
function MemberLastActiveCell({ member }: MemberCellProps) {
    return (
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
    )
}

/** Tool runs and chats in the last 30 days; "—" when this role can't see activity. */
function MemberActivityCell({ member }: MemberCellProps) {
    return (
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
    )
}

/** The join date, and who sent the invitation when that is known. */
function MemberJoinedCell({ member }: MemberCellProps) {
    return (
        <div className="md:col-span-2 text-sm text-muted-foreground pl-13 md:pl-0 min-w-0">
            <div>{formatAbsoluteDate(member.joined_at)}</div>
            {member.invited_by_email && (
                <div className="text-xs truncate" title={`Invited by ${member.invited_by_email}`}>
                    by {member.invited_by_email}
                </div>
            )}
        </div>
    )
}

/** One member: identity, role, last active, activity and joined cells, and Remove (never for an owner). */
export function MemberRow({ member, onRoleChange, onRemove }: MemberRowProps) {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="relative flex flex-col md:grid md:grid-cols-12 gap-2 md:gap-4 p-4 pr-14 items-start md:items-center hover:bg-white/5 transition-colors group"
        >
            <MemberIdentityCell member={member} />

            <MemberRoleCell member={member} onRoleChange={onRoleChange} />

            <MemberLastActiveCell member={member} />

            <MemberActivityCell member={member} />

            <MemberJoinedCell member={member} />

            <div className="absolute right-3 top-3 md:top-1/2 md:-translate-y-1/2">
                {/* Don't allow deleting owner */}
                {member.role !== 'owner' && (
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => onRemove(member.id)}
                        className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-red-400 hover:bg-red-400/10 transition-all"
                        title="Remove member"
                    >
                        <Trash2 className="w-4 h-4" />
                    </Button>
                )}
            </div>
        </motion.div>
    )
}
