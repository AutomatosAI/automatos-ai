'use client'

import { motion } from 'framer-motion'
import { RefreshCw } from 'lucide-react'
import { MemberRow } from './member-row'
import type { TeamMember } from './team-types'

interface MembersTableProps {
    members: TeamMember[]
    loading: boolean
    onRoleChange: (memberId: number, newRole: string) => void
    onRemove: (memberId: number) => void
}

/**
 * The members card: the column headings, then a loading line, the "no members found" line,
 * or one row per member (`members` is already filtered by the search box).
 */
export function MembersTable({ members, loading, onRoleChange, onRemove }: MembersTableProps) {
    return (
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
            ) : members.length === 0 ? (
                <div className="p-12 text-center text-muted-foreground">No members found using that search.</div>
            ) : (
                <div className="divide-y divide-white/5">
                    {members.map((member) => (
                        <MemberRow
                            key={member.id}
                            member={member}
                            onRoleChange={onRoleChange}
                            onRemove={onRemove}
                        />
                    ))}
                </div>
            )}
        </motion.div>
    )
}
