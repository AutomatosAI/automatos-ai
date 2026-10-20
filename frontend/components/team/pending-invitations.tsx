'use client'

import { motion } from 'framer-motion'
import { Mail, Clock, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { PendingInvitation } from './team-types'

interface PendingInvitationsProps {
    pendingInvites: PendingInvitation[]
    onRevoke: (invitationId: number) => void
}

/** The "Pending invitations (n)" card: each invitation's email, role and expiry, with Revoke. */
export function PendingInvitations({ pendingInvites, onRevoke }: PendingInvitationsProps) {
    return (
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
                            onClick={() => onRevoke(inv.id)}
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
    )
}
