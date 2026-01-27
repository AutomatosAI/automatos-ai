'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Mail, ShieldAlert } from 'lucide-react'
import { apiClient } from '@/lib/api-client'
import { useUser } from '@clerk/nextjs'

interface InviteModalProps {
    isOpen: boolean
    onClose: () => void
    onInvite: () => void
    workspaceId: string
}

export function InviteModal({ isOpen, onClose, onInvite, workspaceId }: InviteModalProps) {
    const { user } = useUser()
    const [email, setEmail] = useState('')
    const [role, setRole] = useState('editor')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setError(null)

        // Domain Check Logic
        if (user?.primaryEmailAddress?.emailAddress) {
            const myDomain = user.primaryEmailAddress.emailAddress.split('@')[1]
            const inviteDomain = email.split('@')[1]

            if (myDomain && inviteDomain && myDomain.toLowerCase() !== inviteDomain.toLowerCase()) {
                setError(`Security Restriction: You can only invite users from your organization (${myDomain}).`)
                return
            }
        }

        setLoading(true)

        try {
            await apiClient.request(`/api/workspaces/${workspaceId}/team/invite`, {
                method: 'POST',
                body: { email, role }
            })
            onInvite()
            onClose()
            setEmail('')
            setRole('editor')
        } catch (err: any) {
            console.error("Invite failed:", err)
            setError(err.message || "Failed to send invitation")
        } finally {
            setLoading(false)
        }
    }

    if (!isOpen) return null

    return (
        <AnimatePresence>
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="w-full max-w-md bg-[#0A0A0A] border border-white/10 rounded-xl shadow-2xl overflow-hidden"
                >
                    <div className="p-6 border-b border-white/10 flex justify-between items-center">
                        <h2 className="text-xl font-semibold">Invite Team Member</h2>
                        <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    <form onSubmit={handleSubmit} className="p-6 space-y-4">
                        {error && (
                            <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-start gap-3 text-red-400 text-sm">
                                <ShieldAlert className="w-5 h-5 flex-shrink-0" />
                                <span>{error}</span>
                            </div>
                        )}

                        <div className="space-y-2">
                            <label className="text-sm font-medium text-muted-foreground">Email Address</label>
                            <div className="relative">
                                <Mail className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
                                <input
                                    type="email"
                                    required
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    className="w-full bg-secondary/30 border border-white/10 rounded-lg pl-9 pr-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-orange-500/50 transition-all placeholder:text-muted-foreground/50"
                                    placeholder="colleague@company.com"
                                />
                            </div>
                            <p className="text-xs text-muted-foreground">
                                Only users with the same email domain are allowed.
                            </p>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium text-muted-foreground">Role</label>
                            <select
                                value={role}
                                onChange={(e) => setRole(e.target.value)}
                                className="w-full bg-secondary/30 border border-white/10 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-orange-500/50 max-h-40" // added max-h to prevent overly large dropdown if browser defaults vary
                            >
                                <option value="admin">Admin</option>
                                <option value="editor">Editor</option>
                                <option value="viewer">Viewer</option>
                            </select>
                        </div>

                        <div className="pt-4 flex justify-end gap-3">
                            <button
                                type="button"
                                onClick={onClose}
                                className="px-4 py-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
                                disabled={loading}
                            >
                                Cancel
                            </button>
                            <button
                                type="submit"
                                disabled={loading}
                                className="px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                            >
                                {loading ? 'Sending...' : 'Send Invitation'}
                            </button>
                        </div>
                    </form>
                </motion.div>
            </div>
        </AnimatePresence>
    )
}
