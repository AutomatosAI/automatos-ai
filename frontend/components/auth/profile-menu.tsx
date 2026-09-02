'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useUser, useClerk } from '@/lib/auth-hooks'
import { User, Settings, LogOut, ChevronDown } from 'lucide-react'
import * as DropdownMenu from '@radix-ui/react-dropdown-menu'

/**
 * Profile Menu Component
 * 
 * Dropdown menu triggered by profile icon with:
 * - User info display
 * - Profile navigation
 * - Settings navigation
 * - Logout action
 */

export function ProfileMenu() {
    const router = useRouter()
    const { user, isLoaded } = useUser()
    const { signOut } = useClerk()
    const [open, setOpen] = useState(false)

    const handleSignOut = async () => {
        localStorage.removeItem('last_active_workspace')
        localStorage.removeItem('last_active_org')
        await signOut()
        router.push('/')
    }

    if (!isLoaded) {
        return (
            <div className="w-9 h-9 rounded-full bg-muted animate-pulse" />
        )
    }

    // Show sign-in button if not authenticated
    if (!user) {
        return (
            <button
                onClick={() => router.push('/sign-in')}
                className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
                <User className="w-4 h-4" />
                <span>Sign In</span>
            </button>
        )
    }

    return (
        <DropdownMenu.Root open={open} onOpenChange={setOpen}>
            <DropdownMenu.Trigger asChild>
                <button
                    className="flex items-center space-x-2 rounded-full p-1.5 text-primary hover:text-primary/80 hover:bg-primary/5 transition-colors focus:outline-none focus:ring-2 focus:ring-primary/50"
                    aria-label="User menu"
                >
                    {user.imageUrl ? (
                        <img
                            src={user.imageUrl}
                            alt={user.fullName || 'User'}
                            className="w-8 h-8 rounded-full border-2 border-primary/30"
                        />
                    ) : (
                        <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                            <User className="w-4 h-4 text-primary-foreground" />
                        </div>
                    )}
                    <ChevronDown className={`w-4 h-4 transition-transform ${open ? 'rotate-180' : ''}`} />
                </button>
            </DropdownMenu.Trigger>

            <DropdownMenu.Portal>
                <DropdownMenu.Content
                    className="min-w-[240px] bg-card border border-border rounded-xl p-2 animate-in fade-in-0 zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=top]:slide-in-from-bottom-2"
                    sideOffset={8}
                    align="end"
                >
                    {/* User Info Header */}
                    <div className="px-3 py-3 mb-2 border-b border-border">
                        <div className="flex items-center space-x-3">
                            {user.imageUrl ? (
                                <img
                                    src={user.imageUrl}
                                    alt={user.fullName || 'User'}
                                    className="w-10 h-10 rounded-full border-2 border-primary/30"
                                />
                            ) : (
                                <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center">
                                    <User className="w-5 h-5 text-primary-foreground" />
                                </div>
                            )}
                            <div className="flex-1 min-w-0">
                                <p className="text-sm font-semibold text-foreground truncate">
                                    {user.fullName || 'User'}
                                </p>
                                <p className="text-xs text-muted-foreground truncate">
                                    {user.primaryEmailAddress?.emailAddress}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Menu Items */}
                    <DropdownMenu.Item
                        className="flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm text-foreground hover:text-foreground hover:bg-accent cursor-pointer outline-none transition-colors"
                        onSelect={() => {
                            router.push('/settings/profile')
                            setOpen(false)
                        }}
                    >
                        <User className="w-4 h-4 text-primary" />
                        <span>Profile</span>
                    </DropdownMenu.Item>

                    <DropdownMenu.Item
                        className="flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm text-foreground hover:text-foreground hover:bg-accent cursor-pointer outline-none transition-colors"
                        onSelect={() => {
                            router.push('/settings')
                            setOpen(false)
                        }}
                    >
                        <Settings className="w-4 h-4 text-primary" />
                        <span>Settings</span>
                    </DropdownMenu.Item>

                    <DropdownMenu.Separator className="h-px bg-border my-2" />

                    <DropdownMenu.Item
                        className="flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm text-destructive hover:text-destructive/80 hover:bg-destructive/10 cursor-pointer outline-none transition-colors"
                        onSelect={handleSignOut}
                    >
                        <LogOut className="w-4 h-4" />
                        <span>Logout</span>
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Portal>
        </DropdownMenu.Root>
    )
}
