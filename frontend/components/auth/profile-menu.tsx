'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useUser, useClerk } from '@clerk/nextjs'
import { User, Settings, LogOut, ChevronDown, Sparkles } from 'lucide-react'
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
        await signOut()
        router.push('/')
    }

    if (!isLoaded) {
        return (
            <div className="w-9 h-9 rounded-full bg-slate-700 animate-pulse" />
        )
    }

    // Show sign-in button if not authenticated
    if (!user) {
        return (
            <button
                onClick={() => router.push('/sign-in')}
                className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-orange-500 hover:bg-orange-600 text-white font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-orange-500/50"
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
                    className="flex items-center space-x-2 rounded-full p-1.5 text-orange-400 hover:text-orange-300 hover:bg-orange-500/5 transition-colors focus:outline-none focus:ring-2 focus:ring-orange-500/50"
                    aria-label="User menu"
                >
                    {user.imageUrl ? (
                        <img
                            src={user.imageUrl}
                            alt={user.fullName || 'User'}
                            className="w-8 h-8 rounded-full border-2 border-orange-500/30"
                        />
                    ) : (
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-500 to-purple-600 flex items-center justify-center">
                            <User className="w-4 h-4 text-white" />
                        </div>
                    )}
                    <ChevronDown className={`w-4 h-4 transition-transform ${open ? 'rotate-180' : ''}`} />
                </button>
            </DropdownMenu.Trigger>

            <DropdownMenu.Portal>
                <DropdownMenu.Content
                    className="min-w-[240px] bg-slate-900/95 backdrop-blur-xl border border-slate-700/50 rounded-xl shadow-2xl p-2 animate-in fade-in-0 zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=top]:slide-in-from-bottom-2"
                    sideOffset={8}
                    align="end"
                >
                    {/* User Info Header */}
                    <div className="px-3 py-3 mb-2 border-b border-slate-700/50">
                        <div className="flex items-center space-x-3">
                            {user.imageUrl ? (
                                <img
                                    src={user.imageUrl}
                                    alt={user.fullName || 'User'}
                                    className="w-10 h-10 rounded-full border-2 border-orange-500/30"
                                />
                            ) : (
                                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-500 to-purple-600 flex items-center justify-center">
                                    <User className="w-5 h-5 text-white" />
                                </div>
                            )}
                            <div className="flex-1 min-w-0">
                                <p className="text-sm font-semibold text-white truncate">
                                    {user.fullName || 'User'}
                                </p>
                                <p className="text-xs text-slate-400 truncate">
                                    {user.primaryEmailAddress?.emailAddress}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Menu Items */}
                    <DropdownMenu.Item
                        className="flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-slate-800/80 cursor-pointer outline-none transition-colors"
                        onSelect={() => {
                            router.push('/settings/profile')
                            setOpen(false)
                        }}
                    >
                        <User className="w-4 h-4 text-orange-400" />
                        <span>Profile</span>
                    </DropdownMenu.Item>

                    <DropdownMenu.Item
                        className="flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-slate-800/80 cursor-pointer outline-none transition-colors"
                        onSelect={() => {
                            router.push('/settings')
                            setOpen(false)
                        }}
                    >
                        <Settings className="w-4 h-4 text-orange-400" />
                        <span>Settings</span>
                    </DropdownMenu.Item>

                    <DropdownMenu.Item
                        className="flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-slate-800/80 cursor-pointer outline-none transition-colors"
                        onSelect={() => {
                            setOpen(false)
                            // Small delay so the dropdown closes first
                            setTimeout(async () => {
                                if (user) {
                                    const { createFirstLoginTour } = await import('@/lib/shepherd/first-login-tour')
                                    const tour = createFirstLoginTour(user.id)
                                    tour.start()
                                }
                            }, 150)
                        }}
                    >
                        <Sparkles className="w-4 h-4 text-orange-400" />
                        <span>Start Tour</span>
                    </DropdownMenu.Item>

                    <DropdownMenu.Separator className="h-px bg-slate-700/50 my-2" />

                    <DropdownMenu.Item
                        className="flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 cursor-pointer outline-none transition-colors"
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
