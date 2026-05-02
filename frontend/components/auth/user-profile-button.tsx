'use client'

import Link from 'next/link'
import { UserButton, useUser } from '@clerk/nextjs'

/**
 * PRD-37: User Profile Button Component
 * 
 * Shows sign in button when logged out, user menu when logged in.
 * Drop-in component for navigation bars.
 */

interface UserProfileButtonProps {
    afterSignInUrl?: string
    afterSignUpUrl?: string
}

export function UserProfileButton({
    afterSignInUrl = '/dashboard',
    afterSignUpUrl = '/dashboard',
}: UserProfileButtonProps) {
    const { isSignedIn, isLoaded } = useUser()

    if (!isLoaded) {
        // Loading skeleton
        return (
            <div className="w-8 h-8 rounded-full bg-slate-700 animate-pulse" />
        )
    }

    if (!isSignedIn) {
        return (
            <Link
                href="/sign-in"
                className="px-4 py-2 text-sm font-medium text-white bg-agent rounded-lg hover:bg-agent transition-colors"
            >
                Sign In
            </Link>
        )
    }

    return (
        <UserButton
            afterSignOutUrl="/"
            appearance={{
                elements: {
                    avatarBox: 'w-9 h-9',
                    userButtonPopoverCard: 'bg-slate-800 border border-slate-700',
                    userButtonPopoverFooter: 'hidden',
                    userPreviewMainIdentifier: 'text-white',
                    userPreviewSecondaryIdentifier: 'text-slate-400',
                    userButtonPopoverActionButton: 'text-slate-300 hover:bg-slate-700',
                    userButtonPopoverActionButtonText: 'text-slate-300',
                    userButtonPopoverActionButtonIcon: 'text-slate-400',
                },
            }}
            userProfileMode="navigation"
            userProfileUrl="/settings/profile"
        />
    )
}
