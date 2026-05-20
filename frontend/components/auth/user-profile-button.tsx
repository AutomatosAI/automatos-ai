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
            <div className="w-8 h-8 rounded-full bg-muted animate-pulse" />
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
                    userButtonPopoverCard: 'bg-secondary border border-border',
                    userButtonPopoverFooter: 'hidden',
                    userPreviewMainIdentifier: 'text-foreground',
                    userPreviewSecondaryIdentifier: 'text-muted-foreground',
                    userButtonPopoverActionButton: 'text-foreground hover:bg-secondary',
                    userButtonPopoverActionButtonText: 'text-foreground',
                    userButtonPopoverActionButtonIcon: 'text-muted-foreground',
                },
            }}
            userProfileMode="navigation"
            userProfileUrl="/settings/profile"
        />
    )
}
