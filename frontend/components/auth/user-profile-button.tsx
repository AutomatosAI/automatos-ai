'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { UserButton } from '@clerk/nextjs'
import { useUser } from '@/lib/auth-hooks'
import { isSaaS } from '@/lib/auth-edition'
import { apiClient, type ProfileResponse } from '@/lib/api-client'

/**
 * PRD-37: User Profile Button Component
 * 
 * Shows sign in button when logged out, user menu when logged in.
 * Drop-in component for navigation bars.
 *
 * PRD-233 S6: in the local edition there is no Clerk and no account — the slot
 * shows the operator (avatar or initials + name from GET /api/profile) and
 * links to /settings/profile, where the name Auto greets them by is set.
 */

/** "Ada Lovelace" → "AL"; one word → its first two letters; else the email's first letter. */
export function initialsFor(name: string | null | undefined, email: string | null | undefined): string {
    const parts = (name ?? '').trim().split(/\s+/).filter(Boolean)
    if (parts.length >= 2) return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase()
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase()
    const fromEmail = (email ?? '').trim().charAt(0)
    return fromEmail ? fromEmail.toUpperCase() : '?'
}

const LOCAL_OPERATOR_FALLBACK_LABEL = 'Operator'

function LocalOperatorButton() {
    const [profile, setProfile] = useState<ProfileResponse | null>(null)

    useEffect(() => {
        let cancelled = false
        apiClient
            .getProfile()
            .then((data) => {
                if (!cancelled) setProfile(data)
            })
            .catch(() => {
                // The link still works without a name; the page itself reports errors.
            })
        return () => {
            cancelled = true
        }
    }, [])

    const label = profile?.name?.trim() || profile?.email || LOCAL_OPERATOR_FALLBACK_LABEL

    return (
        <Link
            href="/settings/profile"
            aria-label="Your profile"
            title={label}
            className="flex items-center gap-2 rounded-full pr-2 hover:bg-secondary transition-colors"
        >
            {profile?.avatar_url ? (
                <img
                    src={profile.avatar_url}
                    alt={label}
                    className="w-9 h-9 rounded-full border border-primary/30 object-cover"
                />
            ) : (
                <span
                    aria-hidden="true"
                    className="flex items-center justify-center w-9 h-9 rounded-full bg-primary text-primary-foreground text-xs font-semibold"
                >
                    {initialsFor(profile?.name, profile?.email)}
                </span>
            )}
            <span className="hidden sm:inline text-sm font-medium text-foreground max-w-[10rem] truncate">{label}</span>
        </Link>
    )
}

interface UserProfileButtonProps {
    afterSignInUrl?: string
    afterSignUpUrl?: string
}

export function UserProfileButton({
    afterSignInUrl = '/chat',
    afterSignUpUrl = '/chat',
}: UserProfileButtonProps) {
    const { isSignedIn, isLoaded } = useUser()

    // Local edition: no Clerk, no account — the slot is the operator's profile.
    // (UserButton is a Clerk COMPONENT and would throw without ClerkProvider.)
    if (!isSaaS) {
        return <LocalOperatorButton />
    }

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
