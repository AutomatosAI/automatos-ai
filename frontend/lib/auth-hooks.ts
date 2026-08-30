'use client'

/**
 * Edition-aware auth hooks — THE seam for Clerk identity in components
 * (PRD-209 local-run finding; the PRD-175 §6 residue made concrete).
 *
 * In `saas` these are Clerk's own hooks, re-exported untouched. In `local`
 * there is no <ClerkProvider>, so Clerk hooks would throw at call time —
 * these return fixed anonymous values instead. Null user/organization rides
 * the code paths SaaS personal accounts (no org) already exercise.
 *
 * Rule for the codebase: components import auth hooks from HERE, never from
 * '@clerk/nextjs' directly. The only permitted direct consumers are the
 * saas-only mount chain (providers.tsx AuthBoundary, clerk-api-client-provider)
 * and saas-only routes (sign-in/sign-up/accept-invitation).
 *
 * `isSaaS` is a build-time constant, so each export is a single consistent
 * hook implementation per build — the rules of hooks hold.
 */

import {
    useAuth as clerkUseAuth,
    useUser as clerkUseUser,
    useSession as clerkUseSession,
    useOrganization as clerkUseOrganization,
    useClerk as clerkUseClerk,
} from '@clerk/nextjs'
import { isSaaS } from '@/lib/auth-edition'

// The local session: signed-in-shaped (the backend treats the anonymous local
// session as workspace owner — PRD-195 auth-lane contract), token-less (the
// api-client's token getter is already a null no-op via LocalAuthProvider),
// org-less and user-less (the SaaS personal-account shape).
const LOCAL_AUTH = {
    isLoaded: true,
    isSignedIn: true,
    userId: null,
    sessionId: null,
    orgId: null,
    orgRole: null,
    orgSlug: null,
    getToken: async () => null,
    signOut: async () => {},
} as unknown as ReturnType<typeof clerkUseAuth>

const LOCAL_USER = {
    isLoaded: true,
    isSignedIn: true,
    user: null,
} as unknown as ReturnType<typeof clerkUseUser>

const LOCAL_SESSION = {
    isLoaded: true,
    isSignedIn: true,
    session: null,
} as unknown as ReturnType<typeof clerkUseSession>

const LOCAL_ORGANIZATION = {
    isLoaded: true,
    organization: null,
    membership: null,
} as unknown as ReturnType<typeof clerkUseOrganization>

const LOCAL_CLERK = {
    signOut: async () => {},
    openUserProfile: () => {},
    openOrganizationProfile: () => {},
} as unknown as ReturnType<typeof clerkUseClerk>

export const useAuth = isSaaS ? clerkUseAuth : () => LOCAL_AUTH
export const useUser = isSaaS ? clerkUseUser : () => LOCAL_USER
export const useSession = isSaaS ? clerkUseSession : () => LOCAL_SESSION
export const useOrganization = isSaaS ? clerkUseOrganization : () => LOCAL_ORGANIZATION
export const useClerk = isSaaS ? clerkUseClerk : () => LOCAL_CLERK
