'use client'

import { createContext, useContext, ReactNode } from 'react'
import { useSession } from '@/lib/auth-hooks'

/**
 * System Role Context
 *
 * Provides the system-level role from Clerk's session token metadata,
 * aligned to the backend's system_role set (PRD-195 S8 — the legacy ghost
 * role that existed in no backend role set is deleted).
 * This is separate from workspace roles (owner/admin/editor/viewer).
 *
 * Edition safety (PRD-209): useSession comes from the edition-aware seam
 * (lib/auth-hooks) — in local mode it returns a fixed anonymous session
 * (null user ⇒ system role 'user'), so no Clerk hook ever executes without
 * a ClerkProvider.
 */

type SystemRole = 'super_admin' | 'admin' | 'user'

interface RoleContextType {
    systemRole: SystemRole
    isAdmin: boolean
    isLoading: boolean
}

const RoleContext = createContext<RoleContextType | null>(null)

export function useSystemRole() {
    const context = useContext(RoleContext)
    if (!context) {
        throw new Error('useSystemRole must be used within a RoleProvider')
    }
    return context
}

export function RoleProvider({ children }: { children: ReactNode }) {
    const { isLoaded, session } = useSession()

    // Extract role from user's publicMetadata
    // The publicMetadata is set in Clerk Dashboard for each user
    // Session token is configured to include: { "metadata": "{{user.public_metadata}}" }
    let roleFromToken: string | undefined
    let primaryEmail: string | undefined

    if (session?.user) {
        const publicMetadata = session.user.publicMetadata as { role?: string } | undefined
        roleFromToken = publicMetadata?.role
        primaryEmail = session.user.primaryEmailAddress?.emailAddress?.toLowerCase()
    }

    // Defence-in-depth: Clerk publicMetadata is the source of truth for system
    // role, but a misconfigured tenant user must never be promoted to platform
    // admin. Require an Automatos-staff email domain on top of the metadata.
    const isAutomatosStaff = !!primaryEmail && primaryEmail.endsWith('@automatos.app')
    const isElevated = roleFromToken === 'admin' || roleFromToken === 'super_admin'
    const effectiveRole = isElevated && !isAutomatosStaff ? 'user' : roleFromToken

    const systemRole = (effectiveRole || 'user') as SystemRole

    return (
        <RoleContext.Provider value={{
            systemRole,
            // super_admin ⊇ admin — mirrors the backend hierarchy
            // (modules/policy/roles.py role_satisfies).
            isAdmin: systemRole === 'admin' || systemRole === 'super_admin',
            isLoading: !isLoaded
        }}>
            {children}
        </RoleContext.Provider>
    )
}
