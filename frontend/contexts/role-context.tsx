'use client'

import { createContext, useContext, ReactNode } from 'react'
import { useSession } from '@/lib/auth-hooks'
import { isSaaS } from '@/lib/auth-edition'
import { isEnterprise } from '@/lib/edition-name'
import { systemRoleOf, useEnterpriseSession } from '@/lib/enterprise/session-context'

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

/**
 * PRD-244 W0: the role when a provider exists, `null` when it does not — for
 * chrome that can render in isolation (the Studio rail). An unknown role is
 * not an admin: admin-only items stay hidden until the provider answers.
 */
export function useSystemRoleOptional() {
    return useContext(RoleContext)
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

    // Local edition: there is exactly one operator and it is their machine — the
    // anonymous local session is the instance's super admin (backend mirrors
    // this in hybrid.py's anonymous lane; no platform exists above the operator).
    const editionRole = (!isSaaS ? 'super_admin' : (effectiveRole || 'user')) as SystemRole

    // PRD-277 US-501: enterprise has many people, and each one's install role is the API's
    // (GET /api/v1/auth/me `system_role`), never Clerk's or the browser's; `user` until it answers.
    const enterprise = useEnterpriseSession()
    const systemRole = isEnterprise ? systemRoleOf(enterprise) : editionRole

    return (
        <RoleContext.Provider value={{
            systemRole,
            // super_admin ⊇ admin — mirrors the backend hierarchy
            // (modules/policy/roles.py role_satisfies).
            isAdmin: systemRole === 'admin' || systemRole === 'super_admin',
            isLoading: isEnterprise ? enterprise?.status !== 'signed-in' : !isLoaded
        }}>
            {children}
        </RoleContext.Provider>
    )
}
