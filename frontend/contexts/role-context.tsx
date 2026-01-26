'use client'

import { createContext, useContext, ReactNode } from 'react'
import { useSession } from '@clerk/nextjs'

/**
 * System Role Context
 * 
 * Provides system-level role (admin, user, customer_manager) from Clerk's
 * session token metadata. This is separate from workspace roles (owner/member).
 */

type SystemRole = 'admin' | 'user' | 'customer_manager'

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

    if (session?.user) {
        // Access public metadata from the session user
        const publicMetadata = session.user.publicMetadata as { role?: string } | undefined
        roleFromToken = publicMetadata?.role

        // Auto-admin for @automatos.app domain
        // Anyone with an @automatos.app email is automatically an admin
        // This overrides any publicMetadata settings
        const primaryEmail = session.user.primaryEmailAddress?.emailAddress
        if (primaryEmail?.endsWith('@automatos.app')) {
            roleFromToken = 'admin'
        }
    }

    // Default to 'user' if no role is set
    const systemRole = (roleFromToken || 'user') as SystemRole

    return (
        <RoleContext.Provider value={{
            systemRole,
            isAdmin: systemRole === 'admin',
            isLoading: !isLoaded
        }}>
            {children}
        </RoleContext.Provider>
    )
}
