'use client'

import { useSystemRole } from '@/contexts/role-context'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'

/**
 * RequireRole - Wrapper component for role-based access control
 * 
 * Use this to protect pages or sections that require specific system roles.
 * Can either redirect unauthorized users or show a fallback component.
 */

interface RequireRoleProps {
    /** List of roles that are allowed to access this content */
    roles: ('super_admin' | 'admin' | 'user')[]
    /** Content to render if user has access */
    children: React.ReactNode
    /** Optional fallback to show if user doesn't have access */
    fallback?: React.ReactNode
    /** Optional path to redirect unauthorized users to */
    redirectTo?: string
}

export function RequireRole({
    roles,
    children,
    fallback,
    redirectTo = '/chat'
}: RequireRoleProps) {
    const { systemRole, isLoading } = useSystemRole()
    const router = useRouter()

    const hasAccess = roles.includes(systemRole)

    useEffect(() => {
        if (!isLoading && !hasAccess && redirectTo) {
            router.push(redirectTo)
        }
    }, [isLoading, hasAccess, redirectTo, router])

    // Still loading - show nothing to prevent flash
    if (isLoading) {
        return null
    }

    // No access - show fallback or nothing
    if (!hasAccess) {
        return fallback ? <>{fallback}</> : null
    }

    // Has access - render children
    return <>{children}</>
}

/**
 * RequireAdmin - Convenience wrapper for admin-only content
 */
export function RequireAdmin({
    children,
    fallback,
    redirectTo = '/chat'
}: Omit<RequireRoleProps, 'roles'>) {
    return (
        <RequireRole roles={['super_admin', 'admin']} fallback={fallback} redirectTo={redirectTo}>
            {children}
        </RequireRole>
    )
}
