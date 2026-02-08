'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useAuth, useOrganization } from '@clerk/nextjs'

/**
 * PRD-37: Workspace Context
 * Provides workspace information throughout the app
 */

interface Workspace {
    id: string
    name: string
    slug: string
    plan: 'starter' | 'business' | 'enterprise'
    role: 'owner' | 'member'
    isNewWorkspace: boolean
    planLimits: {
        max_agents: number
        max_workflows: number
        max_documents: number
        max_members: number
    }
}

interface WorkspaceContextType {
    workspace: Workspace | null
    isLoading: boolean
    error: Error | null
    refreshWorkspace: () => Promise<void>
}

const WorkspaceContext = createContext<WorkspaceContextType | null>(null)

export function useWorkspace() {
    const context = useContext(WorkspaceContext)
    if (!context) {
        throw new Error('useWorkspace must be used within a WorkspaceProvider')
    }
    return context
}

export function WorkspaceProvider({ children }: { children: ReactNode }) {
    const { isSignedIn, getToken } = useAuth()
    const { organization } = useOrganization()
    const [workspace, setWorkspace] = useState<Workspace | null>(null)
    const [isLoading, setIsLoading] = useState(true)
    const [error, setError] = useState<Error | null>(null)

    const fetchWorkspace = async () => {
        if (!isSignedIn) {
            setWorkspace(null)
            setIsLoading(false)
            return
        }

        try {
            setIsLoading(true)
            const token = await getToken()

            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/api/workspaces/current`,
                {
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json',
                    },
                }
            )

            if (!response.ok) {
                throw new Error('Failed to fetch workspace')
            }

            const data = await response.json()
            setWorkspace({
                id: data.id,
                name: data.name,
                slug: data.slug,
                plan: data.plan,
                role: data.role,
                isNewWorkspace: data.is_new_workspace ?? false,
                planLimits: data.plan_limits,
            })

            // Persist for api-client.ts
            if (typeof window !== 'undefined') {
                localStorage.setItem('last_active_workspace', data.id)
            }

            setError(null)
        } catch (err) {
            console.error('Error fetching workspace:', err)
            setError(err instanceof Error ? err : new Error('Unknown error'))
        } finally {
            setIsLoading(false)
        }
    }

    useEffect(() => {
        fetchWorkspace()
    }, [isSignedIn, organization?.id])

    const refreshWorkspace = async () => {
        await fetchWorkspace()
    }

    return (
        <WorkspaceContext.Provider value={{ workspace, isLoading, error, refreshWorkspace }}>
            {children}
        </WorkspaceContext.Provider>
    )
}
