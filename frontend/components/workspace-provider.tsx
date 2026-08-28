'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useAuth, useOrganization } from '@clerk/nextjs'
import { usePathname } from 'next/navigation'

const ACCEPT_INVITATION_ROUTE = /^\/accept-invitation(\/|$|\?)/

/**
 * PRD-37: Workspace Context
 * Provides workspace information throughout the app
 */

/**
 * PRD-222 W1S2: server-side Auto-led onboarding state, surfaced on
 * `GET /api/workspaces/current`. The stage machine + trial ledger live in the
 * `workspaces.onboarding` JSONB; the frontend reads this snapshot (never
 * localStorage — D8). Consumed by the onboarding surfaces (US-012..US-015).
 */
export type OnboardingStage =
    | 'not_started'
    | 'questions'
    | 'teach'
    | 'proposal'
    | 'building'
    | 'boom'
    | 'powerup'
    | 'completed'
    | 'skipped'

export type TrialState = 'active' | 'warned' | 'exhausted' | 'converted'

export interface WorkspaceTrial {
    granted_usd: number
    spent_usd: number
    state: TrialState
}

export interface WorkspaceOnboarding {
    stage: OnboardingStage
    /** null until the W1S9 trial is granted at provisioning. */
    trial: WorkspaceTrial | null
}

export interface Workspace {
    id: string
    name: string
    slug: string
    plan: 'starter' | 'business' | 'enterprise'
    role: 'owner' | 'admin' | 'editor' | 'viewer' | 'member'
    planLimits: {
        max_agents: number
        max_workflows: number
        max_documents: number
        max_members: number
    }
    onboarding?: WorkspaceOnboarding
    webhookUrl?: string
    webhookKey?: string
    settings?: {
        integrations?: Record<string, string>
        [key: string]: unknown
    }
}

interface WorkspaceContextType {
    workspace: Workspace | null
    isLoading: boolean
    error: Error | null
    refreshWorkspace: () => Promise<void>
    /**
     * PRD-195 (G5): false when the caller's workspace role is 'viewer' —
     * write affordances (create/edit/delete buttons) should render disabled
     * instead of letting the click 403 against the backend gates.
     * `/api/workspaces/current` now returns the REAL per-tenant role.
     */
    canEdit: boolean
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
    const pathname = usePathname()
    const [workspace, setWorkspace] = useState<Workspace | null>(null)
    const [isLoading, setIsLoading] = useState(true)
    const [error, setError] = useState<Error | null>(null)

    const isAcceptInvitationRoute = ACCEPT_INVITATION_ROUTE.test(pathname || '')

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

            if (response.status === 409) {
                const body = await response.json().catch(() => ({}))
                const detail = body?.detail || body
                if (detail?.code === 'pending_invitation') {
                    const tokenFromServer = detail?.token as string | undefined
                    const tokenFromUrl =
                        typeof window !== 'undefined'
                            ? new URL(window.location.href).searchParams.get('token')
                            : null
                    const token = tokenFromServer || tokenFromUrl
                    const target = token
                        ? `/accept-invitation?token=${encodeURIComponent(token)}`
                        : '/accept-invitation'
                    if (typeof window !== 'undefined') {
                        window.location.href = target
                    }
                    return
                }
            }

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
                planLimits: data.plan_limits,
                onboarding: data.onboarding ?? undefined,
                webhookUrl: data.webhook_url,
                webhookKey: data.webhook_key,
                settings: data.settings || {},
            })

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
        // Skip workspace fetch on the invitation acceptance route — that flow
        // owns workspace selection explicitly via /api/team/accept-invitation.
        // Without this guard, WorkspaceProvider races accept-invitation and
        // can auto-provision a personal workspace before consent fires.
        if (isAcceptInvitationRoute) {
            setIsLoading(false)
            return
        }
        fetchWorkspace()
    }, [isSignedIn, organization?.id, isAcceptInvitationRoute])

    const refreshWorkspace = async () => {
        await fetchWorkspace()
    }

    return (
        <WorkspaceContext.Provider value={{
            workspace,
            isLoading,
            error,
            refreshWorkspace,
            // Fail-open while loading (workspace === null) so solo/local users
            // never see a flash of disabled chrome; the backend gates are the
            // enforcement — this flag is UI honesty for real viewers (G5).
            canEdit: workspace ? workspace.role !== 'viewer' : true,
        }}>
            {children}
        </WorkspaceContext.Provider>
    )
}
