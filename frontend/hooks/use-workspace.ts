'use client'

import { useState, useEffect } from 'react'
import { apiClient } from '@/lib/api-client'
import { useAuth } from '@clerk/nextjs'

export function useWorkspace() {
    const [workspaceId, setWorkspaceId] = useState<string | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const { isLoaded, isSignedIn } = useAuth()

    useEffect(() => {
        const fetchWorkspace = async () => {
            // Wait for Auth to load
            if (!isLoaded) return

            try {
                setLoading(true)
                const response = await apiClient.request<{ id: string }>('/api/workspaces/current')
                if (response && response.id) {
                    setWorkspaceId(response.id)
                } else {
                    setError('No active workspace found')
                }
            } catch (err) {
                console.error('Failed to fetch workspace:', err)
                setError('Failed to load workspace configuration')
            } finally {
                setLoading(false)
            }
        }

        fetchWorkspace()
    }, [isLoaded, isSignedIn])

    return { workspaceId, loading, error }
}
