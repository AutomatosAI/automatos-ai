'use client'

import { useAuth } from '@clerk/nextjs'
import { useCallback } from 'react'
import { useWorkspace } from '@/components/workspace-provider'

/**
 * PRD-37: Authenticated API Hook
 * 
 * Provides fetch with Clerk auth tokens and workspace context.
 * Use this for all API calls that require authentication.
 */

export function useAuthenticatedApi() {
    const { getToken, isSignedIn } = useAuth()
    const { workspace } = useWorkspace()
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

    /**
     * Make an authenticated API request
     */
    const fetchWithAuth = useCallback(
        async (
            endpoint: string,
            options: RequestInit = {}
        ): Promise<Response> => {
            const token = isSignedIn ? await getToken() : null

            const headers: Record<string, string> = {
                'Content-Type': 'application/json',
            }

            // Add auth token
            if (token) {
                headers['Authorization'] = `Bearer ${token}`
            }

            // Add workspace ID
            if (workspace?.id) {
                headers['X-Workspace-ID'] = workspace.id
            }

            // Merge with provided headers
            if (options.headers) {
                Object.assign(headers, options.headers)
            }

            const url = endpoint.startsWith('http')
                ? endpoint
                : `${baseUrl}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`

            return fetch(url, {
                ...options,
                headers,
            })
        },
        [isSignedIn, getToken, baseUrl, workspace?.id]
    )

    /**
     * GET request
     */
    const get = useCallback(
        async <T = any>(endpoint: string): Promise<T> => {
            const response = await fetchWithAuth(endpoint, { method: 'GET' })
            if (!response.ok) {
                const error = await response.json().catch(() => ({}))
                throw new Error(error.detail || `API Error: ${response.status}`)
            }
            return response.json()
        },
        [fetchWithAuth]
    )

    /**
     * POST request
     */
    const post = useCallback(
        async <T = any>(endpoint: string, data?: unknown): Promise<T> => {
            const response = await fetchWithAuth(endpoint, {
                method: 'POST',
                body: data ? JSON.stringify(data) : undefined,
            })
            if (!response.ok) {
                const error = await response.json().catch(() => ({}))
                throw new Error(error.detail || `API Error: ${response.status}`)
            }
            return response.json()
        },
        [fetchWithAuth]
    )

    /**
     * PATCH request
     */
    const patch = useCallback(
        async <T = any>(endpoint: string, data?: unknown): Promise<T> => {
            const response = await fetchWithAuth(endpoint, {
                method: 'PATCH',
                body: data ? JSON.stringify(data) : undefined,
            })
            if (!response.ok) {
                const error = await response.json().catch(() => ({}))
                throw new Error(error.detail || `API Error: ${response.status}`)
            }
            return response.json()
        },
        [fetchWithAuth]
    )

    /**
     * DELETE request
     */
    const del = useCallback(
        async (endpoint: string): Promise<void> => {
            const response = await fetchWithAuth(endpoint, { method: 'DELETE' })
            if (!response.ok && response.status !== 204) {
                const error = await response.json().catch(() => ({}))
                throw new Error(error.detail || `API Error: ${response.status}`)
            }
        },
        [fetchWithAuth]
    )

    return {
        fetchWithAuth,
        get,
        post,
        patch,
        del,
        isAuthenticated: isSignedIn,
        workspace,
    }
}
