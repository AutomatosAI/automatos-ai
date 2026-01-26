'use client'

import { useAuth } from '@clerk/nextjs'
import { useEffect, useRef } from 'react'
import { apiClient } from '@/lib/api-client'

/**
 * Clerk API Client Initializer
 * 
 * Sets up the API client to use Clerk JWT tokens for authentication.
 * This component should be rendered once at the app root level.
 */
export function ClerkApiClientProvider({ children }: { children: React.ReactNode }) {
    const { getToken, isLoaded } = useAuth()

    // Install token getter synchronously (during render) to avoid race conditions
    // where React Query fires before the first useEffect runs.
    const configuredRef = useRef(false)
    const getTokenRef = useRef(getToken)
    getTokenRef.current = getToken

    if (!configuredRef.current) {
        apiClient.setClerkTokenGetter(async () => {
            try {
                // Don't gate on `isLoaded`; `getToken()` is the source of truth and will
                // throw/return null when not ready.
                return await getTokenRef.current()
            } catch (error) {
                console.error('Failed to get Clerk token:', error)
                return null
            }
        })
        configuredRef.current = true
        console.log('✅ API Client configured with Clerk authentication')
    }

    // Keep a log for debugging (optional, harmless)
    useEffect(() => {
        // eslint-disable-next-line no-console
        console.log('[CLERK] isLoaded:', isLoaded)
    }, [isLoaded])

    return <>{children}</>
}
