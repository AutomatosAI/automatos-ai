'use client'

import { useAuth } from '@clerk/nextjs'
import { useEffect } from 'react'
import { apiClient } from '@/lib/api-client'

/**
 * Clerk API Client Initializer
 * 
 * Sets up the API client to use Clerk JWT tokens for authentication.
 * This component should be rendered once at the app root level.
 */
export function ClerkApiClientProvider({ children }: { children: React.ReactNode }) {
    const { getToken, isLoaded } = useAuth()

    useEffect(() => {
        if (isLoaded) {
            // Configure API client to use Clerk tokens
            apiClient.setClerkTokenGetter(async () => {
                try {
                    const token = await getToken()
                    return token
                } catch (error) {
                    console.error('Failed to get Clerk token:', error)
                    return null
                }
            })

            console.log('✅ API Client configured with Clerk authentication')
        }
    }, [isLoaded, getToken])

    return <>{children}</>
}
