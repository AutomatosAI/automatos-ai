'use client'

import { useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Loader2 } from 'lucide-react'
import { apiClient } from '@/lib/api-client'

export default function ComposioCallbackPage() {
    const searchParams = useSearchParams()
    const router = useRouter()

    useEffect(() => {
        const status = searchParams?.get('status')
        const connectionId = searchParams?.get('connection_id')
        const connected = searchParams?.get('connected') // App name (e.g. GMAIL)

        console.log('🔗 OAuth Callback:', { status, connectionId, connected })

        // Notify the backend so it can mark the app as ACTIVE.
        // Always call if we know the app name — connection_id is optional.
        // For API_KEY apps, status param is often missing from the redirect —
        // default to 'active' and let the backend validate at execution time.
        const fireCallback = async () => {
            if (!connected) return

            const appName = connected.toUpperCase()
            const normalizedStatus = (status || 'active').toLowerCase()
            const params = new URLSearchParams({ status: normalizedStatus })
            if (connectionId) {
                params.set('connection_id', connectionId)
            }
            try {
                await apiClient.post(
                    `/api/composio/connect/${encodeURIComponent(appName)}/callback?${params.toString()}`
                )
            } catch (err) {
                console.warn('Failed to sync Composio connection to backend:', err)
            }
        }

        fireCallback()

        // If successful, close the popup
        if (status === 'success' || status === 'active' || connected) {
            if (window.opener) {
                // Notify parent window with trusted origin
                const trustedOrigin = window.location.origin
                window.opener.postMessage({ type: 'COMPOSIO_CONNECTED', status, connectionId }, trustedOrigin)
                window.close()
            } else {
                // If not a popup, redirect to tools page
                router.push('/tools')
            }
        } else if (status === 'error' || status === 'failed') {
            if (window.opener) {
                window.close()
            } else {
                router.push('/tools?error=connection_failed')
            }
        }
    }, [searchParams, router])

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-black text-white">
            <Loader2 className="w-12 h-12 text-orange-500 animate-spin mb-4" />
            <h2 className="text-xl font-semibold">Connecting...</h2>
            <p className="text-muted-foreground">Please wait while we wrap things up.</p>
        </div>
    )
}
