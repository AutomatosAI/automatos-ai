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
        const handleCallback = async () => {
            // Fire backend callback FIRST — must complete before popup closes
            if (connected) {
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
                    console.log('✅ Backend callback succeeded for', appName)
                } catch (err) {
                    console.warn('Failed to sync Composio connection to backend:', err)
                }
            }

            // NOW close the popup (after the API call completes). The message
            // names the app + carries a connected boolean so an inline connect
            // card (PRD-222 US-019) can match its own app and reflect the result
            // — the Tools page ignores the payload and just polls popup.closed,
            // so this stays backward compatible.
            const appName = connected ? connected.toUpperCase() : undefined
            const succeeded = status === 'success' || status === 'active' || !!connected
            if (succeeded) {
                if (window.opener) {
                    const trustedOrigin = window.location.origin
                    window.opener.postMessage(
                        { type: 'COMPOSIO_CONNECTED', app: appName, status, connectionId, connected: true },
                        trustedOrigin,
                    )
                    window.close()
                } else {
                    router.push('/tools')
                }
            } else if (status === 'error' || status === 'failed') {
                if (window.opener) {
                    const trustedOrigin = window.location.origin
                    window.opener.postMessage(
                        { type: 'COMPOSIO_CONNECTED', app: appName, status, connectionId, connected: false },
                        trustedOrigin,
                    )
                    window.close()
                } else {
                    router.push('/tools?error=connection_failed')
                }
            }
        }

        handleCallback()
    }, [searchParams, router])

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-background text-foreground">
            <Loader2 className="w-12 h-12 text-[hsl(var(--info))] animate-spin mb-4" />
            <h2 className="text-xl font-semibold">Connecting</h2>
            <p className="text-muted-foreground">Wrapping things up. This usually finishes in under a second.</p>
        </div>
    )
}
