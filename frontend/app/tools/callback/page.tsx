'use client'

import { useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Loader2 } from 'lucide-react'

export default function ComposioCallbackPage() {
    const searchParams = useSearchParams()
    const router = useRouter()

    useEffect(() => {
        const status = searchParams?.get('status')
        const connectionId = searchParams?.get('connection_id')
        const connected = searchParams?.get('connected') // Fallback param

        console.log('🔗 OAuth Callback:', { status, connectionId, connected })

        // If successful, close the popup
        if (status === 'success' || status === 'active' || connected) {
            if (window.opener) {
                // Notify parent window
                window.opener.postMessage({ type: 'COMPOSIO_CONNECTED', status, connectionId }, '*')
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
