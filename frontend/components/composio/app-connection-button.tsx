/**
 * App Connection Button Component (PRD-36)
 * =========================================
 * 
 * OAuth connection button that opens Composio's hosted auth flow.
 */

'use client'

import { useState } from 'react'
import { ExternalLink, Loader2, Check } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { toast } from 'sonner'
import { useInitiateConnection } from '@/hooks/use-composio-api'

interface AppConnectionButtonProps {
    appName: string
    displayName: string
    isConnected: boolean
    onConnected?: () => void
}

export function AppConnectionButton({
    appName,
    displayName,
    isConnected,
    onConnected,
}: AppConnectionButtonProps) {
    const [isConnecting, setIsConnecting] = useState(false)
    const initiateConnection = useInitiateConnection()

    const handleConnect = async () => {
        setIsConnecting(true)

        try {
            const result = await initiateConnection.mutateAsync({
                appName,
                callbackUrl: `${window.location.origin}/tools/callback?connected=${appName}`,
            })

            // NO_AUTH apps are activated immediately — no OAuth redirect needed
            if (!result.redirect_url) {
                setIsConnecting(false)
                onConnected?.()
                return
            }

            // Open OAuth popup
            const width = 600
            const height = 700
            const left = window.screenX + (window.outerWidth - width) / 2
            const top = window.screenY + (window.outerHeight - height) / 2

            const popup = window.open(
                result.redirect_url,
                `Connect ${displayName}`,
                `width=${width},height=${height},left=${left},top=${top}`
            )

            // Poll for popup close
            const checkClosed = setInterval(() => {
                if (popup?.closed) {
                    clearInterval(checkClosed)
                    setIsConnecting(false)
                    // The callback URL will update the connection status
                    onConnected?.()
                }
            }, 1000)

            // Timeout after 5 minutes
            setTimeout(() => {
                clearInterval(checkClosed)
                if (!popup?.closed) {
                    popup?.close()
                }
                setIsConnecting(false)
            }, 5 * 60 * 1000)

        } catch (error) {
            setIsConnecting(false)
            toast.error('Connection failed', { description: `Failed to connect to ${displayName}. Please try again.` })
        }
    }

    if (isConnected) {
        return (
            <Button
                variant="outline"
                size="sm"
                disabled
                className="text-success border-success/50 bg-success/10"
            >
                <Check className="h-4 w-4 mr-1.5" />
                Connected
            </Button>
        )
    }

    return (
        <Button
            variant="outline"
            size="sm"
            onClick={handleConnect}
            disabled={isConnecting}
            className="bg-secondary hover:bg-secondary/80 border-border"
        >
            {isConnecting ? (
                <>
                    <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
                    Connecting...
                </>
            ) : (
                <>
                    <ExternalLink className="h-4 w-4 mr-1.5" />
                    Connect
                </>
            )}
        </Button>
    )
}
