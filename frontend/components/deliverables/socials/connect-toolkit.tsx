'use client'

/**
 * Connect a toolkit the workspace has not connected yet, through the Composio connect flow
 * (the Tools page's own: POST /api/composio/connect/{app} and its hosted sign-in, in a
 * window of its own). Paid media tools are connected in Composio only (PRD-251 D15). The
 * page that shows it reads its choices again when the window reports back (CONNECTED_MESSAGE).
 */
import { useState } from 'react'
import { Plug } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { useInitiateConnection } from '@/hooks/use-composio-api'

// What the Composio callback page posts to the window that opened it.
export const CONNECTED_MESSAGE = 'COMPOSIO_CONNECTED'
const POPUP_FEATURES = 'width=600,height=700'

interface ConnectToolkitProps {
  toolkit: string
  label: string
  /** What connecting it brings, e.g. "your Fish Audio voices". */
  purpose: string
}

export function ConnectToolkit({ toolkit, label, purpose }: ConnectToolkitProps) {
  const initiate = useInitiateConnection()
  const [failed, setFailed] = useState(false)

  const connect = async () => {
    setFailed(false)
    const appName = toolkit.toUpperCase()
    try {
      const result = await initiate.mutateAsync({
        appName,
        callbackUrl: `${window.location.origin}/tools/callback?connected=${appName}`,
      })
      if (result?.redirect_url) window.open(result.redirect_url, `Connect ${label}`, POPUP_FEATURES)
    } catch {
      setFailed(true)
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button type="button" size="sm" variant="outline" onClick={connect} disabled={initiate.isLoading}>
        <Plug className="mr-1.5 h-4 w-4" aria-hidden />
        Connect {label}
      </Button>
      <span className="text-xs text-muted-foreground">Connect it in Composio to use {purpose}.</span>
      {failed && (
        <span role="alert" className="text-xs text-destructive">
          Could not start the {label} connection. Try again.
        </span>
      )}
    </div>
  )
}
