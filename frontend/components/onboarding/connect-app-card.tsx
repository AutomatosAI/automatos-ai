'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CheckCircle2, ExternalLink, Loader2, Plug, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useConnectedApps, useInitiateConnection } from '@/hooks/use-composio-api'

/**
 * PRD-222 US-019 (W2·S3) — the inline Composio connect card.
 *
 * A chat status card (same shape as US-013/US-015's power-up / intake cards)
 * that connects ONE app Auto asked for, without leaving the conversation. It
 * REUSES the existing Composio OAuth flow verbatim — `useInitiateConnection`
 * (POST /api/composio/connect/{app}) opens the same hosted OAuth popup the Tools
 * page uses; no second OAuth/redirect endpoint is added.
 *
 * The callback lands on `/tools/callback`, which already posts a
 * `COMPOSIO_CONNECTED` message back to `window.opener` (extended by this story to
 * name the app + carry the connected boolean). This card listens for THAT
 * existing message — no new event bus — and also refetches the connection list
 * on window focus / popup close as a fallback, then reflects the result inline:
 * connecting → connected / couldn't-connect.
 */

const CONNECTED_MESSAGE = 'COMPOSIO_CONNECTED'

type ConnectState = 'idle' | 'connecting' | 'connected' | 'failed'

interface ConnectAppCardProps {
  /** Canonical app key Auto asked for (e.g. "GMAIL"); matched case-insensitively. */
  appName: string
  /** Friendly label; defaults to a title-cased app name. */
  displayName?: string
  /** Fired once when this app transitions to connected. */
  onConnected?: (appName: string) => void
}

function titleCase(name: string): string {
  const lower = name.replace(/[_-]+/g, ' ').toLowerCase()
  return lower.charAt(0).toUpperCase() + lower.slice(1)
}

export function ConnectAppCard({ appName, displayName, onConnected }: ConnectAppCardProps) {
  const canonical = (appName || '').toUpperCase()
  const label = displayName || titleCase(appName || '')

  const initiate = useInitiateConnection()
  const { data: connections = [], refetch } = useConnectedApps()

  const [state, setState] = useState<ConnectState>('idle')
  const [error, setError] = useState<string | null>(null)
  const popupRef = useRef<Window | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const firedRef = useRef(false)

  // Clear any live popup-close poll on unmount so no timer leaks past the card.
  useEffect(() => () => {
    if (pollRef.current) clearInterval(pollRef.current)
  }, [])

  // Is this app already active in the workspace's connection list?
  const isActiveInList = useMemo(
    () =>
      connections.some(
        (c) => (c.app_name || '').toUpperCase() === canonical && c.status === 'active',
      ),
    [connections, canonical],
  )

  const markConnected = useCallback(() => {
    setState('connected')
    setError(null)
    if (!firedRef.current) {
      firedRef.current = true
      onConnected?.(canonical)
    }
  }, [canonical, onConnected])

  // The connection list is the source of truth: once this app shows active
  // (initial mount, a refetch, or a lazy pending→active upgrade), reflect it.
  useEffect(() => {
    if (isActiveInList) markConnected()
  }, [isActiveInList, markConnected])

  // Listen for the callback page's existing COMPOSIO_CONNECTED message. Same
  // origin only; only react to THIS card's app (the message now names it).
  useEffect(() => {
    function onMessage(event: MessageEvent) {
      if (event.origin !== window.location.origin) return
      const data = event.data
      if (!data || data.type !== CONNECTED_MESSAGE) return
      const msgApp = (data.app || '').toUpperCase()
      // Older callbacks omit the app name; accept those only while we're the
      // card that opened the popup (state === 'connecting').
      if (msgApp && msgApp !== canonical) return
      if (msgApp === canonical || state === 'connecting') {
        const ok =
          data.connected === true ||
          data.status === 'success' ||
          data.status === 'active'
        refetch()
        if (ok) {
          markConnected()
        } else {
          setState('failed')
          setError(`${label} didn't finish connecting. Try again.`)
        }
      }
    }
    window.addEventListener('message', onMessage)
    return () => window.removeEventListener('message', onMessage)
  }, [canonical, label, state, refetch, markConnected])

  const handleConnect = useCallback(async () => {
    if (state === 'connecting' || state === 'connected') return
    setState('connecting')
    setError(null)
    try {
      const result = await initiate.mutateAsync({
        appName: canonical,
        callbackUrl: `${window.location.origin}/tools/callback?connected=${canonical}`,
      })

      // NO_AUTH apps come back with an empty redirect_url — already active.
      if (!result?.redirect_url) {
        await refetch()
        markConnected()
        return
      }

      const width = 600
      const height = 700
      const left = window.screenX + (window.outerWidth - width) / 2
      const top = window.screenY + (window.outerHeight - height) / 2
      const popup = window.open(
        result.redirect_url,
        `Connect ${label}`,
        `width=${width},height=${height},left=${left},top=${top}`,
      )
      popupRef.current = popup ?? null

      // Fallback bridge: when the popup closes, refetch and let the list-driven
      // effect confirm. If it never went active, surface a retry.
      if (pollRef.current) clearInterval(pollRef.current)
      const poll = setInterval(async () => {
        if (popupRef.current?.closed || !popupRef.current) {
          clearInterval(poll)
          pollRef.current = null
          const fresh = await refetch()
          const nowActive = (fresh.data ?? []).some(
            (c) => (c.app_name || '').toUpperCase() === canonical && c.status === 'active',
          )
          setState((prev) => {
            if (prev === 'connected' || nowActive) return 'connected'
            // popup closed without connecting — back to idle so the user can retry
            return prev === 'connecting' ? 'idle' : prev
          })
        }
      }, 1000)
      pollRef.current = poll

      // Give up polling after 5 minutes so a stuck popup can't leak a timer.
      setTimeout(() => {
        clearInterval(poll)
        if (pollRef.current === poll) pollRef.current = null
      }, 5 * 60 * 1000)
    } catch {
      setState('failed')
      setError(`Couldn't start the ${label} connection. Try again.`)
    }
  }, [state, canonical, label, initiate, refetch, markConnected])

  const connected = state === 'connected'
  const connecting = state === 'connecting'

  return (
    <div
      data-testid="connect-app-card"
      data-app={canonical}
      data-state={state}
      className="bg-card/50 backdrop-blur border border-primary/20 rounded-xl p-4 space-y-3 max-w-md"
    >
      <div className="flex items-center gap-2">
        {connected ? (
          <CheckCircle2 className="w-4 h-4 text-success shrink-0" />
        ) : connecting ? (
          <Loader2 className="w-4 h-4 text-primary shrink-0 animate-spin" />
        ) : (
          <Plug className="w-4 h-4 text-primary shrink-0" />
        )}
        <span className="text-sm font-medium text-foreground">
          {connected ? `${label} connected` : `Connect ${label}`}
        </span>
      </div>

      {!connected && (
        <p className="text-sm text-foreground/80 leading-snug">
          {connecting
            ? `Finish signing in to ${label} in the popup — I'll pick it up here.`
            : `Auto needs ${label} for this setup. Connect it right here — it opens a secure sign-in and comes straight back.`}
        </p>
      )}

      {connected && (
        <p data-testid="connect-app-done" className="text-sm text-foreground/80 leading-snug">
          {label} is connected — Auto can use it now.
        </p>
      )}

      {error && (
        <div
          data-testid="connect-app-error"
          className="flex items-start gap-1.5 rounded-md border border-destructive/30 bg-destructive/10 p-2.5 text-xs text-destructive"
        >
          <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {!connected && (
        <Button
          type="button"
          size="sm"
          onClick={handleConnect}
          disabled={connecting}
          data-testid="connect-app-button"
          className="w-full"
        >
          {connecting ? (
            <>
              <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
              Connecting…
            </>
          ) : (
            <>
              <ExternalLink className="w-3.5 h-3.5 mr-1.5" />
              Connect {label}
            </>
          )}
        </Button>
      )}
    </div>
  )
}
