/**
 * PRD-233 S2 — the honest integration states for the Tools page.
 *
 * Without a Composio key the backend offers NO Composio tools to agents and
 * the marketplace grid is empty. Rendering an empty grid would be a lie about
 * what the platform can do; this card says why and how to fix it. Native
 * platform tools are unaffected, so the rest of the page still renders.
 *
 * A keyed boot with an EMPTY catalogue is the other empty-grid case: the
 * backend syncs the catalogue itself on that boot, so the card says
 * "syncing" (or "last sync failed") instead of nothing.
 */
'use client'

import React from 'react'
import { PlugZap, RefreshCw } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import type { IntegrationsStatus } from '@/hooks/use-tools-api'

export const INTEGRATIONS_DISABLED_NO_KEY_TITLE =
  'Integrations are disabled — no Composio API key is configured.'
export const INTEGRATIONS_DISABLED_TITLE = 'Integrations are disabled.'
export const CATALOGUE_SYNCING_TITLE = 'Integration catalogue is syncing.'
export const CATALOGUE_SYNC_FAILED_TITLE = 'Integration catalogue sync failed.'

interface IntegrationsDisabledCardProps {
  /** `undefined` while loading — nothing renders until the state is known. */
  status: IntegrationsStatus | undefined
}

export function IntegrationsDisabledCard({ status }: IntegrationsDisabledCardProps) {
  if (!status) return null

  if (!status.available) {
    const title = status.key_configured ? INTEGRATIONS_DISABLED_TITLE : INTEGRATIONS_DISABLED_NO_KEY_TITLE
    return (
      <Alert data-testid="integrations-disabled" className="border-[hsl(var(--warning))]/40">
        <PlugZap className="h-4 w-4" />
        <AlertTitle>{title}</AlertTitle>
        <AlertDescription className="space-y-1">
          <p>
            Add <code>COMPOSIO_API_KEY</code> to <code>.env</code>, then{' '}
            <code>docker compose up -d backend</code>; the catalogue syncs automatically on boot.
          </p>
          {status.reason ? <p className="text-muted-foreground">Reason: {status.reason}</p> : null}
          <p className="text-muted-foreground">
            Native platform tools keep working; only third-party app integrations are off.
          </p>
        </AlertDescription>
      </Alert>
    )
  }

  if (status.apps_cached > 0) return null

  const failed = status.sync_status === 'failed'
  return (
    <Alert data-testid={failed ? 'integrations-sync-failed' : 'integrations-syncing'}>
      <RefreshCw className="h-4 w-4" />
      <AlertTitle>{failed ? CATALOGUE_SYNC_FAILED_TITLE : CATALOGUE_SYNCING_TITLE}</AlertTitle>
      <AlertDescription>
        {failed ? (
          <p>
            The last catalogue sync did not complete — check that <code>COMPOSIO_API_KEY</code> is valid
            (backend logs carry the error), then run Sync or restart the backend.
          </p>
        ) : (
          <p>
            A Composio key is configured and the catalogue is being fetched in the background (first
            keyed boot). Refresh this page in a minute.
          </p>
        )}
      </AlertDescription>
    </Alert>
  )
}
