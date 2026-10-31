'use client'

/**
 * PRD-270 US-270-05b (§5.5, K18) — Settings › Automatos Cloud: where this install syncs Marketplace
 * themes from (the URL in use, read-only), the last Sync (its success, check, error and counts), the
 * themes held from Automatos Cloud, Sync now (the Marketplace header's mutation and toast,
 * hooks/use-cloud-sync.ts) and how to turn it off. It mirrors the revocation sync panel
 * (components/admin/themes/revocation-sync-status.tsx).
 *
 * The tab is the local edition's only (by edition, never by role: lib/auth-edition.ts), for a
 * workspace owner or admin (`canManageThemes`, the routes' `workspace:manage`), while the status says
 * the Sync is on: `AUTOMATOS_CLOUD_URL=` (empty) hides it with the Marketplace's button. On the hosted
 * edition and for a member nothing is asked. Should the status the panel reads say off, it says so
 * plainly and offers no Sync now. The server enforces every gate itself; its text renders as text.
 */
import type { UseQueryResult } from '@tanstack/react-query'
import { Cloud, Loader2, RefreshCw } from 'lucide-react'

import {
  CLOUD_LEDE,
  CLOUD_TITLE,
  NOT_SYNCED_YET,
  READING_STATUS,
  SYNC_NOW,
  SYNC_OFF,
  TURN_OFF,
  countsLine,
} from '@/components/marketplace/cloud/copy'
import { canManageThemes } from '@/components/settings/themes/theme-labels'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useWorkspace } from '@/components/workspace-provider'
import { useCloudSync, useCloudSyncStatus } from '@/hooks/use-cloud-sync'
import { isLocal } from '@/lib/auth-edition'
import { formatRelative } from '@/lib/format-relative'
import type { CloudSyncStatus } from '@/lib/marketplace/cloud-sync'
import { themeErrorMessage } from '@/lib/theme-host/api'

/** The Sync's status, asked only on the local edition by a workspace owner or admin. */
function useCloudAccess(): { asks: boolean; status: UseQueryResult<CloudSyncStatus> } {
  const { workspace } = useWorkspace()
  const asks = isLocal && canManageThemes(workspace?.role)
  return { asks, status: useCloudSyncStatus(asks) }
}

/** Whether Settings shows Automatos Cloud: the local edition, an owner or admin, and the Sync on. */
export function useHasAutomatosCloudTab(): boolean {
  const { asks, status } = useCloudAccess()
  return asks && status.data?.enabled === true
}

function Fact({ label, value, testId }: { label: string; value: string; testId: string }) {
  return (
    <div className="flex gap-2 text-sm">
      <dt className="w-36 shrink-0 text-muted-foreground">{label}</dt>
      <dd className="min-w-0 break-words" data-testid={testId}>{value}</dd>
    </div>
  )
}

function CloudFacts({ status }: { status: CloudSyncStatus }) {
  return (
    <dl className="space-y-1">
      <Fact label="URL in use" value={status.cloudUrl ?? 'none'} testId="automatos-cloud-url" />
      <Fact label="Last success" value={formatRelative(status.lastSuccessAt)} testId="automatos-cloud-success" />
      <Fact label="Last check" value={formatRelative(status.lastAttemptAt)} testId="automatos-cloud-attempt" />
      <Fact label="Last error" value={status.lastError ?? 'none'} testId="automatos-cloud-error" />
      <Fact
        label="Last Sync"
        value={status.lastSuccessAt ? countsLine(status.counts) : NOT_SYNCED_YET}
        testId="automatos-cloud-counts"
      />
      <Fact label="Themes held here" value={String(status.themes)} testId="automatos-cloud-themes" />
    </dl>
  )
}

function SyncNow() {
  const sync = useCloudSync()
  const syncing = sync.isLoading
  return (
    <Button type="button" size="sm" variant="outline" disabled={syncing} aria-busy={syncing} onClick={() => sync.mutate()}>
      {syncing ? (
        <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" aria-hidden="true" data-testid="automatos-cloud-spinner" />
      ) : (
        <RefreshCw className="mr-1.5 h-3.5 w-3.5" aria-hidden="true" />
      )}
      {SYNC_NOW}
    </Button>
  )
}

function CloudBody({ status }: { status: UseQueryResult<CloudSyncStatus> }) {
  // A failed re-read keeps the last status (react-query keeps its data), so the facts stay.
  if (!status.data) {
    return status.isError ? (
      <p role="alert" className="text-sm text-destructive">
        {themeErrorMessage(status.error)}
      </p>
    ) : (
      <p className="text-sm text-muted-foreground">{READING_STATUS}</p>
    )
  }
  if (!status.data.enabled) {
    return <p className="text-sm text-muted-foreground" data-testid="automatos-cloud-off">{SYNC_OFF}</p>
  }
  return (
    <>
      <CloudFacts status={status.data} />
      <SyncNow />
      <p className="text-xs text-muted-foreground" data-testid="automatos-cloud-turn-off">{TURN_OFF}</p>
    </>
  )
}

/** Settings › Automatos Cloud: the local edition's Sync, for a workspace owner or admin. */
export function AutomatosCloudPanel() {
  const { asks, status } = useCloudAccess()
  if (!asks) return null
  return (
    <Card data-testid="automatos-cloud">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Cloud className="h-5 w-5" aria-hidden />
          {CLOUD_TITLE}
        </CardTitle>
        <CardDescription>{CLOUD_LEDE}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <CloudBody status={status} />
      </CardContent>
    </Card>
  )
}
