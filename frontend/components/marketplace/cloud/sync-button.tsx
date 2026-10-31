'use client'

/**
 * PRD-270 US-270-05a (§5.5, K18) — Sync with Automatos Cloud in the Marketplace header: the button,
 * and beside it the status line ("Last synced 2m ago · 3 themes · 2 agents", or the last error in plain words).
 * While a Sync runs the button is disabled and shows a spinner; after it, the toast says what it
 * brought (hooks/use-cloud-sync.ts).
 *
 * Shown only on the local edition (by edition, never by role: lib/auth-edition.ts), to a workspace
 * owner or admin (`canManageThemes`, the routes' `workspace:manage`), while the status says the Sync
 * is on (`AUTOMATOS_CLOUD_URL` set). On the hosted edition nothing mounts, so nothing is asked; a
 * member asks nothing either. The server enforces every gate itself. The button is secondary: the
 * Marketplace's header has no ink primary of its own (DESIGN_SYSTEM.md §22).
 */
import { Loader2, RefreshCw } from 'lucide-react'

import { canManageThemes } from '@/components/settings/themes/theme-labels'
import { Button } from '@/components/ui/button'
import { useWorkspace } from '@/components/workspace-provider'
import { useCloudSync, useCloudSyncStatus } from '@/hooks/use-cloud-sync'
import { isLocal } from '@/lib/auth-edition'
import type { CloudSyncStatus } from '@/lib/marketplace/cloud-sync'

import { SYNCING, SYNC_WITH_CLOUD, syncedLine } from './copy'

function SyncStatusLine({ status }: { status: CloudSyncStatus }) {
  if (status.lastError) {
    return (
      <p className="text-xs text-destructive" data-testid="cloud-sync-error">
        {status.lastError}
      </p>
    )
  }
  return (
    <p className="text-xs text-muted-foreground" data-testid="cloud-sync-status">
      {syncedLine(status)}
    </p>
  )
}

function CloudSyncControls() {
  const { workspace } = useWorkspace()
  const canManage = canManageThemes(workspace?.role)
  const status = useCloudSyncStatus(canManage)
  const sync = useCloudSync()
  if (!canManage || !status.data?.enabled) return null
  const syncing = sync.isLoading
  return (
    <div className="flex flex-wrap items-center gap-2" data-testid="cloud-sync">
      <SyncStatusLine status={status.data} />
      <Button
        type="button"
        size="sm"
        variant="outline"
        disabled={syncing}
        aria-busy={syncing}
        title={syncing ? SYNCING : undefined}
        onClick={() => sync.mutate()}
      >
        {syncing ? (
          <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" aria-hidden="true" data-testid="cloud-sync-spinner" />
        ) : (
          <RefreshCw className="mr-1.5 h-3.5 w-3.5" aria-hidden="true" />
        )}
        {SYNC_WITH_CLOUD}
      </Button>
    </div>
  )
}

/** Sync with Automatos Cloud, for the Marketplace header's actions: the local edition only. */
export function SyncButton() {
  return isLocal ? <CloudSyncControls /> : null
}
