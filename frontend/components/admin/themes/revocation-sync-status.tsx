'use client'

/**
 * PRD-261 C-05j — the local edition's revocation sync (C-05h): when this install last checked the
 * Automatos revocation list, what went wrong, how many of its rows it holds and the keys it revoked,
 * and Check now (`POST /api/admin/themes/revocations/sync/pull`).
 *
 * Local edition only: the hosted edition serves the feed and pulls nothing (both sync routes are
 * 404 there), so the panel neither shows this nor asks for it. `stale` is the server's own rule
 * (`revocation_pull.is_stale`: no successful pull in 7 days, or none yet), the same one that lets an
 * admin lift a feed row here; the warning shows only while sync is on.
 */
import type { UseQueryResult } from '@tanstack/react-query'

import { Button } from '@/components/ui/button'
import type { ThemeRevocationSyncStatus } from '@/lib/api-client'
import { formatRelative } from '@/lib/format-relative'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { CHECK_NOW, SYNC_OFF, SYNC_STALE, SYNC_TITLE } from './review-words'
import { usePullRevocations } from './use-theme-revocations'

function Fact({ label, value, testId }: { label: string; value: string; testId: string }) {
  return (
    <div className="flex gap-2 text-xs">
      <dt className="w-32 shrink-0 text-muted-foreground">{label}</dt>
      <dd className="min-w-0 break-words" data-testid={testId}>{value}</dd>
    </div>
  )
}

function SyncFacts({ status }: { status: ThemeRevocationSyncStatus }) {
  const keys = status.revokedKeys.length > 0 ? status.revokedKeys.join(', ') : 'none'
  return (
    <dl className="space-y-1">
      <Fact label="Last success" value={formatRelative(status.lastSuccessAt)} testId="revocation-sync-success" />
      <Fact label="Last check" value={formatRelative(status.lastAttemptAt)} testId="revocation-sync-attempt" />
      <Fact label="Last error" value={status.lastError ?? 'none'} testId="revocation-sync-error" />
      <Fact label="List made" value={formatRelative(status.feedGeneratedAt)} testId="revocation-sync-generated" />
      <Fact label="Rows held here" value={String(status.rows)} testId="revocation-sync-rows" />
      <Fact label="Revoked keys" value={keys} testId="revocation-sync-keys" />
    </dl>
  )
}

function CheckNow({ enabled }: { enabled: boolean }) {
  const pull = usePullRevocations()
  return (
    <div className="space-y-1">
      <Button type="button" size="sm" variant="outline" disabled={!enabled || pull.isLoading} onClick={() => pull.mutate()}>
        {CHECK_NOW}
      </Button>
      {pull.error ? (
        <p role="alert" className="text-xs text-destructive" data-testid="revocation-sync-pull-error">
          {themeErrorMessage(pull.error)}
        </p>
      ) : null}
    </div>
  )
}

function SyncBody({ sync }: { sync: UseQueryResult<ThemeRevocationSyncStatus> }) {
  if (sync.isLoading) return <p className="text-xs text-muted-foreground">Reading the revocation sync…</p>
  if (sync.isError || !sync.data) {
    return (
      <p role="alert" className="text-xs text-destructive">
        {themeErrorMessage(sync.error)}
      </p>
    )
  }
  const status = sync.data
  return (
    <>
      {!status.enabled && <p className="text-xs text-muted-foreground" data-testid="revocation-sync-off">{SYNC_OFF}</p>}
      {status.enabled && status.stale && (
        <p className="text-xs text-warning" data-testid="revocation-sync-stale">{SYNC_STALE}</p>
      )}
      <SyncFacts status={status} />
      <CheckNow enabled={status.enabled} />
    </>
  )
}

export function RevocationSyncStatus({ hosted, sync }: { hosted: boolean; sync: UseQueryResult<ThemeRevocationSyncStatus> }) {
  if (hosted) return null
  return (
    <section aria-label={SYNC_TITLE} className="space-y-2 rounded-lg border border-border/50 p-3" data-testid="revocation-sync">
      <p className="text-sm font-medium">{SYNC_TITLE}</p>
      <SyncBody sync={sync} />
    </section>
  )
}
