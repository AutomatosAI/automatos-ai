'use client'

/**
 * PRD-261 C-05j — the admin's theme revocations (`/admin/themes`, under the review queue): every
 * revocation with its level and origin, the form that writes one, and, in the local edition, the
 * revocation sync's status.
 *
 * A row is `local` (written here, "Set here") or `feed` (pulled from the hosted revocation list,
 * "From the Automatos feed"). A live local row can be lifted, after a confirmation. A feed row is
 * the hosted list's to lift: it has no Lift while this install follows that list (sync on and not
 * stale), and gets one only while sync is off or stale, the server's own rule
 * (`revocation_admin._feed_rows_liftable_here`), which still decides: a refusal (409) shows its
 * sentence and the list is read again.
 *
 * Both editions: the hosted edition has no sync to show (`hosted`), and its rows are all local.
 */
import { useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { ThemeRevocationRow, ThemeRevocationSyncStatus } from '@/lib/api-client'
import { formatRelative } from '@/lib/format-relative'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { RevocationForm } from './revocation-form'
import { RevocationSyncStatus } from './revocation-sync-status'
import {
  CONFIRM_LIFT,
  KEEP_IT,
  LEVEL_WORDS,
  LIFT,
  NO_REVOCATIONS,
  ORIGIN_WORDS,
  REVOCATIONS_LEDE,
  REVOCATIONS_TITLE,
  labelOf,
  revocationTargetWords,
} from './review-words'
import { useLiftRevocation, useRevocationSync, useThemeRevocations } from './use-theme-revocations'

const FEED_ORIGIN = 'feed'

/**
 * Whether a feed row may be lifted here: the hosted edition pulls nothing, and the local edition
 * only while its sync is off or stale. Unknown (the status still loading or unreadable) is no.
 */
export function feedRowsLiftable(hosted: boolean, sync: ThemeRevocationSyncStatus | undefined): boolean {
  if (hosted) return true
  return sync !== undefined && (!sync.enabled || sync.stale)
}

/** Whether `row` shows Lift: live, and written here or a feed row while feed rows are liftable. */
export function showsLift(row: ThemeRevocationRow, feedLiftable: boolean): boolean {
  if (row.liftedAt !== null) return false
  return row.origin !== FEED_ORIGIN || feedLiftable
}

function LiftButton({ revocationId }: { revocationId: string }) {
  const lift = useLiftRevocation()
  const [confirming, setConfirming] = useState(false)
  if (!confirming) {
    return (
      <Button type="button" size="sm" variant="outline" onClick={() => setConfirming(true)}>
        {LIFT}
      </Button>
    )
  }
  return (
    <div className="flex flex-wrap gap-2">
      <Button type="button" size="sm" variant="destructive" disabled={lift.isLoading} onClick={() => lift.mutate(revocationId)}>
        {CONFIRM_LIFT}
      </Button>
      <Button type="button" size="sm" variant="ghost" disabled={lift.isLoading} onClick={() => setConfirming(false)}>
        {KEEP_IT}
      </Button>
    </div>
  )
}

function RevocationItem({ row, feedLiftable }: { row: ThemeRevocationRow; feedLiftable: boolean }) {
  return (
    <li className="space-y-1 rounded-lg border border-border/40 p-2" data-testid={`theme-revocation-${row.id}`}>
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={row.level === 'disable' ? 'destructive' : 'secondary'}>{labelOf(LEVEL_WORDS, row.level)}</Badge>
        <Badge variant="outline" data-testid={`theme-revocation-origin-${row.id}`}>{labelOf(ORIGIN_WORDS, row.origin)}</Badge>
        {row.liftedAt !== null && <Badge variant="secondary">Lifted {formatRelative(row.liftedAt)}</Badge>}
      </div>
      <p className="break-all font-mono text-xs">{revocationTargetWords(row)}</p>
      <p className="text-sm">{row.reason}</p>
      <p className="text-xs text-muted-foreground">Revoked {formatRelative(row.createdAt)}</p>
      {showsLift(row, feedLiftable) && <LiftButton revocationId={row.id} />}
    </li>
  )
}

function RevocationList({ feedLiftable }: { feedLiftable: boolean }) {
  const list = useThemeRevocations()
  if (list.isLoading) return <p className="text-xs text-muted-foreground">Loading the revocations…</p>
  if (list.isError || !list.data) {
    return (
      <p role="alert" className="text-xs text-destructive">
        {themeErrorMessage(list.error)}
      </p>
    )
  }
  if (list.data.revocations.length === 0) return <p className="text-xs text-muted-foreground">{NO_REVOCATIONS}</p>
  return (
    <ul className="space-y-2" aria-label="Revocations">
      {list.data.revocations.map((row) => (
        <RevocationItem key={row.id} row={row} feedLiftable={feedLiftable} />
      ))}
    </ul>
  )
}

export function RevocationsPanel({ hosted }: { hosted: boolean }) {
  const sync = useRevocationSync(!hosted)
  return (
    <section aria-label={REVOCATIONS_TITLE} className="space-y-3" data-testid="theme-revocations">
      <div className="space-y-1">
        <h2 className="text-lg font-semibold">{REVOCATIONS_TITLE}</h2>
        <p className="text-xs text-muted-foreground">{REVOCATIONS_LEDE}</p>
      </div>
      <RevocationSyncStatus hosted={hosted} sync={sync} />
      <RevocationForm />
      <RevocationList feedLiftable={feedRowsLiftable(hosted, sync.data)} />
    </section>
  )
}
