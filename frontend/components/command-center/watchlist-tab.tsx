'use client'

/**
 * WatchlistTab -- PRD-204 S11 (Auto Watcher).
 *
 * The minimal watchlist surface: every launched unit of work Auto is
 * supervising to a verdict, one row each. Title links to the watched
 * mission / playbook execution; status and score come straight off the
 * watch registry (scores are 0-1 internally, displayed x10 by the
 * backend's `final_score_display`); a live watch can be cancelled via the
 * house confirm primitive (PRD-169 -- no window.confirm). A row expands to
 * its recent watch_events timeline (created / terminal / scored /
 * diagnosed / action ...), newest first.
 *
 * The board and the notification bell stay the ACTION surfaces -- this
 * panel is read-and-cancel only, by design (PRD-204 S11).
 */

import { useState } from 'react'
import Link from 'next/link'
import { formatDistanceToNow } from 'date-fns'
import { ChevronDown, ChevronRight, Eye } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { DeleteConfirmation } from '@/components/shared/delete-confirmation'
import {
  useCancelWatch,
  useWatchDetail,
  useWatches,
} from '@/hooks/use-watches-api'
import type { WatchRow } from '@/lib/api-client'

const LIVE_STATUSES = new Set([
  'watching',
  'acting',
  'awaiting_approval',
  'needs_attention',
])

export function isLiveWatch(status: string): boolean {
  return LIVE_STATUSES.has(status)
}

/** Design-token tone per watch status (raw palette classes are banned). */
export function statusToneClass(status: string): string {
  switch (status) {
    case 'passed':
      return 'text-success border-success'
    case 'failed':
    case 'expired':
      return 'text-destructive border-destructive'
    case 'awaiting_approval':
    case 'needs_attention':
    case 'escalated':
      return 'text-warning border-warning'
    case 'cancelled':
      return 'text-muted-foreground border-border'
    default:
      // watching / acting -- healthy, in flight
      return 'text-primary border-primary'
  }
}

/** Where a watch row links: the board card / execution it supervises. */
export function targetHref(watch: WatchRow): string | null {
  switch (watch.target_type) {
    case 'mission':
      return `/assignments?tab=missions&mission=${watch.target_id}`
    case 'playbook_execution':
      return `/assignments?tab=playbooks&execution=${watch.target_id}`
    case 'scheduled_playbook':
      return '/assignments?tab=playbooks'
    case 'board_task':
      return '/command-center?tab=board'
    default:
      return null
  }
}

function relativeLabel(iso: string | null): string {
  if (!iso) return 'never'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return 'never'
  return formatDistanceToNow(d, { addSuffix: true })
}

function Chip({ tone, children }: { tone: string; children: React.ReactNode }) {
  return (
    <span
      className={`inline-block whitespace-nowrap rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${tone}`}
    >
      {children}
    </span>
  )
}

/** Recent watch_events timeline, newest first (loaded on expand). */
function WatchTimeline({ watchId }: { watchId: string }) {
  const { data, isLoading } = useWatchDetail(watchId)

  if (isLoading) {
    return <p className="px-2 py-1.5 text-xs text-muted-foreground">Loading events...</p>
  }
  const events = data?.recent_events ?? []
  if (events.length === 0) {
    return <p className="px-2 py-1.5 text-xs text-muted-foreground">No events yet.</p>
  }
  return (
    <ol className="flex flex-col gap-1 px-2 py-1.5" aria-label="Watch events">
      {events.map((e, i) => (
        <li key={`${e.event_type}-${e.created_at ?? i}`} className="flex items-baseline gap-2 text-xs">
          <span
            className={`shrink-0 font-medium uppercase tracking-wide ${
              e.requires_attention ? 'text-warning' : 'text-muted-foreground'
            }`}
          >
            {e.event_type.replace(/_/g, ' ')}
          </span>
          <span className="min-w-0 flex-1 truncate text-muted-foreground">
            {e.summary ?? ''}
          </span>
          <span className="shrink-0 text-muted-foreground/70">
            {relativeLabel(e.created_at)}
          </span>
        </li>
      ))}
    </ol>
  )
}

function WatchRowItem({ watch }: { watch: WatchRow }) {
  const cancelMut = useCancelWatch()
  const [confirming, setConfirming] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const live = isLiveWatch(watch.status)
  const href = targetHref(watch)
  const scoreLabel =
    watch.final_score != null ? watch.final_score_display : 'unscored'

  const handleCancel = async () => {
    try {
      await cancelMut.mutateAsync(watch.id)
      toast.success('Watch cancelled')
    } catch {
      toast.error('Failed to cancel the watch')
      throw new Error('cancel failed') // keep the confirm dialog open
    }
  }

  return (
    <>
      <tr className="border-b border-border/30">
        <td className="py-2 pr-2 align-top">
          <button
            type="button"
            className="mt-0.5 text-muted-foreground hover:text-foreground"
            aria-label={expanded ? 'Hide events' : 'Show events'}
            aria-expanded={expanded}
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </button>
        </td>
        <td className="max-w-0 w-full py-2 pr-3 align-top">
          {href ? (
            <Link href={href as any} className="block truncate text-sm font-medium hover:underline">
              {watch.title}
            </Link>
          ) : (
            <span className="block truncate text-sm font-medium">{watch.title}</span>
          )}
          {watch.final_verdict && (
            <p className="mt-0.5 truncate text-xs text-muted-foreground">{watch.final_verdict}</p>
          )}
        </td>
        <td className="py-2 pr-3 align-top">
          <Chip tone="text-muted-foreground border-border">
            {watch.watch_type.replace(/_/g, ' ')}
          </Chip>
        </td>
        <td className="py-2 pr-3 align-top">
          <Chip tone={statusToneClass(watch.status)}>
            {watch.status.replace(/_/g, ' ')}
          </Chip>
        </td>
        <td className="whitespace-nowrap py-2 pr-3 align-top text-xs text-muted-foreground">
          {relativeLabel(watch.last_checked_at)}
        </td>
        <td className="whitespace-nowrap py-2 pr-3 align-top text-xs">
          <span className={watch.final_score != null ? 'font-medium' : 'text-muted-foreground'}>
            {scoreLabel}
          </span>
        </td>
        <td className="py-2 align-top">
          {live && (
            <Button
              size="sm"
              variant="outline"
              disabled={cancelMut.isLoading}
              onClick={() => setConfirming(true)}
            >
              Cancel
            </Button>
          )}
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-border/30 bg-background/50">
          <td />
          <td colSpan={6}>
            <WatchTimeline watchId={watch.id} />
          </td>
        </tr>
      )}
      <DeleteConfirmation
        open={confirming}
        onOpenChange={setConfirming}
        title="Stop watching this work?"
        description={`Auto will stop supervising "${watch.title}". The underlying work keeps running.`}
        confirmLabel="Stop watching"
        cancelLabel="Keep watching"
        onConfirm={handleCancel}
        loading={cancelMut.isLoading}
      />
    </>
  )
}

export function WatchlistTab() {
  const [includeClosed, setIncludeClosed] = useState(false)
  const { data, isLoading, isError } = useWatches(includeClosed)

  const watches = data?.watches ?? []

  if (isLoading) {
    return <p className="p-3 text-sm text-muted-foreground">Loading watchlist...</p>
  }
  if (isError) {
    return <p className="p-3 text-sm text-muted-foreground">Could not load the watchlist.</p>
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Watchlist {watches.length > 0 && <span className="text-foreground">({watches.length})</span>}
        </h3>
        <label className="flex cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
          <input
            type="checkbox"
            checked={includeClosed}
            onChange={(e) => setIncludeClosed(e.target.checked)}
          />
          Show closed
        </label>
      </div>

      {watches.length === 0 ? (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <Eye className="h-4 w-4" /> Nothing being watched.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-border text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                <th className="w-6 py-1.5 pr-2" aria-label="Expand" />
                <th className="py-1.5 pr-3">Watch</th>
                <th className="py-1.5 pr-3">Type</th>
                <th className="py-1.5 pr-3">Status</th>
                <th className="py-1.5 pr-3">Last check</th>
                <th className="py-1.5 pr-3">Score</th>
                <th className="w-20 py-1.5" aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {watches.map((w) => (
                <WatchRowItem key={w.id} watch={w} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
