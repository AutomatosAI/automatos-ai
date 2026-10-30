'use client'

/**
 * PRD-268 §6.11.1, §6.12.3 (US-W04) — a work row's status as StatusLabel draws it: the glyph and the word for its
 * `display_state`, ▸ its `next_step`, and Try again when its `retry` is one this row can run.
 *
 * The needs-you widget, the Missions list (Studio's rows and table, Classic's cards) and the board cards (Classic's
 * and Studio's) all draw a row's status through this one component, from the fields the server put on the row
 * (rowStatusOf checks them off the wire). An unknown state draws nothing. Try again runs through useRowRetry, which
 * is mounted only on a row that offers it, so a row without one needs nothing more than StatusLabel does; while its
 * retry is on its way, Try again waits.
 */
import { displayStateLabel } from '@/lib/proactive/display-state'
import { rowStatusOf } from '@/lib/proactive/row-status'
import type { Retry, RowStatus } from '@/lib/proactive/types'
import { StatusLabel } from './status-label'
import { canRetry, useRowRetry, type RetryTarget } from './use-row-retry'

export interface RowStatusLabelProps {
  /** The row as the API sent it; only its status fields are read. */
  readonly row: unknown
  /** What Try again acts on. */
  readonly target: RetryTarget
  /** The row's title: it names Try again ("Try again: Lisbon flights"). */
  readonly title?: string
  readonly className?: string
}

interface DrawnProps {
  readonly status: Required<RowStatus>
  readonly target: RetryTarget
  readonly title?: string
  readonly className?: string
}

function RetryableStatusLabel({ status, target, title, className }: DrawnProps) {
  const { run, busy } = useRowRetry()
  return (
    <StatusLabel
      state={status.display_state}
      nextStep={status.next_step}
      retry={status.retry}
      onRetry={(retry: Retry) => run(retry, target)}
      retrying={busy}
      title={title}
      className={className}
    />
  )
}

/** A row's glyph, word, next step and Try again; nothing when the row's state is unknown. */
export function RowStatusLabel({ row, target, title, className }: RowStatusLabelProps) {
  const status = rowStatusOf(row)
  if (displayStateLabel(status.display_state) === null) return null
  if (status.retry !== null && canRetry(status.retry, target)) {
    return <RetryableStatusLabel status={status} target={target} title={title} className={className} />
  }
  return <StatusLabel state={status.display_state} nextStep={status.next_step} title={title} className={className} />
}
