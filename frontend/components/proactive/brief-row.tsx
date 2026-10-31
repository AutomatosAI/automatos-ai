'use client'

/**
 * PRD-268 §6.11.2, §6.12.3 (US-C21c) — one brief row: its title and why (Auto's words, inside the
 * fence), its time and status, the row action drawn as text ("Start →") and, when the row has
 * somewhere to go, a chevron. A row without `opens` draws no chevron.
 *
 * The action is named by its row for a screen reader ("Start: Lisbon flights"); once pressed it
 * shows "✓ {done_label}" in place. The controls sit outside the fence of Auto's words, and each is
 * drawn only when the surface can run it: with no `onAction` a row shows no action, and with no
 * `onOpen` no chevron, so nothing on the card looks pressable that isn't.
 */
import { ChevronRight } from 'lucide-react'
import type { BriefRow, BriefSection } from '@/lib/proactive/types'
import { TEXT_BUTTON_CLASS } from './card-parts'
import { StatusLabel } from './status-label'

/** The glyph a pressed row's action shows before its done label. */
export const PRESSED_GLYPH = '✓'

export interface BriefRowHandlers {
  /** A row's action was pressed; resolve to show "✓ {done_label}" in place, reject to leave the row as it was. */
  readonly onAction?: (row: BriefRow, section: BriefSection) => unknown
  /** A row with somewhere to go was opened. */
  readonly onOpen?: (row: BriefRow, section: BriefSection) => void
}

export interface BriefRowItemProps extends BriefRowHandlers {
  readonly row: BriefRow
  readonly section: BriefSection
  readonly pressed: boolean
  readonly pending: boolean
  readonly onPress: (row: BriefRow, section: BriefSection) => void
}

function timeOf(at: string | null | undefined): string | null {
  if (!at) return null
  const when = new Date(at)
  return Number.isNaN(when.getTime()) ? null : when.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function RowAction({ row, section, pressed, pending, onAction, onPress }: BriefRowItemProps) {
  const action = row.action
  if (!action) return null
  if (pressed) {
    return (
      <span role="status" className="shrink-0 text-sm font-medium text-foreground">
        <span aria-hidden="true">{PRESSED_GLYPH} </span>
        {action.done_label || action.label}
      </span>
    )
  }
  if (!onAction) return null
  return (
    <button
      type="button"
      aria-label={`${action.label}: ${row.title}`}
      aria-busy={pending}
      disabled={pending}
      onClick={() => onPress(row, section)}
      className={`shrink-0 ${TEXT_BUTTON_CLASS}`}
    >
      {action.label} →
    </button>
  )
}

function RowChevron({ row, section, onOpen }: Pick<BriefRowItemProps, 'row' | 'section' | 'onOpen'>) {
  if (!row.opens || !onOpen) return null
  return (
    <button
      type="button"
      data-row-chevron=""
      aria-label={`Open: ${row.title}`}
      onClick={() => onOpen(row, section)}
      className="shrink-0 rounded-md p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <ChevronRight aria-hidden="true" className="h-4 w-4" />
    </button>
  )
}

/** A row: Auto's words, then its time and status, its action as text, and the chevron when it opens something. */
export function BriefRowItem(props: BriefRowItemProps) {
  const { row } = props
  const time = timeOf(row.at)
  return (
    <li data-row-id={row.id} className="flex items-start gap-3 py-2">
      <div className="min-w-0 flex-1 space-y-1">
        <div data-ai-generated="true">
          <p className="text-sm font-medium text-foreground">
            {row.number ? <span className="mr-1.5 font-mono text-xs text-muted-foreground">{row.number}</span> : null}
            {row.title}
          </p>
          {row.why ? <p className="text-sm text-muted-foreground">{row.why}</p> : null}
        </div>
        {time || row.display_state ? (
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            {time ? <time dateTime={row.at ?? undefined}>{time}</time> : null}
            <StatusLabel state={row.display_state} />
          </div>
        ) : null}
      </div>
      <RowAction {...props} />
      <RowChevron row={row} section={props.section} onOpen={props.onOpen} />
    </li>
  )
}
