'use client'

/**
 * PRD-268 §6.11.3, §6.12.3, K17 (US-C21d) — what Auto depends on: each item's label, its detail, its
 * state as a glyph and a word (✓ Ready, ○ Not checked, ◆ Needs setup), its fix as an in-app link,
 * and Check again.
 *
 * The surface passes Readiness@1 as it reads it (useReadiness) and any rows only the client knows
 * (the browser's microphone and speech, which the server never claims). Check again posts
 * POST /api/readiness/check (useCheckReadiness); while it runs, or while nothing has been read yet,
 * the list says it is checking, and a read that failed (`error`) gets one line in its place. A check over the server's limit is refused with checked_recently and
 * the list says "Checked a moment ago"; any other refusal gets one line. The newest answer is the
 * one shown. Each row reads as one sentence: "Calendar: needs setup. Connect calendar."
 */
import Link from 'next/link'
import { useId } from 'react'
import { checkedRecently, useCheckReadiness } from '@/hooks/use-readiness'
import { readinessStateLabel } from '@/lib/proactive/display-state'
import type { Readiness, ReadinessItem } from '@/lib/proactive/types'
import { appPath } from '@/lib/proactive/wire'
import { cn } from '@/lib/utils'
import { CARD_CLASS, ENTRANCE_CLASS, Eyebrow, QUIET_BUTTON_CLASS } from './card-parts'
import { StatePill } from './status-label'
import { refusedLine } from './use-suggestion-card'

export const READINESS_TITLE = 'What Auto depends on'
export const CHECK_AGAIN = 'Check again'
export const CHECKING = 'Checking…'
export const CHECKED_A_MOMENT_AGO = 'Checked a moment ago'

const FIX_LINK_CLASS =
  'rounded-md text-sm font-medium text-foreground underline underline-offset-4 hover:text-foreground/80 ' +
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

export interface ReadinessListProps {
  /** Readiness@1 as read (useReadiness); null or undefined while it loads. */
  readonly readiness: Readiness | null | undefined
  /** Why the read failed (useReadiness's error): one line in place of "Checking…". */
  readonly error?: unknown
  /** Rows only the client knows (the browser's microphone and speech), drawn after the server's. */
  readonly clientItems?: readonly ReadinessItem[]
  readonly className?: string
}

/** The row as one sentence: "Calendar: needs setup. Connect calendar." */
export function readinessSentence(item: ReadinessItem): string {
  const word = readinessStateLabel(item.state)?.word.toLowerCase() ?? ''
  return item.fix ? `${item.label}: ${word}. ${item.fix.label}.` : `${item.label}: ${word}.`
}

/**
 * Check again's answer, unless the one read is later by checked_at (a refetch after it); when either
 * time can't be read, Check again's. The one present when only one is.
 */
export function newestReadiness(read: Readiness | null | undefined, checked: Readiness | undefined): Readiness | null {
  if (!checked) return read ?? null
  if (!read) return checked
  const readAt = Date.parse(read.checked_at)
  const checkedAt = Date.parse(checked.checked_at)
  return Number.isNaN(readAt) || Number.isNaN(checkedAt) || checkedAt >= readAt ? checked : read
}

function ReadinessRow({ item }: { readonly item: ReadinessItem }) {
  const detailId = useId()
  const label = readinessStateLabel(item.state)
  if (label === null) return null
  const fix = item.fix && appPath(item.fix.href) ? item.fix : null
  return (
    <li
      data-readiness-id={item.id}
      data-state={item.state}
      aria-label={readinessSentence({ ...item, fix })}
      aria-describedby={item.detail ? detailId : undefined}
      className="flex flex-wrap items-start justify-between gap-2 py-2"
    >
      <div className="min-w-0 space-y-0.5">
        <p className="text-sm font-medium text-foreground">{item.label}</p>
        {item.detail ? (
          <p id={detailId} className="text-xs text-muted-foreground">
            {item.detail}
          </p>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <StatePill label={label} />
        {fix ? (
          <Link href={fix.href} className={FIX_LINK_CLASS}>
            {fix.label}
          </Link>
        ) : null}
      </div>
    </li>
  )
}

type ReadinessCheck = ReturnType<typeof useCheckReadiness>

/** Check again, and what became of the last one: "Checked a moment ago", or the refusal's one line. */
function CheckAgain({ check }: { readonly check: ReadinessCheck }) {
  const limited = check.isError && checkedRecently(check.error)
  const refused = check.isError && !limited ? refusedLine(check.error) : null
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" aria-busy={check.isLoading} disabled={check.isLoading} onClick={() => check.mutate()} className={QUIET_BUTTON_CLASS}>
        {check.isLoading ? CHECKING : CHECK_AGAIN}
      </button>
      {limited ? (
        <p role="status" className="text-xs text-muted-foreground">
          {CHECKED_A_MOMENT_AGO}
        </p>
      ) : null}
      {refused ? (
        <p role="alert" className="text-xs text-[hsl(var(--destructive))]">
          {refused}
        </p>
      ) : null}
    </div>
  )
}

/** The items Auto depends on, each with its state and fix, and Check again. */
export function ReadinessList({ readiness, error, clientItems = [], className }: ReadinessListProps) {
  const check = useCheckReadiness()
  const shown = newestReadiness(readiness, check.data)
  const items = shown ? [...shown.items, ...clientItems] : []
  const failed = shown === null && error ? refusedLine(error) : null
  const checking = (shown === null && failed === null) || check.isLoading
  return (
    <section aria-label={READINESS_TITLE} aria-busy={checking} className={cn(CARD_CLASS, ENTRANCE_CLASS, 'space-y-3', className)}>
      <Eyebrow>{READINESS_TITLE}</Eyebrow>
      {failed ? (
        <p role="alert" className="text-sm text-[hsl(var(--destructive))]">
          {failed}
        </p>
      ) : null}
      {shown === null && failed === null ? (
        <p role="status" className="text-sm text-muted-foreground">
          {CHECKING}
        </p>
      ) : null}
      {shown ? (
        <ul className="divide-y divide-border">
          {items.map((item, index) => (
            <ReadinessRow key={`${index}:${item.id}`} item={item} />
          ))}
        </ul>
      ) : null}
      <CheckAgain check={check} />
    </section>
  )
}
