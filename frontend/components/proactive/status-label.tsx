'use client'

/**
 * PRD-268 §6.11.1, §6.12.2, §6.12.3 (US-C21c) — a row's status: the glyph and the word, the
 * running step (`▸ next_step`) and Try again when the row can be retried.
 *
 * The server computes the state; this only draws DISPLAY_STATE_LABELS for it, and an unknown state
 * draws nothing (never a placeholder). The glyph and the word always carry the meaning, so colour is
 * never the only signal. In Studio the pill is `studio-pill`, painted by the role's token per ground
 * tone (STATUS_ROLE_TOKENS: needs-you is warn-ink on a light ground, warning on a dark one); it never
 * goes through `--accent` and never sits on a tinted fill. Classic draws Classic's own status badge
 * (components/shared/status-badge.tsx), so its tokens decide. Try again is drawn only when the row
 * says what retry means and the surface can run it.
 */
import { StatusBadge, type StatusVariant } from '@/components/shared/status-badge'
import { useIsStudio } from '@/hooks/use-studio-theme'
import { NEXT_STEP_GLYPH, displayStateLabel, type StatusRole } from '@/lib/proactive/display-state'
import type { Retry } from '@/lib/proactive/types'
import { cn } from '@/lib/utils'

const RETRIES: ReadonlySet<string> = new Set(['rerun_mission', 'replan_mission', 'retry_ticket'])

/**
 * Studio: each role's token, as literal classes (Tailwind reads them from this file). The light
 * token paints by default and the dark one under `.dark`, as STATUS_ROLE_TOKENS names them.
 */
export const STUDIO_ROLE_CLASS: Readonly<Record<StatusRole, string>> = Object.freeze({
  needs_you: 'text-[hsl(var(--warn-ink))] dark:text-[hsl(var(--warning))]',
  working: 'text-[hsl(var(--running))] dark:text-[hsl(var(--running))]',
  ready: 'text-[hsl(var(--success))] dark:text-[hsl(var(--success))]',
  done: 'text-[hsl(var(--success))] dark:text-[hsl(var(--success))]',
  failed: 'text-[hsl(var(--destructive))] dark:text-[hsl(var(--destructive))]',
  needs_setup: 'text-[hsl(var(--info))] dark:text-[hsl(var(--info))]',
  not_checked: 'text-[hsl(var(--muted-foreground))] dark:text-[hsl(var(--muted-foreground))]',
})

/** Classic: its status badge's variant per role (working is Classic's info until it gains running). */
export const CLASSIC_ROLE_VARIANT: Readonly<Record<StatusRole, StatusVariant>> = Object.freeze({
  needs_you: 'warning',
  working: 'active',
  ready: 'success',
  done: 'success',
  failed: 'error',
  needs_setup: 'info',
  not_checked: 'neutral',
})

export interface StatusLabelProps {
  /** The row's `display_state`; anything that isn't one of §6.11.1's states draws nothing. */
  readonly state: unknown
  /** The running step, at most 90 characters, in the person's language. */
  readonly nextStep?: string | null
  readonly retry?: Retry | null
  /** What Try again runs; without it, Try again is not drawn. */
  readonly onRetry?: (retry: Retry) => void
  /** The row's title, which names Try again for a screen reader ("Try again: Lisbon flights"). */
  readonly title?: string
  readonly className?: string
}

function Pill({ state }: { readonly state: unknown }) {
  const isStudio = useIsStudio()
  const label = displayStateLabel(state)
  if (label === null) return null
  const words = (
    <>
      <span aria-hidden="true">{label.glyph}</span>
      <span>{label.word}</span>
    </>
  )
  if (!isStudio) {
    return (
      <StatusBadge status={CLASSIC_ROLE_VARIANT[label.role]} size="sm" className="gap-1">
        {words}
      </StatusBadge>
    )
  }
  return (
    <span data-status-role={label.role} className={cn('studio-pill gap-1 border border-border bg-transparent', STUDIO_ROLE_CLASS[label.role])}>
      {words}
    </span>
  )
}

/** The glyph and the word for a row's state, its next step and Try again; nothing for an unknown state. */
export function StatusLabel({ state, nextStep, retry, onRetry, title, className }: StatusLabelProps) {
  if (displayStateLabel(state) === null) return null
  const canRetry = typeof retry === 'string' && RETRIES.has(retry) && onRetry !== undefined
  return (
    <span data-display-state={String(state)} className={cn('inline-flex flex-wrap items-center gap-2', className)}>
      <Pill state={state} />
      {nextStep ? (
        <span className="text-xs text-muted-foreground">
          <span aria-hidden="true">{NEXT_STEP_GLYPH} </span>
          {nextStep}
        </span>
      ) : null}
      {canRetry ? (
        <button
          type="button"
          aria-label={title ? `Try again: ${title}` : 'Try again'}
          onClick={() => onRetry?.(retry as Retry)}
          className="rounded-md text-xs font-medium text-foreground underline underline-offset-4 hover:text-foreground/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          Try again
        </button>
      ) : null}
    </span>
  )
}
