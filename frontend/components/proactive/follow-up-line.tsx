'use client'

/**
 * PRD-268 §6.3 'In the thread', §6.12.4, K18, K19 (US-W03b) — what Auto promised to check, under the message that
 * promised it.
 *
 * thread-cards.tsx draws these under an assistant message whose turn scheduled follow-ups (its live `followUps`, else
 * its saved `follow_up@1` parts: lib/chat/turn-parts.ts). Each line reads where its follow-up stands from the caller's
 * own list (hooks/use-follow-ups.ts): "Follow-up at 15:10 · Cancel" while pending, "Cancelled" once cancelled, and
 * "Followed up at 15:10" once fired, the time in the viewer's own locale and zone. A follow-up the list does not hold
 * (another person's, or one the server no longer has) draws nothing. A refused cancel says why in one line, as a
 * refused Start does (suggestion-parts.tsx RefusedLine): the server's sentence, else the cards' own. The label is a
 * studio-eyebrow and Cancel a text button: the composer's Send stays the view's ink primary. The labels are the
 * host's own (K18); the follow-up's reason is never drawn here.
 */
import { useCancelFollowUp, useFollowUps } from '@/hooks/use-follow-ups'
import type { FollowUp } from '@/lib/proactive/follow-ups'
import { cn } from '@/lib/utils'
import type { TurnFollowUp } from '@/types'
import { Eyebrow, TEXT_BUTTON_CLASS } from './card-parts'
import { RefusedLine } from './suggestion-parts'
import { refusedLine } from './use-suggestion-card'

/** The avatar's width and gap (w-8 + space-x-3), so a line lines up with the message's bubble. */
const ALIGNED_CLASS = 'max-w-[92%] space-y-1 pl-11'
const TIME_OPTIONS: Intl.DateTimeFormatOptions = { hour: '2-digit', minute: '2-digit' }
export const CANCEL_LABEL = 'Cancel'
export const CANCELLED_LABEL = 'Cancelled'
export const pendingLabel = (time: string) => `Follow-up at ${time}`
export const firedLabel = (time: string) => `Followed up at ${time}`

/** HH:MM in the viewer's own locale and time zone. */
export function followUpTime(at: string): string {
  return new Date(at).toLocaleTimeString(undefined, TIME_OPTIONS)
}

function stateLabel(status: FollowUp['status'], time: string): string {
  if (status === 'cancelled') return CANCELLED_LABEL
  return status === 'fired' ? firedLabel(time) : pendingLabel(time)
}

interface FollowUpLineProps {
  readonly chatId: string
  readonly followUp: FollowUp
  /** When it fires: the list's time, else the one the message's part names. */
  readonly at: string
}

/** One follow-up's line, with Cancel while it is pending and the reason a refused cancel gives. */
function FollowUpLine({ chatId, followUp, at }: FollowUpLineProps) {
  const cancel = useCancelFollowUp(chatId)
  const time = followUpTime(at)
  return (
    <div data-testid="follow-up-line" data-follow-up-status={followUp.status}>
      <div className="flex flex-wrap items-center gap-x-2" aria-live="polite">
        <Eyebrow>{stateLabel(followUp.status, time)}</Eyebrow>
        {followUp.status === 'pending' ? (
          <>
            <span aria-hidden="true" className="text-xs text-muted-foreground">
              ·
            </span>
            <button
              type="button"
              className={cn(TEXT_BUTTON_CLASS, 'text-xs')}
              aria-label={`${CANCEL_LABEL} the follow-up at ${time}`}
              disabled={cancel.isLoading}
              onClick={() => cancel.mutate(followUp.id)}
            >
              {CANCEL_LABEL}
            </button>
          </>
        ) : null}
      </div>
      <RefusedLine text={refusedLine(cancel.error)} />
    </div>
  )
}

export interface FollowUpLinesProps {
  readonly chatId: string
  /** The follow-ups the message's turn scheduled, as its parts name them. */
  readonly followUps: readonly TurnFollowUp[]
}

/** A line for each follow-up the message's turn scheduled that the caller's list holds; nothing when it holds none. */
export function FollowUpLines({ chatId, followUps }: FollowUpLinesProps) {
  const { data } = useFollowUps(chatId)
  const held = followUps.flatMap((named) => {
    const row = data?.find((followUp) => followUp.id === named.followUpId)
    return row ? [{ row, at: row.at ?? named.at }] : []
  })
  if (held.length === 0) return null
  return (
    <div className={ALIGNED_CLASS} data-testid="follow-up-lines">
      {held.map(({ row, at }) => (
        <FollowUpLine key={row.id} chatId={chatId} followUp={row} at={at} />
      ))}
    </div>
  )
}
