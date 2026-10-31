'use client'

/**
 * PRD-268 §6.11.5, §6.12.3, K15, K19 (US-W06) — the calibration card under the Auto message that carries it.
 *
 * thread-cards.tsx draws this from a message's `calibration@1` part (lib/proactive/thread-parts.ts): W2's
 * CalibrationCard, with the live state from the shared view (hooks/use-calibration.ts), so the lines rejected and
 * "✓ Sounds right" show the same after a reload, in another tab or once answered on the phone. Reject and Sounds right
 * post through useAnswerCalibration, and only while the view's own calibration is this one (it is the person's) and
 * the server offers calibration; otherwise the card shows its lines without them. Something's off sends "Something's
 * off" as the person's own message through the reply-chip path (reply-chip-event.ts), as a chip does, and the lines
 * stay rejectable; a read-only chat has no composer, so there it is disabled. An answer's note (a line kept out of
 * memory) shows under the card. No percentage anywhere: each line says where it came from as a count.
 */
import { useAnswerCalibration, useCalibrationView } from '@/hooks/use-calibration'
import type { CalibrationView } from '@/lib/proactive/calibration'
import type { Calibration } from '@/lib/proactive/types'
import { CalibrationCard, SOMETHING_OFF, type CalibrationAnswered } from './calibration-card'
import { sendReplyChip } from './reply-chip-event'

/** The avatar's width and gap (w-8 + space-x-3), so the card lines up with the message's bubble. */
const ALIGNED_CLASS = 'max-w-[92%] space-y-2 pl-11'
const LOG_PREFIX = '[proactive]'

export interface ThreadCalibrationProps {
  readonly chatId: string
  readonly calibration: Calibration
  /** A read-only chat has no composer to send "Something's off" through. */
  readonly isReadonly?: boolean
}

/** What the server holds as answered, for this calibration only. */
function answeredOf(view: CalibrationView): CalibrationAnswered {
  return { rejectedLines: view.rejected_lines, accepted: view.status === 'answered' }
}

/** The calibration a message carries, as a card with its answers; Something's off goes to the composer. */
export function ThreadCalibration({ chatId, calibration, isReadonly = false }: ThreadCalibrationProps) {
  const { data } = useCalibrationView()
  const answering = useAnswerCalibration()
  const mine = data !== undefined && data.calibration?.id === calibration.id ? data : null
  const somethingOff = () => {
    if (!sendReplyChip(chatId, SOMETHING_OFF)) console.warn(LOG_PREFIX, "no composer sent Something's off: a turn is streaming")
  }
  return (
    <div className={ALIGNED_CLASS} data-testid="thread-calibration">
      <CalibrationCard
        calibration={calibration}
        answered={mine ? answeredOf(mine) : undefined}
        onAnswer={mine && answering.available ? answering.mutate : undefined}
        onSomethingOff={isReadonly ? undefined : somethingOff}
      />
      {mine?.note ? (
        <p role="status" className="text-xs text-muted-foreground">
          {mine.note}
        </p>
      ) : null}
    </div>
  )
}
