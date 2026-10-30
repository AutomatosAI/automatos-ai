'use client'

/**
 * PRD-268 §6.11.5, §6.12.3, K12, K17, K23 (US-C21d) — calibration on day one: what Auto thinks it
 * knows, and what it can do.
 *
 * It renders from its props: Calibration@1's lines (up to three, each with its source count and
 * Reject; calibration-parts.tsx), its three offers, and Sounds right / Something's off. Its routes
 * are US-C22's, so the surface passes how an answer is sent (useAnswerCalibration's mutate) and what
 * Something's off does; without them those controls are disabled. A line is struck through with
 * "I'll forget that" once its Reject is answered, and Sounds right turns into "✓ Sounds right" once
 * accepted; a refused answer gets one line (the server's sentence, else the card's own) and the
 * card stays as it was; a new calibration (another id) starts with none of those answers. With
 * nothing to read yet it says so and shows the offers as starters. No
 * percentage anywhere: a line shows where it came from as a count. The lines and the offers are
 * Auto's words (data-ai-generated); the tiny "Auto is an AI" caption shows once, outside them.
 */
import { useState } from 'react'
import type { Calibration, CalibrationAnswer, CalibrationLine } from '@/lib/proactive/types'
import { cn } from '@/lib/utils'
import { AiCaption, CARD_CLASS, ENTRANCE_CLASS, Eyebrow, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from './card-parts'
import { CalibrationLineItem, CalibrationOffers } from './calibration-parts'
import { refusedLine } from './use-suggestion-card'

export const CALIBRATION_TITLE = 'What Auto knows so far'
export const SOUNDS_RIGHT = 'Sounds right'
export const SOMETHING_OFF = "Something's off"
export const CALIBRATION_EMPTY = 'Nothing to go on yet. Here are three ways to start.'
/** At most three lines and three offers. */
export const CALIBRATION_MAX = 3

const LOG_PREFIX = '[proactive]'

export interface CalibrationCardProps {
  readonly calibration: Calibration
  /** Sends an answer (useAnswerCalibration's mutate); without it Reject and Sounds right are disabled. */
  readonly onAnswer?: (answer: CalibrationAnswer) => Promise<void> | void
  /** Something's off: the surface asks the person what is wrong; without it the button is disabled. */
  readonly onSomethingOff?: () => void
  readonly className?: string
}

/** The answers given on this card: the lines rejected, whether it was accepted, and a refusal's line. */
function useCalibrationAnswers(calibration: Calibration, onAnswer: CalibrationCardProps['onAnswer']) {
  const [rejected, setRejected] = useState<ReadonlySet<string>>(() => new Set())
  const [accepted, setAccepted] = useState(false)
  const [busy, setBusy] = useState(false)
  const [refused, setRefused] = useState<string | null>(null)
  const send = (answer: CalibrationAnswer, answered: () => void) => {
    if (!onAnswer || busy) return
    setBusy(true)
    setRefused(null)
    Promise.resolve()
      .then(() => onAnswer(answer))
      .then(answered, (error: unknown) => {
        console.warn(LOG_PREFIX, 'the calibration answer was not taken', error)
        setRefused(refusedLine(error))
      })
      .finally(() => setBusy(false))
  }
  const reject = (line: CalibrationLine) =>
    send({ calibration_id: calibration.id, answer: 'reject', line_id: line.id }, () =>
      setRejected((before) => new Set([...before, line.id])),
    )
  const accept = () => send({ calibration_id: calibration.id, answer: 'accept' }, () => setAccepted(true))
  return { rejected, accepted, busy, refused, reject, accept }
}

interface AnswersProps {
  readonly accepted: boolean
  readonly busy: boolean
  readonly onAccept?: () => void
  readonly onSomethingOff?: () => void
}

/** Sounds right (the card's one ink primary) and Something's off; "✓ Sounds right" in place once accepted. */
function CalibrationAnswers({ accepted, busy, onAccept, onSomethingOff }: AnswersProps) {
  if (accepted) {
    return (
      <span role="status" className="text-sm font-medium text-foreground">
        <span aria-hidden="true">✓ </span>
        {SOUNDS_RIGHT}
      </span>
    )
  }
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" aria-busy={busy} disabled={busy || !onAccept} onClick={onAccept} className={PRIMARY_BUTTON_CLASS}>
        {SOUNDS_RIGHT}
      </button>
      <button type="button" disabled={!onSomethingOff} onClick={onSomethingOff} className={QUIET_BUTTON_CLASS}>
        {SOMETHING_OFF}
      </button>
    </div>
  )
}

/** Day one's calibration: up to three lines with their sources and Reject, three offers, and the answers. */
export function CalibrationCard(props: CalibrationCardProps) {
  return <CalibrationBody key={props.calibration.id} {...props} />
}

function CalibrationBody({ calibration, onAnswer, onSomethingOff, className }: CalibrationCardProps) {
  const answers = useCalibrationAnswers(calibration, onAnswer)
  const lines = calibration.lines.slice(0, CALIBRATION_MAX)
  const empty = calibration.empty || lines.length === 0
  const canReject = onAnswer !== undefined && !answers.accepted
  return (
    <article
      aria-label={CALIBRATION_TITLE}
      data-calibration-id={calibration.id}
      data-state={empty ? 'empty' : answers.accepted ? 'accepted' : 'open'}
      className={cn(CARD_CLASS, ENTRANCE_CLASS, 'space-y-3', className)}
    >
      <Eyebrow>{CALIBRATION_TITLE}</Eyebrow>
      {empty ? (
        <p className="text-sm text-muted-foreground">{CALIBRATION_EMPTY}</p>
      ) : (
        <ul className="divide-y divide-border">
          {lines.map((line) => (
            <CalibrationLineItem
              key={line.id}
              line={line}
              rejected={answers.rejected.has(line.id)}
              onReject={canReject ? answers.reject : undefined}
              busy={answers.busy}
            />
          ))}
        </ul>
      )}
      <CalibrationOffers offers={calibration.offers.slice(0, CALIBRATION_MAX)} />
      {answers.refused ? (
        <p role="alert" className="text-sm text-[hsl(var(--destructive))]">
          {answers.refused}
        </p>
      ) : null}
      <footer className="flex flex-wrap items-center justify-between gap-2">
        <AiCaption />
        {empty ? null : (
          <CalibrationAnswers
            accepted={answers.accepted}
            busy={answers.busy}
            onAccept={onAnswer ? answers.accept : undefined}
            onSomethingOff={onSomethingOff}
          />
        )}
      </footer>
    </article>
  )
}
