/**
 * PRD-268 US-C21e (SDK 1.3.0) — the cards area's proactive default components: `cards.BriefCard`,
 * `cards.SuggestionCard`, `cards.StatusLabel`, `cards.ReadinessList`, `cards.CalibrationCard` and `cards.ReplyChips`.
 *
 * Each is the host's own component (components/proactive/*, US-C21c and US-C21d) behind its checked props
 * (proactive-props.ts), so a theme draws the host's words, glyphs, AI marks and controls in its look's tokens, and
 * never redraws them. Their calls are the host's: Start and Hide post the host's routes (hooks/use-suggestions.ts),
 * Check again the host's (hooks/use-readiness.ts), the brief's thumbs the digest feedback route, and Listen in the
 * brief card is the SDK's BriefVoice, which reads the brief the host holds, never the text a theme hands the card.
 * Nothing here navigates or sends a message by itself: what a row opens, what Start made, a calibration answer and a
 * chip's reply are handed to the theme.
 */
import { useMemo, type ReactElement } from 'react'

import { BriefCard as HostBriefCard } from '@/components/proactive/brief-card'
import { CalibrationCard as HostCalibrationCard } from '@/components/proactive/calibration-card'
import { ReadinessList as HostReadinessList } from '@/components/proactive/readiness-list'
import { ReplyChips as HostReplyChips } from '@/components/proactive/reply-chips'
import { StatusLabel as HostStatusLabel } from '@/components/proactive/status-label'
import { SuggestionCard as HostSuggestionCard } from '@/components/proactive/suggestion-card'

import { frozenCopy } from '../internal/query-view'
import { BriefVoice } from './brief-voice'
import {
  checkedBrief,
  checkedCalibration,
  checkedChips,
  checkedClientItems,
  checkedFlag,
  checkedReadiness,
  checkedRetry,
  checkedSuggestion,
  checkedVariant,
  handlerOf,
  optionalText,
  readFailure,
  requiredHandler,
  type BriefCardProps,
  type CalibrationCardProps,
  type ReadinessListProps,
  type ReplyChipsProps,
  type StatusLabelProps,
  type SuggestionCardProps,
} from './proactive-props'

/** The host's brief card: the date, the greeting, Auto's Read, the five sections, Listen and the thumbs. */
export function BriefCard(props: BriefCardProps): ReactElement {
  const { brief, listen = true, thumbs = true, onAction, onOpen, onShowMore, className } = props
  const shown = useMemo(() => checkedBrief(brief), [brief])
  const action = handlerOf(onAction, 'BriefCard', 'onAction')
  const open = handlerOf(onOpen, 'BriefCard', 'onOpen')
  const withListen = checkedFlag(listen, 'BriefCard', 'listen')
  return (
    <HostBriefCard
      brief={shown}
      listen={withListen ? <BriefVoice /> : undefined}
      thumbs={checkedFlag(thumbs, 'BriefCard', 'thumbs')}
      onAction={action && ((row, section) => action(frozenCopy(row), section))}
      onOpen={open && ((row, section) => open(frozenCopy(row), section))}
      onShowMore={handlerOf(onShowMore, 'BriefCard', 'onShowMore')}
      className={className}
    />
  )
}

/** The host's suggestion card with Start and Hide, or its line pinned above a composer. */
export function SuggestionCard({ suggestion, variant, onStarted, onOpen, children, className }: SuggestionCardProps): ReactElement {
  const shown = useMemo(() => checkedSuggestion(suggestion), [suggestion])
  const started = handlerOf(onStarted, 'SuggestionCard', 'onStarted')
  return (
    <HostSuggestionCard
      suggestion={shown}
      variant={checkedVariant(variant)}
      onStarted={started && ((answer) => started(frozenCopy(answer)))}
      onOpen={handlerOf(onOpen, 'SuggestionCard', 'onOpen')}
      className={className}
    >
      {children}
    </HostSuggestionCard>
  )
}

/** A row's state as the host draws it: the glyph and the word, ▸ its next step, and Try again. */
export function StatusLabel({ state, nextStep, retry, onRetry, title, className }: StatusLabelProps): ReactElement {
  return (
    <HostStatusLabel
      state={state}
      nextStep={optionalText(nextStep, 'StatusLabel', 'nextStep')}
      retry={checkedRetry(retry)}
      onRetry={handlerOf(onRetry, 'StatusLabel', 'onRetry')}
      title={optionalText(title, 'StatusLabel', 'title') ?? undefined}
      className={className}
    />
  )
}

/** What Auto depends on, each item's state and fix, and Check again. */
export function ReadinessList({ readiness, error, clientItems, className }: ReadinessListProps): ReactElement {
  const shown = useMemo(() => checkedReadiness(readiness), [readiness])
  const client = useMemo(() => checkedClientItems(clientItems), [clientItems])
  return <HostReadinessList readiness={shown} error={readFailure(error)} clientItems={client} className={className} />
}

/** What Auto knows so far, from its props: the lines with Reject, the offers, Sounds right and Something's off. */
export function CalibrationCard({ calibration, onAnswer, onSomethingOff, className }: CalibrationCardProps): ReactElement {
  const shown = useMemo(() => checkedCalibration(calibration), [calibration])
  const answer = handlerOf(onAnswer, 'CalibrationCard', 'onAnswer')
  return (
    <HostCalibrationCard
      calibration={shown}
      onAnswer={answer && ((given) => answer(frozenCopy(given)))}
      onSomethingOff={handlerOf(onSomethingOff, 'CalibrationCard', 'onSomethingOff')}
      className={className}
    />
  )
}

/** Two or three answers under Auto's last message; a tap hands the chip's text to `onReply` once. */
export function ReplyChips({ chips, latest, onReply, className }: ReplyChipsProps): ReactElement {
  return (
    <HostReplyChips
      chips={checkedChips(chips)}
      latest={checkedFlag(latest, 'ReplyChips', 'latest')}
      onReply={requiredHandler(onReply, 'ReplyChips', 'onReply')}
      className={className}
    />
  )
}
