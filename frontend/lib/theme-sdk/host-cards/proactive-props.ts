/**
 * PRD-268 US-C21e (SDK 1.3.0) — the props of the cards area's proactive default components, and their checks.
 *
 * The props come from a theme's code, so the wrappers (proactive-cards.tsx) check them before the host's component
 * sees them: a brief, a suggestion, a readiness answer or a calibration is read the way the host reads one off the
 * wire (lib/proactive/wire.ts), so a row, an item or a line that isn't one is left out, and a suggestion's id, which
 * reaches Start's and Hide's request paths, must be a suggestion's. A value of the wrong kind throws a TypeError
 * naming the card and the prop, and the theme's error boundary reports it. What a card hands back to a theme (a row,
 * Start's answer, a calibration answer) is a frozen copy, never the host's own object.
 */
import type { ReactNode } from 'react'

import type {
  Brief as HostBrief,
  Calibration as HostCalibration,
  CalibrationLine as HostCalibrationLine,
  CalibrationSource as HostCalibrationSource,
  Readiness as HostReadiness,
  ReadinessItem as HostReadinessItem,
  Suggestion as HostSuggestion,
} from '@/lib/proactive/types'
import { briefOf, readinessOf, recordOf, suggestionOf, textOf } from '@/lib/proactive/wire'

import type {
  Brief,
  BriefRow,
  BriefSection,
  Calibration,
  CalibrationAnswer,
  DisplayState,
  Readiness,
  ReadinessItem,
  Retry,
  StartAnswer,
  Suggestion,
} from '../internal/proactive-shapes'

export interface BriefCardProps {
  readonly brief: Brief
  /** Listen (voice.BriefVoice: the brief the host holds) in the card's footer; default true. False when the theme places it itself. */
  readonly listen?: boolean
  /**
   * The thumbs (SDK 1.4.0), showing the person's remembered rating; default true. False when the theme draws its own
   * with proactive.useRateBrief, so the card draws none.
   */
  readonly thumbs?: boolean
  /** A row's action was pressed: resolve to show "✓ done_label" in place, reject to leave the row as it was. */
  readonly onAction?: (row: BriefRow, section: BriefSection) => unknown
  /** A row with somewhere to go was opened; without it no row is a link. */
  readonly onOpen?: (row: BriefRow, section: BriefSection) => void
  /** "Show N more" was pressed for a folded section; without it the line is text. */
  readonly onShowMore?: (section: BriefSection) => void
  readonly className?: string
}

export type SuggestionCardVariant = 'card' | 'pinned'

export interface SuggestionCardProps {
  readonly suggestion: Suggestion
  /** The card (the default), or the one-line variant pinned above a composer. */
  readonly variant?: SuggestionCardVariant
  /** Start was accepted: what it made, for the theme to open. */
  readonly onStarted?: (answer: StartAnswer) => void
  /** The pinned line's title opens the full card. */
  readonly onOpen?: () => void
  /** What the theme draws under the suggestion's text, such as its evidence. */
  readonly children?: ReactNode
  readonly className?: string
}

export interface StatusLabelProps {
  /** The row's `display_state`; anything else draws nothing. */
  readonly state?: DisplayState | null
  /** The row's `next_step`, drawn after ▸. */
  readonly nextStep?: string | null
  readonly retry?: Retry | null
  /** What Try again runs; without it Try again is not drawn. */
  readonly onRetry?: (retry: Retry) => void
  /** The row's title, which names Try again for a screen reader. */
  readonly title?: string
  readonly className?: string
}

export interface ReadinessListProps {
  /** Readiness@1 as useReadiness read it; null while it loads. */
  readonly readiness: Readiness | null | undefined
  /** Why the read failed (useReadiness's error): one line in place of "Checking…". */
  readonly error?: string | null
  /** Rows only the browser knows (its microphone, its speech), drawn after the server's. */
  readonly clientItems?: readonly ReadinessItem[]
  readonly className?: string
}

export interface CalibrationCardProps {
  readonly calibration: Calibration
  /** Sends an answer (useAnswerCalibration's runAsync); without it Reject and Sounds right are disabled. */
  readonly onAnswer?: (answer: CalibrationAnswer) => Promise<void> | void
  /** Something's off: the theme asks the person what is wrong; without it the button is disabled. */
  readonly onSomethingOff?: () => void
  readonly className?: string
}

export interface ReplyChipsProps {
  /** The answers Auto offered with its message; the host draws two or three of at most 24 characters. */
  readonly chips: readonly string[]
  /** True while the message is Auto's last one in the thread. */
  readonly latest: boolean
  /** Sends the chip's text as the person's message; a rejected promise gives the chips back. */
  readonly onReply: (text: string) => void | Promise<unknown>
  readonly className?: string
}

const SDK_PREFIX = 'Automatos theme SDK: '
const VARIANTS: readonly SuggestionCardVariant[] = ['card', 'pinned']
const RETRIES: readonly Retry[] = ['rerun_mission', 'replan_mission', 'retry_ticket']
const SOURCE_TYPES: readonly HostCalibrationSource['type'][] = ['deliverable', 'conversation', 'document', 'skill']

function rejected(card: string, what: string): TypeError {
  return new TypeError(`${SDK_PREFIX}${card} needs ${what}.`)
}

/** A theme's handler: a function, or left out; anything else throws. */
export function handlerOf<F extends (...args: never[]) => unknown>(value: F | undefined, card: string, prop: string): F | undefined {
  if (value === undefined || typeof value === 'function') return value
  throw rejected(card, `${prop} to be a function`)
}

/** Some text, null where null is allowed, or left out; anything else throws. */
export function optionalText(value: unknown, card: string, prop: string): string | null | undefined {
  if (value === undefined || value === null || typeof value === 'string') return value
  throw rejected(card, `${prop} to be some text`)
}

export function checkedBrief(value: unknown): HostBrief {
  const brief = briefOf(value)
  if (brief === null) throw rejected('BriefCard', 'brief to be a Brief@1 (v 1, with its date, greeting and prose)')
  return brief
}

export function checkedSuggestion(value: unknown): HostSuggestion {
  const suggestion = suggestionOf(value)
  if (suggestion === null) throw rejected('SuggestionCard', "suggestion to be one of useSuggestions' rows (its id, title and proposed_action)")
  return suggestion
}

export function checkedVariant(value: unknown): SuggestionCardVariant | undefined {
  if (value === undefined) return undefined
  const variant = VARIANTS.find((known) => known === value)
  if (variant === undefined) throw rejected('SuggestionCard', `variant to be one of ${VARIANTS.join(', ')}`)
  return variant
}

export function checkedRetry(value: unknown): Retry | null | undefined {
  if (value === undefined || value === null) return value
  const retry = RETRIES.find((known) => known === value)
  if (retry === undefined) throw rejected('StatusLabel', `retry to be one of ${RETRIES.join(', ')}`)
  return retry
}

/** Readiness@1 as the host reads it, or null while it loads. */
export function checkedReadiness(value: unknown): HostReadiness | null {
  if (value === null || value === undefined) return null
  if (recordOf(value) === null) throw rejected('ReadinessList', 'readiness to be a Readiness@1, or null while it loads')
  return readinessOf(value)
}

/** The browser's own rows, read as the server's are: an item that isn't one is left out. */
export function checkedClientItems(value: unknown): HostReadinessItem[] {
  if (value === undefined) return []
  if (!Array.isArray(value)) throw rejected('ReadinessList', 'clientItems to be a list of readiness items')
  return readinessOf({ items: value }).items
}

/** The read's failure as the host's list takes it: an error whose message it never shows (it says its own line). */
export function readFailure(value: unknown): Error | undefined {
  const message = optionalText(value, 'ReadinessList', 'error')
  return message ? new Error(message) : undefined
}

function sourceOf(value: unknown): HostCalibrationSource | null {
  const fields = recordOf(value)
  const type = SOURCE_TYPES.find((known) => known === fields?.type)
  const id = textOf(fields?.id)
  if (type === undefined || id === undefined) return null
  return { type, id, label: textOf(fields?.label) ?? id }
}

function lineOf(value: unknown): HostCalibrationLine | null {
  const fields = recordOf(value)
  const id = textOf(fields?.id)
  const text = textOf(fields?.text)
  const sources = fields?.sources
  if (id === undefined || text === undefined) return null
  const read = Array.isArray(sources) ? sources.map(sourceOf) : []
  return { id, text, sources: read.filter((source): source is HostCalibrationSource => source !== null) }
}

/** Calibration@1 as the host's card takes it: a line without its id and text is left out, and so is an offer that isn't text. */
export function checkedCalibration(value: unknown): HostCalibration {
  const fields = recordOf(value)
  const id = textOf(fields?.id)
  const lines = fields?.lines
  const offers = fields?.offers
  if (fields?.v !== 1 || id === undefined || !Array.isArray(lines) || !Array.isArray(offers)) {
    throw rejected('CalibrationCard', 'calibration to be a Calibration@1 (v 1, its id, lines and offers)')
  }
  return {
    v: 1,
    id,
    lines: lines.map(lineOf).filter((line): line is HostCalibrationLine => line !== null),
    offers: offers.filter((offer): offer is string => typeof offer === 'string'),
    empty: fields?.empty === true,
  }
}

/** The chips a theme gave, as a list the host's ReplyChips filters; throws when there is no list. */
export function checkedChips(value: unknown): readonly string[] {
  if (!Array.isArray(value)) throw rejected('ReplyChips', 'chips to be a list of short answers')
  return value.filter((chip): chip is string => typeof chip === 'string')
}

export function checkedFlag(value: unknown, card: string, prop: string): boolean {
  if (typeof value === 'boolean') return value
  throw rejected(card, `${prop} to be true or false`)
}

export function requiredHandler<F extends (...args: never[]) => unknown>(value: F | undefined, card: string, prop: string): F {
  const handler = handlerOf(value, card, prop)
  if (handler === undefined) throw rejected(card, `${prop} to be a function`)
  return handler
}
