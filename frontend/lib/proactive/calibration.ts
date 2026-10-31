/**
 * PRD-268 §6.11.5, K15, K17 (US-W06) — calibration, checked off the wire.
 *
 * GET /api/auto/calibration answers `{available, offer, consent_line, status, calibration, rejected_lines,
 * accepted_lines}` (US-C22b, api/auto_calibration.py): `offer` is true while the person may still answer the consent
 * line, `status` is null before that (else declined, open or answered) and `calibration` is the Calibration@1 once
 * built. POST /api/auto/calibration answers `{status, calibration, chat_id, message_id}`, and a 409 already_offered
 * carries the `record` as it is; Reject and Sounds right answer the record with its `note`. The calibration is posted
 * as one Auto message whose `calibration@1` part carries it. A body that is not what it says reads as not available,
 * a line or source that isn't one is left out (the rest are kept), and an id is checked before it reaches a request
 * path. A line or an offer that carries a percentage is left out too, as the server's own check does (§12-Q26): a
 * confidence number would be one the model made up. The lines, offers and labels are Auto's words: text, never
 * markup. Nothing here mutates what it reads.
 */
import type { Calibration, CalibrationLine, CalibrationSource } from './types'
import { isProactiveId, recordOf, textOf } from './wire'

export const CALIBRATION_PART = 'calibration@1'
export const CALIBRATION_STATUSES = ['declined', 'open', 'answered'] as const
export type CalibrationStatus = (typeof CALIBRATION_STATUSES)[number]

/** The person's calibration as the server holds it: whether to offer it, and where it stands. */
export interface CalibrationView {
  readonly available: boolean
  /** Ask the consent line: never offered (or the offer was abandoned), and the person may answer it. */
  readonly offer: boolean
  /** The server's consent line, in the person's language. */
  readonly consent_line: string
  readonly status: CalibrationStatus | null
  readonly calibration: Calibration | null
  readonly rejected_lines: readonly string[]
  readonly accepted_lines: readonly string[]
  /** The last answer's note (a line kept out of memory, said in a sentence); null when it had none. */
  readonly note: string | null
}

/** What the consent POST answered: where it stands now, and the chat the calibration was posted into. */
export interface CalibrationConsented {
  readonly status: CalibrationStatus | null
  readonly calibration: Calibration | null
  readonly chat_id: string | null
}

export const NOT_AVAILABLE: CalibrationView = Object.freeze({
  available: false,
  offer: false,
  consent_line: '',
  status: null,
  calibration: null,
  rejected_lines: Object.freeze([]),
  accepted_lines: Object.freeze([]),
  note: null,
})

const SOURCE_TYPES: ReadonlySet<unknown> = new Set(['deliverable', 'conversation', 'document', 'skill'])
const STATUSES: ReadonlySet<unknown> = new Set(CALIBRATION_STATUSES)
/** A line's id as the routes take it ("line-1"). */
const LINE_ID = /^[A-Za-z0-9_-]{1,32}$/
/** A percentage, in any of the person's languages (services/calibration/answer.py _PERCENT). */
const PERCENT = /%|\b(?:percent|per\s?cent|pct|por\s?cento|porcentagem|percentual)\b/i

/** True for a line's id: what the reject route takes in its path. */
export function isCalibrationLineId(value: unknown): value is string {
  return typeof value === 'string' && LINE_ID.test(value)
}

function listOf(value: unknown): readonly unknown[] {
  return Array.isArray(value) ? value : []
}

function wordsOf(value: unknown): string | undefined {
  const text = textOf(value)
  return text !== undefined && !PERCENT.test(text) ? text : undefined
}

function sourceOf(value: unknown): CalibrationSource | null {
  const fields = recordOf(value)
  const id = textOf(fields?.id)
  const label = textOf(fields?.label)
  if (!SOURCE_TYPES.has(fields?.type) || id === undefined || label === undefined) return null
  return { type: fields?.type as CalibrationSource['type'], id, label }
}

function lineOf(value: unknown): CalibrationLine | null {
  const fields = recordOf(value)
  const id = fields?.id
  const text = wordsOf(fields?.text)
  if (!isCalibrationLineId(id) || text === undefined) return null
  const sources = listOf(fields?.sources).map(sourceOf).filter((source): source is CalibrationSource => source !== null)
  return { id, text, sources }
}

/** A Calibration@1, or null without its version, its id, and its lines and offers as lists. */
export function calibrationOf(value: unknown): Calibration | null {
  const fields = recordOf(value)
  const id = fields?.id
  if (fields === null || fields.v !== 1 || !isProactiveId(id) || !Array.isArray(fields.lines) || !Array.isArray(fields.offers)) return null
  return {
    v: 1,
    id,
    lines: listOf(fields.lines).map(lineOf).filter((line): line is CalibrationLine => line !== null),
    offers: listOf(fields.offers).map(wordsOf).filter((offer): offer is string => offer !== undefined),
    empty: fields.empty === true,
  }
}

/** The calibration a message's `calibration@1` part carries; null when it carries none that is one. */
export function calibrationOfParts(parts: unknown): Calibration | null {
  const part = listOf(parts).map(recordOf).find((candidate) => candidate?.type === CALIBRATION_PART)
  return part ? calibrationOf(part.calibration) : null
}

function statusOf(value: unknown): CalibrationStatus | null {
  return STATUSES.has(value) ? (value as CalibrationStatus) : null
}

function lineIdsOf(value: unknown): readonly string[] {
  return listOf(value).filter(isCalibrationLineId)
}

/** GET /api/auto/calibration's answer; a body that isn't one reads as not available. */
export function calibrationViewOf(body: unknown): CalibrationView {
  const fields = recordOf(body)
  if (fields === null || fields.available !== true) return NOT_AVAILABLE
  return {
    available: true,
    offer: fields.offer === true,
    consent_line: textOf(fields.consent_line) ?? '',
    status: statusOf(fields.status),
    calibration: calibrationOf(fields.calibration),
    rejected_lines: lineIdsOf(fields.rejected_lines),
    accepted_lines: lineIdsOf(fields.accepted_lines),
    note: null,
  }
}

/** The view once an answer came back with the record (Reject, Sounds right, or a 409's `record`): no longer offered. */
export function answeredView(before: CalibrationView, body: unknown): CalibrationView {
  const record = recordOf(body)
  if (record === null) return { ...before, offer: false }
  return {
    ...before,
    offer: false,
    status: statusOf(record.status) ?? before.status,
    calibration: calibrationOf(record.calibration) ?? before.calibration,
    rejected_lines: Array.isArray(record.rejected_lines) ? lineIdsOf(record.rejected_lines) : before.rejected_lines,
    accepted_lines: Array.isArray(record.accepted_lines) ? lineIdsOf(record.accepted_lines) : before.accepted_lines,
    note: textOf(record.note) ?? null,
  }
}

/** POST /api/auto/calibration's answer: where it stands, and the chat it was posted into (null when declined). */
export function consentedOf(body: unknown): CalibrationConsented {
  const fields = recordOf(body)
  const chatId = fields?.chat_id
  return {
    status: statusOf(fields?.status),
    calibration: calibrationOf(fields?.calibration),
    chat_id: isProactiveId(chatId) ? chatId : null,
  }
}

/** The page's language, as POST /api/auto/calibration takes it; undefined when the browser names none. */
export function pageLocale(): string | undefined {
  const language = typeof navigator !== 'undefined' ? navigator.language : ''
  return language || undefined
}
