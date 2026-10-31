/**
 * PRD-268 §6.3 'In the thread', K8 (US-W03b) — the person's follow-ups, checked off the wire.
 *
 * GET /api/proactive/follow-ups answers `{follow_ups: [{id, at, reason, status, chat_id}]}` and its cancel answers
 * `{follow_up}` (US-C11e): the caller's own follow-ups only, `at` the UTC ISO time it fires, `status` pending, fired,
 * cancelled or missed (its time passed and the scheduler never ran it). A row that is not what it says is left out and
 * the rest are kept, and an id is checked before it reaches a request path. The reason was written by the model in
 * the person's own chat: it stays text. Nothing here mutates what it reads.
 */
import { recordOf, textOf } from './wire'

export const FOLLOW_UP_STATUSES = ['pending', 'fired', 'cancelled', 'missed'] as const
export type FollowUpStatus = (typeof FOLLOW_UP_STATUSES)[number]

const STATUSES: ReadonlySet<unknown> = new Set(FOLLOW_UP_STATUSES)

/** One of the caller's follow-ups, as the list writes it. */
export interface FollowUp {
  readonly id: number
  /** When it fires (UTC ISO), or null when the server has no time for it. */
  readonly at: string | null
  readonly reason: string
  readonly status: FollowUpStatus
  readonly chat_id: string | null
}

/** True for a follow-up's number: the row id the routes take, a positive safe integer. */
export function isFollowUpId(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0
}

function timeOf(value: unknown): string | null {
  const at = textOf(value)
  return at !== undefined && !Number.isNaN(Date.parse(at)) ? at : null
}

/** One follow-up, or null without its id and a status it can have. */
export function followUpOf(value: unknown): FollowUp | null {
  const fields = recordOf(value)
  if (fields === null || !isFollowUpId(fields.id) || !STATUSES.has(fields.status)) return null
  return {
    id: fields.id,
    at: timeOf(fields.at),
    reason: textOf(fields.reason) ?? '',
    status: fields.status as FollowUpStatus,
    chat_id: textOf(fields.chat_id) ?? null,
  }
}

/** GET /api/proactive/follow-ups' rows; one that is not a follow-up is left out. */
export function followUpsOf(body: unknown): FollowUp[] {
  const rows = recordOf(body)?.follow_ups
  return (Array.isArray(rows) ? rows : []).map(followUpOf).filter((row): row is FollowUp => row !== null)
}

/** The follow-up a cancel answered with, now cancelled; null when the answer names none. */
export function cancelledOf(body: unknown): FollowUp | null {
  return followUpOf(recordOf(body)?.follow_up)
}
