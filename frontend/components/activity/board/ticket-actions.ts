/**
 * PRD-252 R7 — what a person can do to a ticket from the board, and how a
 * cancelled one says who stopped it.
 *
 * An Inbox ticket refused with "Assign an agent first" could not be fixed on
 * the board, and a cancelled or closed ticket opened to an empty viewer.
 * Mission tickets are the mission's: the mission engine runs them, so their
 * actions live on the mission's page (PRD-252 D6).
 */
import type { BoardTask } from '@/types/board'
import { isLocal } from '@/lib/auth-edition'

type TicketRef = Pick<BoardTask, 'type' | 'status'>

const FINISHED = new Set(['done', 'failed', 'cancelled', 'closed'])
const ASSIGNABLE = new Set(['inbox', 'assigned', 'blocked', 'failed'])

/** A mission's own card, or one of its steps. */
export function isMissionTicket(task: Pick<BoardTask, 'type'>): boolean {
  return task.type === 'mission'
}

export function canAssign(task: TicketRef): boolean {
  return !isMissionTicket(task) && ASSIGNABLE.has(task.status)
}

export function canCancel(task: TicketRef): boolean {
  return !isMissionTicket(task) && !FINISHED.has(task.status)
}

export interface Stopped {
  /** Who, in words; null when nothing recorded who. */
  by: string | null
  at: string | null
  reason: string | null
  closed: boolean
}

/**
 * Who cancelled (or closed) a ticket, and when. The cancel records itself in
 * `runtime_ref.cancelled`; a person's status move in `runtime_ref.operator_stop`;
 * a ticket closed without either still has its finish time.
 */
export function whoStopped(task: Pick<BoardTask, 'status' | 'runtime_ref' | 'completed_at'>): Stopped | null {
  if (task.status !== 'cancelled' && task.status !== 'closed') return null
  const ref = (task.runtime_ref ?? {}) as Record<string, unknown>
  const record = (ref.cancelled ?? ref.operator_stop) as Record<string, unknown> | undefined
  return {
    by: personLabel(record?.by),
    at: typeof record?.at === 'string' ? record.at : task.completed_at ?? null,
    reason: typeof record?.reason === 'string' && record.reason ? record.reason : null,
    closed: task.status === 'closed',
  }
}

/** A person's record ("operator", "user:<id>") reads "you" on the one-operator
 * local edition; hosted, it could be any member of the workspace. */
function personLabel(by: unknown): string | null {
  if (typeof by !== 'string' || !by) return null
  if (by === 'operator' || by.startsWith('user:')) return isLocal ? 'you' : 'a workspace member'
  return by
}
