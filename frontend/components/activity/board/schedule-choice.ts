/**
 * "When" for a board task — the pure half of the Create Task dialog's
 * scheduling row. `now` creates the ticket immediately (the existing path);
 * every other mode becomes a scheduled task (POST /api/v1/scheduled-tasks)
 * that waits on the Command Centre calendar and is filed on the board when it
 * fires.
 *
 * Recurring modes are 5-field crons in UTC (the scheduler's clock), built from
 * the UTC parts of the chosen local time. A daylight-saving change therefore
 * shifts a standing schedule by an hour on the wall clock — the known limit of
 * a timezone-less cron, stated here rather than hidden.
 */

export type ScheduleMode = 'now' | 'later' | 'daily' | 'weekdays' | 'weekly'

export const SCHEDULE_MODE_OPTIONS: { value: ScheduleMode; label: string }[] = [
  { value: 'now', label: 'Now' },
  { value: 'later', label: 'Later (once)' },
  { value: 'daily', label: 'Every day' },
  { value: 'weekdays', label: 'Weekdays' },
  { value: 'weekly', label: 'Every week' },
]

export interface SchedulePayload {
  task_type: 'one_shot' | 'recurring'
  schedule: string
}

const DEFAULT_HOUR = 9

/** Parse a `datetime-local` input value ("2026-09-11T09:00") as local time. */
export function parseLocalDateTime(value: string): Date | null {
  if (!value) return null
  const at = new Date(value)
  return Number.isNaN(at.getTime()) ? null : at
}

/** Tomorrow at 09:00 local, formatted for a `datetime-local` input. */
export function defaultScheduleAt(now: Date = new Date()): string {
  const d = new Date(now)
  d.setDate(d.getDate() + 1)
  d.setHours(DEFAULT_HOUR, 0, 0, 0)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

export function isScheduleInFuture(value: string, now: Date = new Date()): boolean {
  const at = parseLocalDateTime(value)
  return at !== null && at.getTime() > now.getTime()
}

/** The scheduled-task payload for a mode, or null for `now` / an unparseable time. */
export function buildSchedulePayload(mode: ScheduleMode, value: string): SchedulePayload | null {
  if (mode === 'now') return null
  const at = parseLocalDateTime(value)
  if (!at) return null
  if (mode === 'later') return { task_type: 'one_shot', schedule: at.toISOString() }
  const minute = at.getUTCMinutes()
  const hour = at.getUTCHours()
  const dow = mode === 'daily' ? '*' : mode === 'weekdays' ? '1-5' : String(at.getUTCDay())
  return { task_type: 'recurring', schedule: `${minute} ${hour} * * ${dow}` }
}

/** Human line for the toast: "for Thu 11 Sep, 09:00" / "every weekday at 09:00". */
export function describeSchedule(mode: ScheduleMode, value: string): string {
  const at = parseLocalDateTime(value)
  if (!at) return ''
  const time = at.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
  if (mode === 'later') {
    return `for ${at.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })}, ${time}`
  }
  if (mode === 'daily') return `every day at ${time}`
  if (mode === 'weekdays') return `every weekday at ${time}`
  return `every ${at.toLocaleDateString('en-GB', { weekday: 'long' })} at ${time}`
}
