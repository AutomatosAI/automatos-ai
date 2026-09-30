/**
 * The calendar's model (split out of calendar-tab.tsx, PRD-251 US-307): its layout
 * constants, what a day cell and an event are, and how a feed item becomes the
 * events a window shows (recurrence expansion). Pure: no React.
 */
import type { ScheduleItem } from '@/hooks/use-activity-api'
import { isDeadlineItem } from './calendar-actions'
import { isSocialItem, socialTimeLabel } from './calendar-social'

export type ViewMode = 'day' | 'week' | 'month'

export const HOUR_PX = 44
export const START_HR = 0
export const END_HR = 22
export const HOURS = Array.from({ length: END_HR - START_HR + 1 }, (_, i) => i + START_HR)
/** Smallest rendered event box: the time line plus the title line (28px
 *  clipped the title). The overlap layout uses the same floor, so two short
 *  events closer together than this share the column instead of drawing on
 *  top of each other. */
export const MIN_EVENT_PX = 38
export const MIN_EVENT_MIN = (MIN_EVENT_PX / HOUR_PX) * 60
/** Lanes a column can show legibly per view; a slot needing more becomes one
 *  stacked card with a submenu per member. */
export const MAX_LANES: Record<ViewMode, number> = { day: 6, week: 2, month: 1 }
/** Routines this frequent or more (heartbeats every 5/15/30/60 min) live in the
 *  always-on band ONLY. Plotted, an hourly heartbeat is 12+ blocks a day per
 *  agent and buries the one-off work the grid is for (2026-09-11). */
export const BAND_MAX_INTERVAL_MIN = 60
/** Next Up keeps an overdue deadline this long; after that it is the board's problem. */
export const NEXTUP_OVERDUE_MAX_MS = 7 * 24 * 60 * 60_000
/** Safety cap per item so a frequent routine can't run away across a month. */
export const MAX_OCCURRENCES = 200
export const ROUTINE_MIN_DUR_MIN = 15
export const ROUTINE_MAX_DUR_MIN = 45
export const RECURRING_DUR_MIN = 20
export const SINGLE_DUR_MIN = 15
export const OVERDUE_TONE = 'hsl(45 80% 55%)'


export interface DayCell {
  short: string
  iso: string
  n: number
  month: number
  today: boolean
  date: Date
}

export interface CalEvent {
  id: string
  hour: number
  min: number
  durMin: number
  name: string
  agent: string | null
  dayKey: string
  date: Date
  /** the feed item behind this occurrence — the event menu acts on it */
  item: ScheduleItem
  /** true when this event is a synthesised recurrence (not the literal next_run_at) */
  recurring?: boolean
  /** an SLA deadline (mission or board task), not a run */
  due?: boolean
}

export interface WindowSpan {
  start: Date
  end: Date
}

/**
 * Parse "Every 30m", "Every 8h", "Every 1d" → minutes. Returns null on
 * raw cron expressions (those get handled by getRecurringDays).
 */
export function parseIntervalMinutes(frequency: string | null | undefined): number | null {
  if (!frequency) return null
  const m = frequency.match(/every\s+(\d+)\s*(m|min|h|hr|d|day)/i)
  if (!m) return null
  const v = parseInt(m[1], 10)
  const u = m[2].toLowerCase()
  if (u.startsWith('m')) return v
  if (u.startsWith('h')) return v * 60
  if (u.startsWith('d')) return v * 1440
  return null
}

/**
 * Cron field → day-of-week indices (0 = Sun ... 6 = Sat).
 * Returns null when the cron expression doesn't have a parseable DOW field.
 */
export function cronDaysOfWeek(frequency: string | null | undefined): number[] | null {
  if (!frequency) return null
  const m = frequency.match(
    /^[\d,/*-]+\s+[\d,/*-]+\s+[\d,/*-]+\s+[\d,/*-]+\s+([\d,/*-]+)$/,
  )
  if (!m) return null
  const f = m[1]
  if (f === '*') return [0, 1, 2, 3, 4, 5, 6]
  const range = f.match(/^(\d)-(\d)$/)
  if (range) {
    const out: number[] = []
    for (let i = parseInt(range[1], 10); i <= parseInt(range[2], 10); i++) out.push(i)
    return out
  }
  if (/^[\d,]+$/.test(f)) return f.split(',').map(Number)
  return null
}

/**
 * Cron's hour + minute fields → hour-of-day. Returns null when not parseable.
 */
export function cronHourMin(frequency: string | null | undefined): { h: number; m: number } | null {
  if (!frequency) return null
  const m = frequency.match(/^(\d+)\s+(\d+)\s+/)
  if (!m) return null
  return { m: parseInt(m[1], 10), h: parseInt(m[2], 10) }
}

export function startOfWeek(d: Date): Date {
  const out = new Date(d)
  out.setHours(0, 0, 0, 0)
  out.setDate(out.getDate() - out.getDay())
  return out
}
export function startOfMonth(d: Date): Date {
  const out = new Date(d)
  out.setHours(0, 0, 0, 0)
  out.setDate(1)
  return out
}
export function startOfMonthGrid(d: Date): Date {
  return startOfWeek(startOfMonth(d))
}
export function addDays(d: Date, n: number): Date {
  const out = new Date(d)
  out.setDate(out.getDate() + n)
  return out
}

export function toDayCell(d: Date): DayCell {
  return {
    short: d.toLocaleDateString('en-GB', { weekday: 'short' }).toUpperCase(),
    iso: d.toISOString().slice(0, 10),
    n: d.getDate(),
    month: d.getMonth(),
    today: d.toDateString() === new Date().toDateString(),
    date: d,
  }
}

export function buildWeek(anchor: Date): DayCell[] {
  const sun = startOfWeek(anchor)
  return Array.from({ length: 7 }, (_, i) => toDayCell(addDays(sun, i)))
}

export function buildMonthGrid(anchor: Date): DayCell[] {
  const start = startOfMonthGrid(anchor)
  return Array.from({ length: 42 }, (_, i) => toDayCell(addDays(start, i)))
}

/** Relative "in 12m / in 3h / in 2d" label for the Next Up list. */
export function formatNextRun(iso: string | null): string {
  if (!iso) return ''
  const ms = new Date(iso).getTime() - Date.now()
  if (Number.isNaN(ms)) return ''
  if (ms <= 0) return 'now'
  const mins = Math.round(ms / 60000)
  if (mins < 60) return `in ${mins}m`
  const hrs = Math.round(mins / 60)
  if (hrs < 24) return `in ${hrs}h`
  return `in ${Math.round(hrs / 24)}d`
}

export function occurrence(item: ScheduleItem, d: Date, durMin: number, recurring: boolean): CalEvent {
  return {
    id: `${item.id}-${d.getTime()}`,
    hour: d.getHours(),
    min: d.getMinutes(),
    durMin,
    name: item.name,
    agent: item.agent_name,
    dayKey: d.toDateString(),
    date: d,
    item,
    recurring,
    due: isDeadlineItem(item),
  }
}

/** [start, end] in ms — what the overlap layout compares. The end is the
 *  rendered box, never shorter than MIN_EVENT_MIN. */
export const eventSpan = (evt: CalEvent): [number, number] => [
  evt.date.getTime(),
  evt.date.getTime() + Math.max(evt.durMin, MIN_EVENT_MIN) * 60_000,
]

/** Walk an interval backwards and forwards from its anchor across the window. */
export function expandInterval(
  item: ScheduleItem,
  anchor: Date,
  intervalMin: number,
  span: WindowSpan,
  durMin: number,
): CalEvent[] {
  const out: CalEvent[] = []
  const step = intervalMin * 60_000
  let count = 0
  let t = anchor.getTime()
  while (t >= span.start.getTime() && count < MAX_OCCURRENCES) {
    if (t <= span.end.getTime()) {
      out.push(occurrence(item, new Date(t), durMin, true))
      count++
    }
    t -= step
  }
  t = anchor.getTime() + step
  while (t <= span.end.getTime() && count < MAX_OCCURRENCES) {
    if (t >= span.start.getTime()) {
      out.push(occurrence(item, new Date(t), durMin, true))
      count++
    }
    t += step
  }
  return out
}

/** Every occurrence of one feed item inside the window. */
export function expandItem(item: ScheduleItem, span: WindowSpan): CalEvent[] {
  if (item.type === 'routine') {
    // Structured recurrence from the feed — no string parsing. Routines at the
    // band cadence or faster live in the always-on band only; a routine with
    // no next run (outside its active hours for the whole horizon) has
    // nothing to place.
    const interval = item.recurrence?.interval_minutes ?? null
    if (interval === null || interval <= BAND_MAX_INTERVAL_MIN || !item.next_run_at) return []
    const dur = Math.min(Math.max(interval, ROUTINE_MIN_DUR_MIN), ROUTINE_MAX_DUR_MIN)
    return expandInterval(item, new Date(item.next_run_at), interval, span, dur)
  }

  const interval = parseIntervalMinutes(item.frequency)
  if (interval !== null && interval > 60) {
    const anchorDate = item.next_run_at ? new Date(item.next_run_at) : new Date()
    return expandInterval(item, anchorDate, interval, span, RECURRING_DUR_MIN)
  }

  // Cron with day-of-week + hour fields (e.g. "0 9 * * 1-5") — an event on
  // every matching day in the window.
  const dow = cronDaysOfWeek(item.frequency)
  const cronHm = cronHourMin(item.frequency)
  if (dow && cronHm) {
    const out: CalEvent[] = []
    for (let i = 0; i < 42; i++) {
      const d = addDays(span.start, i)
      if (d > span.end) break
      if (!dow.includes(d.getDay())) continue
      d.setHours(cronHm.h, cronHm.m, 0, 0)
      out.push(occurrence(item, new Date(d), RECURRING_DUR_MIN, true))
    }
    return out
  }

  // Single occurrence at next_run_at (one-shot tasks, SLA deadlines).
  if (item.next_run_at) {
    const d = new Date(item.next_run_at)
    if (d >= span.start && d <= span.end) return [occurrence(item, d, SINGLE_DUR_MIN, false)]
  }
  return []
}

export const hhmm = (evt: CalEvent) =>
  `${String(evt.hour).padStart(2, '0')}:${String(evt.min).padStart(2, '0')}`

/** The event's time: a social post's in its own timezone (PRD-251 US-307). */
export const timeLabel = (evt: CalEvent) => (isSocialItem(evt.item) ? socialTimeLabel(evt.item) : hhmm(evt))
