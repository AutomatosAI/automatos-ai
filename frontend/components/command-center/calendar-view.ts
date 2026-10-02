/**
 * What the calendar shows for a view (split out of calendar-tab.tsx, PRD-251 US-307):
 * the window a mode and anchor cover, its title, where Prev / Today / Next move it,
 * and the feed's items as the grid's events, the 24/7 band and Next Up.
 */
import { useMemo } from 'react'

import type { ScheduleItem } from '@/hooks/use-activity-api'
import type { ScheduleItemType } from './calendar-kinds'
import {
  BAND_MAX_INTERVAL_MIN,
  NEXTUP_OVERDUE_MAX_MS,
  addDays,
  buildMonthGrid,
  buildWeek,
  expandItem,
  startOfMonthGrid,
  startOfWeek,
  toDayCell,
  type CalEvent,
  type DayCell,
  type ViewMode,
  type WindowSpan,
} from './calendar-model'

/** The window covered by the current view (used to expand recurring events). */
export function windowSpanFor(mode: ViewMode, anchor: Date): WindowSpan {
  if (mode === 'day') {
    const start = new Date(anchor)
    start.setHours(0, 0, 0, 0)
    const end = new Date(start)
    end.setHours(23, 59, 59, 999)
    return { start, end }
  }
  // week → 7 days; month → the 6-week grid
  const start = mode === 'week' ? startOfWeek(anchor) : startOfMonthGrid(anchor)
  const end = addDays(start, mode === 'week' ? 7 : 42)
  end.setMilliseconds(end.getMilliseconds() - 1)
  return { start, end }
}

export function titleFor(mode: ViewMode, anchor: Date, week: DayCell[]): string {
  if (mode === 'day') {
    return anchor.toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })
  }
  if (mode === 'week') {
    const day = (d: Date) => d.toLocaleDateString('en-GB', { month: 'long', day: 'numeric' })
    return `${day(week[0].date)} — ${day(week[6].date)}, ${week[6].date.getFullYear()}`
  }
  return anchor.toLocaleDateString('en-GB', { month: 'long', year: 'numeric' })
}

/** Prev (-1) / Today (0) / Next (1). */
export function shiftedAnchor(anchor: Date, mode: ViewMode, direction: -1 | 0 | 1): Date {
  if (direction === 0) return new Date()
  const next = new Date(anchor)
  if (mode === 'day') next.setDate(anchor.getDate() + direction)
  else if (mode === 'week') next.setDate(anchor.getDate() + 7 * direction)
  else next.setMonth(anchor.getMonth() + direction)
  return next
}

/** Ported from the deleted classic ActivityCalendar (PRD-162 S4): the soonest
 *  upcoming items. A deadline overdue by more than a week drops out. */
function nextUpOf(items: ScheduleItem[]): ScheduleItem[] {
  const floor = Date.now() - NEXTUP_OVERDUE_MAX_MS
  return [...items]
    .filter((i) => i.next_run_at && new Date(i.next_run_at).getTime() >= floor)
    .sort((a, b) => new Date(a.next_run_at as string).getTime() - new Date(b.next_run_at as string).getTime())
    .slice(0, 6)
}

/** Routines frequent enough to summarise rather than plot. */
function alwaysOnOf(items: ScheduleItem[]): ScheduleItem[] {
  return items.filter(
    (s) => s.type === 'routine' && (s.recurrence?.interval_minutes ?? Number.POSITIVE_INFINITY) <= BAND_MAX_INTERVAL_MIN,
  )
}

export interface CalendarView {
  week: DayCell[]
  monthCells: DayCell[]
  visibleDays: DayCell[]
  events: CalEvent[]
  alwaysOn: ScheduleItem[]
  nextUp: ScheduleItem[]
}

/** The feed's items (legend chips hide a kind everywhere) as this view shows them. */
export function useCalendarView(
  items: ScheduleItem[],
  hiddenKinds: ReadonlySet<ScheduleItemType>,
  mode: ViewMode,
  anchor: Date,
): CalendarView {
  const week = useMemo(() => buildWeek(anchor), [anchor])
  const monthCells = useMemo(() => buildMonthGrid(anchor), [anchor])
  const visible = useMemo(() => items.filter((i) => !hiddenKinds.has(i.type)), [items, hiddenKinds])
  const visibleDays = useMemo(() => (mode === 'day' ? [toDayCell(new Date(anchor))] : week), [mode, anchor, week])
  const events = useMemo(() => {
    const span = windowSpanFor(mode, anchor)
    return visible.flatMap((item) => expandItem(item, span))
  }, [visible, mode, anchor])
  const alwaysOn = useMemo(() => alwaysOnOf(visible), [visible])
  const nextUp = useMemo(() => nextUpOf(visible), [visible])
  return { week, monthCells, visibleDays, events, alwaysOn, nextUp }
}
