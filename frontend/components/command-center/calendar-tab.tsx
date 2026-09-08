'use client'

/**
 * CalendarTab — Studio calendar for scheduled work.
 *
 * Three real view modes (all wired):
 *  - Day: single column for the selected day, hour-by-hour
 *  - Week: 7 columns (Sun–Sat), hour-by-hour (default)
 *  - Month: classic 6×7 grid with event chips per day
 *
 * ONE data source: `useActivitySchedule` (GET /api/activity/schedule), the
 * DB-first feed PRD-162 made identical on every worker. It carries five kinds
 * of item (see ScheduleItemType): heartbeat routines with a structured
 * recurrence, cron playbooks, agent-scheduled tasks, mission SLA deadlines and
 * board-task SLA deadlines. The always-on band and the grid's routine rows are
 * built from the routine items' `recurrence.interval_minutes` — NOT from
 * /api/heartbeat/workspace, which sits behind the PRD-143 router-wide
 * super-admin lock (a 403 for every other user, polled every 30s) and read
 * `next_run_at` from the one worker that hosts APScheduler (null on the others,
 * so routines anchored at "now" and shifted on every refresh).
 *
 * Events are actionable: clicking one opens a small menu — open the agent /
 * playbook / mission / board card, pause a routine, pause or cancel a
 * scheduled task. Every action rides an endpoint that already exists (see
 * calendar-actions.ts). Deadline items carry a DUE tag; overdue ones go amber.
 *
 * Prev / Today / Next actually move the visible window. No speculative
 * "Schedule task" / "Filter" / "Export" CTAs — those don't belong on a
 * monitoring surface.
 *
 * Mounted by BOTH the studio CommandCenterShell and the classic ActivityPage
 * (dark / light / matte themes). Every `cc-cal-*` rule in globals.css is
 * scoped `:is(.studio, .cc-cal-root)`, so the `.cc-cal-root` wrapper below is
 * what styles the calendar when no `.studio` ancestor exists — drop it and
 * the classic Command Centre renders the grid as a stack of unstyled divs.
 */

import { useMemo, useState, type ReactNode } from 'react'
import { useRouter } from 'next/navigation'
import { ChevronLeft, ChevronRight, Zap } from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  useActivitySchedule,
  useSchedulerHealth,
  type ScheduleItem,
} from '@/hooks/use-activity-api'
import { useToggleHeartbeat } from '@/hooks/use-heartbeats-api'
import { useUpdateScheduledTaskStatus } from '@/hooks/use-scheduled-tasks-api'
import { toneFor } from './agent-tones'
import {
  buildEventActions,
  isDeadlineItem,
  type EventAction,
  type EventActionDeps,
} from './calendar-actions'

type ViewMode = 'day' | 'week' | 'month'

const HOUR_PX = 44
const START_HR = 0
const END_HR = 22
const HOURS = Array.from({ length: END_HR - START_HR + 1 }, (_, i) => i + START_HR)
/** Routines more frequent than this stay in the always-on band: on the grid
 *  they would clutter every column. */
const GRID_MIN_INTERVAL_MIN = 30
/** Routines at least this frequent are summarised in the always-on band. */
const BAND_MAX_INTERVAL_MIN = 60
/** Safety cap per item so a frequent routine can't run away across a month. */
const MAX_OCCURRENCES = 200
const ROUTINE_MIN_DUR_MIN = 15
const ROUTINE_MAX_DUR_MIN = 45
const RECURRING_DUR_MIN = 20
const SINGLE_DUR_MIN = 15
const OVERDUE_TONE = 'hsl(45 80% 55%)'

interface DayCell {
  short: string
  iso: string
  n: number
  month: number
  today: boolean
  date: Date
}

interface CalEvent {
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

interface WindowSpan {
  start: Date
  end: Date
}

/**
 * Parse "Every 30m", "Every 8h", "Every 1d" → minutes. Returns null on
 * raw cron expressions (those get handled by getRecurringDays).
 */
function parseIntervalMinutes(frequency: string | null | undefined): number | null {
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
function cronDaysOfWeek(frequency: string | null | undefined): number[] | null {
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
function cronHourMin(frequency: string | null | undefined): { h: number; m: number } | null {
  if (!frequency) return null
  const m = frequency.match(/^(\d+)\s+(\d+)\s+/)
  if (!m) return null
  return { m: parseInt(m[1], 10), h: parseInt(m[2], 10) }
}

function startOfWeek(d: Date): Date {
  const out = new Date(d)
  out.setHours(0, 0, 0, 0)
  out.setDate(out.getDate() - out.getDay())
  return out
}
function startOfMonth(d: Date): Date {
  const out = new Date(d)
  out.setHours(0, 0, 0, 0)
  out.setDate(1)
  return out
}
function startOfMonthGrid(d: Date): Date {
  return startOfWeek(startOfMonth(d))
}
function addDays(d: Date, n: number): Date {
  const out = new Date(d)
  out.setDate(out.getDate() + n)
  return out
}

function toDayCell(d: Date): DayCell {
  return {
    short: d.toLocaleDateString('en-GB', { weekday: 'short' }).toUpperCase(),
    iso: d.toISOString().slice(0, 10),
    n: d.getDate(),
    month: d.getMonth(),
    today: d.toDateString() === new Date().toDateString(),
    date: d,
  }
}

function buildWeek(anchor: Date): DayCell[] {
  const sun = startOfWeek(anchor)
  return Array.from({ length: 7 }, (_, i) => toDayCell(addDays(sun, i)))
}

function buildMonthGrid(anchor: Date): DayCell[] {
  const start = startOfMonthGrid(anchor)
  return Array.from({ length: 42 }, (_, i) => toDayCell(addDays(start, i)))
}

/** Relative "in 12m / in 3h / in 2d" label for the Next Up list. */
function formatNextRun(iso: string | null): string {
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

function occurrence(item: ScheduleItem, d: Date, durMin: number, recurring: boolean): CalEvent {
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

/** Walk an interval backwards and forwards from its anchor across the window. */
function expandInterval(
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
function expandItem(item: ScheduleItem, span: WindowSpan): CalEvent[] {
  if (item.type === 'routine') {
    // Structured recurrence from the feed — no string parsing. Sub-30-minute
    // routines live in the always-on band only; a routine with no next run
    // (outside its active hours for the whole horizon) has nothing to place.
    const interval = item.recurrence?.interval_minutes ?? null
    if (interval === null || interval < GRID_MIN_INTERVAL_MIN || !item.next_run_at) return []
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

function EventMenu({ actions, children }: { actions: EventAction[]; children: ReactNode }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-[180px]">
        {actions.map((a) => (
          <DropdownMenuItem
            key={a.label}
            onSelect={() => a.run()}
            className={a.tone === 'danger' ? 'text-destructive focus:text-destructive' : undefined}
          >
            {a.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function DueTag({ overdue }: { overdue: boolean }) {
  const tone = overdue ? 'hsl(var(--destructive))' : OVERDUE_TONE
  return (
    <span
      className="due"
      style={{
        marginLeft: 6,
        padding: '0 4px',
        borderRadius: 3,
        fontWeight: 700,
        letterSpacing: 0.4,
        background: overdue ? 'hsl(var(--destructive) / 0.15)' : 'hsl(45 80% 55% / 0.18)',
        color: tone,
      }}
    >
      {overdue ? 'OVERDUE' : 'DUE'}
    </span>
  )
}

export function CalendarTab() {
  const router = useRouter()
  const [mode, setMode] = useState<ViewMode>('week')
  const [anchor, setAnchor] = useState<Date>(() => new Date())

  const range = mode === 'month' ? '30d' : '7d'
  const { data: schedule, isLoading, isError, refetch } = useActivitySchedule(range)
  const { data: health } = useSchedulerHealth()
  const { mutate: toggleHeartbeat } = useToggleHeartbeat()
  const { mutate: updateScheduledTask } = useUpdateScheduledTaskStatus()

  const actionDeps = useMemo<EventActionDeps>(
    () => ({
      navigate: (href) => router.push(href as any),
      pauseRoutine: (agentId) => toggleHeartbeat(agentId),
      setScheduledTaskStatus: (taskId, status) => updateScheduledTask({ taskId, status }),
    }),
    [router, toggleHeartbeat, updateScheduledTask],
  )

  const week = useMemo(() => buildWeek(anchor), [anchor])
  const monthCells = useMemo(() => buildMonthGrid(anchor), [anchor])

  // Ported from the deleted classic ActivityCalendar (PRD-162 S4): the soonest
  // upcoming items, straight from the DB-first schedule feed.
  const nextUp = useMemo(() => {
    const items = schedule?.scheduled ?? []
    return [...items]
      .filter((i) => i.next_run_at)
      .sort(
        (a, b) =>
          new Date(a.next_run_at as string).getTime() -
          new Date(b.next_run_at as string).getTime(),
      )
      .slice(0, 6)
  }, [schedule])

  const visibleDays = useMemo<DayCell[]>(
    () => (mode === 'day' ? [toDayCell(new Date(anchor))] : week),
    [mode, anchor, week],
  )

  // Window covered by the current view (used to expand recurring events).
  const windowSpan = useMemo<WindowSpan>(() => {
    if (mode === 'day') {
      const start = new Date(anchor)
      start.setHours(0, 0, 0, 0)
      const end = new Date(start)
      end.setHours(23, 59, 59, 999)
      return { start, end }
    }
    if (mode === 'week') {
      const start = startOfWeek(anchor)
      const end = addDays(start, 7)
      end.setMilliseconds(end.getMilliseconds() - 1)
      return { start, end }
    }
    // month → 6-week grid
    const start = startOfMonthGrid(anchor)
    const end = addDays(start, 42)
    end.setMilliseconds(end.getMilliseconds() - 1)
    return { start, end }
  }, [mode, anchor])

  const events = useMemo<CalEvent[]>(
    () => (schedule?.scheduled ?? []).flatMap((item) => expandItem(item, windowSpan)),
    [schedule, windowSpan],
  )

  // Routines frequent enough to summarise rather than plot, from the same feed.
  const alwaysOn = useMemo(
    () =>
      (schedule?.scheduled ?? []).filter(
        (s) =>
          s.type === 'routine' &&
          (s.recurrence?.interval_minutes ?? Number.POSITIVE_INFINITY) <= BAND_MAX_INTERVAL_MIN,
      ),
    [schedule],
  )

  const now = new Date()
  const nowHourPos =
    (now.getHours() - START_HR) * HOUR_PX + (now.getMinutes() / 60) * HOUR_PX

  const shiftAnchor = (direction: -1 | 0 | 1) => {
    if (direction === 0) {
      setAnchor(new Date())
      return
    }
    setAnchor((a) => {
      const next = new Date(a)
      if (mode === 'day') next.setDate(a.getDate() + direction)
      else if (mode === 'week') next.setDate(a.getDate() + 7 * direction)
      else next.setMonth(a.getMonth() + direction)
      return next
    })
  }

  const monthLabel = (() => {
    if (mode === 'day') {
      return anchor.toLocaleDateString('en-GB', {
        weekday: 'long',
        day: 'numeric',
        month: 'long',
        year: 'numeric',
      })
    }
    if (mode === 'week') {
      const w = week
      return `${w[0].date.toLocaleDateString('en-GB', { month: 'long', day: 'numeric' })} — ${w[6].date.toLocaleDateString('en-GB', { month: 'long', day: 'numeric' })}, ${w[6].date.getFullYear()}`
    }
    return anchor.toLocaleDateString('en-GB', { month: 'long', year: 'numeric' })
  })()

  return (
    <div className="cc-cal-root">
      <div className="cc-cal-toolbar">
        <div className="cc-seg" role="group" aria-label="Calendar mode">
          <button
            type="button"
            className={mode === 'day' ? 'on' : ''}
            onClick={() => setMode('day')}
          >
            Day
          </button>
          <button
            type="button"
            className={mode === 'week' ? 'on' : ''}
            onClick={() => setMode('week')}
          >
            Week
          </button>
          <button
            type="button"
            className={mode === 'month' ? 'on' : ''}
            onClick={() => setMode('month')}
          >
            Month
          </button>
        </div>
        <div style={{ display: 'inline-flex', gap: 4 }}>
          <button
            type="button"
            className="cc-btn"
            style={{ width: 30, padding: 0 }}
            onClick={() => shiftAnchor(-1)}
            aria-label="Previous"
          >
            <ChevronLeft style={{ width: 13, height: 13 }} />
          </button>
          <button
            type="button"
            className="cc-btn"
            style={{ fontSize: 11.5, padding: '0 10px' }}
            onClick={() => shiftAnchor(0)}
          >
            Today
          </button>
          <button
            type="button"
            className="cc-btn"
            style={{ width: 30, padding: 0 }}
            onClick={() => shiftAnchor(1)}
            aria-label="Next"
          >
            <ChevronRight style={{ width: 13, height: 13 }} />
          </button>
        </div>
        <span
          style={{
            fontFamily: 'var(--font-newsreader, serif)',
            fontSize: 17,
            fontWeight: 500,
            color: 'hsl(var(--foreground))',
          }}
        >
          {monthLabel}
        </span>
      </div>

      {health?.healthy === false && (
        <div
          className="cc-cal-health"
          role="status"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '6px 10px',
            marginBottom: 8,
            fontSize: 12,
            border: '1px solid hsl(45 80% 55% / 0.4)',
            borderRadius: 8,
            background: 'hsl(45 80% 55% / 0.08)',
            color: OVERDUE_TONE,
          }}
        >
          <Zap style={{ width: 12, height: 12 }} />
          Scheduler hasn’t fired recently — configured schedules still shown below.
        </div>
      )}

      {isError && (
        <div
          className="cc-panel-empty"
          role="alert"
          style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}
        >
          <span>Couldn’t load the schedule.</span>
          <button type="button" className="cc-btn" onClick={() => refetch()}>
            Retry
          </button>
        </div>
      )}

      {mode !== 'month' && alwaysOn.length > 0 && (
        <div className="cc-cal-alwayson">
          <div className="lbl">
            <Zap style={{ width: 12, height: 12, color: OVERDUE_TONE }} />
            24/7
          </div>
          <div className="pills">
            {alwaysOn.map((s) => {
              const tone = toneFor(s.agent_name)
              return (
                <EventMenu key={s.id} actions={buildEventActions(s, actionDeps)}>
                  <button
                    type="button"
                    className="pill"
                    style={{ borderLeftColor: tone.bg, cursor: 'pointer' }}
                    title={`${s.agent_name} routine — click for actions`}
                  >
                    <span className="dot" />
                    {s.agent_name} · every {s.recurrence?.interval_minutes}m
                  </button>
                </EventMenu>
              )
            })}
          </div>
        </div>
      )}

      {nextUp.length > 0 && (
        <div
          className="cc-cal-nextup"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            flexWrap: 'wrap',
            padding: '8px 10px',
            marginBottom: 8,
            border: '1px solid hsl(var(--border))',
            borderRadius: 8,
            background: 'hsl(var(--card) / 0.4)',
          }}
        >
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: 0.4,
              color: 'hsl(var(--muted-foreground))',
            }}
          >
            Next up
          </span>
          {nextUp.map((item) => {
            const overdue = item.next_run_at
              ? new Date(item.next_run_at).getTime() < Date.now()
              : false
            const accent = overdue ? OVERDUE_TONE : 'hsl(var(--muted-foreground))'
            const due = isDeadlineItem(item)
            return (
              <span
                key={item.id}
                style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12 }}
              >
                <span
                  style={{ width: 6, height: 6, borderRadius: 999, background: accent }}
                />
                <span style={{ fontWeight: 500 }}>{item.name}</span>
                <span style={{ color: accent }}>
                  {overdue ? (due ? 'overdue' : 'missed') : `${due ? 'due ' : ''}${formatNextRun(item.next_run_at)}`}
                </span>
              </span>
            )
          })}
        </div>
      )}

      {mode === 'month' ? (
        <MonthGrid
          cells={monthCells}
          events={events}
          anchorMonth={anchor.getMonth()}
          actionDeps={actionDeps}
        />
      ) : (
        <div className="cc-cal-frame">
          <div
            className="cc-cal-head"
            style={{
              gridTemplateColumns:
                mode === 'day' ? '60px 1fr' : '60px repeat(7, 1fr)',
            }}
          >
            <div className="h" />
            {visibleDays.map((d) => (
              <div key={d.iso} className={`h${d.today ? ' today' : ''}`}>
                <div className="d">{d.short}</div>
                <div className="n">{d.today ? <span>{d.n}</span> : d.n}</div>
              </div>
            ))}
          </div>

          <div
            className="cc-cal-grid"
            style={{
              gridTemplateColumns:
                mode === 'day' ? '60px 1fr' : '60px repeat(7, 1fr)',
              gridTemplateRows: `repeat(${HOURS.length}, ${HOUR_PX}px)`,
              backgroundSize: `100% ${HOUR_PX}px`,
              backgroundImage: `linear-gradient(to bottom, transparent 0, transparent calc(${HOUR_PX}px - 1px), hsl(var(--border)) calc(${HOUR_PX}px - 1px), hsl(var(--border)) ${HOUR_PX}px)`,
            }}
          >
            <div
              style={{
                gridColumn: 1,
                gridRow: `1 / span ${HOURS.length}`,
                borderRight: '1px solid hsl(var(--border))',
              }}
            >
              {HOURS.map((h) => (
                <div
                  key={h}
                  className="cc-cal-hourlabel"
                  style={{ height: HOUR_PX }}
                >
                  {String(h).padStart(2, '0')}:00
                </div>
              ))}
            </div>

            {visibleDays.map((d, di) => {
              const dayEvents = events.filter(
                (e) => e.dayKey === d.date.toDateString(),
              )
              return (
                <div
                  key={d.iso}
                  className={`cc-cal-daycol${d.today ? ' today' : ''}`}
                  style={{
                    gridColumn: di + 2,
                    gridRow: `1 / span ${HOURS.length}`,
                    height: HOURS.length * HOUR_PX,
                  }}
                >
                  {dayEvents.map((evt) => {
                    const top =
                      (evt.hour - START_HR) * HOUR_PX +
                      (evt.min / 60) * HOUR_PX
                    const height = Math.max((evt.durMin / 60) * HOUR_PX, 28)
                    const tone = toneFor(evt.agent)
                    const overdue = Boolean(evt.due) && evt.date.getTime() < Date.now()
                    return (
                      <EventMenu key={evt.id} actions={buildEventActions(evt.item, actionDeps)}>
                        <button
                          type="button"
                          className="cc-cal-event"
                          style={{
                            top,
                            height,
                            borderLeftColor: overdue ? OVERDUE_TONE : tone.bg,
                            background: 'hsl(var(--secondary))',
                          }}
                          title={`${evt.name} — click for actions`}
                        >
                          <div className="nm">
                            {String(evt.hour).padStart(2, '0')}:
                            {String(evt.min).padStart(2, '0')}
                            {evt.agent && (
                              <span style={{ marginLeft: 6, opacity: 0.85 }}>
                                · {evt.agent}
                              </span>
                            )}
                            {evt.due && <DueTag overdue={overdue} />}
                          </div>
                          <div className="ttl">{evt.name}</div>
                        </button>
                      </EventMenu>
                    )
                  })}
                  {d.today && (
                    <div
                      className="cc-cal-nowline"
                      style={{ top: nowHourPos }}
                    />
                  )}
                </div>
              )
            })}
          </div>

          {isLoading && events.length === 0 && (
            <div className="cc-panel-empty">Loading schedule…</div>
          )}
          {!isLoading && !isError && events.length === 0 && alwaysOn.length === 0 && (
            <div className="cc-panel-empty">
              No scheduled work in this window. Enable a heartbeat in /agents
              to populate the calendar.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function MonthGrid({
  cells,
  events,
  anchorMonth,
  actionDeps,
}: {
  cells: DayCell[]
  events: CalEvent[]
  anchorMonth: number
  actionDeps: EventActionDeps
}) {
  const todayStart = new Date()
  todayStart.setHours(0, 0, 0, 0)
  return (
    <div className="cc-cal-month">
      <div className="cc-cal-month-head">
        {['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'].map((d) => (
          <div key={d} className="h">
            {d}
          </div>
        ))}
      </div>
      <div className="cc-cal-month-grid">
        {cells.map((d) => {
          const dayEvents = events.filter(
            (e) => e.dayKey === d.date.toDateString(),
          )
          const outside = d.month !== anchorMonth
          const past = d.date.getTime() < todayStart.getTime()
          return (
            <div
              key={d.iso}
              className={`cc-cal-month-cell${outside ? ' outside' : ''}${d.today ? ' today' : ''}${past ? ' past' : ''}`}
            >
              <div className="n">{d.today ? <span>{d.n}</span> : d.n}</div>
              <div className="evts">
                {dayEvents.slice(0, 3).map((evt) => {
                  const tone = toneFor(evt.agent)
                  const overdue = Boolean(evt.due) && evt.date.getTime() < Date.now()
                  return (
                    <EventMenu key={evt.id} actions={buildEventActions(evt.item, actionDeps)}>
                      <button
                        type="button"
                        className="cc-cal-month-evt"
                        style={{
                          borderLeftColor: overdue ? OVERDUE_TONE : tone.bg,
                          opacity: past && !evt.due ? 0.5 : 1,
                        }}
                        title={evt.name}
                      >
                        <span className="t">
                          {String(evt.hour).padStart(2, '0')}:
                          {String(evt.min).padStart(2, '0')}
                        </span>{' '}
                        {evt.due && <DueTag overdue={overdue} />}{' '}
                        <span className="n2">{evt.name}</span>
                      </button>
                    </EventMenu>
                  )
                })}
                {dayEvents.length > 3 && (
                  <div className="cc-cal-month-more">
                    +{dayEvents.length - 3} more
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
