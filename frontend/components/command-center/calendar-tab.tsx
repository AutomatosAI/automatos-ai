'use client'

/**
 * CalendarTab — Studio calendar for scheduled routines.
 *
 * Three real view modes (all wired):
 *  - Day: single column for the selected day, hour-by-hour
 *  - Week: 7 columns (Sun–Sat), hour-by-hour (default)
 *  - Month: classic 6×7 grid with event chips per day
 *
 * Events come from `useActivitySchedule`. Always-on heartbeats from
 * `useHeartbeats()` render in a band above week/day views. Events are
 * clickable — clicking routes to the assigned agent's detail page so the
 * user can inspect / pause / tune the routine.
 *
 * Prev / Today / Next actually move the visible window. No speculative
 * "Schedule task" / "Filter" / "Export" CTAs — those don't belong on a
 * monitoring surface.
 */

import { useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import { ChevronLeft, ChevronRight, Zap } from 'lucide-react'
import { useActivitySchedule } from '@/hooks/use-activity-api'
import { useHeartbeats } from '@/hooks/use-heartbeats-api'
import { toneFor } from './agent-tones'

type ViewMode = 'day' | 'week' | 'month'

const HOUR_PX = 44
const START_HR = 0
const END_HR = 22
const HOURS = Array.from({ length: END_HR - START_HR + 1 }, (_, i) => i + START_HR)

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
  agentId: number | null
  dayKey: string
  date: Date
  /** true when this event is a synthesised recurrence (not the literal next_run_at) */
  recurring?: boolean
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

function buildWeek(anchor: Date): DayCell[] {
  const sun = startOfWeek(anchor)
  return Array.from({ length: 7 }, (_, i) => {
    const d = addDays(sun, i)
    return {
      short: d.toLocaleDateString('en-GB', { weekday: 'short' }).toUpperCase(),
      iso: d.toISOString().slice(0, 10),
      n: d.getDate(),
      month: d.getMonth(),
      today: d.toDateString() === new Date().toDateString(),
      date: d,
    }
  })
}

function buildMonthGrid(anchor: Date): DayCell[] {
  const start = startOfMonthGrid(anchor)
  return Array.from({ length: 42 }, (_, i) => {
    const d = addDays(start, i)
    return {
      short: d.toLocaleDateString('en-GB', { weekday: 'short' }).toUpperCase(),
      iso: d.toISOString().slice(0, 10),
      n: d.getDate(),
      month: d.getMonth(),
      today: d.toDateString() === new Date().toDateString(),
      date: d,
    }
  })
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

export function CalendarTab() {
  const router = useRouter()
  const [mode, setMode] = useState<ViewMode>('week')
  const [anchor, setAnchor] = useState<Date>(() => new Date())

  const range = mode === 'month' ? '30d' : '7d'
  const { data: schedule, isLoading, isError, refetch } = useActivitySchedule(range)
  const { data: heartbeats } = useHeartbeats()

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

  const visibleDays = useMemo<DayCell[]>(() => {
    if (mode === 'day') {
      return [
        {
          short: anchor
            .toLocaleDateString('en-GB', { weekday: 'short' })
            .toUpperCase(),
          iso: anchor.toISOString().slice(0, 10),
          n: anchor.getDate(),
          month: anchor.getMonth(),
          today: anchor.toDateString() === new Date().toDateString(),
          date: new Date(anchor),
        },
      ]
    }
    return week
  }, [mode, anchor, week])

  // Window covered by the current view (used to expand recurring events).
  const windowSpan = useMemo(() => {
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

  const events = useMemo<CalEvent[]>(() => {
    const out: CalEvent[] = []
    const MAX_OCCURRENCES = 200 // safety cap so a 5-min heartbeat doesn't run away

    // Pass 1: heartbeats (routines). Use structured interval_minutes — no
    // string parsing — and expand across the visible window.
    if (heartbeats?.heartbeats) {
      heartbeats.heartbeats
        .filter((h) => h.enabled)
        .forEach((h) => {
          const interval = h.interval_minutes
          // Sub-30min heartbeats are summarised in the always-on band only.
          // Putting them on the grid would clutter every column.
          if (interval < 30) return
          const anchorDate = h.next_run_at
            ? new Date(h.next_run_at)
            : new Date()
          const name = `${h.agent_name} routine`
          let count = 0
          // Walk back
          let t = anchorDate.getTime()
          while (t >= windowSpan.start.getTime() && count < MAX_OCCURRENCES) {
            if (t <= windowSpan.end.getTime()) {
              const d = new Date(t)
              out.push({
                id: `hb-${h.id}-${t}`,
                hour: d.getHours(),
                min: d.getMinutes(),
                durMin: Math.min(Math.max(interval, 15), 45),
                name,
                agent: h.agent_name,
                agentId: h.agent_id,
                dayKey: d.toDateString(),
                date: d,
                recurring: true,
              })
              count++
            }
            t -= interval * 60_000
          }
          // Walk forward
          t = anchorDate.getTime() + interval * 60_000
          while (t <= windowSpan.end.getTime() && count < MAX_OCCURRENCES) {
            if (t >= windowSpan.start.getTime()) {
              const d = new Date(t)
              out.push({
                id: `hb-${h.id}-${t}`,
                hour: d.getHours(),
                min: d.getMinutes(),
                durMin: Math.min(Math.max(interval, 15), 45),
                name,
                agent: h.agent_name,
                agentId: h.agent_id,
                dayKey: d.toDateString(),
                date: d,
                recurring: true,
              })
              count++
            }
            t += interval * 60_000
          }
        })
    }

    // Pass 2: items from the schedule endpoint that aren't heartbeats
    // (recipes / one-offs). Fall through if next_run_at is in window.
    if (!schedule?.scheduled) return out
    schedule.scheduled.forEach((s, idx) => {
      // Skip routine items — heartbeats already cover those above and
      // contain structured interval data we don't need to re-parse.
      if (s.type === 'routine') return

      const interval = parseIntervalMinutes(s.frequency)
      const dow = cronDaysOfWeek(s.frequency)
      const cronHm = cronHourMin(s.frequency)

      if (interval !== null && interval > 60) {
        // Anchor at next_run_at and walk backwards + forwards across window.
        const anchorDate = s.next_run_at ? new Date(s.next_run_at) : new Date()
        // Walk back
        let t = anchorDate.getTime()
        while (t > windowSpan.start.getTime() - 1) {
          if (t <= windowSpan.end.getTime()) {
            const d = new Date(t)
            out.push({
              id: `${s.id}-${t}`,
              hour: d.getHours(),
              min: d.getMinutes(),
              durMin: 20,
              name: s.name,
              agent: s.agent_name,
              agentId: s.agent_id,
              dayKey: d.toDateString(),
              date: d,
              recurring: true,
            })
          }
          t -= interval * 60_000
        }
        // Walk forward
        t = anchorDate.getTime() + interval * 60_000
        while (t < windowSpan.end.getTime() + 1) {
          if (t >= windowSpan.start.getTime()) {
            const d = new Date(t)
            out.push({
              id: `${s.id}-${t}`,
              hour: d.getHours(),
              min: d.getMinutes(),
              durMin: 20,
              name: s.name,
              agent: s.agent_name,
              agentId: s.agent_id,
              dayKey: d.toDateString(),
              date: d,
              recurring: true,
            })
          }
          t += interval * 60_000
        }
        return
      }

      // Cron with day-of-week + hour fields (e.g. "0 9 * * 1-5") — render
      // an event on every matching day in the window.
      if (dow && cronHm) {
        for (let i = 0; i < 42; i++) {
          const d = addDays(windowSpan.start, i)
          if (d > windowSpan.end) break
          if (!dow.includes(d.getDay())) continue
          d.setHours(cronHm.h, cronHm.m, 0, 0)
          out.push({
            id: `${s.id}-${d.toISOString()}`,
            hour: cronHm.h,
            min: cronHm.m,
            durMin: 20,
            name: s.name,
            agent: s.agent_name,
            agentId: s.agent_id,
            dayKey: d.toDateString(),
            date: new Date(d),
            recurring: true,
          })
        }
        return
      }

      // Fallback: single occurrence at next_run_at when we can't parse
      // anything else. Better than nothing — the user still sees something.
      if (s.next_run_at) {
        const d = new Date(s.next_run_at)
        if (d >= windowSpan.start && d <= windowSpan.end) {
          out.push({
            id: `${s.id}-${idx}`,
            hour: d.getHours(),
            min: d.getMinutes(),
            durMin: 15,
            name: s.name,
            agent: s.agent_name,
            agentId: s.agent_id,
            dayKey: d.toDateString(),
            date: d,
          })
        }
      }
    })
    return out
  }, [schedule, windowSpan, heartbeats])

  const alwaysOn = useMemo(() => {
    if (!heartbeats?.heartbeats) return []
    return heartbeats.heartbeats
      .filter((h) => h.enabled && h.interval_minutes <= 60)
      .map((h) => ({
        id: h.id,
        name: `${h.agent_name} · every ${h.interval_minutes}m`,
        agent: h.agent_name,
      }))
  }, [heartbeats])

  const now = new Date()
  const nowHourPos =
    (now.getHours() - START_HR) * HOUR_PX + (now.getMinutes() / 60) * HOUR_PX

  const handleEventClick = (evt: CalEvent) => {
    if (evt.agentId) {
      router.push(`/agents?agent=${evt.agentId}` as any)
    } else {
      router.push('/command-center?tab=activity' as any)
    }
  }

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
    <>
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
            <Zap style={{ width: 12, height: 12, color: 'hsl(45 80% 55%)' }} />
            24/7
          </div>
          <div className="pills">
            {alwaysOn.map((p) => {
              const tone = toneFor(p.agent)
              return (
                <span
                  key={p.id}
                  className="pill"
                  style={{ borderLeftColor: tone.bg }}
                >
                  <span className="dot" />
                  {p.name}
                </span>
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
          {nextUp.map((item) => (
            <span
              key={item.id}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12 }}
            >
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: 999,
                  background: 'hsl(var(--muted-foreground))',
                }}
              />
              <span style={{ fontWeight: 500 }}>{item.name}</span>
              <span style={{ color: 'hsl(var(--muted-foreground))' }}>
                {formatNextRun(item.next_run_at)}
              </span>
            </span>
          ))}
        </div>
      )}

      {mode === 'month' ? (
        <MonthGrid
          cells={monthCells}
          events={events}
          anchorMonth={anchor.getMonth()}
          onEventClick={handleEventClick}
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
                    return (
                      <button
                        key={evt.id}
                        type="button"
                        className="cc-cal-event"
                        style={{
                          top,
                          height,
                          borderLeftColor: tone.bg,
                          background: 'hsl(var(--secondary))',
                        }}
                        title={`${evt.name} — click for agent details`}
                        onClick={() => handleEventClick(evt)}
                      >
                        <div className="nm">
                          {String(evt.hour).padStart(2, '0')}:
                          {String(evt.min).padStart(2, '0')}
                          {evt.agent && (
                            <span style={{ marginLeft: 6, opacity: 0.85 }}>
                              · {evt.agent}
                            </span>
                          )}
                        </div>
                        <div className="ttl">{evt.name}</div>
                      </button>
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
          {!isLoading && events.length === 0 && alwaysOn.length === 0 && (
            <div className="cc-panel-empty">
              No scheduled work in this window. Enable a heartbeat in /agents
              to populate the calendar.
            </div>
          )}
        </div>
      )}
    </>
  )
}

function MonthGrid({
  cells,
  events,
  anchorMonth,
  onEventClick,
}: {
  cells: DayCell[]
  events: CalEvent[]
  anchorMonth: number
  onEventClick: (e: CalEvent) => void
}) {
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
          return (
            <div
              key={d.iso}
              className={`cc-cal-month-cell${outside ? ' outside' : ''}${d.today ? ' today' : ''}`}
            >
              <div className="n">{d.today ? <span>{d.n}</span> : d.n}</div>
              <div className="evts">
                {dayEvents.slice(0, 3).map((evt) => {
                  const tone = toneFor(evt.agent)
                  return (
                    <button
                      key={evt.id}
                      type="button"
                      className="cc-cal-month-evt"
                      style={{ borderLeftColor: tone.bg }}
                      onClick={() => onEventClick(evt)}
                      title={evt.name}
                    >
                      <span className="t">
                        {String(evt.hour).padStart(2, '0')}:
                        {String(evt.min).padStart(2, '0')}
                      </span>{' '}
                      <span className="n2">{evt.name}</span>
                    </button>
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
