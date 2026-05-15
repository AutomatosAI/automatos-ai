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

export function CalendarTab() {
  const router = useRouter()
  const [mode, setMode] = useState<ViewMode>('week')
  const [anchor, setAnchor] = useState<Date>(() => new Date())

  const range = mode === 'month' ? '30d' : '7d'
  const { data: schedule, isLoading } = useActivitySchedule(range)
  const { data: heartbeats } = useHeartbeats()

  const week = useMemo(() => buildWeek(anchor), [anchor])
  const monthCells = useMemo(() => buildMonthGrid(anchor), [anchor])

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

  const events = useMemo<CalEvent[]>(() => {
    if (!schedule?.scheduled) return []
    const visibleKeys = new Set(visibleDays.map((d) => d.date.toDateString()))
    const allEvts = schedule.scheduled
      .map((s, idx) => {
        if (!s.next_run_at) return null
        const d = new Date(s.next_run_at)
        return {
          id: `${s.id}-${idx}`,
          hour: d.getHours(),
          min: d.getMinutes(),
          durMin: 15,
          name: s.name,
          agent: s.agent_name,
          agentId: s.agent_id,
          dayKey: d.toDateString(),
          date: d,
        }
      })
      .filter((e): e is CalEvent => e !== null)
    if (mode === 'month') return allEvts
    return allEvts.filter((e) => visibleKeys.has(e.dayKey))
  }, [schedule, visibleDays, mode])

  const alwaysOn = useMemo(() => {
    if (!heartbeats?.heartbeats) return []
    return heartbeats.heartbeats
      .filter((h) => h.enabled && h.interval_minutes <= 30)
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
