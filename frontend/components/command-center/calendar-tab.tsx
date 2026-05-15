'use client'

/**
 * CalendarTab — CD round-4 week view with a real time-of-day grid.
 *
 * Y axis: hours (00:00 → 22:00, 40px per hour). X axis: days of the
 * current week (Sun → Sat). Events are positioned at their actual scheduled
 * time using `next_run_at` from `useActivitySchedule('7d')`. Always-on
 * heartbeats are listed in a band above the grid.
 *
 * Now-line is drawn on today's column at the current minute. Each event's
 * left border colour matches the assigned agent's tone.
 */

import { useMemo, useState } from 'react'
import { ChevronLeft, ChevronRight, Filter, Download, Plus, Zap } from 'lucide-react'
import { useActivitySchedule } from '@/hooks/use-activity-api'
import { useHeartbeats } from '@/hooks/use-heartbeats-api'
import { toneFor } from './agent-tones'

const HOUR_PX = 40
const START_HR = 0
const END_HR = 22
const HOURS = Array.from({ length: END_HR - START_HR + 1 }, (_, i) => i + START_HR)

interface DayCell {
  short: string
  iso: string
  n: number
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
  dayIdx: number
}

function buildWeek(): DayCell[] {
  const now = new Date()
  const dow = now.getDay() // 0 = Sun
  const sunday = new Date(now)
  sunday.setHours(0, 0, 0, 0)
  sunday.setDate(now.getDate() - dow)
  return Array.from({ length: 7 }, (_, i) => {
    const d = new Date(sunday)
    d.setDate(sunday.getDate() + i)
    return {
      short: d.toLocaleDateString('en-GB', { weekday: 'short' }).toUpperCase(),
      iso: d.toISOString().slice(0, 10),
      n: d.getDate(),
      today: d.toDateString() === now.toDateString(),
      date: d,
    }
  })
}

export function CalendarTab() {
  const { data: schedule, isLoading } = useActivitySchedule('7d')
  const { data: heartbeats } = useHeartbeats()

  const week = useMemo(buildWeek, [])
  const todayIdx = week.findIndex((d) => d.today)

  const events = useMemo<CalEvent[]>(() => {
    if (!schedule?.scheduled) return []
    return schedule.scheduled
      .map((s, idx) => {
        if (!s.next_run_at) return null
        const d = new Date(s.next_run_at)
        const dayIdx = week.findIndex(
          (cell) => cell.date.toDateString() === d.toDateString(),
        )
        if (dayIdx === -1) return null
        return {
          id: `${s.id}-${idx}`,
          hour: d.getHours(),
          min: d.getMinutes(),
          durMin: 15, // ScheduleItem doesn't carry duration — render fixed 15min slot
          name: s.name,
          agent: s.agent_name,
          dayIdx,
        }
      })
      .filter((e): e is CalEvent => e !== null)
  }, [schedule, week])

  const alwaysOn = useMemo(() => {
    if (!heartbeats?.heartbeats) return []
    // Heartbeats with very short interval (<= 30min) get "always-on" treatment
    return heartbeats.heartbeats
      .filter((h) => h.enabled && h.interval_minutes <= 30)
      .map((h) => ({
        id: h.id,
        name: `${h.agent_name} · every ${h.interval_minutes}m`,
        agent: h.agent_name,
      }))
  }, [heartbeats])

  const now = new Date()
  const nowHourPos = (now.getHours() - START_HR) * HOUR_PX + (now.getMinutes() / 60) * HOUR_PX
  const monthLabel = `${week[0].date.toLocaleDateString('en-GB', { month: 'long', day: 'numeric' })} — ${week[6].date.toLocaleDateString('en-GB', { month: 'long', day: 'numeric' })}, ${week[6].date.getFullYear()}`

  return (
    <>
      <div className="cc-cal-toolbar">
        <div className="cc-seg" role="group" aria-label="Calendar mode">
          <button type="button">Day</button>
          <button type="button" className="on">Week</button>
          <button type="button">Month</button>
        </div>
        <div style={{ display: 'inline-flex', gap: 4 }}>
          <button type="button" className="cc-btn" style={{ width: 30, padding: 0 }}>
            <ChevronLeft style={{ width: 13, height: 13 }} />
          </button>
          <button type="button" className="cc-btn" style={{ fontSize: 11.5, padding: '0 10px' }}>
            Today
          </button>
          <button type="button" className="cc-btn" style={{ width: 30, padding: 0 }}>
            <ChevronRight style={{ width: 13, height: 13 }} />
          </button>
        </div>
        <span
          style={{
            fontFamily: 'var(--font-newsreader, serif)',
            fontSize: 16,
            fontWeight: 500,
            color: 'hsl(var(--foreground))',
          }}
        >
          {monthLabel}
        </span>
        <div style={{ marginLeft: 'auto', display: 'inline-flex', gap: 6 }}>
          <button type="button" className="cc-btn">
            <Filter style={{ width: 12, height: 12 }} /> Filter
          </button>
          <button type="button" className="cc-btn">
            <Download style={{ width: 12, height: 12 }} /> Export
          </button>
          <button type="button" className="cc-btn primary">
            <Plus style={{ width: 12, height: 12 }} /> Schedule task
          </button>
        </div>
      </div>

      {/* Always-on band */}
      {alwaysOn.length > 0 && (
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

      {/* Time grid */}
      <div className="cc-cal-frame">
        <div className="cc-cal-head">
          <div className="h" />
          {week.map((d) => (
            <div key={d.iso} className={`h${d.today ? ' today' : ''}`}>
              <div className="d">{d.short}</div>
              <div className="n">
                {d.today ? <span>{d.n}</span> : d.n}
              </div>
            </div>
          ))}
        </div>

        <div
          className="cc-cal-grid"
          style={{
            gridTemplateRows: `repeat(${HOURS.length}, ${HOUR_PX}px)`,
          }}
        >
          {/* Hour labels column */}
          <div
            style={{
              gridColumn: 1,
              gridRow: `1 / span ${HOURS.length}`,
              borderRight: '1px solid hsl(var(--border))',
            }}
          >
            {HOURS.map((h) => (
              <div key={h} className="cc-cal-hourlabel">
                {String(h).padStart(2, '0')}:00
              </div>
            ))}
          </div>

          {week.map((d, di) => {
            const dayEvents = events.filter((e) => e.dayIdx === di)
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
                  const top = (evt.hour - START_HR) * HOUR_PX + (evt.min / 60) * HOUR_PX
                  const height = Math.max((evt.durMin / 60) * HOUR_PX, 24)
                  const tone = toneFor(evt.agent)
                  return (
                    <div
                      key={evt.id}
                      className="cc-cal-event"
                      style={{
                        top,
                        height,
                        borderLeftColor: tone.bg,
                        background: 'hsl(var(--secondary))',
                      }}
                      title={evt.name}
                    >
                      <div className="nm">
                        {String(evt.hour).padStart(2, '0')}:
                        {String(evt.min).padStart(2, '0')}
                      </div>
                      <div className="ttl">{evt.name}</div>
                    </div>
                  )
                })}
                {d.today && (
                  <div className="cc-cal-nowline" style={{ top: nowHourPos }} />
                )}
              </div>
            )
          })}
        </div>

        {isLoading && (
          <div className="cc-panel-empty" style={{ position: 'absolute', inset: 0 }}>
            Loading schedule…
          </div>
        )}
        {!isLoading && events.length === 0 && alwaysOn.length === 0 && (
          <div className="cc-panel-empty">
            No scheduled work this week. Enable a heartbeat or schedule a recipe to populate the calendar.
          </div>
        )}
      </div>
    </>
  )
}
