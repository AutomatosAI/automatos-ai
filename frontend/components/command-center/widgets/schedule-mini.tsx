'use client'

/**
 * ScheduleMini — week strip + "Upcoming today" list for the Summary tab.
 * Live data via `useActivitySchedule('7d')`.
 *
 * Calendar dates are computed locally so the week strip reflects the user's
 * actual week, not a static fixture.
 */

import { useMemo } from 'react'
import { useActivitySchedule } from '@/hooks/use-activity-api'
import { toneFor } from '../agent-tones'

interface DayCell {
  label: string
  n: number
  today: boolean
}

interface UpcomingItem {
  time: string
  name: string
  agent: string | null
  state: 'now' | 'next' | null
}

function buildWeek(): DayCell[] {
  const now = new Date()
  const dow = now.getDay() // 0 = Sun
  const sunday = new Date(now)
  sunday.setDate(now.getDate() - dow)
  return Array.from({ length: 7 }, (_, i) => {
    const d = new Date(sunday)
    d.setDate(sunday.getDate() + i)
    return {
      label: d.toLocaleDateString('en-GB', { weekday: 'short' }),
      n: d.getDate(),
      today: d.toDateString() === now.toDateString(),
    }
  })
}

function isToday(dateStr: string | null): boolean {
  if (!dateStr) return false
  try {
    return new Date(dateStr).toDateString() === new Date().toDateString()
  } catch {
    return false
  }
}

function formatTime(dateStr: string | null): string {
  if (!dateStr) return '—'
  try {
    return new Date(dateStr).toLocaleTimeString('en-GB', {
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return '—'
  }
}

export function ScheduleMini({ limit = 6 }: { limit?: number }) {
  const { data, isLoading } = useActivitySchedule('7d')
  const week = useMemo(buildWeek, [])

  const upcoming = useMemo<UpcomingItem[]>(() => {
    if (!data?.scheduled) return []
    const todayItems = data.scheduled.filter((s) => isToday(s.next_run_at))
    const now = Date.now()
    return todayItems
      .map((s) => {
        const t = s.next_run_at ? new Date(s.next_run_at).getTime() : 0
        const delta = t - now
        let state: 'now' | 'next' | null = null
        if (delta < 0 && delta > -10 * 60 * 1000) state = 'now'
        return {
          time: formatTime(s.next_run_at),
          name: s.name,
          agent: s.agent_name,
          state,
        }
      })
      .sort((a, b) => a.time.localeCompare(b.time))
      .map((item, i, arr) => {
        if (item.state === null && i > 0 && arr[i - 1].state === 'now') {
          return { ...item, state: 'next' as const }
        }
        return item
      })
      .slice(0, limit)
  }, [data, limit])

  return (
    <div className="cc-sched">
      <div className="cc-sched-week">
        {week.map((d) => (
          <span key={d.label + d.n} className="lbl">
            {d.label}
          </span>
        ))}
      </div>
      <div className="cc-sched-week">
        {week.map((d) => (
          <span key={`n-${d.label}-${d.n}`} className={`d${d.today ? ' today' : ''}`}>
            <span>{d.n}</span>
          </span>
        ))}
      </div>

      <p className="cc-eyebrow-sm" style={{ marginTop: 8 }}>
        Upcoming · today
      </p>
      {isLoading ? (
        <div
          style={{
            padding: '12px 0',
            fontSize: 12,
            color: 'hsl(var(--muted-foreground))',
          }}
        >
          Loading…
        </div>
      ) : upcoming.length === 0 ? (
        <div
          style={{
            padding: '12px 0',
            fontSize: 12,
            color: 'hsl(var(--muted-foreground))',
            fontFamily: 'var(--font-newsreader, serif)',
          }}
        >
          Nothing else scheduled today.
        </div>
      ) : (
        <div className="cc-sched-list">
          {upcoming.map((it, i) => {
            const tone = toneFor(it.agent)
            return (
              <div key={`${it.time}-${it.name}-${i}`} className="cc-sched-row">
                <span className="t">{it.time}</span>
                <span className="bar" style={{ background: tone.bg }} />
                <span className="nm">{it.name}</span>
                {it.state === 'now' && <span className="cc-pill-ok">NOW</span>}
                {it.state === 'next' && <span className="cc-pill-info">NEXT</span>}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
