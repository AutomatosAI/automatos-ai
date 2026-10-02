'use client'

/**
 * The calendar's panels above the grid (split out of calendar-tab.tsx, PRD-251
 * US-307): the scheduler-health banner, the load error, the 24/7 band of frequent
 * heartbeats and Next Up.
 */
import { Zap } from 'lucide-react'

import type { ScheduleItem } from '@/hooks/use-activity-api'
import { toneFor } from './agent-tones'
import { buildEventActions, isDeadlineItem, type EventActionDeps } from './calendar-actions'
import { EventMenu } from './calendar-event-menu'
import { kindTone } from './calendar-kinds'
import { OVERDUE_TONE, formatNextRun } from './calendar-model'

export function HealthBanner() {
  return (
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
  )
}

export function LoadError({ onRetry }: { onRetry: () => void }) {
  return (
    <div
      className="cc-panel-empty"
      role="alert"
      style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}
    >
      <span>Couldn’t load the schedule.</span>
      <button type="button" className="cc-btn" onClick={onRetry}>
        Retry
      </button>
    </div>
  )
}

export function AlwaysOnBand({ items, actionDeps }: { items: ScheduleItem[]; actionDeps: EventActionDeps }) {
  return (
    <div className="cc-cal-alwayson">
      <div className="lbl">
        <Zap style={{ width: 12, height: 12, color: OVERDUE_TONE }} />
        24/7
      </div>
      <div className="pills">
        {items.map((s) => (
          <EventMenu key={s.id} actions={buildEventActions(s, actionDeps)}>
            <button
              type="button"
              className="pill"
              style={{ borderLeftColor: kindTone('routine'), cursor: 'pointer' }}
              title={`${s.agent_name} heartbeat — click for actions`}
            >
              <span className="dot" style={{ background: toneFor(s.agent_name).bg }} />
              {s.agent_name} · every {s.recurrence?.interval_minutes}m
            </button>
          </EventMenu>
        ))}
      </div>
    </div>
  )
}

function NextUpItem({ item }: { item: ScheduleItem }) {
  const overdue = item.next_run_at ? new Date(item.next_run_at).getTime() < Date.now() : false
  const accent = overdue ? OVERDUE_TONE : kindTone(item.type)
  const due = isDeadlineItem(item)
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
      <span style={{ width: 6, height: 6, borderRadius: 999, background: accent }} />
      <span style={{ fontWeight: 500 }}>{item.name}</span>
      <span style={{ color: accent }}>
        {overdue ? (due ? 'overdue' : 'missed') : `${due ? 'due ' : ''}${formatNextRun(item.next_run_at)}`}
      </span>
    </span>
  )
}

export function NextUp({ items }: { items: ScheduleItem[] }) {
  return (
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
      {items.map((item) => (
        <NextUpItem key={item.id} item={item} />
      ))}
    </div>
  )
}
