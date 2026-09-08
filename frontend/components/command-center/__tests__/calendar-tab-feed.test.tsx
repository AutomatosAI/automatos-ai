/**
 * The calendar draws everything from the schedule feed (GET /api/activity/schedule).
 * It used to take its routine rows and 24/7 band from /api/heartbeat/workspace,
 * a router behind the PRD-143 super-admin lock (403 for everyone else, polled
 * every 30s) whose next_run_at came from the one worker hosting APScheduler.
 * These tests pin the feed-only contract and the new deadline items.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'
import type { ScheduleItem } from '@/hooks/use-activity-api'

const feed = vi.hoisted(() => ({ items: [] as unknown[] }))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))
vi.mock('@/hooks/use-activity-api', () => ({
  activityQueryKeys: { all: ['activity'] },
  useActivitySchedule: () => ({
    data: { scheduled: feed.items },
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  }),
  useSchedulerHealth: () => ({ data: { healthy: null, last_fired_at: null } }),
}))
vi.mock('@/hooks/use-heartbeats-api', () => ({
  useToggleHeartbeat: () => ({ mutate: vi.fn() }),
}))
vi.mock('@/hooks/use-scheduled-tasks-api', () => ({
  useUpdateScheduledTaskStatus: () => ({ mutate: vi.fn() }),
}))

import { CalendarTab } from '../calendar-tab'

const src = readFileSync(path.resolve(__dirname, '..', 'calendar-tab.tsx'), 'utf8')

function routine(id: number, name: string, intervalMinutes: number, nextRunAt: Date): ScheduleItem {
  return {
    id: `routine-${id}`,
    name: `${name} Routine`,
    type: 'routine',
    next_run_at: nextRunAt.toISOString(),
    frequency: `Every ${intervalMinutes}m`,
    agent_name: name,
    agent_id: id,
    recurrence: {
      cron_expression: null,
      interval_minutes: intervalMinutes,
      timezone: 'UTC',
      active_hours: null,
    },
  }
}

/** A moment inside today's visible week that is never near a day boundary. */
function midToday(): Date {
  const d = new Date()
  d.setHours(12, 30, 0, 0)
  return d
}

describe('CalendarTab — feed-only data contract', () => {
  it('never calls the super-admin-locked heartbeat list', () => {
    expect(src).not.toContain('useHeartbeats(')
    expect(src).not.toContain('/api/heartbeat/workspace')
    // The schedule feed is the one read; the toggle hook is only a mutation.
    expect(src).toContain('useActivitySchedule(')
    expect(src).toContain('useToggleHeartbeat')
  })

  it('a frequent routine is summarised in the 24/7 band, not plotted', () => {
    feed.items = [routine(1, 'Ops', 15, midToday())]
    const { container } = render(<CalendarTab />)
    expect(screen.getByText(/Ops · every 15m/)).toBeInTheDocument()
    expect(container.querySelector('.cc-cal-event')).toBeNull()
  })

  it('an hourly routine is plotted from its structured recurrence and next run', () => {
    feed.items = [routine(2, 'Research', 60, midToday())]
    const { container } = render(<CalendarTab />)
    const plotted = container.querySelectorAll('.cc-cal-event')
    expect(plotted.length).toBeGreaterThan(1) // expanded across the day, not one anchor
    expect(plotted[0].textContent).toContain('Research')
  })

  it('a board-task SLA deadline is a DUE event and sits in Next Up', () => {
    const due = new Date(midToday().getTime() + 30 * 60_000)
    feed.items = [
      {
        id: 'board-42',
        board_task_id: 42,
        name: 'Fix the calendar',
        type: 'task_due',
        next_run_at: due.toISOString(),
        frequency: 'SLA deadline',
        agent_name: 'Ops',
        agent_id: 1,
        status: 'assigned',
        priority: 'high',
      } satisfies ScheduleItem,
    ]
    const { container } = render(<CalendarTab />)
    const evt = container.querySelector('.cc-cal-event')
    expect(evt).not.toBeNull()
    expect(evt!.textContent).toContain('Fix the calendar')
    expect(evt!.textContent).toMatch(/DUE|OVERDUE/)
    expect(screen.getByText('Next up')).toBeInTheDocument()
  })
})
