/**
 * The calendar draws everything from the schedule feed (GET /api/activity/schedule).
 * It used to take its routine rows and 24/7 band from /api/heartbeat/workspace,
 * a router behind the PRD-143 super-admin lock (403 for everyone else, polled
 * every 30s) whose next_run_at came from the one worker hosting APScheduler.
 * These tests pin the feed-only contract, the deadline items, and the
 * 2026-09-11 rules: heartbeats at the band cadence stay off the grid, colour
 * is by kind, overlapping events share the column, legend chips hide a kind.
 */
import { describe, it, expect, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'
import type { ScheduleItem } from '@/hooks/use-activity-api'
import { KIND_META } from '../calendar-kinds'

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

// Code only: the header comment is allowed to NAME the endpoint it no longer calls.
const src = readFileSync(path.resolve(__dirname, '..', 'calendar-tab.tsx'), 'utf8')
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/^\s*\/\/.*$/gm, '')

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

function deadline(id: number, name: string, due: Date): ScheduleItem {
  return {
    id: `board-${id}`,
    board_task_id: id,
    name,
    type: 'task_due',
    next_run_at: due.toISOString(),
    frequency: 'SLA deadline',
    agent_name: 'Ops',
    agent_id: 1,
    status: 'assigned',
    priority: 'high',
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

  it('an hourly heartbeat stays in the band and off the grid', () => {
    // WATCHTOWER hourly 08:00-20:00 was 13 grey blocks a day on the prod calendar.
    feed.items = [routine(3, 'Watchtower', 60, midToday())]
    const { container } = render(<CalendarTab />)
    expect(screen.getByText(/Watchtower · every 60m/)).toBeInTheDocument()
    expect(container.querySelector('.cc-cal-event')).toBeNull()
  })

  it('a slower routine is plotted from its structured recurrence and next run', () => {
    feed.items = [routine(2, 'Research', 120, midToday())]
    const { container } = render(<CalendarTab />)
    const plotted = container.querySelectorAll('.cc-cal-event')
    expect(plotted.length).toBeGreaterThan(1) // expanded across the day, not one anchor
    expect(plotted[0].textContent).toContain('Research')
  })

  it('a board-task SLA deadline is a DUE event and sits in Next Up', () => {
    feed.items = [deadline(42, 'Fix the calendar', new Date(midToday().getTime() + 30 * 60_000))]
    const { container } = render(<CalendarTab />)
    const evt = container.querySelector('.cc-cal-event')
    expect(evt).not.toBeNull()
    expect(evt!.textContent).toContain('Fix the calendar')
    expect(evt!.textContent).toMatch(/DUE|OVERDUE/)
    expect(screen.getByText('Next up')).toBeInTheDocument()
  })
})

describe('CalendarTab — kinds, lanes, legend', () => {
  it('colours an event by its kind, with the agent as the dot', () => {
    feed.items = [
      routine(2, 'Research', 120, midToday()),
      deadline(42, 'Fix the calendar', new Date(midToday().getTime() + 5 * 60 * 60_000)),
    ]
    const { container } = render(<CalendarTab />)
    const events = Array.from(container.querySelectorAll<HTMLElement>('.cc-cal-event'))
    const byTitle = (re: RegExp) => events.find((e) => re.test(e.title))!
    // The kind decides the border colour (calendar-kinds.ts); jsdom's style
    // parser is unreliable with modern hsl() syntax, so assert the kind hook.
    expect(byTitle(/^Heartbeat:/).dataset.kind).toBe('routine')
    expect(byTitle(/^Task deadline:/).dataset.kind).toBe('task_due')
    expect(KIND_META.routine.tone).not.toBe(KIND_META.task_due.tone)
    expect(byTitle(/^Heartbeat:/).querySelector('.agent-dot')).not.toBeNull()
  })

  it('overlapping events share the column', () => {
    const at = new Date(midToday().getTime() + 2 * 60 * 60_000)
    feed.items = [deadline(1, 'A', at), deadline(2, 'B', at)]
    const { container } = render(<CalendarTab />)
    const lanes = Array.from(container.querySelectorAll('.cc-cal-event')).map((e) => [
      e.getAttribute('data-lane'),
      e.getAttribute('data-lanes'),
    ])
    expect(lanes).toEqual([
      ['0', '2'],
      ['1', '2'],
    ])
  })

  it('short events closer together than the minimum box height share the column', () => {
    // 15-minute deadlines 20 minutes apart: their 28px boxes would overlap.
    const at = new Date(midToday().getTime() + 2 * 60 * 60_000)
    feed.items = [deadline(1, 'A', at), deadline(2, 'B', new Date(at.getTime() + 20 * 60_000))]
    const { container } = render(<CalendarTab />)
    const lanes = Array.from(container.querySelectorAll('.cc-cal-event')).map((e) =>
      e.getAttribute('data-lanes'),
    )
    expect(lanes).toEqual(['2', '2'])
  })

  it('a legend chip hides its kind from the grid, band and Next Up', () => {
    feed.items = [
      routine(1, 'Ops', 15, midToday()),
      routine(2, 'Research', 120, midToday()),
      deadline(42, 'Fix the calendar', new Date(midToday().getTime() + 30 * 60_000)),
    ]
    const { container } = render(<CalendarTab />)
    expect(container.querySelectorAll('.cc-cal-event').length).toBeGreaterThan(1)

    fireEvent.click(screen.getByRole('button', { name: 'Heartbeat' }))

    expect(screen.queryByText(/Ops · every 15m/)).toBeNull()
    const left = Array.from(container.querySelectorAll<HTMLElement>('.cc-cal-event'))
    expect(left).toHaveLength(1)
    expect(left[0].title).toMatch(/^Task deadline:/)
    expect(screen.getByRole('button', { name: 'Heartbeat' })).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('button', { name: 'Task deadline' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('Next Up drops a deadline overdue by more than a week', () => {
    const soon = new Date(midToday().getTime() + 30 * 60_000)
    const ancient = new Date(Date.now() - 8 * 24 * 60 * 60_000)
    feed.items = [deadline(1, 'Soon ticket', soon), deadline(2, 'Ancient ticket', ancient)]
    render(<CalendarTab />)
    expect(screen.getAllByText('Soon ticket').length).toBeGreaterThan(0) // Next Up and the grid
    expect(screen.queryByText('Ancient ticket')).toBeNull()
  })
})
