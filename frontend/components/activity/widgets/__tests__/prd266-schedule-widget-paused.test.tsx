/**
 * PRD-266 P266W3-RVW-1 (D27): the schedule feed now carries a paused scheduled task
 * (`paused: true`, US-018a) so the calendar can grey it. The Command Center's Schedule
 * widget reads the same feed, and a paused task will not run: it is neither listed under
 * 'Upcoming' nor counted as a dot on its day, as the calendar's Next up skips it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { format } from 'date-fns'
import type { ScheduleItem } from '@/hooks/use-activity-api'

const query = vi.hoisted(() => ({
  data: undefined as { scheduled: unknown[]; scheduler_active: boolean } | undefined,
  isLoading: false,
  isError: false,
}))
vi.mock('@/hooks/use-activity-api', () => ({ useActivitySchedule: () => query }))

import { ScheduleWidget } from '@/components/activity/widgets/schedule-widget'

/** Today at the given hour: inside the widget's week, far from a day boundary. */
function todayAt(hours: number): Date {
  const d = new Date()
  d.setHours(hours, 0, 0, 0)
  return d
}

/** A one-off scheduled task as the feed sends it (services/activity_scheduled_task_items.py). */
function task(id: number, name: string, paused?: boolean): ScheduleItem {
  return {
    id: `task-${id}`,
    scheduled_task_id: id,
    task_type: 'one_shot',
    name,
    type: 'task',
    next_run_at: todayAt(14).toISOString(),
    frequency: 'One-off',
    agent_name: 'Ops',
    agent_id: 3,
    ...(paused === undefined ? {} : { paused }),
  }
}

/** The dots under today's date in the week strip. */
function todaysDots(): Element[] {
  const dayNumber = format(new Date(), 'd')
  const cell = Array.from(document.querySelectorAll('.w-7.h-7')).find((e) => e.textContent === dayNumber)
  if (!cell?.parentElement) throw new Error('no week-strip cell for today')
  return Array.from(cell.parentElement.querySelectorAll('.w-1\\.5.h-1\\.5'))
}

beforeEach(() => {
  query.data = undefined
  query.isLoading = false
  query.isError = false
})
afterEach(() => cleanup())

describe('ScheduleWidget with a paused task (PRD-266 D27)', () => {
  it("lists only the active task under 'Upcoming' and dots its day once", () => {
    query.data = { scheduled: [task(1, 'Morning briefing'), task(2, 'Weekly digest', true)], scheduler_active: true }
    render(<ScheduleWidget />)
    expect(screen.getByText('Morning briefing')).toBeInTheDocument()
    expect(screen.queryByText('Weekly digest')).toBeNull()
    expect(todaysDots()).toHaveLength(1)
  })

  it("says 'No upcoming schedules' when only a paused task remains, with no dot", () => {
    query.data = { scheduled: [task(2, 'Weekly digest', true)], scheduler_active: true }
    render(<ScheduleWidget />)
    expect(screen.getByText('No upcoming schedules')).toBeInTheDocument()
    expect(screen.queryByText('Weekly digest')).toBeNull()
    expect(todaysDots()).toHaveLength(0)
  })

  it('an active task with paused: false is still upcoming', () => {
    query.data = { scheduled: [task(1, 'Morning briefing', false)], scheduler_active: true }
    render(<ScheduleWidget />)
    expect(screen.getByText('Morning briefing')).toBeInTheDocument()
    expect(todaysDots()).toHaveLength(1)
  })
})
