/**
 * PRD-266 US-018b (D27): a paused scheduled task stays on the calendar instead of
 * vanishing. The schedule feed marks it `paused: true` (US-018a); the calendar greys it
 * with the word 'Paused' beside its name in the week grid and the month grid (never
 * colour alone), its menu offers Resume and Cancel instead of Pause, Resume sends
 * status active through the existing scheduled-task mutation, and Next up skips it.
 * An active task's menu is unchanged.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { ScheduleItem } from '@/hooks/use-activity-api'

const feed = vi.hoisted(() => ({ items: [] as unknown[] }))
const statusMutation = vi.hoisted(() => ({ mutate: vi.fn() }))

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock('@/hooks/use-activity-api', () => ({
  activityQueryKeys: { all: ['activity'] },
  useActivitySchedule: () => ({ data: { scheduled: feed.items }, isLoading: false, isError: false, refetch: vi.fn() }),
  useSchedulerHealth: () => ({ data: { healthy: null, last_fired_at: null } }),
}))
vi.mock('@/hooks/use-heartbeats-api', () => ({ useToggleHeartbeat: () => ({ mutate: vi.fn() }) }))
vi.mock('@/hooks/use-scheduled-tasks-api', () => ({ useUpdateScheduledTaskStatus: () => statusMutation }))

import { CalendarTab } from '../calendar-tab'
import { buildEventActions, isPausedItem, type EventActionDeps } from '../calendar-actions'
import { PAUSED_OPACITY } from '../calendar-model'

/** Today at the given hour: inside the visible week and month, far from a day boundary. */
function todayAt(hours: number): Date {
  const d = new Date()
  d.setHours(hours, 0, 0, 0)
  return d
}

/** A one-off scheduled task as the feed sends it (services/activity_scheduled_task_items.py). */
function task(id: number, name: string, at: Date, paused?: boolean): ScheduleItem {
  return {
    id: `task-${id}`,
    scheduled_task_id: id,
    task_type: 'one_shot',
    deliver_as: 'chat',
    name,
    type: 'task',
    next_run_at: at.toISOString(),
    frequency: 'One-off',
    agent_name: 'Ops',
    agent_id: 3,
    ...(paused === undefined ? {} : { paused }),
  }
}

function deps(): EventActionDeps {
  return { navigate: vi.fn(), pauseRoutine: vi.fn(), setScheduledTaskStatus: vi.fn(), rescheduleSocialPost: vi.fn() }
}

const labels = (actions: { label: string }[]) => actions.map((a) => a.label)

function eventNamed(selector: string, name: string): HTMLElement {
  const found = Array.from(document.querySelectorAll<HTMLElement>(selector)).find((e) => e.textContent?.includes(name))
  if (!found) throw new Error(`no ${selector} for ${name}`)
  return found
}

function showMonth() {
  const modes = screen.getByRole('group', { name: 'Calendar mode' })
  fireEvent.click(within(modes).getByRole('button', { name: 'Month' }))
}

beforeEach(() => {
  statusMutation.mutate.mockReset()
  feed.items = []
})
afterEach(() => cleanup())

describe("a scheduled task's menu", () => {
  it('a paused task offers Resume and Cancel, never Pause, and Resume sends status active', () => {
    const d = deps()
    const actions = buildEventActions(task(12, 'Weekly digest', todayAt(14), true), d)
    expect(labels(actions)).toEqual(['Resume', 'Cancel', 'Open agent'])
    expect(labels(actions)).not.toContain('Pause')
    actions[0].run()
    expect(d.setScheduledTaskStatus).toHaveBeenCalledWith(12, 'active')
    expect(actions[1].tone).toBe('danger')
    actions[1].run()
    expect(d.setScheduledTaskStatus).toHaveBeenCalledWith(12, 'cancelled')
  })

  it("an active task's actions are unchanged: Pause, Cancel, then the agent", () => {
    for (const paused of [undefined, false]) {
      const d = deps()
      const actions = buildEventActions(task(12, 'Weekly digest', todayAt(14), paused), d)
      expect(labels(actions)).toEqual(['Pause', 'Cancel', 'Open agent'])
      actions[0].run()
      expect(d.setScheduledTaskStatus).toHaveBeenCalledWith(12, 'paused')
    }
  })

  it('only a scheduled task can be paused', () => {
    expect(isPausedItem({ type: 'task', paused: true })).toBe(true)
    expect(isPausedItem({ type: 'task', paused: false })).toBe(false)
    expect(isPausedItem({ type: 'task' })).toBe(false)
    expect(isPausedItem({ type: 'routine', paused: true })).toBe(false)
  })
})

describe('the week grid', () => {
  it("draws a paused task greyed, with the word 'Paused' beside its name", () => {
    feed.items = [task(12, 'Weekly digest', todayAt(14), true), task(13, 'Daily brief', todayAt(17))]
    render(<CalendarTab />)
    const paused = eventNamed('.cc-cal-event[data-kind="task"]', 'Weekly digest')
    expect(paused.dataset.paused).toBe('true')
    expect(within(paused).getByText('Paused')).toBeInTheDocument()
    expect(paused.querySelector('.ttl')?.textContent).toBe('PausedWeekly digest')
    expect(Number(paused.style.opacity)).toBe(PAUSED_OPACITY)

    const active = eventNamed('.cc-cal-event[data-kind="task"]', 'Daily brief')
    expect(active.dataset.paused).toBeUndefined()
    expect(within(active).queryByText('Paused')).toBeNull()
    expect(active.style.opacity).toBe('')
  })

  it("Resume in a paused event's menu sets the task active through the existing mutation", async () => {
    feed.items = [task(12, 'Weekly digest', todayAt(14), true)]
    render(<CalendarTab />)
    fireEvent.keyDown(eventNamed('.cc-cal-event', 'Weekly digest'), { key: 'Enter' })
    const resume = await screen.findByText('Resume')
    expect(screen.queryByText('Pause')).toBeNull()
    expect(screen.getByText('Cancel')).toBeInTheDocument()
    fireEvent.click(resume)
    await waitFor(() => expect(statusMutation.mutate).toHaveBeenCalledWith({ taskId: 12, status: 'active' }))
  })
})

describe('the month grid', () => {
  it("draws a paused task's chip greyed, with the word 'Paused' beside its name", () => {
    feed.items = [task(12, 'Weekly digest', todayAt(14), true), task(13, 'Daily brief', todayAt(17))]
    render(<CalendarTab />)
    showMonth()
    const paused = eventNamed('.cc-cal-month-evt', 'Weekly digest')
    expect(paused.dataset.paused).toBe('true')
    expect(within(paused).getByText('Paused')).toBeInTheDocument()
    expect(Number(paused.style.opacity)).toBeLessThanOrEqual(PAUSED_OPACITY)

    const active = eventNamed('.cc-cal-month-evt', 'Daily brief')
    expect(active.dataset.paused).toBeUndefined()
    expect(within(active).queryByText('Paused')).toBeNull()
  })
})

describe('Next up', () => {
  it('skips a paused task and keeps the active one', () => {
    feed.items = [task(12, 'Weekly digest', todayAt(14), true), task(13, 'Daily brief', todayAt(17))]
    render(<CalendarTab />)
    const nextUp = document.querySelector('.cc-cal-nextup') as HTMLElement
    expect(nextUp).toBeTruthy()
    expect(within(nextUp).getByText('Daily brief')).toBeInTheDocument()
    expect(within(nextUp).queryByText('Weekly digest')).toBeNull()
  })

  it('is not shown when the only upcoming item is paused', () => {
    feed.items = [task(12, 'Weekly digest', todayAt(14), true)]
    render(<CalendarTab />)
    expect(screen.queryByText('Next up')).toBeNull()
    expect(eventNamed('.cc-cal-event', 'Weekly digest').dataset.paused).toBe('true')
  })
})
