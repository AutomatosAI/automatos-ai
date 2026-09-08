/**
 * The calendar's event menu is a pure function of the feed item: every item
 * kind gets the actions its backing endpoint supports, nothing more. Routes and
 * statuses are asserted here so a rename in the pages or the scheduled-task
 * API can't silently strand a menu entry.
 */
import { describe, it, expect, vi } from 'vitest'
import type { ScheduleItem } from '@/hooks/use-activity-api'
import { buildEventActions, isDeadlineItem, type EventActionDeps } from '../calendar-actions'

function deps(): EventActionDeps & { navigate: ReturnType<typeof vi.fn> } {
  return { navigate: vi.fn(), pauseRoutine: vi.fn(), setScheduledTaskStatus: vi.fn() }
}

function item(over: Partial<ScheduleItem>): ScheduleItem {
  return {
    id: 'x',
    name: 'X',
    type: 'routine',
    next_run_at: null,
    frequency: '',
    agent_name: null,
    agent_id: null,
    ...over,
  }
}

const labels = (actions: { label: string }[]) => actions.map((a) => a.label)

describe('buildEventActions', () => {
  it('routine: open the agent, pause the heartbeat', () => {
    const d = deps()
    const actions = buildEventActions(item({ type: 'routine', agent_id: 7, agent_name: 'Ops' }), d)
    expect(labels(actions)).toEqual(['Open agent', 'Pause routine'])
    actions[0].run()
    expect(d.navigate).toHaveBeenCalledWith('/agents?agent=7')
    actions[1].run()
    expect(d.pauseRoutine).toHaveBeenCalledWith(7)
  })

  it('scheduled task: pause, cancel (danger), then the agent', () => {
    const d = deps()
    const actions = buildEventActions(
      item({ type: 'task', scheduled_task_id: 12, agent_id: 3 }),
      d,
    )
    expect(labels(actions)).toEqual(['Pause', 'Cancel', 'Open agent'])
    expect(actions[1].tone).toBe('danger')
    actions[0].run()
    expect(d.setScheduledTaskStatus).toHaveBeenCalledWith(12, 'paused')
    actions[1].run()
    expect(d.setScheduledTaskStatus).toHaveBeenCalledWith(12, 'cancelled')
  })

  it('playbook and mission open their own pages', () => {
    const d = deps()
    buildEventActions(item({ type: 'recipe', playbook_id: 5 }), d)[0].run()
    expect(d.navigate).toHaveBeenCalledWith('/assignments?tab=playbooks&id=5')
    buildEventActions(item({ type: 'mission', mission_id: 9 }), d)[0].run()
    expect(d.navigate).toHaveBeenCalledWith('/missions/9')
  })

  it('board-task deadline opens the card on the board via ?task_id=', () => {
    const d = deps()
    const actions = buildEventActions(item({ type: 'task_due', board_task_id: 42, agent_id: 3 }), d)
    expect(labels(actions)).toEqual(['Open on board', 'Open agent'])
    actions[0].run()
    expect(d.navigate).toHaveBeenCalledWith('/command-center?tab=board&task_id=42')
  })

  it('an item with nothing to act on still opens the activity view', () => {
    const d = deps()
    const actions = buildEventActions(item({ type: 'recipe' }), d)
    expect(labels(actions)).toEqual(['Open activity'])
    actions[0].run()
    expect(d.navigate).toHaveBeenCalledWith('/command-center?tab=activity')
  })

  it('deadline items are missions and board tasks only', () => {
    expect(isDeadlineItem({ type: 'mission' })).toBe(true)
    expect(isDeadlineItem({ type: 'task_due' })).toBe(true)
    expect(isDeadlineItem({ type: 'routine' })).toBe(false)
    expect(isDeadlineItem({ type: 'task' })).toBe(false)
  })
})
