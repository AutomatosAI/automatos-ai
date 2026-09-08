/**
 * What a click on a calendar event can do — one pure function so the menu
 * wiring in CalendarTab stays thin and the rules are unit-testable.
 *
 * Every action rides an endpoint or route that already exists:
 *   routine   → the agent page, or pause the heartbeat (PATCH /api/heartbeat/{agent}/toggle)
 *   task      → pause / cancel the scheduled task (PATCH /api/v1/scheduled-tasks/{id}/status)
 *   recipe    → the playbook in the Assignments hub
 *   mission   → the mission page
 *   task_due  → the board, with the card opened (?task_id=)
 */

import type { ScheduleItem } from '@/hooks/use-activity-api'
import type { ScheduledTaskStatus } from '@/hooks/use-scheduled-tasks-api'

export interface EventAction {
  label: string
  run: () => void
  tone?: 'default' | 'danger'
}

export interface EventActionDeps {
  navigate: (href: string) => void
  pauseRoutine: (agentId: number) => void
  setScheduledTaskStatus: (taskId: number, status: ScheduledTaskStatus) => void
}

/** Deadline items (mission / board-task SLA) are due times, not runs. */
export function isDeadlineItem(item: Pick<ScheduleItem, 'type'>): boolean {
  return item.type === 'mission' || item.type === 'task_due'
}

export function agentHref(agentId: number): string {
  return `/agents?agent=${agentId}`
}

export function boardTaskHref(boardTaskId: number): string {
  return `/command-center?tab=board&task_id=${boardTaskId}`
}

export function playbookHref(playbookId: number): string {
  return `/assignments?tab=playbooks&id=${playbookId}`
}

export function missionHref(missionId: number): string {
  return `/missions/${missionId}`
}

const ACTIVITY_FALLBACK_HREF = '/command-center?tab=activity'

export function buildEventActions(item: ScheduleItem, deps: EventActionDeps): EventAction[] {
  const actions: EventAction[] = []
  const agentId = item.agent_id
  const openAgent: EventAction | null =
    agentId != null ? { label: 'Open agent', run: () => deps.navigate(agentHref(agentId)) } : null

  switch (item.type) {
    case 'routine':
      if (openAgent) actions.push(openAgent)
      if (agentId != null) {
        actions.push({ label: 'Pause routine', run: () => deps.pauseRoutine(agentId) })
      }
      break
    case 'task': {
      const taskId = item.scheduled_task_id
      if (taskId != null) {
        actions.push({ label: 'Pause', run: () => deps.setScheduledTaskStatus(taskId, 'paused') })
        actions.push({
          label: 'Cancel',
          tone: 'danger',
          run: () => deps.setScheduledTaskStatus(taskId, 'cancelled'),
        })
      }
      if (openAgent) actions.push(openAgent)
      break
    }
    case 'recipe':
      if (item.playbook_id != null) {
        const playbookId = item.playbook_id
        actions.push({ label: 'Open playbook', run: () => deps.navigate(playbookHref(playbookId)) })
      }
      break
    case 'mission':
      if (item.mission_id != null) {
        const missionId = item.mission_id
        actions.push({ label: 'Open mission', run: () => deps.navigate(missionHref(missionId)) })
      }
      break
    case 'task_due':
      if (item.board_task_id != null) {
        const boardTaskId = item.board_task_id
        actions.push({ label: 'Open on board', run: () => deps.navigate(boardTaskHref(boardTaskId)) })
      }
      if (openAgent) actions.push(openAgent)
      break
  }

  if (actions.length === 0) {
    actions.push({ label: 'Open activity', run: () => deps.navigate(ACTIVITY_FALLBACK_HREF) })
  }
  return actions
}
