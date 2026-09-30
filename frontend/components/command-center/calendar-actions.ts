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
 *   social    → the post in the Socials tab, or reschedule it
 *               (POST /api/socials/posts/{id}/schedule: the slot moves, the approval stands)
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
  /** social: open the reschedule dialog for the post (PRD-251 US-307) */
  rescheduleSocialPost: (item: ScheduleItem) => void
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

export function socialPostHref(postId: string): string {
  return `/deliverables?tab=socials&post=${encodeURIComponent(postId)}`
}

const ACTIVITY_FALLBACK_HREF = '/command-center?tab=activity'

type KindActions = (item: ScheduleItem, deps: EventActionDeps, openAgent: EventAction | null) => EventAction[]

const routineActions: KindActions = (item, deps, openAgent) => {
  const agentId = item.agent_id
  const pause = agentId != null ? [{ label: 'Pause routine', run: () => deps.pauseRoutine(agentId) }] : []
  return [...(openAgent ? [openAgent] : []), ...pause]
}

const taskActions: KindActions = (item, deps, openAgent) => {
  const taskId = item.scheduled_task_id
  const own: EventAction[] =
    taskId != null
      ? [
          { label: 'Pause', run: () => deps.setScheduledTaskStatus(taskId, 'paused') },
          { label: 'Cancel', tone: 'danger', run: () => deps.setScheduledTaskStatus(taskId, 'cancelled') },
        ]
      : []
  return [...own, ...(openAgent ? [openAgent] : [])]
}

const recipeActions: KindActions = (item, deps) => {
  const playbookId = item.playbook_id
  return playbookId != null ? [{ label: 'Open playbook', run: () => deps.navigate(playbookHref(playbookId)) }] : []
}

const missionActions: KindActions = (item, deps) => {
  const missionId = item.mission_id
  return missionId != null ? [{ label: 'Open mission', run: () => deps.navigate(missionHref(missionId)) }] : []
}

const taskDueActions: KindActions = (item, deps, openAgent) => {
  const boardTaskId = item.board_task_id
  const open = boardTaskId != null ? [{ label: 'Open on board', run: () => deps.navigate(boardTaskHref(boardTaskId)) }] : []
  return [...open, ...(openAgent ? [openAgent] : [])]
}

const socialActions: KindActions = (item, deps) => {
  const postId = item.post_id
  if (!postId) return []
  return [
    { label: 'Open post', run: () => deps.navigate(socialPostHref(postId)) },
    { label: 'Reschedule…', run: () => deps.rescheduleSocialPost(item) },
  ]
}

/** Each kind's own actions: one small builder per kind. */
function kindActions(item: ScheduleItem, deps: EventActionDeps, openAgent: EventAction | null): EventAction[] {
  switch (item.type) {
    case 'routine':
      return routineActions(item, deps, openAgent)
    case 'task':
      return taskActions(item, deps, openAgent)
    case 'recipe':
      return recipeActions(item, deps, openAgent)
    case 'mission':
      return missionActions(item, deps, openAgent)
    case 'task_due':
      return taskDueActions(item, deps, openAgent)
    case 'social':
      return socialActions(item, deps, openAgent)
    default:
      return []
  }
}

export function buildEventActions(item: ScheduleItem, deps: EventActionDeps): EventAction[] {
  const agentId = item.agent_id
  const openAgent: EventAction | null =
    agentId != null ? { label: 'Open agent', run: () => deps.navigate(agentHref(agentId)) } : null
  const actions = kindActions(item, deps, openAgent)
  return actions.length > 0 ? actions : [{ label: 'Open activity', run: () => deps.navigate(ACTIVITY_FALLBACK_HREF) }]
}
