/**
 * PRD-252 R1 — every ticket reference opens that ticket.
 *
 * The board opens its viewer for `?task_id=<id>`, fetching the ticket by id, so
 * a filtered ticket, one done long ago or a mission step opens all the same.
 * `&question=<grant id>` opens it at that question. F218: the classic activity
 * feed built `?task=<id>`, which nothing read, and Needs you linked to the whole
 * board, so a click landed on a column of tickets instead of the one that needed
 * the owner. Every surface builds its link here.
 */
import type { ApprovalGrant } from './api-client'
import type { ActivityFeedItem } from '@/hooks/use-activity-api'

export const BOARD_HREF = '/command-center?tab=board'
export const QUESTIONS_HREF = '/command-center?tab=questions'

type QuestionRef = Pick<ApprovalGrant, 'id' | 'subject_type' | 'subject_id' | 'owner'>

/** The board, opened at one ticket and, when given, at one of its questions. */
export function ticketHref(taskId: string | number, questionId?: string | number | null): string {
  const params = new URLSearchParams({ tab: 'board', task_id: String(taskId) })
  if (questionId != null) params.set('question', String(questionId))
  return `/command-center?${params.toString()}`
}

/** A question known only by its id (a notification): the board finds its ticket. */
export function questionIdHref(questionId: string | number): string {
  return `/command-center?${new URLSearchParams({ tab: 'board', question: String(questionId) }).toString()}`
}

/** The ticket a question was asked on, or null for a question about anything else. */
export function questionTicketId(q: Omit<QuestionRef, 'id'>): string | null {
  if (q.subject_type === 'board_task' && q.subject_id) return String(q.subject_id)
  const owned = q.owner?.ticket?.id
  return owned != null ? String(owned) : null
}

/** A question opens inside its ticket; one asked about anything else opens the Questions tab. */
export function questionHref(q: QuestionRef): string {
  const ticket = questionTicketId(q)
  return ticket ? ticketHref(ticket, q.id) : QUESTIONS_HREF
}

/** A mission's own page, where its plan is approved or rejected. */
export function missionHref(missionId: string | number): string {
  return `/missions/${encodeURIComponent(String(missionId))}`
}

/**
 * Where an activity row opens: the thing itself (the Activity tab, the classic
 * feed, the Activity widget and global search). Only playbooks (recipes) belong
 * in the ExecutionKitchen viewer. Routines are heartbeats: they go to the report
 * they produced, else to the agent that owns them.
 */
export function feedItemHref(item: ActivityFeedItem): string | null {
  const id = item.source_id
  switch (item.type) {
    case 'mission':
      return id ? missionHref(id) : null
    case 'chat':
      // A query param, so client-side auth loads the thread; the /chat/[id]
      // server route 404s without a server-side auth context.
      return id ? `/chat?chatId=${encodeURIComponent(id)}` : null
    case 'recipe':
      // The feed id is "recipe-<execId>"; source_id is the recipe id.
      return id
        ? `/activity/execution?id=${encodeURIComponent(item.id.replace(/^recipe-/, ''))}&recipeId=${encodeURIComponent(id)}`
        : null
    case 'task':
      return id ? ticketHref(id) : null
    case 'routine':
      if (item.source_url?.startsWith('/deliverables/explorer')) return item.source_url
      return item.agent?.id ? `/agents?agent=${item.agent.id}&panel=reports` : null
    default:
      return null
  }
}
