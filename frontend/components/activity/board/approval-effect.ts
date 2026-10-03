/**
 * PRD-252 R2 — what approving a ticket in review does, in words.
 *
 * The owner asked: "Approve — does this just mean the ticket is closed?" It does,
 * unless the ticket carries `planning_data.approval_action`, which the approve
 * route runs first (api/board_tasks.py `_run_approval_action`): `publish_blog`
 * publishes a post, `create_blog` starts a blog mission, and any other type is
 * approved without a side effect. The button, the line above it and the toast
 * after it all come from here.
 */
import type { ApproveResult } from '@/hooks/use-board-tasks'

export interface ApprovalAction {
  type: string
  topic?: string
  [key: string]: unknown
}

/** The approve button: Approve and <what it does>. */
export function approveLabel(action?: ApprovalAction | null): string {
  if (action?.type === 'publish_blog') return 'Approve and publish'
  if (action?.type === 'create_blog') return 'Approve and start the post'
  return 'Approve and mark done'
}

/** The line above the buttons: what approving does, before anyone clicks. */
export function approveEffect(action?: ApprovalAction | null): string {
  if (action?.type === 'publish_blog') return 'Approving publishes the blog post to the live site.'
  if (action?.type === 'create_blog') {
    return action.topic
      ? `Approving starts a blog post on "${action.topic}".`
      : 'Approving starts the blog post.'
  }
  return 'Approving marks the ticket done. Nothing else runs.'
}

/** The toast after an approval: what happened. ``ticket`` is its number (#0042), else its id. */
export function approvedMessage(result: ApproveResult | undefined, ticket: string): string {
  const done = result?.action_result
  if (done?.type === 'publish_blog') return done.title ? `Approved. "${done.title}" is published.` : 'Approved. The post is published.'
  if (done?.type === 'create_blog') return done.topic ? `Approved. The blog post on "${done.topic}" has started.` : 'Approved. The blog post has started.'
  return `Approved. Ticket ${ticket} is done.`
}
