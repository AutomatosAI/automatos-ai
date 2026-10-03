/**
 * PRD-252 R3 — every Review or Blocked card says why it is there, in words.
 *
 * The codes come from the board's API (`review_reason`, `blocked_code`; see
 * orchestrator/core/services/ticket_reasons.py). A chip names the reason on
 * the card; the sentence says it in the viewer. A mission step being checked
 * by its mission reads "Mission checking", not Review: it is not the owner's.
 */
import type { BoardTask } from '@/types/board'

export interface StageReason {
  /** The stage's own name when it differs from the column's ("Mission checking"). */
  stage?: string
  chip: string
  says: string
}

const REVIEW: Record<string, StageReason> = {
  mission_checking: { stage: 'Mission checking', chip: 'Mission checking', says: 'Its mission is checking this step. Nothing here needs you.' },
  mission_plan: { stage: 'Plan to approve', chip: 'Plan to approve', says: "Its mission waits for you to approve its plan. Approve or reject it here, or change it on the mission's page." },
  file_missing: { chip: 'File missing', says: 'The file it named is not in the workspace. Check the result before you approve it.' },
  nothing_done: { chip: 'Did nothing', says: 'It finished without doing anything: its tool calls were skipped, or the result is empty.' },
  held_command: { chip: 'Command refused', says: 'A held command was refused, so the result was not checked end to end.' },
  retries_used_up: { chip: 'Out of retries', says: 'It ran out of attempts. Whatever it produced is on the ticket.' },
  approval_action: { chip: 'Your OK', says: 'It is ready and waits for your OK to run its action.' },
  stopped_with_work: { chip: 'Stopped part-way', says: 'The playbook stopped after part of the work. What it finished is here.' },
  moved_by_you: { chip: 'Moved by you', says: 'You moved it to Review.' },
  asked: { chip: 'You asked', says: 'You asked to review it before it closes.' },
  ends_on_a_question: {
    chip: 'Question for you',
    says: 'It ends on a question for you. Approve to close it, or Reject with your answer to run it again.',
  },
  unexplained: { chip: 'Review', says: 'Waiting for your verdict.' },
}

const BLOCKED: Record<string, StageReason> = {
  question: { chip: 'Question', says: 'Waiting for your answer to its question.' },
  approval: { chip: 'Approval', says: 'Waiting for your approval before it runs.' },
  spend_ceiling: {
    chip: 'Spend ceiling',
    says: "Held: this window's spend is over its ceiling. It goes back to Assigned on its own when you raise the ceiling or the window rolls over.",
  },
  mission_paused: { chip: 'Mission paused', says: 'Its mission is paused. Resume it from the mission.' },
  owner_check: {
    chip: 'Your check',
    says: 'Its mission waits for you to check a step: approve that step to go on, or reject it to have it redone.',
  },
  step_failed: { chip: 'Step failed', says: 'This step failed. Its mission decides what happens next.' },
  stopped_by_you: { chip: 'Stopped by you', says: 'You stopped it. Move it to In progress, or press Run Now, to start it again.' },
  waiting: { chip: 'Blocked', says: 'Waiting: see the reason on the ticket.' },
}

/** Why a ticket in Review or Blocked is there (or that a closed one was closed); null otherwise. */
export function stageReason(task: Pick<BoardTask, 'status' | 'review_reason' | 'blocked_code'>): StageReason | null {
  if (task.status === 'review') return REVIEW[task.review_reason ?? ''] ?? REVIEW.unexplained
  if (task.status === 'blocked') return BLOCKED[task.blocked_code ?? ''] ?? BLOCKED.waiting
  if (task.status === 'closed') return { chip: 'Closed', says: 'Closed without a claim about the work.' }
  return null
}
