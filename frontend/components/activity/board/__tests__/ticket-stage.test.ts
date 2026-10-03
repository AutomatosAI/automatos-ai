/**
 * PRD-252 R3 — every Review or Blocked card says why it is there; a mission
 * step being checked by its mission reads "Mission checking"; no LLM review
 * option is offered (it had no reviewer and behaved as Human).
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
import { stageReason } from '../ticket-stage'

const REVIEW_CODES = ['mission_plan', 'file_missing', 'nothing_done', 'held_command', 'retries_used_up', 'approval_action',
  'stopped_with_work', 'moved_by_you', 'asked', 'ends_on_a_question', 'unexplained']
const BLOCKED_CODES = ['question', 'approval', 'spend_ceiling', 'mission_paused', 'owner_check', 'step_failed',
  'stopped_by_you', 'waiting']
const read = (rel: string) => readFileSync(path.resolve(__dirname, '..', '..', '..', '..', rel), 'utf8')

describe('stageReason', () => {
  it('names every review cause and every block, on the card and in words', () => {
    for (const code of REVIEW_CODES) {
      const r = stageReason({ status: 'review', review_reason: code, blocked_code: null })!
      expect(r.chip.length, code).toBeGreaterThan(0)
      expect(r.says.length, code).toBeGreaterThan(0)
    }
    for (const code of BLOCKED_CODES) {
      expect(stageReason({ status: 'blocked', review_reason: null, blocked_code: code })!.chip, code).toBeTruthy()
    }
    expect(stageReason({ status: 'review', review_reason: 'file_missing', blocked_code: null })!.chip).toBe('File missing')
    expect(stageReason({ status: 'blocked', review_reason: null, blocked_code: 'spend_ceiling' })!.says).toContain('goes back to Assigned')
  })

  it('F242: a run that ends on a question, and a mission waiting for a check, say so', () => {
    expect(stageReason({ status: 'review', review_reason: 'ends_on_a_question', blocked_code: null })!.chip)
      .toBe('Question for you')
    expect(stageReason({ status: 'blocked', review_reason: null, blocked_code: 'owner_check' })!.says)
      .toContain('approve that step to go on')
  })

  it('a mission step under its mission’s check is not a Review that needs the owner', () => {
    expect(stageReason({ status: 'review', review_reason: 'mission_checking', blocked_code: null })!.stage).toBe('Mission checking')
  })

  it('never leaves a Review or Blocked card without a reason, even an unknown or missing code', () => {
    expect(stageReason({ status: 'review', review_reason: null, blocked_code: null })!.says).toBe('Waiting for your verdict.')
    expect(stageReason({ status: 'blocked', review_reason: null, blocked_code: 'new_kind' })!.chip).toBe('Blocked')
    expect(stageReason({ status: 'in_progress', review_reason: null, blocked_code: null })).toBeNull()
  })

  it('both boards show it on the card', () => {
    expect(read('components/command-center/board-tab.tsx')).toContain('stageReason(task)')
    expect(read('components/activity/board/board-card.tsx')).toContain('stageReason(task)')
  })

  it("a mission's card is approved on its mission, never by ticket verdict (D6)", () => {
    // Review of #857: ticket Approve marked the card done and the mission never started.
    expect(stageReason({ status: 'review', review_reason: 'mission_plan', blocked_code: null })!.stage).toBe('Plan to approve')
    const viewer = read('components/activity/board/board-task-viewer.tsx')
    expect(viewer).toContain('const missionDecides = isMissionTicket(task)')
    // 3b: the mission's own Approve and Reject (mission-verdict.test.tsx)
    expect(viewer).toContain('<MissionVerdict task={task} onDecided={onDecided} />')
  })

  it('offers no LLM review mode when creating a ticket', () => {
    expect(read('components/activity/board/create-task-steps.tsx')).not.toContain('value="llm"')
  })
})
