/** PRD-252 R1 — every ticket reference opens that ticket: one way to build the link. */
import { describe, it, expect } from 'vitest'
import {
  feedItemHref,
  missionHref,
  questionHref,
  questionIdHref,
  questionTicketId,
  ticketHref,
} from '@/lib/ticket-links'

const feedItem = { id: 'x', name: '', status: 'completed', started_at: null, completed_at: null, duration_seconds: null, agent: null, agents: [], summary: '', source_id: null, source_url: null, trigger: null, error_message: null } as const

describe('ticketHref', () => {
  it('opens the board at the ticket, and at one of its questions when given', () => {
    expect(ticketHref(1169)).toBe('/command-center?tab=board&task_id=1169')
    expect(ticketHref('612', 41)).toBe('/command-center?tab=board&task_id=612&question=41')
  })

  it('never builds the ?task= parameter the board does not read (F218)', () => {
    expect(ticketHref(7)).not.toMatch(/[?&]task=/)
  })
})

describe('questions', () => {
  it('a question asked on a ticket opens inside it', () => {
    expect(questionTicketId({ subject_type: 'board_task', subject_id: '612' })).toBe('612')
    expect(questionHref({ id: 41, subject_type: 'board_task', subject_id: '612' })).toBe(
      '/command-center?tab=board&task_id=612&question=41',
    )
  })

  it('a question about anything else names its ticket through its owner, or opens the Questions tab', () => {
    const onAStep = { id: 9, subject_type: 'orchestration_task', subject_id: 'abc', owner: { ticket: { id: 77 } } }
    expect(questionHref(onAStep)).toBe('/command-center?tab=board&task_id=77&question=9')
    expect(questionHref({ id: 10, subject_type: 'chat', subject_id: 'c1' })).toBe('/command-center?tab=questions')
  })

  it('a question known only by its id lets the board find its ticket', () => {
    expect(questionIdHref(41)).toBe('/command-center?tab=board&question=41')
  })
})

describe('feedItemHref', () => {
  it('opens the ticket by its source id, not the feed id ("task-<id>")', () => {
    // Global search linked task_id=task-1169: the viewer never opened.
    expect(feedItemHref({ ...feedItem, id: 'task-1169', type: 'task', source_id: '1169' } as any)).toBe(
      '/command-center?tab=board&task_id=1169',
    )
  })

  it('opens a mission at its page and a chat at its thread', () => {
    expect(feedItemHref({ ...feedItem, type: 'mission', source_id: 'm-1' } as any)).toBe(missionHref('m-1'))
    expect(feedItemHref({ ...feedItem, type: 'chat', source_id: 'c 1' } as any)).toBe('/chat?chatId=c%201')
  })
})
