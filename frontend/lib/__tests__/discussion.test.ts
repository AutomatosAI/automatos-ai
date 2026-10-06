/**
 * PRD-252 R2 — Discuss: a chat with the ticket in its context, and the brief
 * Auto proposed in it, ready to go back onto the ticket.
 */
import { describe, it, expect } from 'vitest'
import type { BoardTask } from '@/types/board'
import type { ChatMessage } from '@/types'
import { discussHref, discussionLabel, discussionOf, discussionSelection, proposedBrief } from '@/lib/discussion'

function ticket(over: Partial<BoardTask> = {}): BoardTask {
  return { id: '612', type: 'task', name: 'Welcome email', status: 'review', priority: 'medium', tags: [],
    review_mode: 'human', source_id: '612', number: '#0042', assignee: { agent_id: 3, agent_name: 'Words' }, ...over }
}

function reply(role: 'user' | 'assistant', text: string): ChatMessage {
  return { id: `${role}-${text.length}`, role, parts: [{ type: 'text', text }] } as unknown as ChatMessage
}

describe('a discussion', () => {
  it('opens the chat at the ticket and names it by its number', () => {
    expect(discussHref('612')).toBe('/chat?ticket=612')
    const discussion = discussionOf(ticket())
    expect(discussion).toEqual({ ticketId: '612', number: '#0042', title: 'Welcome email', agentName: 'Words', missionId: null })
    expect(discussionSelection(discussion)).toEqual({ type: 'board_task', id: '#0042' })   // the number Auto's tools take
    expect(discussionSelection({ ...discussion, number: null })).toEqual({ type: 'board_task', id: '612' })
    expect(discussionLabel(discussion)).toBe('ticket #0042')
    expect(discussionLabel({ number: null, ticketId: '612' })).toBe('ticket 612')
  })

  it("on a mission's ticket, discusses the mission (D4)", () => {
    const discussion = discussionOf(ticket({ type: 'mission', mission_id: 'run-9' }))
    expect(discussionSelection(discussion)).toEqual({ type: 'mission', id: 'run-9' })
  })
})

describe('the brief Auto proposed', () => {
  it("is the fenced block in Auto's last reply", () => {
    const messages = [
      reply('assistant', 'Earlier thought'),
      reply('user', 'Yes, that one.'),
      reply('assistant', 'Agreed. Here it is:\n```\nWrite to Priya at Gull & Anchor: two short paragraphs, no prices.\n```\nShall I send it back?'),
    ]
    expect(proposedBrief(messages)).toBe('Write to Priya at Gull & Anchor: two short paragraphs, no prices.')
  })

  it('is the whole reply when it has no block, and empty with no reply', () => {
    expect(proposedBrief([reply('assistant', '  Two paragraphs, no prices.  ')])).toBe('Two paragraphs, no prices.')
    expect(proposedBrief([reply('user', 'Hello')])).toBe('')
  })
})
