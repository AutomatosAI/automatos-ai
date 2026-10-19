// @vitest-environment node
/**
 * PRD-260 T-006 — the Conversations list as pure data (lib/play/conversations.ts),
 * ported from Automatos Play's tests/conversations/list.test.ts onto the core's
 * Chat: the group edges (local midnight, the seventh day), the relative times,
 * titles, the newest line and the title search. Local dates throughout, so the
 * tests pass in any time zone.
 */
import { describe, expect, it } from 'vitest'

import {
  type ConversationSummary,
  conversationTitle,
  filterByTitle,
  groupConversations,
  groupOf,
  lastActive,
  previewLine,
  relativeTime,
} from '../conversations'

const NOW = new Date(2026, 9, 14, 15, 30) // Wed 14 Oct 2026, 15:30

function chat(id: string, at: Date | null, extra: Partial<ConversationSummary> = {}): ConversationSummary {
  const iso = at ? at.toISOString() : ''
  return { id, title: `Chat ${id}`, createdAt: iso, updatedAt: iso, lastMessagePreview: null, ...extra }
}

describe('groupOf — midnight', () => {
  it('is Today from local midnight, through the rest of the day', () => {
    expect(groupOf(new Date(2026, 9, 14, 0, 0), NOW)).toBe('Today')
    expect(groupOf(new Date(2026, 9, 14, 0, 1), NOW)).toBe('Today')
    expect(groupOf(new Date(2026, 9, 14, 15, 29), NOW)).toBe('Today')
  })

  it('is This week a minute before midnight', () => {
    expect(groupOf(new Date(2026, 9, 13, 23, 59), NOW)).toBe('This week')
  })

  it('moves at midnight: a minute after it, 23:59 the night before is yesterday', () => {
    const justAfterMidnight = new Date(2026, 9, 14, 0, 1)
    expect(groupOf(new Date(2026, 9, 13, 23, 59), justAfterMidnight)).toBe('This week')
    expect(groupOf(new Date(2026, 9, 14, 0, 0), justAfterMidnight)).toBe('Today')
  })
})

describe('groupOf — 7 days', () => {
  it('is This week for the six days before today', () => {
    expect(groupOf(new Date(2026, 9, 8, 0, 0), NOW)).toBe('This week')
    expect(groupOf(new Date(2026, 9, 8, 23, 59), NOW)).toBe('This week')
  })

  it('is Earlier from the seventh day back', () => {
    expect(groupOf(new Date(2026, 9, 7, 23, 59), NOW)).toBe('Earlier')
    expect(groupOf(new Date(2026, 8, 1, 9, 0), NOW)).toBe('Earlier')
  })

  it('counts calendar days across a clock change', () => {
    // 25 Oct 2026 is the UK's clock change; a local calendar still counts days.
    const afterChange = new Date(2026, 9, 28, 12, 0)
    expect(groupOf(new Date(2026, 9, 22, 0, 0), afterChange)).toBe('This week')
    expect(groupOf(new Date(2026, 9, 21, 23, 0), afterChange)).toBe('Earlier')
  })

  it('puts a conversation with no date in Earlier, and a clock-skewed future one in Today', () => {
    expect(groupOf(null, NOW)).toBe('Earlier')
    expect(groupOf(new Date(2026, 9, 15, 9, 0), NOW)).toBe('Today')
  })
})

describe('groupConversations', () => {
  it("groups in the mock's order, keeps the history's order inside a group and drops empty groups", () => {
    const chats = [
      chat('a', new Date(2026, 9, 14, 15, 28)),
      chat('b', new Date(2026, 9, 13, 10, 0)),
      chat('c', new Date(2026, 9, 1, 10, 0)),
      chat('d', new Date(2026, 9, 12, 10, 0)),
      chat('e', new Date(2026, 9, 14, 9, 0)),
    ]
    const groups = groupConversations(chats, NOW)
    expect(groups.map((group) => [group.label, group.chats.map((c) => c.id)])).toEqual([
      ['Today', ['a', 'e']],
      ['This week', ['b', 'd']],
      ['Earlier', ['c']],
    ])
    expect(groupConversations([chat('c', new Date(2026, 9, 1, 10, 0))], NOW).map((g) => g.label)).toEqual(['Earlier'])
    expect(groupConversations([], NOW)).toEqual([])
  })

  it('dates a conversation by its last activity, falling back to when it began', () => {
    const moved = chat('a', new Date(2026, 8, 1), { updatedAt: new Date(2026, 9, 14, 12, 0).toISOString() })
    expect(groupConversations([moved], NOW)[0]?.label).toBe('Today')
    const old = chat('b', new Date(2026, 8, 1), { updatedAt: '' })
    expect(lastActive(old)?.getMonth()).toBe(8)
    expect(lastActive(chat('c', null, { updatedAt: 'not a date' }))).toBeNull()
  })
})

describe('relativeTime', () => {
  it('counts minutes and hours today', () => {
    expect(relativeTime(new Date(2026, 9, 14, 15, 29, 40), NOW)).toBe('Just now')
    expect(relativeTime(new Date(2026, 9, 14, 15, 28), NOW)).toBe('2 min')
    expect(relativeTime(new Date(2026, 9, 14, 12, 0), NOW)).toBe('3 h')
  })

  it('says Yesterday, then the weekday, then the date', () => {
    expect(relativeTime(new Date(2026, 9, 13, 22, 0), NOW)).toBe('Yesterday')
    expect(relativeTime(new Date(2026, 9, 12, 9, 0), NOW)).toBe('Mon')
    expect(relativeTime(new Date(2026, 9, 8, 9, 0), NOW)).toBe('Thu')
    expect(relativeTime(new Date(2026, 9, 7, 9, 0), NOW)).toBe('7 Oct')
    expect(relativeTime(new Date(2025, 11, 24, 9, 0), NOW)).toBe('24 Dec 2025')
  })

  it('is empty when the date is unknown', () => {
    expect(relativeTime(null, NOW)).toBe('')
  })
})

describe('titles, the newest line and search', () => {
  it('names an untitled conversation and tidies white space', () => {
    expect(conversationTitle(chat('a', NOW, { title: '  ' }))).toBe('New chat')
    expect(conversationTitle(chat('a', NOW, { title: 'Laptop\n for  uni' }))).toBe('Laptop for uni')
  })

  it("shows the newest line as written (the core's preview may be the user's own), none when empty", () => {
    expect(previewLine(chat('a', NOW, { lastMessagePreview: 'Done!  Three laptops\nmake the cut.' }))).toBe(
      'Done! Three laptops make the cut.',
    )
    expect(previewLine(chat('a', NOW, { lastMessagePreview: '  ' }))).toBeNull()
    expect(previewLine(chat('a', NOW))).toBeNull()
  })

  it('filters by title, ignoring case and surrounding spaces, never by the preview', () => {
    const chats = [
      chat('a', NOW, { title: 'Laptop for uni' }),
      chat('b', NOW, { title: 'Revision plan for chemistry', lastMessagePreview: 'laptop' }),
    ]
    expect(filterByTitle(chats, '  LAPTOP ').map((c) => c.id)).toEqual(['a'])
    expect(filterByTitle(chats, 'for').map((c) => c.id)).toEqual(['a', 'b'])
    expect(filterByTitle(chats, '')).toBe(chats)
    expect(filterByTitle(chats, 'lisbon')).toEqual([])
  })
})
