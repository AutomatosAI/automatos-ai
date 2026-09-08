/**
 * PRD-237: the conversation session — storage semantics and tab rules.
 */
import { describe, it, expect, beforeEach } from 'vitest'
import {
  EMPTY_CHAT_SESSION,
  MAX_OPEN_CHATS,
  MAX_TRACKED_THREADS,
  chatSessionKey,
  isChatUnread,
  loadChatSession,
  mergeServerSession,
  saveChatSession,
  sessionTabs,
  threadTimeAgo,
  toServerSession,
  withChatIdAssigned,
  withChatOpened,
  withChatTouched,
  withDraft,
  withDraftClosed,
  withTabClosed,
  withThreadRead,
  withTitles,
  type ChatSession,
} from './chat-session'

const NOW = 1_800_000_000_000
const KEY = chatSessionKey('ws-1', 'user-1')

/** Open `ids` in order, then focus `active` (default: the last one). */
const open = (ids: string[], active = ids[ids.length - 1]): ChatSession => {
  const opened = ids.reduce<ChatSession>((s, id, i) => withChatOpened(s, id, NOW + i), EMPTY_CHAT_SESSION)
  return withChatOpened(opened, active, NOW + ids.length)
}

describe('chatSessionKey', () => {
  it('scopes by workspace and user, with anon/default fallbacks', () => {
    expect(chatSessionKey('ws-1', 'u-1')).not.toBe(chatSessionKey('ws-2', 'u-1'))
    expect(chatSessionKey('ws-1', 'u-1')).not.toBe(chatSessionKey('ws-1', 'u-2'))
    expect(chatSessionKey(null, undefined)).toBe('automatos:chat:session:default:anon')
  })
})

describe('load/save', () => {
  beforeEach(() => localStorage.clear())

  it('round-trips and never expires', () => {
    const session = withChatOpened(EMPTY_CHAT_SESSION, 'c1', NOW)
    saveChatSession(KEY, session)
    expect(loadChatSession(KEY)).toEqual(session)
  })

  it('drops corrupt JSON, invalid shapes and the old widget-session shape', () => {
    localStorage.setItem(KEY, '{not json')
    expect(loadChatSession(KEY)).toEqual(EMPTY_CHAT_SESSION)
    localStorage.setItem(KEY, JSON.stringify({ activeChatId: 'c1', lastActiveAt: NOW, lastReadAt: {}, closedThreadIds: [] }))
    expect(loadChatSession(KEY)).toEqual(EMPTY_CHAT_SESSION)
  })
})

describe('tabs', () => {
  it('an empty session shows exactly the draft tab', () => {
    expect(sessionTabs(EMPTY_CHAT_SESSION)).toEqual([{ id: null, isDraft: true }])
  })

  it('opening a chat adds a tab, activates it and dismisses an empty draft', () => {
    const draft = withDraft(EMPTY_CHAT_SESSION, NOW)
    const next = withChatOpened(draft, 'c1', NOW + 1)
    expect(next.activeChatId).toBe('c1')
    expect(next.openChatIds).toEqual(['c1'])
    expect(next.draftOpen).toBe(false)
    expect(sessionTabs(next)).toEqual([{ id: 'c1', isDraft: false }])
    expect(draft.draftOpen).toBe(true) // input untouched
  })

  it('New Chat twice is one draft; a draft survives as the active tab', () => {
    const s = withChatOpened(EMPTY_CHAT_SESSION, 'c1', NOW)
    const once = withDraft(s, NOW + 1)
    expect(withDraft(once, NOW + 2)).toBe(once)
    expect(sessionTabs(once)).toEqual([{ id: 'c1', isDraft: false }, { id: null, isDraft: true }])
    expect(once.activeChatId).toBeNull()
  })

  it('the first message swaps the draft for the real id in place', () => {
    const draft = withDraft(withChatOpened(EMPTY_CHAT_SESSION, 'c1', NOW), NOW + 1)
    const named = withChatIdAssigned(draft, 'c2', NOW + 2)
    expect(named.openChatIds).toEqual(['c1', 'c2'])
    expect(named.activeChatId).toBe('c2')
    expect(named.draftOpen).toBe(false)
    // every later turn re-announces the id — a no-op, same reference
    expect(withChatIdAssigned(named, 'c2', NOW + 3)).toBe(named)
  })

  it('closing the active tab focuses the right neighbour, else the left, else the draft', () => {
    const s = open(['a', 'b', 'c'], 'b')
    const closedB = withTabClosed(s, 'b', NOW + 10)
    expect(closedB.openChatIds).toEqual(['a', 'c'])
    expect(closedB.activeChatId).toBe('c')
    const closedC = withTabClosed(closedB, 'c', NOW + 11)
    expect(closedC.activeChatId).toBe('a')
    const closedA = withTabClosed(closedC, 'a', NOW + 12)
    expect(closedA.openChatIds).toEqual([])
    expect(closedA.activeChatId).toBeNull()
    expect(closedA.draftOpen).toBe(true)
  })

  it('closing an inactive tab keeps focus; closing an unknown id is a no-op', () => {
    const s = open(['a', 'b'], 'b')
    expect(withTabClosed(s, 'a', NOW + 1).activeChatId).toBe('b')
    expect(withTabClosed(s, 'zzz', NOW + 1)).toBe(s)
  })

  it('the draft can be closed only when another tab remains', () => {
    const lonely = withDraft(EMPTY_CHAT_SESSION, NOW)
    expect(withDraftClosed(lonely, NOW + 1)).toBe(lonely)
    const s = withDraft(withChatOpened(EMPTY_CHAT_SESSION, 'a', NOW), NOW + 1)
    const closed = withDraftClosed(s, NOW + 2)
    expect(closed.draftOpen).toBe(false)
    expect(closed.activeChatId).toBe('a')
  })

  it('evicts the oldest inactive tab beyond the cap, never the active one', () => {
    const ids = Array.from({ length: MAX_OPEN_CHATS }, (_, i) => `c${i}`)
    const full = open(ids, 'c0')
    const over = withChatOpened(full, 'extra', NOW + 100)
    expect(over.openChatIds).toHaveLength(MAX_OPEN_CHATS)
    expect(over.openChatIds).toContain('c0') // the tab the user was just on — protected
    expect(over.openChatIds).not.toContain('c1') // oldest unprotected — evicted
    expect(over.openChatIds[over.openChatIds.length - 1]).toBe('extra')
  })
})

describe('read / unread', () => {
  it('flags an open background tab, never the active or an unopened one', () => {
    const s = open(['a', 'b'], 'b')
    expect(withChatTouched(s, 'b')).toBe(s)
    expect(withChatTouched(s, 'nope')).toBe(s)
    const touched = withChatTouched(s, 'a')
    expect(isChatUnread(touched, 'a')).toBe(true)
    expect(withChatTouched(touched, 'a')).toBe(touched)
    expect(isChatUnread(withChatOpened(touched, 'a', NOW + 5), 'a')).toBe(false)
    expect(isChatUnread(withThreadRead(touched, 'a', NOW + 5), 'a')).toBe(false)
  })

  it('withThreadRead is a no-op for null ids and never mutates', () => {
    const s = open(['a'])
    const before = JSON.parse(JSON.stringify(s)) as ChatSession
    expect(withThreadRead(s, null, NOW)).toBe(s)
    withThreadRead(s, 'a', NOW + 1)
    expect(s).toEqual(before)
  })

  it('caps tracked read stamps', () => {
    let s = EMPTY_CHAT_SESSION
    for (let i = 0; i < MAX_TRACKED_THREADS + 10; i++) s = withThreadRead(s, `t${i}`, NOW + i)
    expect(Object.keys(s.lastReadAt)).toHaveLength(MAX_TRACKED_THREADS)
    expect(s.lastReadAt[`t${MAX_TRACKED_THREADS + 9}`]).toBe(NOW + MAX_TRACKED_THREADS + 9)
    expect(s.lastReadAt.t0).toBeUndefined()
  })
})

describe('titles', () => {
  it('merges labels and is a no-op when nothing changes', () => {
    const s = open(['a', 'b'])
    const titled = withTitles(s, { a: 'Alpha', b: 'Beta', other: 'Elsewhere' })
    expect(titled.titles).toEqual({ a: 'Alpha', b: 'Beta', other: 'Elsewhere' })
    expect(withTitles(titled, { a: 'Alpha' })).toBe(titled)
    expect(withTitles(titled, { a: '' })).toBe(titled)
  })
})

describe('server reconciliation (hosted edition)', () => {
  const at = (ms: number) => new Date(ms).toISOString()

  it('a fresh device inherits the server tabs', () => {
    const merged = mergeServerSession(
      EMPTY_CHAT_SESSION,
      { activeChatId: 'b', draftOpen: false, openChatIds: ['a', 'b'], lastReadAt: { a: 5 }, updatedAt: at(NOW) },
      NOW + 1,
    )
    expect(merged.openChatIds).toEqual(['a', 'b'])
    expect(merged.activeChatId).toBe('b')
    expect(merged.lastReadAt).toEqual({ a: 5 })
    expect(merged.lastActiveAt).toBe(NOW)
  })

  it('a newer local change beats an older server copy', () => {
    const local = withChatOpened(EMPTY_CHAT_SESSION, 'local', NOW + 10)
    const merged = mergeServerSession(local, { activeChatId: 'srv', draftOpen: false, openChatIds: ['srv'], lastReadAt: {}, updatedAt: at(NOW) }, NOW + 20)
    expect(merged).toBe(local)
  })

  it('an active id the server lists nowhere falls back to the draft; malformed payloads are ignored', () => {
    const merged = mergeServerSession(EMPTY_CHAT_SESSION, { activeChatId: 'ghost', draftOpen: false, openChatIds: ['a'], lastReadAt: {}, updatedAt: at(NOW) }, NOW)
    expect(merged.activeChatId).toBeNull()
    expect(merged.draftOpen).toBe(true)
    expect(mergeServerSession(EMPTY_CHAT_SESSION, null)).toBe(EMPTY_CHAT_SESSION)
    expect(mergeServerSession(EMPTY_CHAT_SESSION, { openChatIds: 'nope' } as never)).toBe(EMPTY_CHAT_SESSION)
    // the server's empty default (never written) leaves the device alone — the cold-start fallback still runs
    expect(
      mergeServerSession(EMPTY_CHAT_SESSION, { activeChatId: null, draftOpen: false, openChatIds: [], lastReadAt: {}, updatedAt: null }),
    ).toBe(EMPTY_CHAT_SESSION)
  })

  it('toServerSession carries ids and stamps only — no titles, no unread', () => {
    const s = withTitles(withChatTouched(open(['a', 'b'], 'b'), 'a'), { a: 'Alpha' })
    expect(toServerSession(s)).toEqual({
      activeChatId: 'b',
      draftOpen: false,
      openChatIds: ['a', 'b'],
      lastReadAt: s.lastReadAt,
    })
  })
})

describe('threadTimeAgo', () => {
  it('formats compact relative times', () => {
    expect(threadTimeAgo(new Date(NOW - 30_000).toISOString(), NOW)).toBe('now')
    expect(threadTimeAgo(new Date(NOW - 5 * 60_000).toISOString(), NOW)).toBe('5m')
    expect(threadTimeAgo(new Date(NOW - 3 * 3_600_000).toISOString(), NOW)).toBe('3h')
    expect(threadTimeAgo(new Date(NOW - 49 * 3_600_000).toISOString(), NOW)).toBe('2d')
    expect(threadTimeAgo('garbage', NOW)).toBe('')
  })
})
