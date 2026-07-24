/**
 * PRD-220: widget chat session persistence — storage semantics.
 */
import { describe, it, expect, beforeEach } from 'vitest'
import {
  EMPTY_WIDGET_SESSION,
  MAX_WIDGET_THREADS,
  WIDGET_SESSION_TTL_MS,
  isThreadUnread,
  loadWidgetSession,
  saveWidgetSession,
  threadTimeAgo,
  visibleThreads,
  widgetSessionKey,
  withActiveChat,
  withThreadClosed,
  withThreadRead,
  withoutActiveChat,
  type WidgetChatSession,
} from './widget-session'

const NOW = 1_800_000_000_000
const KEY = widgetSessionKey('ws-1', 'user-1')

describe('widgetSessionKey', () => {
  it('scopes by workspace and user', () => {
    expect(widgetSessionKey('ws-1', 'u-1')).not.toBe(widgetSessionKey('ws-2', 'u-1'))
    expect(widgetSessionKey('ws-1', 'u-1')).not.toBe(widgetSessionKey('ws-1', 'u-2'))
  })

  it('falls back for missing scope parts', () => {
    expect(widgetSessionKey(null, undefined)).toBe('automatos:widget:chat:default:anon')
  })
})

describe('load/save round-trip', () => {
  beforeEach(() => localStorage.clear())

  it('returns the empty session when nothing is stored', () => {
    expect(loadWidgetSession(KEY, NOW)).toEqual(EMPTY_WIDGET_SESSION)
  })

  it('round-trips a saved session', () => {
    const session = withActiveChat(EMPTY_WIDGET_SESSION, 'chat-1', NOW)
    saveWidgetSession(KEY, session)
    expect(loadWidgetSession(KEY, NOW)).toEqual(session)
  })

  it('drops corrupt JSON and invalid shapes', () => {
    localStorage.setItem(KEY, '{not json')
    expect(loadWidgetSession(KEY, NOW)).toEqual(EMPTY_WIDGET_SESSION)

    localStorage.setItem(KEY, JSON.stringify({ activeChatId: 7, lastActiveAt: 'x' }))
    expect(loadWidgetSession(KEY, NOW)).toEqual(EMPTY_WIDGET_SESSION)
  })

  it('expires only the resume pointer after the TTL, keeping read markers', () => {
    const session = withActiveChat(EMPTY_WIDGET_SESSION, 'chat-1', NOW)
    saveWidgetSession(KEY, session)

    const later = NOW + WIDGET_SESSION_TTL_MS + 1
    const loaded = loadWidgetSession(KEY, later)
    expect(loaded.activeChatId).toBeNull()
    expect(loaded.lastReadAt['chat-1']).toBe(NOW)
  })

  it('resumes within the TTL', () => {
    saveWidgetSession(KEY, withActiveChat(EMPTY_WIDGET_SESSION, 'chat-1', NOW))
    expect(loadWidgetSession(KEY, NOW + WIDGET_SESSION_TTL_MS - 1).activeChatId).toBe('chat-1')
  })
})

describe('immutable updaters', () => {
  it('withActiveChat never mutates and reopens closed threads', () => {
    const closed = withThreadClosed(EMPTY_WIDGET_SESSION, 'chat-1', NOW)
    const before = JSON.parse(JSON.stringify(closed)) as WidgetChatSession

    const next = withActiveChat(closed, 'chat-1', NOW + 1)

    expect(closed).toEqual(before)
    expect(next.activeChatId).toBe('chat-1')
    expect(next.closedThreadIds).not.toContain('chat-1')
    expect(next.lastReadAt['chat-1']).toBe(NOW + 1)
  })

  it('withoutActiveChat clears the pointer only', () => {
    const active = withActiveChat(EMPTY_WIDGET_SESSION, 'chat-1', NOW)
    const next = withoutActiveChat(active, NOW + 5)
    expect(next.activeChatId).toBeNull()
    expect(next.lastReadAt).toEqual(active.lastReadAt)
    expect(active.activeChatId).toBe('chat-1')
  })

  it('withThreadClosed clears the active pointer when closing the active thread', () => {
    const active = withActiveChat(EMPTY_WIDGET_SESSION, 'chat-1', NOW)
    const next = withThreadClosed(active, 'chat-1', NOW + 1)
    expect(next.activeChatId).toBeNull()
    expect(next.closedThreadIds).toContain('chat-1')
  })

  it('withThreadRead is a no-op for null ids', () => {
    expect(withThreadRead(EMPTY_WIDGET_SESSION, null, NOW)).toBe(EMPTY_WIDGET_SESSION)
  })
})

describe('isThreadUnread', () => {
  const at = (ms: number) => new Date(ms).toISOString()

  it('flags threads that moved since last read', () => {
    const session = withThreadRead(EMPTY_WIDGET_SESSION, 'chat-1', NOW)
    expect(isThreadUnread(session, { id: 'chat-1', updatedAt: at(NOW + 1000) })).toBe(true)
    expect(isThreadUnread(session, { id: 'chat-1', updatedAt: at(NOW - 1000) })).toBe(false)
  })

  it('never flags the active thread or never-opened threads', () => {
    const session = withActiveChat(EMPTY_WIDGET_SESSION, 'chat-1', NOW)
    expect(isThreadUnread(session, { id: 'chat-1', updatedAt: at(NOW + 9999) })).toBe(false)
    // chat-2 has no baseline — no badge.
    expect(isThreadUnread(session, { id: 'chat-2', updatedAt: at(NOW + 9999) })).toBe(false)
  })
})

describe('visibleThreads', () => {
  const threads = Array.from({ length: 8 }, (_, i) => ({ id: `chat-${i}` }))

  it('filters closed threads and caps the list', () => {
    const session = withThreadClosed(EMPTY_WIDGET_SESSION, 'chat-0', NOW)
    const visible = visibleThreads(threads, session)
    expect(visible).toHaveLength(MAX_WIDGET_THREADS)
    expect(visible.map((t) => t.id)).not.toContain('chat-0')
    expect(visible[0].id).toBe('chat-1')
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
