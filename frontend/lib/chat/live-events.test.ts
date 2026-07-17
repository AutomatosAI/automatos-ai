/**
 * PRD-205 S7 -- the client bridge for background->chat delivery.
 *
 * Pins:
 *   1. `parseChatChangedPayload` accepts exactly the notify_chat_event shape
 *      and rejects board payloads / malformed data (frames share the
 *      `board_changed` SSE event name, so payload routing is the ONLY gate).
 *   2. The per-user filter drops other users' events once the viewer's
 *      integer id is known, and fails open while it is unknown (listeners
 *      are self-scoped, so a false positive is just a cheap refetch).
 *   3. `routeChatChangedPayload` dispatches the window CustomEvent for
 *      deliverable chat payloads and claims (returns true for) ALL chat
 *      payloads so the caller never also refetches the board for them.
 *   4. `mergeBackgroundMessages` is append-only, no-dupe, createdAt-ordered,
 *      ignores in-turn rows (no persisted source), and returns the same
 *      reference when nothing new landed.
 *   5. `attachPersistedSource` lifts a row's `source` into metadata.source
 *      without mutating the input row.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import {
  CHAT_CHANGED_EVENT,
  attachPersistedSource,
  dispatchChatChanged,
  getViewerUserId,
  hasBackgroundSource,
  mergeBackgroundMessages,
  parseChatChangedPayload,
  rememberViewerUserId,
  resetViewerUserIdForTests,
  routeChatChangedPayload,
  shouldDeliverChatEvent,
  type ChatChangedDetail,
} from './live-events'

const CHAT_PAYLOAD = {
  workspace_id: 'ws-1',
  chat_id: 'chat-1',
  user_id: 7,
  event: 'chat_changed',
}

const BOARD_PAYLOAD = {
  workspace_id: 'ws-1',
  task_id: 42,
  status: 'done',
  event: 'task_changed',
}

describe('parseChatChangedPayload', () => {
  it('parses the exact notify_chat_event payload', () => {
    expect(parseChatChangedPayload(CHAT_PAYLOAD)).toEqual({
      chatId: 'chat-1',
      userId: 7,
    })
  })

  it('returns null for board payloads and the connect frame', () => {
    expect(parseChatChangedPayload(BOARD_PAYLOAD)).toBeNull()
    expect(parseChatChangedPayload({ workspace_id: 'ws-1' })).toBeNull()
  })

  it('returns null for malformed data', () => {
    expect(parseChatChangedPayload(null)).toBeNull()
    expect(parseChatChangedPayload('chat_changed')).toBeNull()
    expect(parseChatChangedPayload({ ...CHAT_PAYLOAD, chat_id: '' })).toBeNull()
    expect(parseChatChangedPayload({ ...CHAT_PAYLOAD, user_id: '7' })).toBeNull()
    expect(parseChatChangedPayload({ ...CHAT_PAYLOAD, user_id: 7.5 })).toBeNull()
  })
})

describe('viewer id cache + per-user filter', () => {
  beforeEach(() => resetViewerUserIdForTests())
  afterEach(() => resetViewerUserIdForTests())

  it('remembers only integer ids', () => {
    rememberViewerUserId('user_2abc')
    rememberViewerUserId(3.14)
    expect(getViewerUserId()).toBeNull()
    rememberViewerUserId(7)
    expect(getViewerUserId()).toBe(7)
  })

  it('drops foreign-user events once the viewer is known', () => {
    expect(shouldDeliverChatEvent(8, 7)).toBe(false)
    expect(shouldDeliverChatEvent(7, 7)).toBe(true)
  })

  it('fails open while the viewer is unknown', () => {
    expect(shouldDeliverChatEvent(8, null)).toBe(true)
  })
})

describe('routeChatChangedPayload', () => {
  let received: string[]
  const listener = (event: Event) =>
    received.push((event as CustomEvent<ChatChangedDetail>).detail.chatId)

  beforeEach(() => {
    resetViewerUserIdForTests()
    received = []
    window.addEventListener(CHAT_CHANGED_EVENT, listener)
  })
  afterEach(() => {
    window.removeEventListener(CHAT_CHANGED_EVENT, listener)
    resetViewerUserIdForTests()
  })

  it('dispatches the window event for the viewer and claims the payload', () => {
    rememberViewerUserId(7)
    expect(routeChatChangedPayload(CHAT_PAYLOAD)).toBe(true)
    expect(received).toEqual(['chat-1'])
  })

  it('claims but does NOT dispatch a foreign user event', () => {
    rememberViewerUserId(99)
    expect(routeChatChangedPayload(CHAT_PAYLOAD)).toBe(true)
    expect(received).toEqual([])
  })

  it('leaves board payloads to the caller', () => {
    expect(routeChatChangedPayload(BOARD_PAYLOAD)).toBe(false)
    expect(received).toEqual([])
  })

  it('dispatchChatChanged emits the detail shape listeners rely on', () => {
    dispatchChatChanged('chat-9')
    expect(received).toEqual(['chat-9'])
  })
})

// ---------------------------------------------------------------------------
// Merge + source mapping
// ---------------------------------------------------------------------------

const SOURCE = {
  origin: 'watcher',
  label: 'Auto \u00b7 background',
  link_type: 'watch',
  link_id: 'w-1',
}

function backgroundMsg(id: string, createdAt: string) {
  return { id, role: 'assistant', createdAt, metadata: { source: { ...SOURCE } } }
}

describe('mergeBackgroundMessages', () => {
  const local = [
    { id: 'u-1', role: 'user' },
    { id: 'a-1', role: 'assistant' }, // in-flight placeholder: no createdAt
  ]

  it('appends missing background messages after existing entries', () => {
    const fetched = [
      { id: 'db-1', role: 'user', createdAt: '2026-07-17T00:00:01Z' }, // in-turn, no source
      backgroundMsg('bg-1', '2026-07-17T00:00:02Z'),
    ]
    const merged = mergeBackgroundMessages(local as any, fetched as any)
    expect(merged.map((m: any) => m.id)).toEqual(['u-1', 'a-1', 'bg-1'])
    // Existing entries (incl. the placeholder) are the same objects, untouched.
    expect(merged[0]).toBe(local[0])
    expect(merged[1]).toBe(local[1])
  })

  it('never duplicates an already-merged id', () => {
    const first = mergeBackgroundMessages(
      local as any,
      [backgroundMsg('bg-1', '2026-07-17T00:00:02Z')] as any
    )
    const second = mergeBackgroundMessages(
      first,
      [backgroundMsg('bg-1', '2026-07-17T00:00:02Z')] as any
    )
    expect(second).toBe(first) // same reference -- no re-render
    expect(second.filter((m: any) => m.id === 'bg-1')).toHaveLength(1)
  })

  it('returns the same reference when the fetch adds nothing', () => {
    const fetched = [{ id: 'db-1', role: 'user', createdAt: '2026-07-17T00:00:01Z' }]
    expect(mergeBackgroundMessages(local as any, fetched as any)).toBe(local)
  })

  it('orders multiple additions by createdAt', () => {
    const fetched = [
      backgroundMsg('bg-2', '2026-07-17T00:00:05Z'),
      backgroundMsg('bg-1', '2026-07-17T00:00:03Z'),
    ]
    const merged = mergeBackgroundMessages([] as any, fetched as any)
    expect(merged.map((m: any) => m.id)).toEqual(['bg-1', 'bg-2'])
  })

  it('ignores in-turn rows -- id-merge alone would duplicate the thread', () => {
    // The fetched history re-sends the whole conversation under DB ids that
    // never match the client-generated local ids; only source-carrying rows
    // may merge.
    const fetched = [
      { id: 'db-u-1', role: 'user', createdAt: '2026-07-17T00:00:01Z' },
      { id: 'db-a-1', role: 'assistant', createdAt: '2026-07-17T00:00:02Z' },
      backgroundMsg('bg-1', '2026-07-17T00:00:03Z'),
    ]
    const merged = mergeBackgroundMessages(local as any, fetched as any)
    expect(merged.map((m: any) => m.id)).toEqual(['u-1', 'a-1', 'bg-1'])
  })
})

describe('attachPersistedSource (badge mapping)', () => {
  it('lifts a row source into metadata.source without mutating the row', () => {
    const row = {
      id: 'm-1',
      role: 'assistant',
      parts: [{ type: 'text', text: 'Watched it. 8.4/10, passed.' }],
      createdAt: '2026-07-17T00:00:03Z',
      source: { ...SOURCE },
    }
    const mapped: any = attachPersistedSource(row as any)
    expect(mapped.metadata.source).toEqual(SOURCE)
    expect(mapped).not.toBe(row)
    expect((row as any).metadata).toBeUndefined()
    expect(hasBackgroundSource(mapped)).toBe(true)
  })

  it('passes rows without a source through untouched', () => {
    const row = { id: 'm-2', role: 'user', source: null }
    expect(attachPersistedSource(row as any)).toBe(row)
    expect(hasBackgroundSource(row)).toBe(false)
  })

  it('preserves existing metadata keys', () => {
    const row = {
      id: 'm-3',
      metadata: { processing_time: 1.2 },
      source: { ...SOURCE },
    }
    const mapped: any = attachPersistedSource(row as any)
    expect(mapped.metadata.processing_time).toBe(1.2)
    expect(mapped.metadata.source).toEqual(SOURCE)
  })
})
