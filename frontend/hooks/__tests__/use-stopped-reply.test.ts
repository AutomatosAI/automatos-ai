/**
 * PRD-260 decision F4 (P260-FX-30) — useStoppedReplies: Stop keeps the reply
 * Auto was writing. The conversation Play shows is useChat's with each stopped
 * reply kept: as it was when Stop was pressed while useChat still holds it, put
 * back after the message it followed once useChat drops it (its catch on the
 * abort), and gone with that message when Try again takes it. The chat itself
 * is a plain object here; lib/chat-controller's tests run Stop through the
 * core's real useChat (Play's own chat tests moved to the theme kit, W-K6).
 */
import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import {
  replyBeingWritten,
  type StoppableChat,
  type StoppedReply,
  useStoppedReplies,
  withStoppedReplies,
} from '@/hooks/use-stopped-reply'
import type { ChatMessage } from '@/types'

const message = (id: string, role: 'user' | 'assistant', text: string) =>
  ({ id, role, content: text, parts: [{ type: 'text', text }] }) as unknown as ChatMessage

const HELLO = message('u1', 'user', 'Write me a story')
const PARTIAL = message('r1', 'assistant', 'The keeper climbed')
const LONGER = message('r1', 'assistant', 'The keeper climbed the stairs')
const NEXT_ASK = message('u2', 'user', 'A shorter one?')
const NEXT_REPLY = message('r2', 'assistant', 'Here is one.')
const STOPPED: StoppedReply = { reply: PARTIAL, after: 'u1' }

describe('the reply being written', () => {
  it('is the newest message while the chat streams, when it is a reply that follows a message', () => {
    expect(replyBeingWritten({ messages: [HELLO, PARTIAL], status: 'streaming' })).toEqual(STOPPED)
    expect(replyBeingWritten({ messages: [HELLO, PARTIAL], status: 'idle' })).toBeNull()
    expect(replyBeingWritten({ messages: [HELLO], status: 'streaming' })).toBeNull()
    expect(replyBeingWritten({ messages: [PARTIAL], status: 'streaming' })).toBeNull()
  })

  it('is none while the page only awaits a reply the server finishes detached', () => {
    expect(replyBeingWritten({ messages: [HELLO, PARTIAL], status: 'streaming', awaitingReply: true })).toBeNull()
  })
})

describe('the conversation with the stopped replies kept', () => {
  it('is the same list when nothing was stopped', () => {
    const messages = [HELLO, PARTIAL]
    expect(withStoppedReplies(messages, [])).toBe(messages)
  })

  it('shows a reply useChat still holds as it was when Stop was pressed', () => {
    expect(withStoppedReplies([HELLO, LONGER], [STOPPED])).toEqual([HELLO, PARTIAL])
  })

  it('puts a reply useChat dropped back after the message it followed, through the turns that follow', () => {
    expect(withStoppedReplies([HELLO], [STOPPED])).toEqual([HELLO, PARTIAL])
    expect(withStoppedReplies([HELLO, NEXT_ASK, NEXT_REPLY], [STOPPED])).toEqual([HELLO, PARTIAL, NEXT_ASK, NEXT_REPLY])
  })

  it('drops a stopped reply whose message left the conversation, and never mutates the list', () => {
    const messages = [NEXT_ASK, NEXT_REPLY]
    expect(withStoppedReplies(messages, [STOPPED])).toEqual([NEXT_ASK, NEXT_REPLY])
    const before = [HELLO, NEXT_ASK]
    withStoppedReplies(before, [STOPPED])
    expect(before).toEqual([HELLO, NEXT_ASK])
  })
})

describe('useStoppedReplies', () => {
  function mount(chat: StoppableChat) {
    return renderHook((props: StoppableChat) => useStoppedReplies(props), { initialProps: chat })
  }

  it("Stop keeps the reply being written, then calls the chat's stop(); the reply stays once useChat drops it", () => {
    const stop = vi.fn()
    const view = mount({ messages: [HELLO, PARTIAL], status: 'streaming', stop })
    act(() => view.result.current.stop())
    expect(stop).toHaveBeenCalledTimes(1)
    expect(view.result.current.isStopped('r1')).toBe(true)

    view.rerender({ messages: [HELLO], status: 'idle', stop })
    expect(view.result.current.messages).toEqual([HELLO, PARTIAL])
    view.rerender({ messages: [HELLO, NEXT_ASK, NEXT_REPLY], status: 'idle', stop })
    expect(view.result.current.messages).toEqual([HELLO, PARTIAL, NEXT_ASK, NEXT_REPLY])
    expect(view.result.current.isStopped('r2')).toBe(false)
  })

  it('keeps nothing when Stop is pressed while the page only awaits a detached reply, and still stops', () => {
    const stop = vi.fn()
    const view = mount({ messages: [HELLO, PARTIAL], status: 'streaming', awaitingReply: true, stop })
    act(() => view.result.current.stop())
    expect(stop).toHaveBeenCalledTimes(1)
    expect(view.result.current.isStopped('r1')).toBe(false)
    view.rerender({ messages: [HELLO], status: 'idle', stop })
    expect(view.result.current.messages).toEqual([HELLO])
  })
})
