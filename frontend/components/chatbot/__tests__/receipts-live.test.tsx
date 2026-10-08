/**
 * PRD-256 FX-001 — the live receipts reach the reply without touching `useChat` or `Message`.
 *
 * `useChat` hands a `tool-data` frame to its data callback; the backend sends the receipts
 * frame's data there too, under `receipts`. `useChatWithReceipts` puts it on the reply being
 * streamed and then runs the page's own callback; `MessageWithReceipts` draws the block above
 * the reply's bubble.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, renderHook, screen, cleanup } from '@testing-library/react'
import React from 'react'
import type { ChatMessage, Receipt } from '@/types'

const chatState: { options?: any; messages: ChatMessage[] } = { messages: [] }
vi.mock('@/lib/chat/hooks', () => ({
  useChat: (options: any) => {
    chatState.options = options
    return {
      messages: chatState.messages,
      setMessages: (next: (prev: ChatMessage[]) => ChatMessage[]) => {
        chatState.messages = next(chatState.messages)
      },
    }
  },
}))
vi.mock('../message', () => ({
  Message: ({ message }: { message: ChatMessage }) => <div data-testid="the-message">{message.content}</div>,
}))

import { useChatWithReceipts } from '@/lib/chat/use-chat-with-receipts'
import { MessageWithReceipts } from '../message-with-receipts'
import { liveReceiptsFromToolData, withReceiptsOnTheReply } from '@/lib/chat/receipts'

afterEach(cleanup)

const MOVED: Receipt = {
  action: 'platform_update_task_status', kind: 'write', status: 'done', subject: '#0422',
  effect: 'moved to Done', link: '/command-center?tab=board&task_id=422', reason: null,
}
const TRIED = "I tried to approve the mission and it didn't go through: Mission m-1 is not waiting for approval."
const FRAME_DATA = { receipts: [MOVED], model: 'google/gemini-2.5-flash', above: [TRIED] }
const TOOL_DATA = { type: 'tool-data', data: { receipts: FRAME_DATA } }
const user = { id: 'u1', role: 'user', content: 'Approve #0422', parts: [] } as unknown as ChatMessage
const reply = { id: 'a1', role: 'assistant', content: 'Done.', parts: [] } as unknown as ChatMessage

describe('the live receipts', () => {
  it('are read from a forwarded tool-data part, and from nothing else', () => {
    expect(liveReceiptsFromToolData(TOOL_DATA)).toEqual({ receipts: [MOVED], receiptsAbove: [TRIED] })
    expect(liveReceiptsFromToolData({ type: 'tool-data', data: { documents: [] } })).toBeUndefined()
    expect(liveReceiptsFromToolData({ type: 'tool-end', data: { receipts: FRAME_DATA } })).toBeUndefined()
    expect(liveReceiptsFromToolData(null)).toBeUndefined()
  })

  it('go on the reply being streamed, never mutating the list', () => {
    const before = [user, reply]
    const after = withReceiptsOnTheReply(before, { receipts: [MOVED], receiptsAbove: [TRIED] })
    expect(after[1]).toEqual({ ...reply, receipts: [MOVED], receiptsAbove: [TRIED] })
    expect(after[0]).toBe(user)
    expect(before[1]).not.toHaveProperty('receipts')
    expect(withReceiptsOnTheReply([user], { receipts: [MOVED] })).toEqual([user])
  })

  it('are set by the hook, then the page gets the part as before', () => {
    const pageOnData = vi.fn()
    chatState.messages = [user, reply]
    renderHook(() => useChatWithReceipts({ id: 'c1', onData: pageOnData }))

    chatState.options.onData(TOOL_DATA)
    expect(chatState.messages[1]).toMatchObject({ receipts: [MOVED], receiptsAbove: [TRIED] })
    expect(pageOnData).toHaveBeenCalledWith(TOOL_DATA)

    chatState.options.onData({ type: 'data-usage', data: { totalTokens: 3 } })
    expect(pageOnData).toHaveBeenCalledTimes(2)
  })

  it('marks an ask\'s tool-end waiting on the reply, then the page gets the part (P256-FIX-RVW-22)', () => {
    const pageOnData = vi.fn()
    const closed = { toolCallId: 'call_1', toolName: 'platform_execute', state: 'error' as const }
    chatState.messages = [user, { ...reply, toolCalls: [closed] }]
    renderHook(() => useChatWithReceipts({ id: 'c1', onData: pageOnData }))

    const end = { type: 'tool-end', data: { toolCallId: 'call_1', toolName: 'platform_execute', success: false, waiting: true } }
    chatState.options.onData(end)
    expect(chatState.messages[1].toolCalls![0]).toMatchObject({ state: 'completed', waiting: true })
    expect(pageOnData).toHaveBeenCalledWith(end)
  })
})

describe('MessageWithReceipts', () => {
  it('draws the receipts and the lines above, directly above the reply', () => {
    render(<MessageWithReceipts chatId="c1" setMessages={vi.fn() as any} regenerate={vi.fn()} isReadonly
      message={{ ...reply, receipts: [MOVED], receiptsAbove: [TRIED] }} />)
    const block = screen.getByTestId('receipts-block')
    expect(block).toHaveTextContent('#0422: moved to Done')
    expect(screen.getByTestId('receipts-above')).toHaveTextContent(TRIED)
    expect(block.compareDocumentPosition(screen.getByTestId('the-message')) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy()
  })

  it('reads a reloaded reply from its saved part', () => {
    const saved = { ...reply, parts: [{ type: 'receipts', receipts: [MOVED] }] } as unknown as ChatMessage
    render(<MessageWithReceipts chatId="c1" setMessages={vi.fn() as any} regenerate={vi.fn()} isReadonly message={saved} />)
    expect(screen.getByTestId('receipts-block')).toHaveTextContent('#0422: moved to Done')
  })

  it('leaves a message from before receipts, and a user message, as Message draws them', () => {
    render(<MessageWithReceipts chatId="c1" setMessages={vi.fn() as any} regenerate={vi.fn()} isReadonly message={reply} />)
    render(<MessageWithReceipts chatId="c1" setMessages={vi.fn() as any} regenerate={vi.fn()} isReadonly
      message={{ ...user, receipts: [MOVED] }} />)
    expect(screen.queryByTestId('receipts-block')).toBeNull()
    expect(screen.queryByTestId('message-with-receipts')).toBeNull()
    expect(screen.getAllByTestId('the-message')).toHaveLength(2)
  })
})
