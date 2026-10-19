/**
 * PRD-261 A-11a (M28) — the chat controller (lib/chat-controller/controller.ts)
 * on its own, with no theme around it: the stream's messages and parts reach
 * the view model (the theme's own reading first, then the turn's), Stop keeps
 * the reply being written once useChat drops it (P260-FX-30), Try again asks
 * the last message again and Send and Try again clear the turn's suggestion
 * (both refused while a turn runs), the approvals a reply raises are filed
 * once and go with a new chat's name, and a reply that failed in words is told
 * once. useChatWithReceipts is a plain stand-in here; the PRD-260 theme's
 * chat tests run the same controller under that theme's chat screen.
 */
import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/types'

const chat = vi.hoisted(() => ({
  options: null as null | Record<string, any>,
  status: 'idle' as 'idle' | 'streaming' | 'error',
  isLoading: false,
  messages: null as null | unknown[],
  sendMessage: vi.fn(),
  stop: vi.fn(),
  reload: vi.fn(),
}))
const session = vi.hoisted(() => ({ chatIdAssigned: vi.fn() }))

vi.mock('@/lib/chat/use-chat-with-receipts', () => ({
  useChatWithReceipts: (options: Record<string, any>) => {
    chat.options = options
    return {
      messages: chat.messages ?? options.initialMessages,
      setMessages: vi.fn(),
      sendMessage: chat.sendMessage,
      reload: chat.reload,
      stop: chat.stop,
      status: chat.status,
      isLoading: chat.isLoading,
      awaitingReply: false,
      usage: undefined,
      errorCode: null,
    }
  },
}))
vi.mock('@/lib/page-context', () => ({ usePageContext: () => ({ page: 'chat', route: '/chat' }) }))
vi.mock('@/hooks/use-refresh-history-on-turn-end', () => ({ useRefreshHistoryOnTurnEnd: vi.fn() }))
vi.mock('@/stores/chat-session-store', () => ({ useChatSessionStore: { getState: () => session } }))

import { cardsOfChat, useChatApprovalsStore } from '@/lib/chat-controller/approvals'
import { useChatController } from '@/lib/chat-controller/controller'
import type { ChatConversationView, ChatTurnEvent } from '@/lib/chat-controller/view-model'

const message = (id: string, role: 'user' | 'assistant', text: string, extra: Record<string, unknown> = {}) =>
  ({ id, role, content: text, parts: [{ type: 'text', text }], ...extra }) as unknown as ChatMessage

const HELLO = message('u1', 'user', 'Find me a laptop')
const PARTIAL = message('r1', 'assistant', 'Here are three')
const NEXT_ASK = message('u2', 'user', 'And a cheaper one?')
const FAILED = message('r2', 'assistant', '', { error: { message: 'The model took too long to answer.', code: 'timeout' } })

const STORED: ChatConversationView = { key: 'c1', chatId: 'c1', title: null, messages: [HELLO], awaitingReply: false }
const NEW_CHAT: ChatConversationView = { key: 'draft-1', chatId: null, title: null, messages: [], awaitingReply: false }

const DOCUMENT_PART = {
  type: 'tool-data',
  data: { generated_document: { deliverable_id: 'd-1', title: 'Laptop shortlist', format: 'pdf' } },
}
const SUGGESTION_PART = { type: 'mission-suggestion', data: { goal: 'Compare laptops weekly', complexity: 'low' } }
const TOOL_PART = { type: 'tool-data', data: { tool_approval: { grant_id: 42, action: 'platform_delete_document' } } }

function render(view: ChatConversationView) {
  const events: Array<ChatTurnEvent | 'theme-part'> = []
  const onEvent = (event: ChatTurnEvent) => events.push(event)
  const onData = vi.fn(() => events.push('theme-part'))
  const onChatIdAssigned = vi.fn()
  const hook = renderHook(() => useChatController({ view, onEvent, onData, onChatIdAssigned }))
  return { ...hook, events, onData, onChatIdAssigned }
}

const typesOf = (events: ReadonlyArray<ChatTurnEvent | 'theme-part'>) =>
  events.map((event) => (typeof event === 'string' ? event : event.type))

beforeEach(() => {
  chat.options = null
  chat.status = 'idle'
  chat.isLoading = false
  chat.messages = null
  chat.sendMessage.mockReset()
  chat.stop.mockReset()
  chat.reload.mockReset()
  session.chatIdAssigned.mockReset()
  useChatApprovalsStore.setState({ filed: [] })
})

describe('the chat controller: the stream', () => {
  it("opens the stream on the conversation on screen and shows the stream's messages", () => {
    chat.messages = [HELLO, PARTIAL]
    chat.status = 'streaming'
    const { result } = render(STORED)
    expect(chat.options).toMatchObject({ id: 'c1', initialMessages: STORED.messages, initialAwaitingReply: false })
    expect(chat.options?.pageContext).toEqual({ page: 'chat', route: '/chat' })
    expect(result.current.messages).toEqual([HELLO, PARTIAL])
    expect(result.current.liveMessages).toEqual([HELLO, PARTIAL])
    expect(result.current.streaming).toBe(true)
    expect(result.current.status).toBe('streaming')
  })

  it("hands each data part to the theme first, then files the turn's Deliverable under the reply being written", () => {
    chat.messages = [HELLO, PARTIAL]
    chat.status = 'streaming'
    const { result, events, onData } = render(STORED)
    act(() => chat.options?.onData(DOCUMENT_PART))
    expect(onData).toHaveBeenCalledWith(DOCUMENT_PART)
    expect(typesOf(events)).toEqual(['theme-part', 'deliverable-added'])
    expect(result.current.deliverablesOf('r1')).toEqual([{ id: 'd-1', title: 'Laptop shortlist', kind: 'document' }])
    act(() => chat.options?.onData(DOCUMENT_PART))
    expect(typesOf(events)).toEqual(['theme-part', 'deliverable-added', 'theme-part'])
    expect(result.current.deliverablesOf('r1')).toHaveLength(1)
  })

  it('keeps the Mission Auto suggests until the next message goes', () => {
    chat.messages = [HELLO, PARTIAL]
    const { result } = render(STORED)
    act(() => chat.options?.onData(SUGGESTION_PART))
    expect(result.current.suggestion).toEqual({ goal: 'Compare laptops weekly', complexity: 'low' })
    act(() => {
      expect(result.current.send('Go on')).toBe(true)
    })
    expect(chat.sendMessage).toHaveBeenCalledWith('Go on')
    expect(result.current.suggestion).toBeNull()
  })
})

describe('the chat controller: Stop keeps the reply being written (P260-FX-30)', () => {
  it('stops the chat and keeps the partial reply once useChat drops it', () => {
    chat.messages = [HELLO, PARTIAL]
    chat.status = 'streaming'
    const { result, rerender } = render(STORED)
    act(() => result.current.stop())
    expect(chat.stop).toHaveBeenCalledTimes(1)
    // useChat's catch on the abort removes the reply it was writing.
    chat.messages = [HELLO]
    chat.status = 'idle'
    rerender()
    expect(result.current.liveMessages).toEqual([HELLO])
    expect(result.current.messages).toEqual([HELLO, PARTIAL])
    expect(result.current.isStopped('r1')).toBe(true)
    expect(result.current.streaming).toBe(false)
  })

  it('keeps nothing when no reply is being written', () => {
    chat.messages = [HELLO]
    const { result } = render(STORED)
    act(() => result.current.stop())
    expect(chat.stop).toHaveBeenCalledTimes(1)
    expect(result.current.messages).toEqual([HELLO])
    expect(result.current.isStopped('r1')).toBe(false)
  })
})

describe('the chat controller: Try again', () => {
  it('asks the last message again and clears the suggestion', () => {
    chat.messages = [HELLO, PARTIAL]
    const { result } = render(STORED)
    act(() => chat.options?.onData(SUGGESTION_PART))
    act(() => {
      expect(result.current.retry()).toBe(true)
    })
    expect(chat.reload).toHaveBeenCalledTimes(1)
    expect(result.current.suggestion).toBeNull()
  })

  it('refuses Try again and Send while a turn runs', () => {
    chat.messages = [HELLO, PARTIAL]
    chat.isLoading = true
    chat.status = 'streaming'
    const { result } = render(STORED)
    act(() => chat.options?.onData(SUGGESTION_PART))
    act(() => {
      expect(result.current.retry()).toBe(false)
      expect(result.current.send('Another')).toBe(false)
    })
    expect(chat.reload).not.toHaveBeenCalled()
    expect(chat.sendMessage).not.toHaveBeenCalled()
    expect(result.current.suggestion).not.toBeNull()
  })
})

describe('the chat controller: approval filing', () => {
  it('files a card once under the reply that raised it, and tells the theme it waits', () => {
    chat.messages = [HELLO, PARTIAL]
    chat.status = 'streaming'
    const { result, events } = render(STORED)
    act(() => chat.options?.onData(TOOL_PART))
    act(() => chat.options?.onData(TOOL_PART))
    expect(result.current.approvals.of('r1')).toEqual([
      expect.objectContaining({ key: 'tool-42', replyId: 'r1', status: 'waiting' }),
    ])
    expect(result.current.approvals.waiting).toBe(true)
    expect(typesOf(events).filter((type) => type === 'approval-waiting')).toHaveLength(1)
    expect(cardsOfChat(useChatApprovalsStore.getState().filed, 'c1')).toHaveLength(1)
  })

  it('answers and closes a card, and tells the theme once none waits', () => {
    chat.messages = [HELLO, PARTIAL]
    const { result, events } = render(STORED)
    act(() => chat.options?.onData(TOOL_PART))
    act(() => result.current.approvals.answered('tool-42'))
    expect(result.current.approvals.of('r1')).toEqual([expect.objectContaining({ status: 'answered' })])
    expect(result.current.approvals.waiting).toBe(false)
    expect(typesOf(events)).toContain('approval-settled')
    act(() => result.current.approvals.closed('tool-42'))
    expect(result.current.approvals.of('r1')).toEqual([])
  })

  it("moves a new chat's cards to the name the backend gives it, and tells the screen and the session", () => {
    chat.messages = [HELLO, PARTIAL]
    chat.status = 'streaming'
    const { result, onChatIdAssigned } = render(NEW_CHAT)
    expect(chat.options?.id).toBe('')
    act(() => chat.options?.onData(TOOL_PART))
    expect(cardsOfChat(useChatApprovalsStore.getState().filed, '')).toHaveLength(1)
    act(() => chat.options?.onChatIdUpdate('c9'))
    expect(cardsOfChat(useChatApprovalsStore.getState().filed, '')).toEqual([])
    expect(cardsOfChat(useChatApprovalsStore.getState().filed, 'c9')).toHaveLength(1)
    expect(result.current.approvals.of('r1')).toHaveLength(1)
    expect(onChatIdAssigned).toHaveBeenCalledWith('c9')
    expect(session.chatIdAssigned).toHaveBeenCalledWith('c9')
  })
})

describe('the chat controller: a reply that failed in words', () => {
  it('is told once (turn-failed), and a failure already on screen when the chat opened stays quiet', () => {
    chat.messages = [HELLO, { ...PARTIAL, error: { message: 'Earlier', code: 'old' } }]
    const { events, rerender } = render(STORED)
    expect(typesOf(events)).not.toContain('turn-failed')
    chat.messages = [HELLO, PARTIAL, NEXT_ASK, FAILED]
    rerender()
    rerender()
    expect(events.filter((event) => typeof event !== 'string' && event.type === 'turn-failed')).toEqual([
      { type: 'turn-failed', code: 'timeout' },
    ])
  })
})
