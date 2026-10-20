/**
 * PRD-261 A-11b — SDK area `chat` (lib/theme-sdk/chat.ts) over the host's chat controller (A-11a). The
 * controller's own behaviour (stream, Stop keeping the partial reply, Try again, approval filing) is proved by
 * lib/chat-controller/__tests__; here it is a stand-in, and these tests hold the SDK to what it adds: the two
 * calls of useChatController (the conversation on screen, then that conversation's chat), the frozen,
 * theme-neutral copies (no raw core message, no store), a forged or switched call refused, and the list, New chat
 * and receipt names. The browser proof is e2e/browser/specs/sdk-chat.spec.ts (fixtures/hello-a11b).
 */
import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage, Receipt } from '@/types'

type Options = Record<string, unknown>

const host = vi.hoisted(() => ({
  onScreen: { view: null as unknown, adopt: (_chatId: string) => undefined, draft: null as unknown, takeDraft: () => undefined },
  conversationOptions: [] as Record<string, unknown>[],
  controllerOptions: [] as Record<string, unknown>[],
  core: null as unknown,
  freshDrafts: 0,
  planning: null as string | null,
  activeChatId: null as string | null,
  list: { status: 'loading' } as unknown,
  listLimits: [] as unknown[],
  recent: [] as unknown[],
  start: { startChat: () => undefined, ready: true },
}))

vi.mock('@/lib/chat-controller/conversation', () => ({
  useChatConversation: (options: Options) => {
    host.conversationOptions.push(options)
    return host.onScreen
  },
}))
vi.mock('@/lib/chat-controller/controller', () => ({
  useChatController: (options: Options) => {
    host.controllerOptions.push(options)
    return host.core
  },
}))
vi.mock('@/stores/play-chat-store', () => ({
  usePlayChatStore: (select: (state: { freshDrafts: number }) => unknown) => select({ freshDrafts: host.freshDrafts }),
}))
vi.mock('@/stores/mission-store', () => ({
  useMissionStore: (select: (state: { activePlanningMissionId: string | null }) => unknown) =>
    select({ activePlanningMissionId: host.planning }),
}))
vi.mock('@/stores/chat-session-store', () => ({
  useChatSessionStore: (select: (state: { session: { activeChatId: string | null } }) => unknown) =>
    select({ session: { activeChatId: host.activeChatId } }),
}))
vi.mock('@/hooks/use-conversation-list', () => ({
  useConversationList: (limit: unknown) => {
    host.listLimits.push(limit)
    return host.list
  },
}))
vi.mock('@/hooks/use-recent-chats', () => ({ useRecentChats: () => host.recent }))
vi.mock('@/hooks/use-start-chat', () => ({ useStartChat: () => host.start }))

import {
  movedSomethingOnTheBoard,
  receiptLine,
  receiptLink,
  useChatController,
  useConversationList,
  useRecentChats,
  useStartChat,
  type HostChatConversation,
} from '../chat'

const message = (id: string, role: 'user' | 'assistant', text: string, extra: Record<string, unknown> = {}) =>
  ({ id, role, content: text, parts: [{ type: 'text', text }], ...extra }) as unknown as ChatMessage

const MOVED: Receipt = { action: 'move_task', kind: 'write', status: 'done', subject: '#0422', effect: 'moved to Done', link: '/board?task=422', reason: null }
const REFUSED: Receipt = { action: 'send_email', kind: 'write', status: 'refused', subject: 'Letter to Maya', effect: '', link: '//evil.example', reason: 'no connection' }
const TICKET = { id: 7, number: '#0007', title: 'Order laptops', status: 'in_progress', assigned_agent: 'Scout' }
const ASK = message('u1', 'user', 'Find me a laptop')
const REPLY = message('r1', 'assistant', 'Here are three', {
  receipts: [MOVED],
  receiptsAbove: ['Work said done was not done.'],
  taskCards: [TICKET],
  progress: ['Searching shops'],
})
const FAILED = message('r2', 'assistant', '', { error: { message: 'The model took too long to answer.', code: 'timeout' } })
const STORED_VIEW = { key: 'c1', chatId: 'c1', title: null, messages: [ASK], awaitingReply: false }
const DRAFT_LINK = { param: 'draft', read: () => 'Ask about this' }
const DOCUMENT = { id: 'd-1', title: 'Laptop shortlist', kind: 'document' }

function filedCard(key: string, replyId: string) {
  return { kind: 'tool', key, replyId, status: 'waiting', filedAt: new Date(0), data: { grant_id: 42, action: 'platform_delete_document' } }
}

function coreView(overrides: Record<string, unknown> = {}) {
  const cards = [filedCard('grant-42', 'r1')]
  const approvals = {
    of: (replyId: string) => cards.filter((card) => card.replyId === replyId),
    loose: [filedCard('grant-9', 'gone')],
    answered: vi.fn(),
    closed: vi.fn(),
    waiting: true,
  }
  return {
    messages: [ASK, REPLY],
    liveMessages: [ASK, REPLY],
    status: 'idle',
    errorCode: null,
    streaming: false,
    isStopped: (id: string) => id === 'r1',
    suggestion: { goal: 'Compare laptops weekly', complexity: 'low' },
    deliverablesOf: (id: string) => (id === 'r1' ? [DOCUMENT] : []),
    approvals,
    send: vi.fn(() => true),
    retry: vi.fn(() => true),
    stop: vi.fn(),
    ...overrides,
  }
}

function lastOf<T>(list: readonly T[]): T | undefined {
  return list[list.length - 1]
}

/** The conversation the first call hands out for `view`. */
function handedOut(view: unknown = STORED_VIEW): HostChatConversation {
  host.onScreen = { ...host.onScreen, view }
  const { result } = renderHook(() => useChatController())
  const { conversation } = result.current
  if (conversation === null) throw new Error('no conversation handed out')
  return conversation
}

function openChat(core: unknown = coreView(), options: Options = {}) {
  const conversation = handedOut()
  host.core = core
  return { conversation, ...renderHook(() => useChatController(conversation, options)) }
}

afterEach(() => {
  host.onScreen = { view: null, adopt: () => undefined, draft: null, takeDraft: () => undefined }
  host.conversationOptions.length = 0
  host.controllerOptions.length = 0
  host.core = null
  host.freshDrafts = 0
  host.planning = null
  host.activeChatId = null
  host.list = { status: 'loading' }
  host.listLimits.length = 0
  host.recent = []
})

describe('PRD-261 A-11b — useChatController(): the conversation on screen', () => {
  it('is null while the conversation opens, with the draft and takeDraft passed through', () => {
    const takeDraft = vi.fn()
    host.onScreen = { ...host.onScreen, draft: { text: 'Ask about this', serial: 1 }, takeDraft }
    const { result } = renderHook(() => useChatController())
    expect(result.current).toEqual({ conversation: null, draft: { text: 'Ask about this', serial: 1 }, takeDraft })
    expect(Object.isFrozen(result.current)).toBe(true)
  })

  it("follows New chat's count and the theme's own draft link", () => {
    host.freshDrafts = 3
    renderHook(() => useChatController({ draftLink: DRAFT_LINK }))
    expect(lastOf(host.conversationOptions)).toEqual({ freshDrafts: 3, draftLink: DRAFT_LINK })
  })

  it('hands out the key, chat id and title only, frozen, the same object while the view is the same', () => {
    host.onScreen = { ...host.onScreen, view: STORED_VIEW }
    const { result, rerender } = renderHook(() => useChatController())
    const first = result.current.conversation
    expect(first).toEqual({ key: 'c1', chatId: 'c1', title: null })
    expect(Object.keys(first ?? {})).toEqual(['key', 'chatId', 'title'])
    expect(JSON.stringify(first)).not.toContain('Find me a laptop')
    expect(Object.isFrozen(first)).toBe(true)
    rerender()
    expect(result.current.conversation).toBe(first)
  })
})

describe("PRD-261 A-11b — useChatController(conversation): one conversation's chat", () => {
  it("opens the controller on the view and adopt the first call handed out, with the theme's onData and onEvent", () => {
    const adopt = vi.fn()
    host.onScreen = { ...host.onScreen, adopt }
    const onData = vi.fn()
    const onEvent = vi.fn()
    openChat(coreView(), { onData, onEvent })
    expect(lastOf(host.controllerOptions)).toEqual({ view: STORED_VIEW, onChatIdAssigned: adopt, onData, onEvent })
  })

  it('copies each message for a theme: words, receipts, Deliverables, cards, tickets, stopped', () => {
    const { result } = openChat()
    const [ask, reply] = result.current.messages
    expect(ask).toMatchObject({ id: 'u1', role: 'user', text: 'Find me a laptop', stopped: false, receipts: null, approvals: [] })
    expect(reply).toMatchObject({
      id: 'r1',
      role: 'assistant',
      text: 'Here are three',
      activity: ['Searching shops'],
      error: null,
      writing: false,
      stopped: true,
      receipts: [MOVED],
      above: ['Work said done was not done.'],
      deliverables: [DOCUMENT],
      taskCards: [TICKET],
    })
    expect(Object.keys(reply)).not.toContain('parts')
    expect(Object.isFrozen(reply)).toBe(true)
    expect(Object.isFrozen(result.current.messages)).toBe(true)
  })

  it('binds each approval card to its outcomes, in the reply and after the conversation', () => {
    const core = coreView()
    const { result } = openChat(core)
    const [card] = result.current.messages[1].approvals
    expect(card).toMatchObject({ key: 'grant-42', kind: 'tool', replyId: 'r1', status: 'waiting', chatId: 'c1' })
    expect(card.data).toEqual({ grant_id: 42, action: 'platform_delete_document' })
    card.answered()
    card.closed()
    expect(core.approvals.answered).toHaveBeenCalledWith('grant-42')
    expect(core.approvals.closed).toHaveBeenCalledWith('grant-42')
    expect(result.current.looseApprovals.map((loose) => loose.key)).toEqual(['grant-9'])
    expect(result.current.approvalWaiting).toBe(true)
  })

  it('marks the reply being written, and a reply that failed in words', () => {
    const { result } = openChat(coreView({ messages: [ASK, REPLY, FAILED], streaming: true, status: 'streaming' }))
    expect(result.current.messages.map((shown) => shown.writing)).toEqual([false, false, true])
    expect(result.current.messages[2].error).toEqual({ message: 'The model took too long to answer.', code: 'timeout' })
    expect(result.current).toMatchObject({ status: 'streaming', streaming: true })
  })

  it("carries the turn's suggestion, the Mission being planned, the title and a spent trial", () => {
    host.planning = 'm-1'
    const { result } = openChat(coreView({ errorCode: 'trial_exhausted', status: 'error' }))
    expect(result.current).toMatchObject({
      conversation: { key: 'c1', chatId: 'c1', title: null },
      title: 'Find me a laptop',
      suggestion: { goal: 'Compare laptops weekly', complexity: 'low' },
      planningMissionId: 'm-1',
      errorCode: 'trial_exhausted',
      trialExhausted: true,
    })
  })

  it("sends, tries again and stops through the controller's own actions", () => {
    const core = coreView()
    const { result } = openChat(core)
    expect(result.current.send('And a cheaper one?')).toBe(true)
    expect(result.current.retry()).toBe(true)
    result.current.stop()
    expect(core.send).toHaveBeenCalledWith('And a cheaper one?')
    expect(core.retry).toHaveBeenCalledTimes(1)
    expect(core.stop).toHaveBeenCalledTimes(1)
    expect(Object.isFrozen(result.current)).toBe(true)
  })

  it('refuses a conversation it did not hand out', () => {
    const forged = { key: 'c1', chatId: 'c1', title: null }
    expect(() => renderHook(() => useChatController(forged))).toThrow('opens only a conversation that useChatController() handed out')
    expect(host.controllerOptions).toEqual([])
  })

  it('refuses a component that switches between the two calls', () => {
    const conversation = handedOut()
    host.core = coreView()
    const { rerender } = renderHook((props: { open: boolean }) => (props.open ? useChatController(conversation) : useChatController()), {
      initialProps: { open: false },
    })
    expect(() => rerender({ open: true })).toThrow('for the life of a component')
  })
})

describe('PRD-261 A-11b — the lists and New chat', () => {
  it('copies the conversation list, its failure and its loading line', () => {
    const chat = { id: 'c1', userId: 'u-1', title: 'Laptops', createdAt: '2026-10-01', updatedAt: '2026-10-02', visibility: 'private', lastMessagePreview: 'Here are three' }
    host.list = { status: 'loaded', chats: [chat] }
    const loaded = renderHook(() => useConversationList(20))
    expect(loaded.result.current).toEqual({ status: 'loaded', chats: [{ id: 'c1', title: 'Laptops', updatedAt: '2026-10-02', preview: 'Here are three' }] })
    expect(JSON.stringify(loaded.result.current)).not.toContain('u-1')
    expect(host.listLimits).toEqual([20])

    const retry = vi.fn()
    host.list = { status: 'failed', retry }
    expect(renderHook(() => useConversationList()).result.current).toEqual({ status: 'failed', retry })
    host.list = { status: 'loading' }
    expect(renderHook(() => useConversationList()).result.current).toEqual({ status: 'loading' })
  })

  it('gives the recent conversations with the one on screen', () => {
    host.recent = [{ id: 'c2', userId: 'u-1', title: 'Shoes', createdAt: 'x', updatedAt: 'y', visibility: 'private' }]
    host.activeChatId = 'c2'
    const { result } = renderHook(() => useRecentChats())
    expect(result.current).toEqual({ chats: [{ id: 'c2', title: 'Shoes', updatedAt: 'y', preview: null }], activeChatId: 'c2' })
    expect(Object.isFrozen(result.current.chats)).toBe(true)
  })

  it("is New chat: the host's startChat and whether it can be pressed yet", () => {
    const { result } = renderHook(() => useStartChat())
    expect(result.current).toEqual({ startChat: host.start.startChat, ready: true })
  })
})

describe('PRD-261 A-11b — the receipt lines', () => {
  it("says each receipt in the host's words, the same in every theme", () => {
    expect(receiptLine(MOVED)).toBe('#0422: moved to Done')
    expect(receiptLine(REFUSED)).toBe('Letter to Maya: tried, refused: no connection')
  })

  it('links only a done receipt, and only to a page in this app', () => {
    expect(receiptLink(MOVED)).toBe('/board?task=422')
    expect(receiptLink(REFUSED)).toBeNull()
    expect(receiptLink({ ...MOVED, link: '//evil.example' })).toBeNull()
    expect(receiptLink({ ...MOVED, link: 'https://evil.example' })).toBeNull()
  })

  it('tells whether a done write moved a card on the board', () => {
    expect(movedSomethingOnTheBoard([MOVED])).toBe(true)
    expect(movedSomethingOnTheBoard([REFUSED])).toBe(false)
    expect(movedSomethingOnTheBoard(Object.freeze([]))).toBe(false)
  })
})
