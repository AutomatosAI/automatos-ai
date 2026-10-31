// Theme voice — a voice session's words in the conversation's messages (lib/chat/live-voice-turns.ts), through the
// one stream hook Studio's chat and the theme chat controller share (useChatWithReceipts): the in-flight
// voice-live-user and voice-live-assistant entries end the conversation they belong to, never another, never
// useChat's own state, and they go when chat_changed brings the saved rows. useChat is a stand-in here.
import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/types'

const chat = vi.hoisted(() => ({ messages: [] as unknown[], setMessages: vi.fn() }))
const session = vi.hoisted(() => ({ chatIdAssigned: vi.fn() }))

vi.mock('@/lib/chat/hooks', () => ({
  useChat: () => ({
    messages: chat.messages,
    setMessages: chat.setMessages,
    sendMessage: vi.fn(),
    stop: vi.fn(),
    status: 'idle',
    isLoading: false,
    awaitingReply: false,
    usage: undefined,
    errorCode: null,
  }),
}))
vi.mock('@/lib/page-context', () => ({ usePageContext: () => ({ page: 'chat', route: '/chat' }) }))
vi.mock('@/hooks/use-refresh-history-on-turn-end', () => ({ useRefreshHistoryOnTurnEnd: vi.fn() }))
vi.mock('@/stores/chat-session-store', () => ({ useChatSessionStore: { getState: () => session } }))

import { useChatController } from '@/lib/chat-controller/controller'
import type { ChatConversationView } from '@/lib/chat-controller/view-model'
import {
  VOICE_LIVE_ASSISTANT_ID,
  VOICE_LIVE_USER_ID,
  withLiveVoiceTurn,
} from '@/lib/chat/live-voice-turns'
import { useChatWithReceipts } from '@/lib/chat/use-chat-with-receipts'
import { CHAT_CHANGED_EVENT, useLiveTurnStore, watchSavedVoiceTurns } from '@/lib/voice/live-turns'

const message = (id: string, role: 'user' | 'assistant', text: string) =>
  ({ id, role, content: text, parts: [{ type: 'text', text }] }) as unknown as ChatMessage
const HELLO = message('u1', 'user', 'Morning')
const REPLY = message('r1', 'assistant', 'Morning! What first?')
const STORED: ChatConversationView = { key: 'c1', chatId: 'c1', title: null, messages: [HELLO, REPLY], awaitingReply: false }

const idsOf = (messages: readonly ChatMessage[]) => messages.map((entry) => entry.id)
const speak = (userText: string, agentText = '', chatId = 'c1') =>
  act(() => useLiveTurnStore.getState().publish({ chatId, userText, agentText }))
const saved = (chatId: string) =>
  act(() => {
    window.dispatchEvent(new CustomEvent(CHAT_CHANGED_EVENT, { detail: { chat_id: chatId } }))
  })

let unwatch: () => void = () => undefined

beforeEach(() => {
  chat.messages = [HELLO, REPLY]
  chat.setMessages.mockReset()
  unwatch = watchSavedVoiceTurns(window)
})

afterEach(() => {
  unwatch()
  useLiveTurnStore.getState().clear()
})

describe('withLiveVoiceTurn', () => {
  it('ends the messages with the words, as the user and the assistant, each only when there are words', () => {
    const shown = withLiveVoiceTurn([HELLO], { userText: 'what is on today', agentText: 'Two meetings' })
    expect(idsOf(shown)).toEqual(['u1', VOICE_LIVE_USER_ID, VOICE_LIVE_ASSISTANT_ID])
    expect(shown[1]).toMatchObject({ role: 'user', content: 'what is on today', parts: [{ type: 'text', text: 'what is on today' }] })
    expect(shown[2]).toMatchObject({ role: 'assistant', content: 'Two meetings' })
    expect(idsOf(withLiveVoiceTurn([HELLO], { userText: 'hi', agentText: '' }))).toEqual(['u1', VOICE_LIVE_USER_ID])
  })

  it('is the messages themselves when there are no words', () => {
    const messages = [HELLO]
    expect(withLiveVoiceTurn(messages, null)).toBe(messages)
  })
})

describe("Studio's stream hook (useChatWithReceipts)", () => {
  it("shows the words in their own conversation, and keeps useChat's state free of them", () => {
    const { result } = renderHook(() => useChatWithReceipts({ id: 'c1' }))
    expect(result.current.messages).toBe(chat.messages)
    speak('what is on today', 'Two meetings')
    expect(idsOf(result.current.messages)).toEqual(['u1', 'r1', VOICE_LIVE_USER_ID, VOICE_LIVE_ASSISTANT_ID])
    expect(result.current.messages[3]).toMatchObject({ content: 'Two meetings' })
    expect(chat.setMessages).not.toHaveBeenCalled()
  })

  it("shows nothing in another conversation, or in a new chat the session hasn't named", () => {
    speak('hello', 'Hi', 'c1')
    expect(idsOf(renderHook(() => useChatWithReceipts({ id: 'c2' })).result.current.messages)).toEqual(['u1', 'r1'])
    expect(idsOf(renderHook(() => useChatWithReceipts({ id: '' })).result.current.messages)).toEqual(['u1', 'r1'])
  })

  it('drops the words when chat_changed brings the saved rows, and Auto catching up keeps them dropped', () => {
    const { result } = renderHook(() => useChatWithReceipts({ id: 'c1' }))
    speak('what is on today', 'Two meet')
    saved('c1')
    expect(idsOf(result.current.messages)).toEqual(['u1', 'r1'])
    speak('what is on today', 'Two meetings, then a call.')
    expect(idsOf(result.current.messages)).toEqual(['u1', 'r1'])
    speak('and tomorrow')
    expect(idsOf(result.current.messages)).toEqual(['u1', 'r1', VOICE_LIVE_USER_ID])
  })
})

describe('the theme chat controller', () => {
  it("has the words at the end of the conversation's messages, and drops them on chat_changed", () => {
    const { result } = renderHook(() => useChatController({ view: STORED }))
    speak('what is on today', 'Two meetings')
    expect(idsOf(result.current.messages)).toEqual(['u1', 'r1', VOICE_LIVE_USER_ID, VOICE_LIVE_ASSISTANT_ID])
    expect(idsOf(result.current.liveMessages)).toEqual(['u1', 'r1', VOICE_LIVE_USER_ID, VOICE_LIVE_ASSISTANT_ID])
    saved('c1')
    expect(idsOf(result.current.messages)).toEqual(['u1', 'r1'])
  })
})
