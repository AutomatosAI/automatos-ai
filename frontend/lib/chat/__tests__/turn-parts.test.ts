/**
 * PRD-268 §6.3, §6.6, K12, K13 (US-W03a) — the turn's parts, read off the wire: the live `turn_parts`
 * tool-data frame (consumers/chatbot/turn_parts.py turn_parts_frames) and a saved message's `reply_chips@1`
 * and `follow_up@1` parts. A part that is not what it says is left out and never thrown on; the rest are kept.
 * useChatWithReceipts puts the live ones on the reply being streamed, as it does the receipts, then runs the
 * page's own data callback. The voice and the theme chats read text parts only, so they never say the chips.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, renderHook } from '@testing-library/react'
import type { ChatMessage } from '@/types'

const chatState: { options?: { onData?: (part: unknown) => void }; messages: ChatMessage[] } = { messages: [] }
vi.mock('@/lib/chat/hooks', () => ({
  useChat: (options: { onData?: (part: unknown) => void }) => {
    chatState.options = options
    return {
      messages: chatState.messages,
      setMessages: (next: (prev: ChatMessage[]) => ChatMessage[]) => {
        chatState.messages = next(chatState.messages)
      },
    }
  },
}))

import { messageText } from '@/lib/chat/message-text'
import {
  LIVE_KEY,
  followUpsOfMessage,
  liveTurnPartsFromToolData,
  replyChipsOfMessage,
  turnPartsOf,
  withTurnPartsOnTheReply,
} from '@/lib/chat/turn-parts'
import { useChatWithReceipts } from '@/lib/chat/use-chat-with-receipts'

afterEach(cleanup)

const AT = '2026-10-31T15:10:00+00:00'
const CHIPS_PART = { type: 'reply_chips@1', replies: ['Window', 'Aisle', 'No preference'] }
const FOLLOW_UP = { type: 'follow_up@1', follow_up_id: 41, at: AT }
const toolData = (data: unknown) => ({ type: 'tool-data', data })
const message = (role: string, parts: unknown[], extra: Record<string, unknown> = {}) =>
  ({ id: `${role}-1`, role, parts, ...extra }) as unknown as ChatMessage

describe('the live turn_parts frame', () => {
  it('is read from a forwarded tool-data part: the reply chips and each follow-up', () => {
    expect(LIVE_KEY).toBe('turn_parts')
    expect(liveTurnPartsFromToolData(toolData({ turn_parts: [CHIPS_PART, FOLLOW_UP] }))).toEqual({
      replyChips: ['Window', 'Aisle', 'No preference'],
      followUps: [{ followUpId: 41, at: AT }],
    })
    expect(liveTurnPartsFromToolData(toolData({ turn_parts: [FOLLOW_UP] }))).toEqual({ followUps: [{ followUpId: 41, at: AT }] })
  })

  it('is nothing for any other part, another key, or a frame with nothing that can be read', () => {
    for (const part of [
      null, undefined, 'turn_parts', [], { type: 'tool-end', data: { turn_parts: [CHIPS_PART] } },
      toolData({ receipts: { receipts: [] } }), toolData(null), toolData({ turn_parts: CHIPS_PART }),
      toolData({ turn_parts: [] }), toolData({ turn_parts: [{ type: 'reply_chips@1', replies: 'Window' }, null] }),
    ]) {
      expect(liveTurnPartsFromToolData(part)).toBeUndefined()
    }
  })

  it('goes on the last assistant message, leaving the others and the list it was given as they were', () => {
    const before = [message('assistant', []), message('user', []), { ...message('assistant', []), id: 'a2' }, message('user', [])]
    const after = withTurnPartsOnTheReply(before, { replyChips: ['Yes', 'No'] })
    expect(after[2]).toMatchObject({ id: 'a2', replyChips: ['Yes', 'No'] })
    expect(after[0]).toBe(before[0])
    expect(before[2]).not.toHaveProperty('replyChips')
    const users = [message('user', [])]
    expect(withTurnPartsOnTheReply(users, { replyChips: ['Yes', 'No'] })).toBe(users)
  })

  it("reaches the reply being streamed through useChatWithReceipts, then the page's own data callback", () => {
    chatState.messages = [message('user', []), message('assistant', [])]
    const pageOnData = vi.fn()
    renderHook(() => useChatWithReceipts({ id: 'c1', onData: pageOnData }))
    const part = toolData({ turn_parts: [CHIPS_PART, FOLLOW_UP] })
    chatState.options?.onData?.(part)
    expect(chatState.messages[1]).toMatchObject({
      replyChips: ['Window', 'Aisle', 'No preference'],
      followUps: [{ followUpId: 41, at: AT }],
    })
    expect(chatState.messages[0]).not.toHaveProperty('replyChips')
    expect(pageOnData).toHaveBeenCalledWith(part)
  })
})

describe("a saved message's parts", () => {
  it('reads the reply chips and the follow-ups after the text part', () => {
    const saved = message('assistant', [{ type: 'text', text: 'Window or aisle?' }, CHIPS_PART, FOLLOW_UP])
    expect(replyChipsOfMessage(saved)).toEqual(['Window', 'Aisle', 'No preference'])
    expect(followUpsOfMessage(saved)).toEqual([{ followUpId: 41, at: AT }])
  })

  it('prefers the live chips and follow-ups, and reads none on a user message or a message without them', () => {
    const live = message('assistant', [CHIPS_PART], { replyChips: ['Yes', 'No'], followUps: [{ followUpId: 7, at: AT }] })
    expect(replyChipsOfMessage(live)).toEqual(['Yes', 'No'])
    expect(followUpsOfMessage(live)).toEqual([{ followUpId: 7, at: AT }])
    expect(replyChipsOfMessage(message('user', [CHIPS_PART]))).toBeUndefined()
    expect(followUpsOfMessage(message('user', [FOLLOW_UP]))).toEqual([])
    expect(replyChipsOfMessage(message('assistant', [{ type: 'text', text: 'Done.' }]))).toBeUndefined()
    expect(followUpsOfMessage(message('assistant', undefined as unknown as unknown[]))).toEqual([])
  })

  it('names each follow-up once, in order, and keeps the first chips part with a reply', () => {
    const second = { ...FOLLOW_UP, follow_up_id: 42 }
    const parts = [FOLLOW_UP, CHIPS_PART, second, FOLLOW_UP, { type: 'reply_chips@1', replies: ['Later', 'Never'] }]
    expect(turnPartsOf(parts)).toEqual({
      replyChips: ['Window', 'Aisle', 'No preference'],
      followUps: [{ followUpId: 41, at: AT }, { followUpId: 42, at: AT }],
    })
  })

  it('is never said by the voice: a message reads as its text parts only', () => {
    const saved = message('assistant', [{ type: 'text', text: 'Window or aisle?' }, CHIPS_PART, FOLLOW_UP])
    expect(messageText(saved)).toBe('Window or aisle?')
  })
})

describe('malformed input', () => {
  it('leaves out a chips part that is not a list of text, and never throws', () => {
    for (const replies of [undefined, null, 'Window', { 0: 'Window' }, ['Window', 2], ['Window', null]]) {
      expect(() => turnPartsOf([{ type: 'reply_chips@1', replies }])).not.toThrow()
      expect(turnPartsOf([{ type: 'reply_chips@1', replies }])).toEqual({})
    }
  })

  it('keeps chips as trimmed text, without empty ones or a slash command', () => {
    const parts = [{ type: 'reply_chips@1', replies: ['  Window ', '', '   ', '/mission book it', 'Aisle'] }]
    expect(turnPartsOf(parts).replyChips).toEqual(['Window', 'Aisle'])
    expect(turnPartsOf([{ type: 'reply_chips@1', replies: ['/mission x', ' '] }])).toEqual({})
    expect(turnPartsOf([{ type: 'reply_chips@1', replies: ['/mission x'] }, { type: 'reply_chips@1', replies: ['Yes', 'No'] }]).replyChips)
      .toEqual(['Yes', 'No'])
  })

  it('leaves out a follow-up without a row id or a time it can read, and keeps the good ones', () => {
    const bad = [
      { type: 'follow_up@1', follow_up_id: '41', at: AT }, { type: 'follow_up@1', follow_up_id: 0, at: AT },
      { type: 'follow_up@1', follow_up_id: -3, at: AT }, { type: 'follow_up@1', follow_up_id: 4.5, at: AT },
      { type: 'follow_up@1', follow_up_id: true, at: AT }, { type: 'follow_up@1', follow_up_id: 41 },
      { type: 'follow_up@1', follow_up_id: 41, at: 'at ten past three' }, { type: 'follow_up@1', follow_up_id: 41, at: 1730387400 },
    ]
    expect(turnPartsOf(bad)).toEqual({})
    expect(turnPartsOf([...bad, FOLLOW_UP]).followUps).toEqual([{ followUpId: 41, at: AT }])
  })

  it('passes over parts that are not objects, and lists that are not lists', () => {
    for (const parts of [null, undefined, 'reply_chips@1', 41, { type: 'reply_chips@1' }]) {
      expect(turnPartsOf(parts)).toEqual({})
    }
    expect(turnPartsOf([null, 'x', 3, [CHIPS_PART], { type: 'weather@1' }, CHIPS_PART]).replyChips).toEqual(['Window', 'Aisle', 'No preference'])
  })
})
