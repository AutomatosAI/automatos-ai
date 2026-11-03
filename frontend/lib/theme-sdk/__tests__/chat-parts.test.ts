/**
 * PRD-268 K17 (SDK 1.4.0) — a chat message's parts reach a theme (HostChatMessage.knownParts,
 * lib/theme-sdk/internal/message-parts.ts), so a theme that owns /chat finds the day's brief by its part, not by
 * matching its prose (the 1.3.0 limit).
 *
 * - The known types come typed: text, brief@1 (the Brief checked as the host's thread checks it), suggestion@1 (its
 *   id, else the message's source.suggestion_id) and reply_chips@1 (the chips the host's ReplyChips would draw).
 * - A part of an unknown type, or one that isn't what its type says, is dropped, never thrown on; brief, suggestion and
 *   chip parts count on Auto's messages only.
 * - Everything handed back is frozen, down to the brief's rows and the chips.
 * - The host's message copies (internal/chat-view.ts hostMessages) carry them, never the raw parts.
 * - A reply's chips are read as the web chat reads them (lib/chat/turn-parts.ts replyChipsOfMessage), so a reply being
 *   streamed has its reply_chips@1 as soon as its turn_parts frame lands, not only once reloaded.
 */
import { describe, expect, it, vi } from 'vitest'

import type { ChatConversationView } from '@/lib/chat-controller/view-model'
import { liveTurnPartsFromToolData, withTurnPartsOnTheReply } from '@/lib/chat/turn-parts'
import type { ChatMessage } from '@/types'

import type { HostBriefPart, HostMessagePart, HostReplyChipsPart } from '../chat'
import { hostMessages, type ControllerReading } from '../internal/chat-view'
import { hostParts } from '../internal/message-parts'

const SUGGESTION_ID = '3c9e1f7a-6b2d-4e8f-a1c3-5d7e9f0b2a4c'
const SOURCE_ID = '7d1c0a52-3e0b-4c55-9a1e-2f6f0c1d9b10'
const PROSE = 'Two things need you today.'
const ROW = { id: 'sugg:1', title: 'Lisbon flights', why: 'Prices dropped', action: { verb: 'start', label: 'Start' } }
const BRIEF = {
  v: 1, date: '2026-10-31', greeting: 'Morning, Sam.', prose: PROSE, spoken: 'Morning, Sam. 1. Lisbon flights.',
  sections: { needs_you: [], suggestions: [ROW], today: [], since: [], unchecked: [] },
  counts: { needs_you: 0, suggestions: 1, today: 0, since: 0 },
  labels: { sections: { suggestions: 'Suggestions' }, more: {}, empty: 'Nothing needs you.' },
  state_hash: 'h1', generated_at: '2026-10-31T07:00:00Z',
}
const CHIPS = { type: 'reply_chips@1', replies: ['Window', 'Aisle'] }

function message(role: 'user' | 'assistant', parts: readonly unknown[], extra: Record<string, unknown> = {}): ChatMessage {
  return { id: `${role}-1`, role, content: '', parts, ...extra } as unknown as ChatMessage
}

const assistant = (parts: readonly unknown[], extra: Record<string, unknown> = {}) => message('assistant', parts, extra)

/** What a theme writes to find the day's brief in a message: its part, by type. */
function briefOf(parts: readonly HostMessagePart[]) {
  return parts.find((part): part is HostBriefPart => part.type === 'brief@1')?.brief ?? null
}

function chipsOf(parts: readonly HostMessagePart[]) {
  return parts.find((part): part is HostReplyChipsPart => part.type === 'reply_chips@1')?.replies ?? null
}

/** What the SDK's copies read off the host's chat controller; `streaming` while the last reply is being written. */
function reading(messages: ChatMessage[], streaming = false): ControllerReading {
  const approvals = { of: () => [], loose: [], answered: vi.fn(), closed: vi.fn(), waiting: false }
  return {
    messages,
    streaming,
    isStopped: () => false,
    deliverablesOf: () => [],
    approvals,
  } as unknown as ControllerReading
}
const VIEW = { key: 'c1', chatId: 'c1', title: null } as unknown as ChatConversationView

describe('SDK 1.4.0 — knownParts: the known types, typed', () => {
  it('hands a theme the text, brief@1, suggestion@1 and reply_chips@1 parts, in the message’s order', () => {
    const parts = hostParts(assistant([
      { type: 'text', text: PROSE },
      { type: 'brief@1', brief: BRIEF },
      { type: 'suggestion@1', suggestion_id: SUGGESTION_ID },
      CHIPS,
    ]))
    expect(parts.map((part) => part.type)).toEqual(['text', 'brief@1', 'suggestion@1', 'reply_chips@1'])
    expect(parts[0]).toEqual({ type: 'text', text: PROSE })
    expect(briefOf(parts)).toMatchObject({ v: 1, date: '2026-10-31', greeting: 'Morning, Sam.', prose: PROSE, state_hash: 'h1' })
    expect(briefOf(parts)?.sections.suggestions).toEqual([expect.objectContaining({ id: 'sugg:1', title: 'Lisbon flights' })])
    expect(parts[2]).toEqual({ type: 'suggestion@1', suggestion_id: SUGGESTION_ID })
    expect(chipsOf(parts)).toEqual(['Window', 'Aisle'])
  })

  it('takes a suggestion part’s id from the message’s source when the part has none, as the host’s card does', () => {
    const parts = hostParts(assistant([{ type: 'suggestion@1' }], { source: { origin: 'suggestion', suggestion_id: SOURCE_ID } }))
    expect(parts).toEqual([{ type: 'suggestion@1', suggestion_id: SOURCE_ID }])
  })

  it('hands over the chips the host would draw: trimmed, at most 24 characters, no repeats, no slash command, three at most', () => {
    const replies = ['  Window ', 'Aisle', 'Aisle', '/mission plan it', 'A reply that is far too long to tap', 'Middle', 'Either']
    expect(chipsOf(hostParts(assistant([{ type: 'reply_chips@1', replies }])))).toEqual(['Window', 'Aisle', 'Middle'])
  })
})

describe('SDK 1.4.0 — knownParts: what isn’t known is dropped, never thrown', () => {
  const malformed = [
    { type: 'weather@1', forecast: 'rain' },
    { type: 'reasoning', reasoning: 'thinking' },
    { type: 'tool-invocation', toolInvocation: {} },
    { type: 'constructor' },
    { type: '__proto__' },
    { type: 'text', text: 7 },
    { type: 'brief@1', brief: { v: 2 } },
    { type: 'brief@1' },
    { type: 'suggestion@1', suggestion_id: '../../admin' },
    { type: 'reply_chips@1', replies: ['Window', 3] },
    { type: 'reply_chips@1', replies: 'Window, Aisle' },
    { type: 'reply_chips@1', replies: ['Only one'] },
    { type: 7 },
    'text',
    null,
    [],
  ]

  it('drops every part of an unknown type and every part that isn’t what its type says, keeping the rest', () => {
    expect(() => hostParts(assistant(malformed))).not.toThrow()
    expect(hostParts(assistant(malformed))).toEqual([])
    expect(hostParts(assistant([...malformed, { type: 'text', text: PROSE }]))).toEqual([{ type: 'text', text: PROSE }])
  })

  it('reads no parts on a message without a list of them', () => {
    expect(hostParts(assistant(undefined as unknown as unknown[]))).toEqual([])
    expect(hostParts(assistant('brief@1' as unknown as unknown[]))).toEqual([])
  })

  it('counts brief, suggestion and chip parts on Auto’s messages only; a person’s message keeps its text', () => {
    const parts = hostParts(message('user', [
      { type: 'text', text: 'Brief me' },
      { type: 'brief@1', brief: BRIEF },
      { type: 'suggestion@1', suggestion_id: SUGGESTION_ID },
      CHIPS,
    ]))
    expect(parts).toEqual([{ type: 'text', text: 'Brief me' }])
  })
})

describe('SDK 1.4.0 — knownParts: frozen copies', () => {
  it('freezes the list, each part, the brief down to its rows, and the chips', () => {
    const raw = [{ type: 'text', text: PROSE }, { type: 'brief@1', brief: BRIEF }, CHIPS]
    const parts = hostParts(assistant(raw))
    const brief = briefOf(parts)
    expect(Object.isFrozen(parts)).toBe(true)
    parts.forEach((part) => expect(Object.isFrozen(part)).toBe(true))
    expect(Object.isFrozen(brief)).toBe(true)
    expect(Object.isFrozen(brief?.sections.suggestions)).toBe(true)
    expect(Object.isFrozen(brief?.sections.suggestions[0])).toBe(true)
    expect(Object.isFrozen(chipsOf(parts))).toBe(true)
    expect(() => (parts as HostMessagePart[]).push({ type: 'text', text: 'more' })).toThrow(TypeError)
    expect(() => (chipsOf(parts) as string[]).push('Middle')).toThrow(TypeError)
  })

  it('copies: the brief a theme holds is never the object the message carries', () => {
    const parts = hostParts(assistant([{ type: 'brief@1', brief: BRIEF }]))
    expect(briefOf(parts)).not.toBe(BRIEF)
    expect(briefOf(parts)?.sections.suggestions[0]).not.toBe(ROW)
    expect(Object.isFrozen(BRIEF)).toBe(false)
  })
})

describe('SDK 1.4.0 — the brief found by its part', () => {
  it('finds the brief by its brief@1 part even when the words are not its prose', () => {
    const parts = hostParts(assistant([{ type: 'text', text: 'Here is your day.' }, { type: 'brief@1', brief: BRIEF }]))
    expect(briefOf(parts)?.greeting).toBe('Morning, Sam.')
  })

  it('finds none in a message whose words are the brief’s prose but that has no brief@1 part', () => {
    expect(briefOf(hostParts(assistant([{ type: 'text', text: PROSE }])))).toBeNull()
  })
})

describe('SDK 1.4.0 — the host’s message copies carry knownParts', () => {
  it('puts each message’s known parts on its copy, frozen, and never its raw parts', () => {
    const raw = [{ type: 'text', text: PROSE }, { type: 'brief@1', brief: BRIEF }, { type: 'weather@1' }]
    const ask = message('user', [{ type: 'text', text: 'Brief me' }])
    const { list } = hostMessages(reading([ask, assistant(raw)]), VIEW)
    expect(list[0].knownParts).toEqual([{ type: 'text', text: 'Brief me' }])
    expect(list[1].knownParts.map((part) => part.type)).toEqual(['text', 'brief@1'])
    expect(briefOf(list[1].knownParts)?.prose).toBe(PROSE)
    expect(Object.isFrozen(list[1].knownParts)).toBe(true)
    expect(list[1].knownParts).not.toBe(raw)
    expect(Object.keys(list[1])).not.toContain('parts')
  })
})

describe('PRD-268 — a live reply’s chips reach knownParts at once, read as the web chat reads them', () => {
  const ASK = 'Window or aisle?'
  const ASKED = { type: 'text', text: ASK }
  const FOLLOW_UP = { type: 'follow_up@1', follow_up_id: 41, at: '2026-10-31T15:10:00+00:00' }
  const OFFERED = { type: 'reply_chips@1', replies: ['Window', 'Aisle'] }

  /** The conversation as useChatWithReceipts holds it mid-turn: the reply's streamed parts, and its turn_parts frame on it. */
  function streamed(replyParts: readonly unknown[], frame: readonly unknown[]): ChatMessage[] {
    const live = liveTurnPartsFromToolData({ type: 'tool-data', data: { turn_parts: frame } })
    if (!live) throw new Error('the frame carries no turn parts')
    return withTurnPartsOnTheReply([message('user', [{ type: 'text', text: 'Book my seat' }]), assistant(replyParts)], live)
  }

  it('shows a streamed reply’s reply_chips@1 in its knownParts before any reload', () => {
    const messages = streamed([ASKED], [CHIPS, FOLLOW_UP])
    expect(messages[1].parts).toEqual([ASKED])
    const { list } = hostMessages(reading(messages, true), VIEW)
    expect(list[1].writing).toBe(true)
    expect(list[1].knownParts).toEqual([ASKED, OFFERED])
    expect(Object.isFrozen(list[1].knownParts)).toBe(true)
    expect(Object.isFrozen(chipsOf(list[1].knownParts))).toBe(true)
    expect(list[0].knownParts).toEqual([{ type: 'text', text: 'Book my seat' }])
  })

  it('leaves a stored message’s chips as they were: the same part, in its place, once, and the live reply matches it', () => {
    const saved = assistant([ASKED, CHIPS, FOLLOW_UP])
    expect(hostParts(saved)).toEqual([ASKED, OFFERED])
    expect(hostParts(assistant([CHIPS, ASKED]))).toEqual([OFFERED, ASKED])
    expect(hostParts(assistant([{ type: 'reply_chips@1', replies: ['Window', 3] }, ASKED, CHIPS]))).toEqual([ASKED, OFFERED])
    expect(hostParts({ ...saved, replyChips: ['Window', 'Aisle'] })).toEqual([ASKED, OFFERED])
    expect(hostParts(streamed([ASKED], [CHIPS, FOLLOW_UP])[1])).toEqual(hostParts(saved))
  })

  it('still drops an unknown part on a live reply, keeping its text and its chips', () => {
    const weather = { type: 'weather@1', forecast: 'rain' }
    const messages = streamed([weather, ASKED, { type: 'tool-invocation', toolInvocation: {} }, FOLLOW_UP], [CHIPS])
    expect(hostParts(messages[1])).toEqual([ASKED, OFFERED])
  })
})
