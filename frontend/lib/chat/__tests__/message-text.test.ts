// @vitest-environment node
/**
 * PRD-260 §4, PRD-261 A-11a (M28, M29), W-A8 — the core's chat messages as every theme's chat reads
 * them (lib/chat/message-text.ts, through the SDK's chat view): the conversation's name (chatTitle:
 * its stored title, else its first message cut at 60 characters, else "New chat") and a failed turn
 * the server saved (savedTurnError: its words without the core's warning sign, else SAVED_ERROR).
 * Builtin Play's caption test held these cases until it left core; a theme only sees the result.
 */
import { describe, expect, it } from 'vitest'

import { chatTitle, NEW_CHAT, SAVED_ERROR, savedTurnError } from '@/lib/chat/message-text'
import type { ChatMessage } from '@/types'

/** The longest name chatTitle keeps whole (TITLE_CHARS in message-text.ts). */
const TITLE_CHARS = 60
const ELLIPSIS = '…'
/** The core's warning sign in front of a saved turn error's words, with and without its emoji form. */
const WARNING = '⚠️'
const WARNING_TEXT_FORM = '⚠'
/** The provenance the backend saves on a failed turn (consumers/chatbot/service.py _persist_turn_error). */
const TURN_ERROR = { origin: 'turn_error', label: 'Error', code: 'model_unavailable' }

function message(fields: Record<string, unknown>): ChatMessage {
  return { id: 'm', role: 'assistant', parts: [], ...fields } as unknown as ChatMessage
}

function said(text: string) {
  return [{ type: 'text', text }]
}

function asked(text: string): ChatMessage {
  return message({ id: 'u', role: 'user', parts: said(text) })
}

function savedFailure(fields: Record<string, unknown> = {}): ChatMessage {
  return message({ source: TURN_ERROR, ...fields })
}

describe('chatTitle: the 60-character cut', () => {
  it('keeps a first message of 60 characters whole', () => {
    const whole = 'x'.repeat(TITLE_CHARS)
    expect(chatTitle([asked(whole)])).toBe(whole)
  })

  it('cuts a longer one to 59 characters and an ellipsis, 60 in all', () => {
    const cut = chatTitle([asked('y'.repeat(TITLE_CHARS + 20))])
    expect(cut).toBe(`${'y'.repeat(TITLE_CHARS - 1)}${ELLIPSIS}`)
    expect(cut).toHaveLength(TITLE_CHARS)
  })

  it('measures the words with their spaces folded, and drops a space left before the ellipsis', () => {
    const folded = `${'a'.repeat(30)}${' '.repeat(10)}${'b'.repeat(29)}`
    expect(chatTitle([asked(folded)])).toBe(`${'a'.repeat(30)} ${'b'.repeat(29)}`)
    const endsOnSpace = `${'c'.repeat(TITLE_CHARS - 2)} ${'d'.repeat(10)}`
    expect(chatTitle([asked(endsOnSpace)])).toBe(`${'c'.repeat(TITLE_CHARS - 2)}${ELLIPSIS}`)
  })

  it('cuts a first message stored as content the same way', () => {
    const stored = message({ role: 'user', content: 'z'.repeat(80) })
    expect(chatTitle([stored])).toBe(`${'z'.repeat(TITLE_CHARS - 1)}${ELLIPSIS}`)
  })
})

describe('chatTitle: the "New chat" fallback', () => {
  it('is the words "New chat"', () => {
    expect(NEW_CHAT).toBe('New chat')
  })

  it('names a conversation with no messages, or no stored title, "New chat"', () => {
    expect(chatTitle([])).toBe(NEW_CHAT)
    expect(chatTitle([], null)).toBe(NEW_CHAT)
    expect(chatTitle([], '   ')).toBe(NEW_CHAT)
  })

  it('names one whose only messages are replies, or whose first message has no words, "New chat"', () => {
    expect(chatTitle([message({ parts: said('An answer before any question') })])).toBe(NEW_CHAT)
    expect(chatTitle([asked('  \n\t ')])).toBe(NEW_CHAT)
  })

  it('prefers the stored title, trimmed, over the first message', () => {
    expect(chatTitle([asked('What   laptop?')], '  Laptop for uni ')).toBe('Laptop for uni')
    expect(chatTitle([asked('What   laptop?')])).toBe('What laptop?')
  })
})

describe('savedTurnError: a failed turn the server saved', () => {
  it("reads SAVED_ERROR, 'This answer failed.', for a saved failure without words of its own", () => {
    expect(SAVED_ERROR).toBe('This answer failed.')
    expect(savedTurnError(savedFailure())).toBe(SAVED_ERROR)
    expect(savedTurnError(savedFailure({ parts: said('   ') }))).toBe(SAVED_ERROR)
  })

  it("reads SAVED_ERROR when its only words are the core's warning sign", () => {
    expect(savedTurnError(savedFailure({ parts: said(WARNING) }))).toBe(SAVED_ERROR)
    expect(savedTurnError(savedFailure({ parts: said(` ${WARNING_TEXT_FORM}  `) }))).toBe(SAVED_ERROR)
  })

  it('reads its own words, without the warning sign, when it has some', () => {
    expect(savedTurnError(savedFailure({ parts: said(`${WARNING} The model is unavailable`) }))).toBe('The model is unavailable')
    expect(savedTurnError(savedFailure({ content: `${WARNING_TEXT_FORM} Out of credit.` }))).toBe('Out of credit.')
  })

  it('is none for any message the server did not save as a failed turn', () => {
    expect(savedTurnError(message({ parts: said(`${WARNING} careful`) }))).toBeNull()
    expect(savedTurnError(message({ parts: [], source: { origin: 'watcher', label: 'Auto · background' } }))).toBeNull()
    expect(savedTurnError(message({ parts: [], source: null }))).toBeNull()
  })
})
