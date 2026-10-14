/**
 * PRD-260 §4 — the words around Play's chat, ported from Automatos Play's
 * tests/chat (caption and labels): what Auto's caption says, a tool call's
 * detail, and the core's ChatMessage read for the chat (text, title, the newest
 * reply, its activity and its running tool call).
 */
import { describe, expect, it } from 'vitest'

import { chatCaption, NAPPING, ON_THE_SHELF, WAKE_HINT } from '@/components/play/chat/caption'
import { clip, toolDetail } from '@/components/play/chat/labels'
import {
  activityLines,
  chatTitle,
  latestReply,
  messageText,
  NEW_CHAT,
  runningToolInput,
} from '@/components/play/chat/message-text'
import { captionFor } from '@/lib/mascot/media'
import type { ChatMessage } from '@/types'

function message(fields: Record<string, unknown>): ChatMessage {
  return { id: 'm', role: 'assistant', parts: [], ...fields } as unknown as ChatMessage
}

describe('chatCaption', () => {
  it("says the state's own words by default", () => {
    expect(chatCaption('thinking')).toEqual({ caption: captionFor('thinking') })
  })

  it('names the running query beside a busy state', () => {
    expect(chatCaption('searching', { runningInput: { query: '  best   budget laptops ' } })).toEqual({
      caption: captionFor('searching'),
      detail: 'best budget laptops',
    })
    expect(chatCaption('reading', { runningInput: {} })).toEqual({ caption: captionFor('reading') })
  })

  it('says a nap is a nap, and not when Automatos is down', () => {
    expect(chatCaption('sleeping')).toEqual({ caption: NAPPING, detail: WAKE_HINT })
    expect(chatCaption('sleeping', { backendDown: true })).toEqual({ caption: captionFor('sleeping') })
  })

  it('celebrates the Deliverable by its kind', () => {
    expect(chatCaption('celebrate', { deliverableKind: 'report' })).toEqual({
      caption: 'Your report is ready',
      detail: ON_THE_SHELF,
    })
    expect(chatCaption('celebrate')).toEqual({ caption: captionFor('celebrate') })
  })
})

describe('toolDetail', () => {
  it('reads the first text field in order, clipped to one line', () => {
    expect(toolDetail({ url: 'https://example.com', query: 'laptops' })).toBe('laptops')
    expect(toolDetail({ title: 'x'.repeat(100) })).toBe(`${'x'.repeat(79)}…`)
    expect(clip('a\n\nb', 10)).toBe('a b')
  })

  it('is null for an input with no text field, or no input at all', () => {
    expect(toolDetail({ limit: 3, query: '   ' })).toBeNull()
    expect(toolDetail(undefined)).toBeNull()
    expect(toolDetail(['query'])).toBeNull()
  })
})

describe("the core's messages, read for Play", () => {
  it('reads the text parts, else the content', () => {
    expect(messageText(message({ parts: [{ type: 'text', text: 'a' }, { type: 'reasoning', reasoning: 'r' }, { type: 'text', text: 'b' }] }))).toBe('a\nb')
    expect(messageText(message({ parts: [], content: 'stored' }))).toBe('stored')
  })

  it('names the conversation by its title, else its first message, else New chat', () => {
    const asked = message({ role: 'user', parts: [{ type: 'text', text: 'What   laptop?' }] })
    expect(chatTitle([asked], 'Laptop for uni')).toBe('Laptop for uni')
    expect(chatTitle([asked])).toBe('What laptop?')
    expect(chatTitle([message({ role: 'user', content: 'y'.repeat(80) })])).toBe(`${'y'.repeat(59)}…`)
    expect(chatTitle([])).toBe(NEW_CHAT)
  })

  it('finds the newest reply, its activity and its running tool call', () => {
    const reply = message({
      id: 'r2',
      parts: [{ type: 'narration', narration: 'Checking prices first.' }],
      progress: ['3 of 5 pages', ' '],
      toolCalls: [
        { toolCallId: 'a', toolName: 'x', state: 'completed', input: { query: 'old' } },
        { toolCallId: 'b', toolName: 'y', state: 'running', input: { query: 'new' } },
      ],
    })
    const messages = [message({ id: 'r1' }), message({ id: 'u', role: 'user' }), reply]
    expect(latestReply(messages)?.id).toBe('r2')
    expect(activityLines(reply)).toEqual(['Checking prices first.', '3 of 5 pages'])
    expect(runningToolInput(reply)).toEqual({ query: 'new' })
    expect(runningToolInput(undefined)).toBeUndefined()
  })
})
