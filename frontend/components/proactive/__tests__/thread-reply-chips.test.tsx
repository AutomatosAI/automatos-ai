/**
 * PRD-268 §6.6, §6.12.3, §6.12.4, K4(7), K19 (US-W03a) — reply chips on the web, through the chat's own seams.
 *
 * message-with-receipts.tsx registers each message it renders (newest-message.ts) and hands an assistant message's
 * chips (live `replyChips`, or its saved `reply_chips@1` part) to thread-cards.tsx, which draws W2's ReplyChips while
 * the message is the chat's newest: they are gone once a newer one renders. A tap dispatches `automatos:reply-chip`
 * {chatId, text}; the chat's composer (composer-with-suggestion.tsx) sends it once through its sendMessage as the
 * person's own message (role user, a text part), and does nothing while a turn is streaming (the chips come back, as
 * they do when no newer message follows the send within REPLY_FOLLOWS_MS).
 * Each chip is named "Reply: <chip>", its words carry data-ai-generated and are drawn as text. message.tsx, chat.tsx
 * and multimodal-input.tsx are untouched.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ChatMessage } from '@/types'

// Message as a stand-in that shows its text; the composer's box as a stand-in (the chips never type into it).
vi.mock('@/components/chatbot/message', () => ({
  Message: ({ message }: { message: ChatMessage }) => (
    <div data-testid="the-message">
      {(message.parts ?? []).map((part, index) => ('text' in part ? <p key={index}>{(part as { text: string }).text}</p> : null))}
    </div>
  ),
}))
vi.mock('@/components/chatbot/multimodal-input', () => ({
  MultimodalInput: ({ chatId }: { chatId: string }) => <div data-testid="the-composer" data-chat-id={chatId} />,
}))
vi.mock('@/components/proactive/pinned-suggestion', () => ({ PinnedSuggestion: () => null }))

import { MessageWithReceipts } from '@/components/chatbot/message-with-receipts'
import { MultimodalInput } from '@/components/proactive/composer-with-suggestion'
import { REPLY_CHIP_EVENT, sendReplyChip } from '@/components/proactive/reply-chip-event'
import { REPLY_FOLLOWS_MS } from '@/components/proactive/thread-reply-chips'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, ROUNDED_3XL, componentSpans, proactiveSource } from './source-rules'

const CHAT = 'c1'
const CHIPS = ['Window', 'Aisle', 'No preference']
const ASK = 'Window or aisle?'

const user = (id: string, text = 'Book me a seat.') => ({ id, role: 'user', parts: [{ type: 'text', text }] }) as unknown as ChatMessage
const auto = (id: string, extra: Record<string, unknown> = {}, text = ASK) =>
  ({ id, role: 'assistant', parts: [{ type: 'text', text }], ...extra }) as unknown as ChatMessage
const withSavedChips = (id: string, replies: unknown = CHIPS) =>
  auto(id, { parts: [{ type: 'text', text: ASK }, { type: 'reply_chips@1', replies }] })

interface ThreadProps {
  readonly messages: readonly ChatMessage[]
  readonly status?: string
  readonly readOnly?: boolean
}

let sendMessage: ReturnType<typeof vi.fn>

/** The chat as chat.tsx draws it: each message through MessageWithReceipts, then the composer when it has one. */
function Thread({ messages, status = 'idle', readOnly = false }: ThreadProps) {
  return (
    <>
      {messages.map((message) => (
        <MessageWithReceipts key={message.id} chatId={CHAT} message={message} setMessages={vi.fn() as never} regenerate={vi.fn()} isReadonly={readOnly} />
      ))}
      {readOnly ? null : <MultimodalInput chatId={CHAT} status={status} stop={vi.fn()} sendMessage={sendMessage} selectedVisibilityType="private" />}
    </>
  )
}

const chipButtons = () => screen.queryAllByRole('button', { name: /^Reply: / })
const settle = () => act(async () => new Promise<void>((resolve) => setTimeout(resolve, 0)))

beforeEach(() => {
  sendMessage = vi.fn()
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('shown under the newest message only', () => {
  it("draws the saved reply_chips@1 part under Auto's newest message, outside its AI-generated block", () => {
    render(<Thread messages={[user('u1'), withSavedChips('a1')]} />)
    const group = screen.getByRole('group', { name: 'Replies' })
    expect(chipButtons().map((chip) => chip.textContent)).toEqual(CHIPS)
    expect(screen.getByTestId('ai-generated-message')).not.toContainElement(group)
    expect(screen.getByText(ASK)).toBeInTheDocument()
  })

  it('draws the live chips the turn_parts frame put on the reply being streamed', () => {
    render(<Thread messages={[user('u1'), auto('a1', { replyChips: ['Yes', 'No'] })]} />)
    expect(chipButtons().map((chip) => chip.getAttribute('aria-label'))).toEqual(['Reply: Yes', 'Reply: No'])
  })

  it('draws nothing under an older message', () => {
    render(<Thread messages={[withSavedChips('a1'), user('u1'), auto('a2')]} />)
    expect(chipButtons()).toEqual([])
  })

  it('draws nothing for a malformed part, a single chip, or on a read-only chat', () => {
    const { rerender } = render(<Thread messages={[user('u1'), withSavedChips('a1', 'Window')]} />)
    expect(chipButtons()).toEqual([])
    rerender(<Thread messages={[user('u1'), withSavedChips('a2', ['Window'])]} />)
    expect(chipButtons()).toEqual([])
    rerender(<Thread messages={[user('u1'), withSavedChips('a3')]} readOnly />)
    expect(chipButtons()).toEqual([])
  })

  it('goes once a newer message renders: the person types instead, or a background post lands', () => {
    const { rerender } = render(<Thread messages={[user('u1'), withSavedChips('a1')]} />)
    expect(chipButtons()).toHaveLength(3)
    rerender(<Thread messages={[user('u1'), withSavedChips('a1'), user('u2', 'Neither, thanks.')]} />)
    expect(chipButtons()).toEqual([])
    rerender(<Thread messages={[user('u1'), withSavedChips('a1')]} />)
    expect(chipButtons()).toHaveLength(3)
    rerender(<Thread messages={[user('u1'), withSavedChips('a1'), auto('a2', {}, 'Your flight moved.')]} />)
    expect(chipButtons()).toEqual([])
  })
})

describe('a tap', () => {
  it("sends the chip's text once, through the composer, as the person's own message; the chips go after it", async () => {
    const { rerender } = render(<Thread messages={[user('u1'), withSavedChips('a1')]} />)
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Window' }))
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Aisle' }))
    expect(sendMessage).toHaveBeenCalledTimes(1)
    expect(sendMessage).toHaveBeenCalledWith({ role: 'user', content: 'Window', attachment_ids: [], parts: [{ type: 'text', text: 'Window' }] })
    expect(chipButtons()).toHaveLength(3)
    chipButtons().forEach((chip) => expect(chip).toBeDisabled())
    expect(screen.getByTestId('the-composer')).toBeInTheDocument()
    rerender(<Thread messages={[user('u1'), withSavedChips('a1'), user('u2', 'Window')]} />)
    await settle()
    expect(chipButtons()).toEqual([])
    expect(console.warn).not.toHaveBeenCalled()
  })

  it('gives the chips back when no newer message follows the send: the chat passed it over', async () => {
    vi.useFakeTimers()
    render(<Thread messages={[user('u1'), withSavedChips('a1')]} />)
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Window' }))
    expect(sendMessage).toHaveBeenCalledTimes(1)
    expect(chipButtons()).toHaveLength(3)
    chipButtons().forEach((chip) => expect(chip).toBeDisabled())
    await act(async () => {
      await vi.advanceTimersByTimeAsync(REPLY_FOLLOWS_MS)
    })
    expect(chipButtons()).toHaveLength(3)
    chipButtons().forEach((chip) => expect(chip).toBeEnabled())
    expect(console.warn).toHaveBeenCalled()
  })

  it('sends nothing while a turn is streaming, gives the chips back, and sends once the turn is over', async () => {
    const { rerender } = render(<Thread messages={[user('u1'), withSavedChips('a1')]} status="streaming" />)
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Window' }))
    await settle()
    expect(sendMessage).not.toHaveBeenCalled()
    expect(chipButtons()).toHaveLength(3)
    chipButtons().forEach((chip) => expect(chip).toBeEnabled())
    rerender(<Thread messages={[user('u1'), withSavedChips('a1')]} status="idle" />)
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Aisle' }))
    expect(sendMessage).toHaveBeenCalledTimes(1)
    expect(sendMessage).toHaveBeenCalledWith(expect.objectContaining({ role: 'user', content: 'Aisle' }))
  })

  it("is sent by one composer only, and never by another chat's", () => {
    const other = vi.fn()
    render(
      <>
        <Thread messages={[user('u1'), withSavedChips('a1')]} />
        <MultimodalInput chatId={CHAT} status="idle" stop={vi.fn()} sendMessage={other} selectedVisibilityType="private" />
        <MultimodalInput chatId="c2" status="idle" stop={vi.fn()} sendMessage={other} selectedVisibilityType="private" />
      </>,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Window' }))
    expect(sendMessage.mock.calls.length + other.mock.calls.length).toBe(1)
    expect(sendReplyChip('c2', 'Hello')).toBe(true)
    expect(other).toHaveBeenLastCalledWith(expect.objectContaining({ content: 'Hello' }))
    expect(sendReplyChip('c9', 'Hello')).toBe(false)
  })

  it('gives the chips back when no composer is there to send them', async () => {
    render(
      <MessageWithReceipts chatId={CHAT} message={withSavedChips('a1')} setMessages={vi.fn() as never} regenerate={vi.fn()} isReadonly={false} />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Window' }))
    await settle()
    expect(chipButtons()).toHaveLength(3)
    chipButtons().forEach((chip) => expect(chip).toBeEnabled())
    expect(console.warn).toHaveBeenCalled()
  })

  it('reaches the composer as the automatos:reply-chip window event, {chatId, text}', () => {
    const heard = vi.fn()
    window.addEventListener(REPLY_CHIP_EVENT, heard)
    render(<Thread messages={[user('u1'), withSavedChips('a1')]} />)
    fireEvent.click(screen.getByRole('button', { name: 'Reply: No preference' }))
    window.removeEventListener(REPLY_CHIP_EVENT, heard)
    expect(REPLY_CHIP_EVENT).toBe('automatos:reply-chip')
    expect((heard.mock.calls[0][0] as CustomEvent).detail).toEqual({ chatId: CHAT, text: 'No preference' })
  })
})

describe('the names, the marks and the text', () => {
  it('names each chip "Reply: <chip>" and marks its words, not the control, as AI-generated', () => {
    render(<Thread messages={[user('u1'), withSavedChips('a1')]} />)
    for (const chip of CHIPS) {
      const button = screen.getByRole('button', { name: `Reply: ${chip}` })
      expect(button).not.toHaveAttribute('data-ai-generated')
      expect(within(button).getByText(chip)).toHaveAttribute('data-ai-generated', 'true')
    }
  })

  it('draws a chip as text, never markup', () => {
    render(<Thread messages={[user('u1'), withSavedChips('a1', ['<b>Yes</b>', 'No'])]} />)
    const button = screen.getByRole('button', { name: 'Reply: <b>Yes</b>' })
    expect(button.textContent).toBe('<b>Yes</b>')
    expect(button.querySelector('b')).toBeNull()
  })
})

describe('the source rules', () => {
  const chatbot = (file: string) => readFileSync(path.resolve(__dirname, '../../chatbot', file), 'utf8')

  it('keeps the new files to tokens and short components', () => {
    for (const file of ['thread-reply-chips.tsx', 'thread-cards.tsx', 'composer-with-suggestion.tsx']) {
      const source = proactiveSource(file)
      expect(source).not.toMatch(HEX_COLOUR)
      expect(source).not.toMatch(ORANGE_CLASS)
      expect(source).not.toMatch(ROUNDED_3XL)
      componentSpans(source).forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
    }
  })

  it('draws the chips through message-with-receipts.tsx and the composer seam, never in message.tsx, chat.tsx or the box', () => {
    expect(chatbot('message-with-receipts.tsx')).toContain("from '@/components/proactive/newest-message'")
    for (const file of ['message.tsx', 'chat.tsx', 'multimodal-input.tsx']) {
      expect(chatbot(file)).not.toMatch(/reply_chips@1|reply-chip|replyChips|ReplyChips/)
    }
  })
})
