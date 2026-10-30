/**
 * PRD-268 §6.5.1, §6.12.4, K4(4), K17 (US-W01b) — an Auto-thread message's suggestion@1 and brief@1 parts draw as
 * cards under it (components/proactive/thread-cards.tsx, through message-with-receipts.tsx), outside its
 * data-ai-generated block. The suggestion card takes its live state from the open list by the id its part names, so
 * it survives a reload and goes once the suggestion is decided; Start started here stays for "Started. Auto is on
 * it."; Start is never the thread's ink primary, and a card steps aside while its suggestion is pinned above the
 * composer (one live Start and Hide per suggestion). `Message` gets a copy without the card parts, and without the text
 * part when the brief card holds the prose; unknown parts still reach it (it drops them); every other message
 * reaches it unchanged.
 *
 * US-W02: the thread's brief card has thumbs (the digest feedback route, the brief's state_hash) and Listen (the host's
 * BriefVoice over this brief's spoken words): nothing plays until Listen is pressed, and a press reads them in the server
 * voice.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ChatMessage } from '@/types'
import type { SynthesisPort } from '@/lib/voice/quick-ask/speaker'

type Mutation = { mutate: ReturnType<typeof vi.fn>; reset: ReturnType<typeof vi.fn>; isLoading: boolean; isError: boolean; error: unknown }
const state = vi.hoisted(() => ({
  list: undefined as unknown,
  start: null as unknown as Mutation,
  hide: null as unknown as Mutation,
  feedback: { mutate: vi.fn() },
}))
vi.mock('@/hooks/use-suggestions', () => ({
  useSuggestions: () => ({ data: state.list }),
  useStartSuggestion: () => state.start,
  useHideSuggestion: () => state.hide,
}))
vi.mock('@/hooks/use-digest-api', () => ({ useSubmitDigestFeedback: () => state.feedback }))
// Listen's voice: the server-speech fakes, so a press is heard and a mount is seen to ask nothing.
const speech = vi.hoisted(() => ({ make: null as unknown as () => SynthesisPort }))
vi.mock('@/lib/voice/quick-ask/deps', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/voice/quick-ask/deps')>()),
  speechPort: () => speech.make(),
}))
// Message as a stand-in that shows the part types and text it was handed.
vi.mock('@/components/chatbot/message', () => ({
  Message: ({ message }: { message: ChatMessage }) => (
    <div data-testid="the-message" data-part-types={(message.parts ?? []).map((part) => String((part as { type: unknown }).type)).join(',')}>
      {(message.parts ?? []).map((part, index) => ('text' in part ? <p key={index}>{(part as { text: string }).text}</p> : null))}
      {(message as unknown as { content?: string }).content}
    </div>
  ),
}))

import { MessageWithReceipts } from '@/components/chatbot/message-with-receipts'
import { PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from '@/components/proactive/card-parts'
import { markSuggestionPinned } from '@/components/proactive/mission-suggestion-presence'
import { ThreadCards } from '@/components/proactive/thread-cards'
import { STARTED_LINE, STARTED_MS } from '@/components/proactive/use-suggestion-card'
import { threadPartsOf } from '@/lib/proactive/thread-parts'
import { LISTEN_TO_THE_BRIEF } from '@/components/voice/brief-voice'
import { flush } from '@/lib/voice/quick-ask/__tests__/quick-ask-harness'
import { serverSpeechSetup } from '@/lib/voice/quick-ask/__tests__/server-speech-fakes'
import type { Brief, Suggestion } from '@/lib/proactive/types'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, ROUNDED_3XL, componentSpans, proactiveSource } from './source-rules'

const OPEN_ID = '7d1c0a52-3e0b-4c55-9a1e-2f6f0c1d9b10'
const OTHER_ID = '0b6f3e21-94c7-4d0a-8e15-6a2d9c7f4b33'
const TEXT = 'Three runs of the Lisbon flights check failed since Monday.'

function suggestion(overrides: Partial<Suggestion> = {}): Suggestion {
  return {
    id: OPEN_ID, workspace_id: 'ws-1', user_id: 1, source_type: 'check_in', source_agent_id: 4, run_id: null,
    title: 'Look at the Lisbon flights', detail: 'Three runs failed since Monday.', evidence: [],
    proposed_action: { kind: 'board_task', payload: {} }, subject_key: 'ticket:0419', severity: 'today',
    status: 'open', occurrences: 1, last_seen_at: null, started_ref: null, decided_via: null, hide_reason: null,
    decided_at: null, est_credits: null, created_at: null, delivered_at: null, expires_at: null,
    ...overrides,
  }
}

function brief(overrides: Partial<Brief> = {}): Brief {
  return {
    v: 1, date: '2026-10-30', greeting: 'Morning, Sam.', prose: 'Two things moved overnight.',
    spoken: 'Morning, Sam.',
    sections: { needs_you: [], suggestions: [], today: [], since: [], unchecked: [] },
    counts: { needs_you: 0, suggestions: 0, today: 0, since: 0 },
    labels: { sections: {}, more: {}, empty: 'Nothing needs you today.' },
    state_hash: 'hash-7', generated_at: '2026-10-30T06:00:00Z',
    ...overrides,
  }
}

function assistant(parts: unknown[], extra: Record<string, unknown> = {}): ChatMessage {
  return { id: 'm1', role: 'assistant', parts, ...extra } as unknown as ChatMessage
}

const suggestionMessage = (id = OPEN_ID) =>
  assistant([{ type: 'text', text: TEXT }, { type: 'suggestion@1', suggestion_id: id }], { source: { origin: 'suggestion', suggestion_id: id } })
const briefMessage = (part: unknown = { type: 'brief@1', brief: brief() }) =>
  assistant([{ type: 'text', text: 'Two things moved overnight.' }, part], { source: { origin: 'daily_digest', local_date: '2026-10-30' } })

let speechSetup: ReturnType<typeof serverSpeechSetup>

const mutation = (): Mutation => ({ mutate: vi.fn(), reset: vi.fn(), isLoading: false, isError: false, error: null })

function draw(message: ChatMessage) {
  return render(<MessageWithReceipts chatId="c1" setMessages={vi.fn() as never} regenerate={vi.fn()} isReadonly message={message} />)
}

beforeEach(() => {
  state.list = [suggestion()]
  state.start = mutation()
  state.hide = mutation()
  state.feedback.mutate.mockReset()
  speechSetup = serverSpeechSetup()
  speech.make = vi.fn(() => speechSetup.port)
})

afterEach(() => {
  vi.useRealTimers()
})

describe('threadPartsOf, what Message still draws', () => {
  it('hands back the same message when it carries no card part, and every user message', () => {
    const plain = assistant([{ type: 'text', text: TEXT }])
    expect(threadPartsOf(plain)).toEqual({ brief: null, suggestionIds: [], message: plain })
    expect(threadPartsOf(plain).message).toBe(plain)
    const user = { id: 'u1', role: 'user', parts: [{ type: 'suggestion@1', suggestion_id: OPEN_ID }] } as unknown as ChatMessage
    expect(threadPartsOf(user).message).toBe(user)
    expect(threadPartsOf(user).suggestionIds).toEqual([])
  })

  it('takes the suggestion@1 part out, keeps the text, and names the suggestion', () => {
    const message = suggestionMessage()
    const thread = threadPartsOf(message)
    expect(thread.suggestionIds).toEqual([OPEN_ID])
    expect(thread.brief).toBeNull()
    expect(thread.message.parts).toEqual([{ type: 'text', text: TEXT }])
    expect(message.parts).toHaveLength(2)
  })

  it("falls back to the message's source.suggestion_id, names each suggestion once, and drops an id that isn't one", () => {
    const fallback = assistant([{ type: 'suggestion@1' }], { source: { origin: 'suggestion', suggestion_id: OTHER_ID } })
    expect(threadPartsOf(fallback).suggestionIds).toEqual([OTHER_ID])
    const twice = assistant([{ type: 'suggestion@1', suggestion_id: OPEN_ID }, { type: 'suggestion@1', suggestion_id: OPEN_ID }])
    expect(threadPartsOf(twice).suggestionIds).toEqual([OPEN_ID])
    const bad = assistant([{ type: 'text', text: TEXT }, { type: 'suggestion@1', suggestion_id: '../../admin' }])
    expect(threadPartsOf(bad).suggestionIds).toEqual([])
    expect(threadPartsOf(bad).message.parts).toEqual([{ type: 'text', text: TEXT }])
  })

  it('takes the brief@1 part and the text part out once the brief is a Brief@1, since the card holds the prose', () => {
    const thread = threadPartsOf(briefMessage())
    expect(thread.brief?.greeting).toBe('Morning, Sam.')
    expect(thread.message.parts).toEqual([])
    const withContent = threadPartsOf(assistant([{ type: 'brief@1', brief: brief() }], { content: 'Two things moved overnight.' }))
    expect((withContent.message as unknown as { content?: string }).content).toBe('')
  })

  it("keeps the text when the brief@1 part isn't a Brief@1, and still passes unknown parts on for Message to drop", () => {
    const broken = threadPartsOf(briefMessage({ type: 'brief@1', brief: { v: 2 } }))
    expect(broken.brief).toBeNull()
    expect(broken.message.parts).toEqual([{ type: 'text', text: 'Two things moved overnight.' }])
    const unknown = threadPartsOf(assistant([{ type: 'text', text: TEXT }, { type: 'weather@1' }, { type: 'suggestion@1', suggestion_id: OPEN_ID }]))
    expect(unknown.message.parts).toEqual([{ type: 'text', text: TEXT }, { type: 'weather@1' }])
  })
})

describe('the suggestion card in the Auto thread', () => {
  it('draws the card under the message, outside its AI-generated block, with Start and Hide; Message gets the text only', () => {
    draw(suggestionMessage())
    const holder = screen.getByTestId('ai-generated-message')
    expect(screen.getByTestId('the-message')).toHaveAttribute('data-part-types', 'text')
    expect(within(holder).getByText(TEXT)).toBeInTheDocument()
    const card = screen.getByRole('article', { name: 'Auto suggests: Look at the Lisbon flights' })
    expect(holder).not.toContainElement(card)
    expect(screen.getByTestId('thread-cards')).toContainElement(card)
    expect(within(card).getByText('Three runs failed since Monday.').closest('[data-ai-generated="true"]')).not.toBeNull()
    expect(within(card).getByRole('button', { name: 'Start: Look at the Lisbon flights' }).closest('[data-ai-generated]')).toBeNull()
    expect(within(card).getByRole('button', { name: 'Hide: Look at the Lisbon flights' })).toBeEnabled()
  })

  it('takes its state from the open list by id: drawn while open, gone once decided, back after a reload while open', () => {
    const { rerender } = draw(suggestionMessage())
    expect(screen.getByRole('article', { name: /Auto suggests/ })).toHaveAttribute('data-suggestion-id', OPEN_ID)
    state.list = [suggestion({ id: OTHER_ID, title: 'Renew the domain' })]
    rerender(<MessageWithReceipts chatId="c1" setMessages={vi.fn() as never} regenerate={vi.fn()} isReadonly message={suggestionMessage()} />)
    expect(screen.queryByRole('article', { name: /Auto suggests/ })).toBeNull()
    expect(screen.queryByTestId('thread-cards')).toBeNull()
    expect(screen.getByText(TEXT)).toBeInTheDocument()
    state.list = [suggestion()]
    rerender(<MessageWithReceipts chatId="c1" setMessages={vi.fn() as never} regenerate={vi.fn()} isReadonly message={{ ...suggestionMessage() }} />)
    expect(screen.getByRole('article', { name: 'Auto suggests: Look at the Lisbon flights' })).toBeInTheDocument()
  })

  it('draws nothing of the card while the list is unanswered', () => {
    state.list = undefined
    draw(suggestionMessage())
    expect(screen.queryByTestId('thread-cards')).toBeNull()
    expect(screen.getByText(TEXT)).toBeInTheDocument()
  })

  it('Start posts with via web and keeps the card for "Started. Auto is on it." after the list lets it go', () => {
    vi.useFakeTimers()
    const { rerender } = draw(suggestionMessage())
    fireEvent.click(screen.getByRole('button', { name: 'Start: Look at the Lisbon flights' }))
    expect(state.start.mutate).toHaveBeenCalledWith({ id: OPEN_ID, via: 'web' }, expect.anything())
    const options = state.start.mutate.mock.calls[0][1] as { onSuccess: (answer: unknown) => void }
    act(() => options.onSuccess({ suggestion: suggestion({ status: 'started' }), opens: { type: 'board_task', board_task_id: 9 } }))
    state.list = []
    rerender(<MessageWithReceipts chatId="c1" setMessages={vi.fn() as never} regenerate={vi.fn()} isReadonly message={suggestionMessage()} />)
    expect(screen.getByText(STARTED_LINE)).toBeInTheDocument()
    act(() => vi.advanceTimersByTime(STARTED_MS))
    expect(screen.queryByText(STARTED_LINE)).toBeNull()
  })

  it("Hide sends the reason picked from the menu", () => {
    draw(suggestionMessage())
    fireEvent.click(screen.getByRole('button', { name: 'Hide: Look at the Lisbon flights' }))
    fireEvent.click(screen.getByRole('button', { name: 'Not useful' }))
    expect(state.hide.mutate).toHaveBeenCalledWith({ id: OPEN_ID, via: 'web', reason: 'not_useful' }, expect.anything())
  })

  it('steps aside while its suggestion is pinned above the composer, and comes back once the line lets it go', () => {
    draw(suggestionMessage())
    expect(screen.getByRole('article', { name: 'Auto suggests: Look at the Lisbon flights' })).toBeInTheDocument()
    let release: () => void = () => undefined
    act(() => {
      release = markSuggestionPinned(OPEN_ID)
    })
    expect(screen.queryByRole('article', { name: /Auto suggests/ })).toBeNull()
    expect(screen.getByText(TEXT)).toBeInTheDocument()
    act(() => markSuggestionPinned(OTHER_ID)())
    expect(screen.queryByRole('article', { name: /Auto suggests/ })).toBeNull()
    act(() => release())
    expect(screen.getByRole('article', { name: 'Auto suggests: Look at the Lisbon flights' })).toBeInTheDocument()
  })

  it("never makes Start the thread's ink primary: the pinned line holds it", () => {
    draw(suggestionMessage())
    const start = screen.getByRole('button', { name: 'Start: Look at the Lisbon flights' })
    expect(start.className).toBe(QUIET_BUTTON_CLASS)
    expect(start.className).not.toBe(PRIMARY_BUTTON_CLASS)
  })
})

describe('the brief card in the Auto thread', () => {
  it('draws the brief as its card under the message and hands Message neither the brief nor its text', () => {
    draw(briefMessage())
    expect(screen.getByTestId('the-message')).toHaveAttribute('data-part-types', '')
    const card = screen.getByRole('article', { name: 'Morning, Sam.' })
    expect(screen.getByTestId('ai-generated-message')).not.toContainElement(card)
    expect(screen.getAllByText('Two things moved overnight.')).toHaveLength(1)
    expect(card).toContainElement(screen.getByText('Two things moved overnight.'))
  })

  it("leaves the text to Message when the brief@1 part isn't a Brief@1", () => {
    draw(briefMessage({ type: 'brief@1', brief: { v: 1 } }))
    expect(screen.queryByRole('article', { name: 'Morning, Sam.' })).toBeNull()
    expect(screen.getByTestId('the-message')).toHaveTextContent('Two things moved overnight.')
  })

  it("rates the brief with the thumbs, keyed by the brief's state_hash", () => {
    draw(briefMessage())
    const card = screen.getByRole('article', { name: 'Morning, Sam.' })
    fireEvent.click(within(card).getByRole('button', { name: 'Helpful' }))
    expect(state.feedback.mutate).toHaveBeenCalledWith({ state_hash: 'hash-7', rating: 1 }, expect.anything())
  })

  it('has Listen, outside the AI-generated block, and plays nothing until it is pressed', async () => {
    draw(briefMessage())
    const card = screen.getByRole('article', { name: 'Morning, Sam.' })
    const listen = within(card).getByRole('button', { name: LISTEN_TO_THE_BRIEF })
    expect(listen).toBeEnabled()
    expect(screen.getByTestId('ai-generated-message')).not.toContainElement(listen)
    await act(async () => flush())
    expect(speech.make).not.toHaveBeenCalled()
    expect(speechSetup.server.fetchSpeech).not.toHaveBeenCalled()
    expect(speechSetup.audio.player.play).not.toHaveBeenCalled()
  })

  it("reads this brief's spoken words in the server voice once Listen is pressed", async () => {
    draw(briefMessage({ type: 'brief@1', brief: brief({ spoken: 'Morning, Sam. 1. Lisbon flights.' }) }))
    fireEvent.click(screen.getByRole('button', { name: LISTEN_TO_THE_BRIEF }))
    await act(async () => flush())
    expect(speechSetup.server.texts()).toEqual(['Morning, Sam.', '1. Lisbon flights.'])
  })
})

describe('every other message', () => {
  it('renders exactly as before: no cards, the message whole', () => {
    draw(assistant([{ type: 'text', text: TEXT }, { type: 'reasoning', reasoning: 'r' }]))
    expect(screen.queryByTestId('thread-cards')).toBeNull()
    expect(screen.getByTestId('the-message')).toHaveAttribute('data-part-types', 'text,reasoning')
  })

  it('ThreadCards draws nothing without a brief or a suggestion', () => {
    const { container } = render(<ThreadCards brief={null} suggestionIds={[]} />)
    expect(container).toBeEmptyDOMElement()
  })
})

describe('the source rules', () => {
  const chatbot = (file: string) => readFileSync(path.resolve(__dirname, '../../chatbot', file), 'utf8')

  it('keeps the new files to tokens and short components', () => {
    for (const file of ['thread-cards.tsx']) {
      const source = proactiveSource(file)
      expect(source).not.toMatch(HEX_COLOUR)
      expect(source).not.toMatch(ORANGE_CLASS)
      expect(source).not.toMatch(ROUNDED_3XL)
      componentSpans(source).forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
    }
    componentSpans(chatbot('message-with-receipts.tsx')).forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
  })

  it('draws the cards through message-with-receipts.tsx, never inside message.tsx', () => {
    expect(chatbot('message-with-receipts.tsx')).toContain("from '@/components/proactive/thread-cards'")
    expect(chatbot('message.tsx')).not.toMatch(/brief@1|suggestion@1|thread-cards/)
  })
})
