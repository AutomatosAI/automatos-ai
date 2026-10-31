/**
 * PRD-268 §6.11.5, §6.12.4, K15, K18, K19 (US-W06) — the consent line, drawn by composer-with-suggestion.tsx above the
 * composer in any chat, once: only while GET /api/auto/calibration answers available and offer. It says the server's
 * consent line; Go ahead (the view's ink primary) posts consent true with the chat's id and the page's locale, and Not
 * now posts consent false; each is sent once, and the line goes once the server has the answer. A calibration posted
 * into another chat (the Auto thread) gets one line with a link to it. A refused answer keeps the line with the
 * server's sentence. While it shows, the pinned suggestion steps aside. Never shown when not available.
 */
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const requestMock = vi.hoisted(() => vi.fn())
const postMock = vi.hoisted(() => vi.fn())
vi.mock('@/lib/api-client', () => ({
  apiClient: {
    request: (...args: unknown[]) => requestMock(...args),
    post: (...args: unknown[]) => postMock(...args),
  },
}))
vi.mock('@/components/chatbot/multimodal-input', () => ({
  MultimodalInput: ({ chatId }: { chatId: string }) => <div data-testid="the-composer" data-chat-id={chatId} />,
}))
vi.mock('@/components/proactive/pinned-suggestion', () => ({
  PinnedSuggestion: ({ chatId }: { chatId: string }) => <div data-testid="pinned-suggestion" data-chat-id={chatId} />,
}))
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: string; children: ReactNode; className?: string }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}))

import { PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from '@/components/proactive/card-parts'
import { GO_AHEAD, NOT_NOW, OPEN_IT, POSTED_ELSEWHERE } from '@/components/proactive/calibration-offer'
import { MultimodalInput } from '@/components/proactive/composer-with-suggestion'
import { CALIBRATION_PATH } from '@/hooks/use-calibration'
import { chatHref } from '@/lib/ticket-links'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, ROUNDED_3XL, componentSpans, proactiveSource } from './source-rules'

const CHAT = '5b7f2c1e-8d4a-4f6b-9c3e-1a2b3c4d5e6f'
const AUTO_THREAD = '0b6f3e21-94c7-4d0a-8e15-6a2d9c7f4b33'
const CAL_ID = '3f2a1b0c-9d8e-4f7a-b6c5-d4e3f2a1b0c9'
const CONSENT_LINE = 'Auto reads the titles of your recent work to say what it knows.'
const CALIBRATION = { v: 1, id: CAL_ID, lines: [], offers: ['Plan the week'], empty: true }

function view(overrides: Record<string, unknown> = {}) {
  return {
    available: true, offer: true, consent_line: CONSENT_LINE, status: null, calibration: null,
    rejected_lines: [], accepted_lines: [], ...overrides,
  }
}

/** The server's view before the answer, then after it. */
function server(before: unknown, after: unknown = view({ offer: false, status: 'open', calibration: CALIBRATION })) {
  requestMock.mockResolvedValueOnce(before).mockResolvedValue(after)
}

/** An answer the test settles. */
function deferredPost() {
  const pending: { resolve?: (value: unknown) => void } = {}
  postMock.mockImplementation(() => new Promise((resolve) => (pending.resolve = resolve)))
  return (value: unknown) => act(async () => pending.resolve?.(value))
}

const quiet = { log: () => undefined, warn: () => undefined, error: () => undefined }

function draw(chatId = CHAT, ground = '') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } }, logger: quiet })
  const composer = (id: string) => (
    <QueryClientProvider client={client}>
      <div className={ground}>
        <MultimodalInput chatId={id} status="ready" stop={vi.fn()} sendMessage={vi.fn()} selectedVisibilityType="private" />
      </div>
    </QueryClientProvider>
  )
  const drawn = render(composer(chatId))
  return { ...drawn, client, moveTo: (id: string) => drawn.rerender(composer(id)) }
}

const offerLine = () => screen.queryByTestId('calibration-offer')
const goAhead = () => screen.getByRole('button', { name: GO_AHEAD })
const notNow = () => screen.getByRole('button', { name: NOT_NOW })
const settle = () => act(async () => new Promise<void>((resolve) => setTimeout(resolve, 0)))

beforeEach(() => {
  requestMock.mockReset()
  postMock.mockReset()
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('the consent line above the composer', () => {
  it("shows the server's consent line once while offered, Go ahead the ink primary, in the pinned suggestion's place", async () => {
    server(view())
    draw()
    expect(await screen.findByText(CONSENT_LINE)).toBeInTheDocument()
    expect(screen.getAllByTestId('calibration-offer')).toHaveLength(1)
    expect(goAhead().className).toBe(PRIMARY_BUTTON_CLASS)
    expect(notNow().className).toBe(QUIET_BUTTON_CLASS)
    expect(screen.getByTestId('the-composer')).toBeInTheDocument()
    expect(screen.queryByTestId('pinned-suggestion')).toBeNull()
    expect(offerLine()?.querySelector('[data-ai-generated]')).toBeNull()
  })

  it("Go ahead posts consent true once, with the chat's id and the page's locale; then the line goes", async () => {
    server(view())
    const answer = deferredPost()
    draw()
    fireEvent.click(await screen.findByRole('button', { name: GO_AHEAD }))
    fireEvent.click(goAhead())
    fireEvent.click(notNow())
    // The mutation reaches its POST a few microtasks after the press: a second answer would have been sent by now.
    await waitFor(() => expect(postMock).toHaveBeenCalled())
    await settle()
    expect(postMock).toHaveBeenCalledTimes(1)
    expect(postMock).toHaveBeenCalledWith(CALIBRATION_PATH, { consent: true, chat_id: CHAT, locale: navigator.language })
    await waitFor(() => expect(goAhead()).toBeDisabled())
    expect(notNow()).toBeDisabled()

    await answer({ status: 'open', calibration: CALIBRATION, chat_id: CHAT, message_id: CAL_ID })
    await waitFor(() => expect(offerLine()).toBeNull())
    expect(screen.queryByTestId('calibration-posted')).toBeNull()
    expect(screen.getByTestId('pinned-suggestion')).toBeInTheDocument()
    expect(postMock).toHaveBeenCalledTimes(1)
  })

  it('Not now posts consent false once, and the line goes', async () => {
    server(view(), view({ offer: false, status: 'declined' }))
    const answer = deferredPost()
    draw()
    fireEvent.click(await screen.findByRole('button', { name: NOT_NOW }))
    fireEvent.click(notNow())
    await waitFor(() => expect(postMock).toHaveBeenCalled())
    await settle()
    expect(postMock).toHaveBeenCalledTimes(1)
    expect(postMock).toHaveBeenCalledWith(CALIBRATION_PATH, { consent: false, chat_id: CHAT, locale: navigator.language })

    await answer({ status: 'declined', calibration: null, chat_id: null, message_id: null })
    await waitFor(() => expect(offerLine()).toBeNull())
    expect(screen.queryByTestId('calibration-posted')).toBeNull()
  })

  it('says where Auto answered when the calibration went to the Auto thread, with a link to it', async () => {
    server(view())
    postMock.mockResolvedValue({ status: 'open', calibration: CALIBRATION, chat_id: AUTO_THREAD, message_id: CAL_ID })
    const { moveTo } = draw()
    fireEvent.click(await screen.findByRole('button', { name: GO_AHEAD }))
    const posted = await screen.findByTestId('calibration-posted')
    expect(posted).toHaveTextContent(POSTED_ELSEWHERE)
    expect(screen.getByRole('link', { name: OPEN_IT })).toHaveAttribute('href', chatHref(AUTO_THREAD))
    expect(offerLine()).toBeNull()

    moveTo(AUTO_THREAD)
    expect(screen.queryByTestId('calibration-posted')).toBeNull()
  })

  it("keeps the line when the answer is refused, with the server's sentence; a 409 already_offered takes it away", async () => {
    server(view(), view())
    const detail = 'Auto could not do that just now. Try again in a moment.'
    postMock.mockRejectedValueOnce(Object.assign(new Error('HTTP 503'), { status: 503, body: { code: 'calibration_unavailable', detail } }))
    draw()
    fireEvent.click(await screen.findByRole('button', { name: GO_AHEAD }))
    expect(await screen.findByRole('alert')).toHaveTextContent(detail)
    expect(offerLine()).not.toBeNull()
    expect(goAhead()).toBeEnabled()

    requestMock.mockResolvedValue(view({ offer: false, status: 'declined' }))
    postMock.mockRejectedValueOnce(
      Object.assign(new Error('HTTP 409'), { status: 409, body: { code: 'already_offered', detail: 'Auto has offered you this already.', record: { status: 'declined' } } }),
    )
    fireEvent.click(goAhead())
    await waitFor(() => expect(offerLine()).toBeNull())
    expect(postMock).toHaveBeenCalledTimes(2)
  })
})

describe('never shown when not available', () => {
  it('draws nothing and posts nothing while the server answers available false; the pinned suggestion stays', async () => {
    requestMock.mockResolvedValue({ ...view(), available: false, offer: false })
    draw()
    await settle()
    await waitFor(() => expect(requestMock).toHaveBeenCalled())
    await settle()
    expect(offerLine()).toBeNull()
    expect(screen.queryByText(CONSENT_LINE)).toBeNull()
    expect(screen.getByTestId('pinned-suggestion')).toBeInTheDocument()
    expect(postMock).not.toHaveBeenCalled()
  })

  it('draws nothing once answered (or for a person who may not answer), nor while the read is on its way', async () => {
    requestMock.mockResolvedValue(view({ offer: false, status: 'answered', calibration: CALIBRATION }))
    draw()
    expect(offerLine()).toBeNull()
    await waitFor(() => expect(requestMock).toHaveBeenCalled())
    await settle()
    expect(offerLine()).toBeNull()
    expect(postMock).not.toHaveBeenCalled()
  })
})

const LOOKS = [
  { name: 'Studio light', ground: 'studio' },
  { name: 'Studio dark', ground: 'studio dark' },
  { name: 'Classic', ground: '' },
] as const

describe.each(LOOKS)('the consent line in $name', ({ ground }) => {
  it("draws from the token classes each look's token file paints", async () => {
    server(view())
    draw(CHAT, ground)
    await screen.findByText(CONSENT_LINE)
    expect(offerLine()).toHaveClass('bg-card', 'border-border', 'rounded-xl')
    expect(screen.getByText(CONSENT_LINE)).toHaveClass('text-foreground')
    expect(goAhead()).toHaveClass('bg-primary', 'text-primary-foreground')
    expect(notNow()).toHaveClass('border-border', 'text-foreground')
  })
})

describe('calibration offer source', () => {
  it(`keeps to tokens and components of at most ${MAX_COMPONENT_LINES} lines`, () => {
    for (const file of ['calibration-offer.tsx', 'composer-with-suggestion.tsx', 'thread-calibration.tsx']) {
      const source = proactiveSource(file)
      expect(source).not.toMatch(HEX_COLOUR)
      expect(source).not.toMatch(ORANGE_CLASS)
      expect(source).not.toMatch(ROUNDED_3XL)
      expect(source).not.toMatch(/\bdark:/)
      expect(source).not.toMatch(/fetch\(/)
      componentSpans(source).forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
    }
  })
})
