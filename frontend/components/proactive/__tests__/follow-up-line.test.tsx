/**
 * PRD-268 §6.3 'In the thread', §6.12.4, K18, K19 (US-W03b) — the follow-up line under the message that scheduled it.
 *
 * message-with-receipts.tsx hands an assistant message's follow-ups (its live `followUps`, or its saved `follow_up@1`
 * parts) to thread-cards.tsx, which draws follow-up-line.tsx for each one the caller's own list holds
 * (hooks/use-follow-ups.ts: GET /api/proactive/follow-ups?chat_id=&status=all through apiClient, asked again on
 * `automatos:chat-changed`). "Follow-up at HH:MM · Cancel" while pending, "Cancelled" after a cancel and "Followed up at
 * HH:MM" once fired, in the viewer's locale; Cancel posts …/{id}/cancel once and the list is asked again; a refused
 * cancel says why in one line; a follow-up the list does not hold draws nothing.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ChatMessage } from '@/types'

const requestMock = vi.hoisted(() => vi.fn())
const postMock = vi.hoisted(() => vi.fn())
vi.mock('@/lib/api-client', () => ({
  apiClient: {
    request: (...args: unknown[]) => requestMock(...args),
    post: (...args: unknown[]) => postMock(...args),
  },
}))
// Message as a stand-in that shows its text.
vi.mock('@/components/chatbot/message', () => ({
  Message: ({ message }: { message: ChatMessage }) => (
    <div data-testid="the-message">
      {(message.parts ?? []).map((part, index) => ('text' in part ? <p key={index}>{(part as { text: string }).text}</p> : null))}
    </div>
  ),
}))

import { MessageWithReceipts } from '@/components/chatbot/message-with-receipts'
import {
  CANCELLED_LABEL,
  FollowUpLines,
  firedLabel,
  followUpTime,
  pendingLabel,
} from '@/components/proactive/follow-up-line'
import { REFUSED_FALLBACK } from '@/components/proactive/use-suggestion-card'
import {
  FOLLOW_UPS_MAX_WAIT_MS,
  FOLLOW_UPS_POLL_MS,
  FOLLOW_UPS_REFRESH_EVENT,
  followUpsPath,
  nextAskIn,
} from '@/hooks/use-follow-ups'
import { followUpsOf, isFollowUpId, type FollowUp } from '@/lib/proactive/follow-ups'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, ROUNDED_3XL, componentSpans, proactiveSource } from './source-rules'

const CHAT = '4f0c2a1e-8b7d-4e3a-9c5b-1d2e3f4a5b6c'
const AT = '2026-10-31T15:10:00+00:00'
const ID = 7
const REASON = 'See whether the Lisbon flights moved.'
const LIST_PATH = `/api/proactive/follow-ups?chat_id=${CHAT}&status=all`
const CANCEL_PATH = `/api/proactive/follow-ups/${ID}/cancel`
const FIRED_LINE = 'This follow-up already came, so there is nothing to cancel.'
/** Text as the matchers read an element's: any run of spaces (ICU's narrow no-break space in "03:10 PM" too) as one. */
const shown = (text: string) => text.replace(/\s+/g, ' ')
const VIEWER_TIME = shown(new Date(AT).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }))

const row = (status: string, id = ID, at: string | null = AT) => ({ id, at, reason: REASON, status, chat_id: CHAT })
const listOf = (...rows: unknown[]) => ({ follow_ups: rows })
const savedPart = (id: unknown = ID, at: unknown = AT) => ({ type: 'follow_up@1', follow_up_id: id, at })
const auto = (extra: Record<string, unknown> = {}) =>
  ({ id: 'a1', role: 'assistant', parts: [{ type: 'text', text: "I'll check again at 15:10." }, savedPart()], ...extra }) as unknown as ChatMessage

function refusal(status: number, body?: unknown, message = 'refused'): Error {
  return Object.assign(new Error(message), { status, ...(body === undefined ? {} : { body }) })
}

function renderThread(message: ChatMessage = auto(), chatId = CHAT) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MessageWithReceipts chatId={chatId} message={message} setMessages={vi.fn() as never} regenerate={vi.fn()} isReadonly={false} />
    </QueryClientProvider>,
  )
}

const listAsks = () => requestMock.mock.calls.filter(([asked]) => asked === LIST_PATH).length
const CANCEL_NAME = /^Cancel the follow-up at /
const cancelButton = () => screen.queryByRole('button', { name: CANCEL_NAME })
const findCancel = () => screen.findByRole('button', { name: CANCEL_NAME })
const settle = () => act(async () => new Promise<void>((resolve) => setTimeout(resolve, 0)))

beforeEach(() => {
  requestMock.mockReset()
  postMock.mockReset()
  requestMock.mockResolvedValue(listOf(row('pending')))
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('where the follow-up stands', () => {
  it("pending: 'Follow-up at HH:MM · Cancel' under the message, outside its AI-generated block, from the caller's list", async () => {
    renderThread()
    const line = await screen.findByTestId('follow-up-line')
    expect(line).toHaveTextContent(`${pendingLabel(VIEWER_TIME)}·Cancel`)
    expect(line).toHaveAttribute('data-follow-up-status', 'pending')
    expect(cancelButton()).toHaveTextContent('Cancel')
    expect(screen.getByTestId('ai-generated-message')).not.toContainElement(line)
    expect(requestMock).toHaveBeenCalledWith(LIST_PATH)
  })

  it("cancelled: 'Cancelled', with no Cancel", async () => {
    requestMock.mockResolvedValue(listOf(row('cancelled')))
    renderThread()
    expect(await screen.findByTestId('follow-up-line')).toHaveTextContent(CANCELLED_LABEL)
    expect(cancelButton()).toBeNull()
  })

  it("fired: 'Followed up at HH:MM', with no Cancel", async () => {
    requestMock.mockResolvedValue(listOf(row('fired')))
    renderThread()
    expect(await screen.findByTestId('follow-up-line')).toHaveTextContent(firedLabel(VIEWER_TIME))
    expect(cancelButton()).toBeNull()
  })

  it('draws the live followUps the turn_parts frame put on the reply being streamed', async () => {
    renderThread(auto({ parts: [{ type: 'text', text: 'Noted.' }], followUps: [{ followUpId: ID, at: AT }] }))
    expect(await screen.findByTestId('follow-up-line')).toHaveTextContent(pendingLabel(VIEWER_TIME))
  })

  it("never draws the follow-up's reason, and the label is a studio-eyebrow beside a text button, not an ink primary", async () => {
    renderThread()
    const line = await screen.findByTestId('follow-up-line')
    expect(within(line).getByText(pendingLabel(VIEWER_TIME))).toHaveClass('studio-eyebrow')
    expect(cancelButton()?.className).not.toMatch(/\bbg-primary\b/)
    expect(screen.queryByText(REASON)).toBeNull()
  })
})

describe('Cancel', () => {
  it("calls the route once, and the line turns 'Cancelled' from the server's answer", async () => {
    let answer: (value: unknown) => void = () => undefined
    postMock.mockReturnValue(new Promise((resolve) => (answer = resolve)))
    renderThread()
    const cancel = await findCancel()
    // the list asked again after the cancel never answers: the line turns from the cancel's own answer
    requestMock.mockReturnValue(new Promise(() => undefined))
    fireEvent.click(cancel)
    await waitFor(() => expect(cancel).toBeDisabled())
    fireEvent.click(cancel)
    expect(postMock).toHaveBeenCalledTimes(1)
    expect(postMock).toHaveBeenCalledWith(CANCEL_PATH, {})
    await act(async () => answer({ follow_up: row('cancelled') }))
    expect(await screen.findByTestId('follow-up-line')).toHaveTextContent(CANCELLED_LABEL)
    expect(cancelButton()).toBeNull()
    expect(listAsks()).toBe(2)
  })

  it("stays 'Cancelled' when a list fetch that started before the cancel answers late with it still pending", async () => {
    renderThread()
    const cancel = await findCancel()
    let late: (value: unknown) => void = () => undefined
    requestMock.mockReturnValueOnce(new Promise((resolve) => (late = resolve))).mockResolvedValue(listOf(row('cancelled')))
    await act(async () => {
      window.dispatchEvent(new CustomEvent(FOLLOW_UPS_REFRESH_EVENT))
    })
    await waitFor(() => expect(listAsks()).toBe(2))
    postMock.mockResolvedValue({ follow_up: row('cancelled') })
    fireEvent.click(cancel)
    expect(await screen.findByText(CANCELLED_LABEL)).toBeInTheDocument()
    await act(async () => late(listOf(row('pending'))))
    await settle()
    expect(screen.getByTestId('follow-up-line')).toHaveTextContent(CANCELLED_LABEL)
    expect(cancelButton()).toBeNull()
  })

  it("a 409 shows the server's line, and the list asked again says where it stands", async () => {
    postMock.mockRejectedValue(refusal(409, { code: 'follow_up_fired', detail: FIRED_LINE }, FIRED_LINE))
    renderThread()
    const cancel = await findCancel()
    requestMock.mockResolvedValue(listOf(row('fired')))
    fireEvent.click(cancel)
    expect(await screen.findByRole('alert')).toHaveTextContent(FIRED_LINE)
    expect(await screen.findByText(firedLabel(VIEWER_TIME))).toBeInTheDocument()
    expect(cancelButton()).toBeNull()
    expect(postMock).toHaveBeenCalledTimes(1)
  })

  it("a failure with no sentence of its own shows the cards' line, and the follow-up stays pending", async () => {
    postMock.mockRejectedValue(refusal(502, undefined, 'Bad Gateway: upstream connect error'))
    renderThread()
    fireEvent.click(await findCancel())
    expect(await screen.findByRole('alert')).toHaveTextContent(REFUSED_FALLBACK)
    expect(screen.queryByText(/upstream/)).toBeNull()
    expect(cancelButton()).not.toBeNull()
  })
})

describe('the time', () => {
  it("is drawn in the viewer's own locale and zone", async () => {
    const asked: unknown[][] = []
    vi.spyOn(Date.prototype, 'toLocaleTimeString').mockImplementation(function (...args: unknown[]) {
      asked.push(args)
      return args[0] === undefined ? '15.10' : 'not the viewer’s'
    })
    renderThread()
    expect(await screen.findByTestId('follow-up-line')).toHaveTextContent(pendingLabel('15.10'))
    expect(asked[0]).toEqual([undefined, { hour: '2-digit', minute: '2-digit' }])
  })

  it("is the list's time, else the one the message's part names", async () => {
    requestMock.mockResolvedValue(listOf(row('pending', ID, null)))
    renderThread(auto({ parts: [savedPart(ID, '2026-10-31T18:40:00Z')] }))
    expect(await screen.findByTestId('follow-up-line')).toHaveTextContent(shown(pendingLabel(followUpTime('2026-10-31T18:40:00Z'))))
  })
})

describe('nothing to draw', () => {
  it('draws nothing for a follow-up the list does not hold', async () => {
    requestMock.mockResolvedValue(listOf(row('pending', ID + 1)))
    renderThread()
    await waitFor(() => expect(listAsks()).toBe(1))
    await settle()
    expect(screen.queryByTestId('follow-up-line')).toBeNull()
    expect(screen.queryByTestId('follow-up-lines')).toBeNull()
  })

  it('asks nothing for a message that scheduled none, a user message, or a chat id that is not a UUID', async () => {
    const { unmount } = renderThread(auto({ parts: [{ type: 'text', text: 'Done.' }] }))
    unmount()
    const asked = { id: 'u1', role: 'user', parts: [savedPart()] } as unknown as ChatMessage
    renderThread(asked).unmount()
    renderThread(auto(), 'not-a-chat')
    await settle()
    expect(requestMock).not.toHaveBeenCalled()
    expect(screen.queryByTestId('follow-up-line')).toBeNull()
  })
})

describe('the list', () => {
  it('is asked again on automatos:chat-changed', async () => {
    renderThread()
    await screen.findByTestId('follow-up-line')
    expect(listAsks()).toBe(1)
    await act(async () => {
      window.dispatchEvent(new CustomEvent(FOLLOW_UPS_REFRESH_EVENT, { detail: { chat_id: CHAT } }))
    })
    await waitFor(() => expect(listAsks()).toBe(2))
  })

  it('is asked again once the soonest pending follow-up is due, at least a minute and at most six hours on', () => {
    const now = Date.parse(AT)
    const pending = (at: string | null): FollowUp => ({ id: 1, at, reason: '', status: 'pending', chat_id: CHAT })
    const fired: FollowUp = { ...pending(AT), status: 'fired' }
    expect(nextAskIn(undefined, now)).toBe(false)
    expect(nextAskIn([fired], now)).toBe(false)
    expect(nextAskIn([pending('2026-10-31T17:10:00Z')], now)).toBe(2 * 60 * 60_000)
    expect(nextAskIn([pending('2026-10-31T17:10:00Z'), pending('2026-10-31T15:40:00Z')], now)).toBe(30 * 60_000)
    expect(nextAskIn([pending(AT)], now)).toBe(FOLLOW_UPS_POLL_MS)
    expect(nextAskIn([pending(null)], now)).toBe(FOLLOW_UPS_POLL_MS)
    expect(nextAskIn([pending('2026-11-06T15:10:00Z')], now)).toBe(FOLLOW_UPS_MAX_WAIT_MS)
  })

  it("keeps the rows that are follow-ups and leaves out the rest; a chat's path names it once, every status", () => {
    const rows = followUpsOf(listOf(row('pending'), row('fired', 8, 'not a time'), { ...row('pending'), id: '9' }, row('open', 10), null))
    expect(rows.map((kept) => [kept.id, kept.status, kept.at])).toEqual([
      [ID, 'pending', AT],
      [8, 'fired', null],
    ])
    expect(followUpsOf({ follow_ups: 'none' })).toEqual([])
    expect([7, 0, -1, 1.5, '7', Number.MAX_SAFE_INTEGER + 1].map(isFollowUpId)).toEqual([true, false, false, false, false, false])
    expect(followUpsPath(CHAT)).toBe(LIST_PATH)
  })
})

describe('source rules', () => {
  it('tokens only, every component at most 150 lines, and the hook calls the routes through apiClient only', () => {
    const line = proactiveSource('follow-up-line.tsx')
    expect(line).not.toMatch(HEX_COLOUR)
    expect(line).not.toMatch(ORANGE_CLASS)
    expect(line).not.toMatch(ROUNDED_3XL)
    componentSpans(line).forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
    const hook = readFileSync(path.resolve(__dirname, '..', '..', '..', 'hooks', 'use-follow-ups.ts'), 'utf8')
    expect(hook).not.toMatch(/\bfetch\(/)
    expect(hook).toMatch(/apiClient\.request/)
    expect(hook).toMatch(/apiClient\.post/)
  })

  it('draws the line through message-with-receipts.tsx and thread-cards.tsx, never in message.tsx, chat.tsx or the box', () => {
    const chatbot = (file: string) => readFileSync(path.resolve(__dirname, '../../chatbot', file), 'utf8')
    expect(proactiveSource('thread-cards.tsx')).toContain("from './follow-up-line'")
    expect(chatbot('message-with-receipts.tsx')).toContain('followUpsOfMessage')
    for (const file of ['message.tsx', 'chat.tsx', 'multimodal-input.tsx']) {
      expect(chatbot(file)).not.toMatch(/follow_up@1|follow-up-line|followUps|FollowUpLines/)
    }
  })

  it('draws through FollowUpLines with nothing else around it when the list holds none', async () => {
    requestMock.mockResolvedValue(listOf())
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { container } = render(
      <QueryClientProvider client={client}>
        <FollowUpLines chatId={CHAT} followUps={[{ followUpId: ID, at: AT }]} />
      </QueryClientProvider>,
    )
    await waitFor(() => expect(listAsks()).toBe(1))
    await settle()
    expect(container).toBeEmptyDOMElement()
  })
})
