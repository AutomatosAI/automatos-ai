/**
 * PRD-268 §6.11.5, §6.12.3, K15, K19 (US-W06) — the calibration@1 part an Auto message carries draws as W2's
 * CalibrationCard under it (thread-cards.tsx, through message-with-receipts.tsx), outside its AI-generated block, with
 * its live state from GET /api/auto/calibration: each line with its source count and Reject (struck through with
 * "I'll forget that" once rejected), the three offers, Sounds right (accept) and Something's off, which sends
 * "Something's off" as the person's own message through the reply-chip path. The state survives a reload. No
 * percentage anywhere. The lines and offers carry data-ai-generated; the controls do not. `Message` gets neither the
 * part nor the text part the card holds.
 */
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
// Message as a stand-in that shows the part types and text it was handed; the composer's box as a stand-in.
vi.mock('@/components/chatbot/message', () => ({
  Message: ({ message }: { message: ChatMessage }) => (
    <div data-testid="the-message" data-part-types={(message.parts ?? []).map((part) => String((part as { type: unknown }).type)).join(',')}>
      {(message.parts ?? []).map((part, index) => ('text' in part ? <p key={index}>{(part as { text: string }).text}</p> : null))}
    </div>
  ),
}))
vi.mock('@/components/chatbot/multimodal-input', () => ({
  MultimodalInput: ({ chatId }: { chatId: string }) => <div data-testid="the-composer" data-chat-id={chatId} />,
}))
vi.mock('@/components/proactive/pinned-suggestion', () => ({ PinnedSuggestion: () => null }))

import { MessageWithReceipts } from '@/components/chatbot/message-with-receipts'
import { CALIBRATION_TITLE, SOMETHING_OFF, SOUNDS_RIGHT } from '@/components/proactive/calibration-card'
import { FORGET_THAT } from '@/components/proactive/calibration-parts'
import { MultimodalInput } from '@/components/proactive/composer-with-suggestion'
import { replyChipMessage } from '@/components/proactive/reply-chip-event'
import { CALIBRATION_PATH } from '@/hooks/use-calibration'
import { threadPartsOf } from '@/lib/proactive/thread-parts'

const CHAT = '5b7f2c1e-8d4a-4f6b-9c3e-1a2b3c4d5e6f'
const CAL_ID = '3f2a1b0c-9d8e-4f7a-b6c5-d4e3f2a1b0c9'
const OTHER_CAL = '0b6f3e21-94c7-4d0a-8e15-6a2d9c7f4b33'
const BAKERY = 'You run a bakery in Lisbon.'
const NEWSLETTER = 'You send a newsletter on Fridays.'
const TEXT = `Here is what I think I know about your work so far:\n- ${BAKERY}\n- ${NEWSLETTER}`
const OFFERS = ['Draft the newsletter', 'Plan the spring hire', 'Price the new menu']

const deliverables = (count: number) => Array.from({ length: count }, (_, index) => ({ type: 'deliverable', id: `d${index}`, label: `Menu ${index}` }))

const CALIBRATION = {
  v: 1,
  id: CAL_ID,
  lines: [
    { id: 'line-1', text: BAKERY, sources: deliverables(6) },
    { id: 'line-2', text: NEWSLETTER, sources: [{ type: 'conversation', id: 'c1', label: 'Friday' }, { type: 'conversation', id: 'c2', label: 'List' }] },
    { id: 'line-3', text: 'You are 90% sure about the menu.', sources: deliverables(1) },
  ],
  offers: OFFERS,
  empty: false,
}

function view(overrides: Record<string, unknown> = {}) {
  return {
    available: true, offer: false, consent_line: 'Auto reads the titles of your recent work to say what it knows.',
    status: 'open', calibration: CALIBRATION, rejected_lines: [], accepted_lines: [], ...overrides,
  }
}

const calibrationMessage = (part: unknown = { type: 'calibration@1', calibration: CALIBRATION }) =>
  ({ id: 'm1', role: 'assistant', parts: [{ type: 'text', text: TEXT }, part], source: { origin: 'calibration', calibration_id: CAL_ID } }) as unknown as ChatMessage

const quiet = { log: () => undefined, warn: () => undefined, error: () => undefined }
let sendMessage: ReturnType<typeof vi.fn>

/** The chat as chat.tsx draws it: the message through MessageWithReceipts, then the composer unless read-only. */
function draw(message: ChatMessage = calibrationMessage(), readOnly = false, ground = '') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } }, logger: quiet })
  return render(
    <QueryClientProvider client={client}>
      <div className={ground}>
        <MessageWithReceipts chatId={CHAT} message={message} setMessages={vi.fn() as never} regenerate={vi.fn()} isReadonly={readOnly} />
        {readOnly ? null : <MultimodalInput chatId={CHAT} status="ready" stop={vi.fn()} sendMessage={sendMessage} selectedVisibilityType="private" />}
      </div>
    </QueryClientProvider>,
  )
}

const card = () => screen.getByRole('article', { name: CALIBRATION_TITLE })
const lineOf = (id: string) => card().querySelector(`li[data-line-id="${id}"]`) as HTMLElement
const reject = (text: string) => screen.getByRole('button', { name: `Reject: ${text}` })
/** The answers wait for the live view: Reject shows once it says the calibration is the person's own. */
const ready = () => screen.findByRole('button', { name: `Reject: ${BAKERY}` })

beforeEach(() => {
  requestMock.mockReset()
  postMock.mockReset()
  sendMessage = vi.fn()
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('the calibration card from the part', () => {
  it("draws the card under the message, outside its AI-generated block; Message gets neither the part nor the text", async () => {
    requestMock.mockResolvedValue(view())
    draw()
    await ready()
    const holder = screen.getByTestId('ai-generated-message')
    expect(holder).not.toContainElement(card())
    expect(screen.getByTestId('thread-calibration')).toContainElement(card())
    expect(screen.getByTestId('the-message')).toHaveAttribute('data-part-types', '')
    expect(within(card()).getByText(BAKERY)).toBeInTheDocument()
    expect(within(card()).getByText('from 6 Deliverables')).toBeInTheDocument()
    expect(within(card()).getByText('from 2 Conversations')).toBeInTheDocument()
    OFFERS.forEach((offer) => expect(within(card()).getByText(offer)).toBeInTheDocument())
    expect(screen.getByRole('button', { name: SOUNDS_RIGHT })).toBeEnabled()
    expect(screen.getByRole('button', { name: SOMETHING_OFF })).toBeEnabled()
  })

  it("shows no percentage anywhere: a line carrying one is left out", async () => {
    requestMock.mockResolvedValue(view())
    draw()
    await ready()
    expect(card().textContent).not.toContain('%')
    expect(screen.queryByText(/90/)).toBeNull()
    expect(card().querySelectorAll('li[data-line-id]')).toHaveLength(2)
  })

  it("marks the lines and the offers as Auto's words, and no control", async () => {
    requestMock.mockResolvedValue(view())
    draw()
    await ready()
    expect(within(card()).getByText(BAKERY)).toHaveAttribute('data-ai-generated', 'true')
    expect(within(card()).getByText(OFFERS[0])).toHaveAttribute('data-ai-generated', 'true')
    within(card()).getAllByRole('button').forEach((button) => {
      expect(button).not.toHaveAttribute('data-ai-generated')
      expect(button.closest('[data-ai-generated]')).toBeNull()
    })
  })

  it("keeps the text when the part isn't a Calibration@1, and draws no card", () => {
    const broken = threadPartsOf(calibrationMessage({ type: 'calibration@1', calibration: { v: 2 } }))
    expect(broken.calibration).toBeUndefined()
    expect(broken.message.parts).toEqual([{ type: 'text', text: TEXT }])
    const whole = threadPartsOf(calibrationMessage())
    expect(whole.calibration?.id).toBe(CAL_ID)
    expect(whole.message.parts).toEqual([])
  })
})

describe('the answers', () => {
  it('Reject posts the line’s reject, then strikes it through with "I\'ll forget that"', async () => {
    requestMock.mockResolvedValueOnce(view()).mockResolvedValue(view({ rejected_lines: ['line-1'] }))
    postMock.mockResolvedValue({ ...view({ rejected_lines: ['line-1'] }), not_stored: [], note: null })
    draw()
    fireEvent.click(await ready())
    await waitFor(() => expect(lineOf('line-1')).toHaveAttribute('data-state', 'rejected'))
    expect(postMock).toHaveBeenCalledTimes(1)
    expect(postMock).toHaveBeenCalledWith(`${CALIBRATION_PATH}/${CAL_ID}/lines/line-1/reject`, {})
    expect(within(lineOf('line-1')).getByText(FORGET_THAT)).toBeInTheDocument()
    expect(reject(NEWSLETTER)).toBeEnabled()
  })

  it('Sounds right posts accept, then says "✓ Sounds right"; a line kept out of memory says so', async () => {
    const note = 'Not kept in memory: it looks like it holds something private, such as a password or a key.'
    requestMock.mockResolvedValueOnce(view()).mockResolvedValue(view({ status: 'answered', accepted_lines: ['line-1'] }))
    postMock.mockResolvedValue({ ...view({ status: 'answered', accepted_lines: ['line-1'] }), not_stored: ['line-2'], note })
    draw()
    await ready()
    fireEvent.click(screen.getByRole('button', { name: SOUNDS_RIGHT }))
    await waitFor(() => expect(card()).toHaveAttribute('data-state', 'accepted'))
    expect(postMock).toHaveBeenCalledWith(`${CALIBRATION_PATH}/${CAL_ID}/accept`, {})
    expect(within(card()).getByRole('status')).toHaveTextContent(`✓ ${SOUNDS_RIGHT}`)
    expect(screen.queryByRole('button', { name: SOUNDS_RIGHT })).toBeNull()
    expect(await screen.findByText(note)).toBeInTheDocument()
  })

  it("Something's off sends \"Something's off\" as the person's own message, and the lines stay rejectable", async () => {
    requestMock.mockResolvedValue(view())
    draw()
    await ready()
    fireEvent.click(screen.getByRole('button', { name: SOMETHING_OFF }))
    expect(sendMessage).toHaveBeenCalledTimes(1)
    expect(sendMessage).toHaveBeenCalledWith(replyChipMessage(SOMETHING_OFF))
    expect(reject(BAKERY)).toBeEnabled()
    expect(postMock).not.toHaveBeenCalled()
  })

  it("holds Something's off on a read-only chat, which has no composer", async () => {
    requestMock.mockResolvedValue(view())
    draw(calibrationMessage(), true)
    await ready()
    expect(screen.getByRole('button', { name: SOMETHING_OFF })).toBeDisabled()
  })
})

describe('the live state', () => {
  it('survives a reload: the lines rejected and Sounds right come from the server’s record', async () => {
    requestMock.mockResolvedValue(view({ status: 'answered', rejected_lines: ['line-2'], accepted_lines: ['line-1'] }))
    draw()
    await waitFor(() => expect(card()).toHaveAttribute('data-state', 'accepted'))
    expect(lineOf('line-2')).toHaveAttribute('data-state', 'rejected')
    expect(within(lineOf('line-2')).getByText(FORGET_THAT)).toBeInTheDocument()
    expect(lineOf('line-1')).toHaveAttribute('data-state', 'open')
    expect(screen.queryByRole('button', { name: /^Reject: / })).toBeNull()
    expect(within(card()).getAllByRole('status').map((node) => node.textContent)).toContain(`✓ ${SOUNDS_RIGHT}`)
    expect(postMock).not.toHaveBeenCalled()
  })

  it("offers no Reject or Sounds right for a calibration that isn't the viewer's, or once the server offers none", async () => {
    requestMock.mockResolvedValue(view({ calibration: { ...CALIBRATION, id: OTHER_CAL } }))
    const { unmount } = draw()
    await waitFor(() => expect(requestMock).toHaveBeenCalled())
    await act(async () => new Promise<void>((resolve) => setTimeout(resolve, 0)))
    expect(screen.queryByRole('button', { name: /^Reject: / })).toBeNull()
    expect(screen.getByRole('button', { name: SOUNDS_RIGHT })).toBeDisabled()
    unmount()

    requestMock.mockReset()
    requestMock.mockResolvedValue({ ...view(), available: false })
    draw()
    await waitFor(() => expect(requestMock).toHaveBeenCalled())
    await act(async () => new Promise<void>((resolve) => setTimeout(resolve, 0)))
    expect(within(card()).getByText(BAKERY)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^Reject: / })).toBeNull()
    expect(postMock).not.toHaveBeenCalled()
  })
})

const LOOKS = [
  { name: 'Studio light', ground: 'studio' },
  { name: 'Studio dark', ground: 'studio dark' },
  { name: 'Classic', ground: '' },
] as const

describe.each(LOOKS)('the calibration card in $name', ({ ground }) => {
  it("draws from the token classes each look's token file paints; the lines are Auto's words, the controls not", async () => {
    requestMock.mockResolvedValue(view())
    draw(calibrationMessage(), false, ground)
    await ready()
    expect(card()).toHaveClass('bg-card', 'border-border')
    expect(within(card()).getByText(CALIBRATION_TITLE)).toHaveClass('studio-eyebrow')
    expect(screen.getByRole('button', { name: SOUNDS_RIGHT })).toHaveClass('bg-primary', 'text-primary-foreground')
    expect(within(card()).getByText(BAKERY)).toHaveAttribute('data-ai-generated', 'true')
    expect(reject(BAKERY)).not.toHaveAttribute('data-ai-generated')
  })
})
