/**
 * PRD-268 §6.11.5, §6.12.3, K12, K17, K23 (US-C21d) — CalibrationCard, from its props: up to three
 * lines, each with its source count (never a percentage) and Reject named by the line; three
 * offers; Sounds right (the one ink primary) and Something's off; the empty state with the offers
 * as starters; a rejected line struck through with "I'll forget that"; "✓ Sounds right" once
 * accepted; a refused answer's one line; data-ai-generated on the lines and the offers and not on
 * the controls; the AI caption once; and the same parts in Studio light, Studio dark and Classic.
 */
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const postMock = vi.hoisted(() => vi.fn())
// The server offers no calibration (CALIBRATION_ENABLED ships off): its read says so, and no answer is sent.
vi.mock('@/lib/api-client', () => ({ apiClient: { request: async () => ({ available: false }), post: postMock } }))

import {
  CALIBRATION_EMPTY,
  CALIBRATION_TITLE,
  CalibrationCard,
  SOMETHING_OFF,
  SOUNDS_RIGHT,
} from '@/components/proactive/calibration-card'
import { FORGET_THAT, OFFERS_LABEL, sourceCountOf } from '@/components/proactive/calibration-parts'
import { REFUSED_FALLBACK } from '@/components/proactive/use-suggestion-card'
import { CALIBRATION_UNAVAILABLE, useAnswerCalibration } from '@/hooks/use-calibration'
import type { Calibration, CalibrationSource } from '@/lib/proactive/types'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, ROUNDED_3XL, componentSpans, proactiveSource } from './source-rules'

const AI_CAPTION = 'Auto is an AI'
const BAKERY = 'You run a bakery in Lisbon.'
const NEWSLETTER = 'You send a newsletter every Friday.'

const deliverables = (count: number): CalibrationSource[] =>
  Array.from({ length: count }, (_, index) => ({ type: 'deliverable', id: `d${index}`, label: `Menu ${index}` }))

function calibration(overrides: Partial<Calibration> = {}): Calibration {
  return {
    v: 1,
    id: 'cal-1',
    lines: [
      { id: 'l1', text: BAKERY, sources: [...deliverables(6), { type: 'conversation', id: 'c1', label: 'Rent' }, { type: 'conversation', id: 'c2', label: 'Flour' }] },
      { id: 'l2', text: NEWSLETTER, sources: [{ type: 'document', id: 'k1', label: 'Newsletter plan' }] },
      { id: 'l3', text: 'You hire in spring.', sources: [{ type: 'skill', id: 's1', label: 'Hiring' }] },
      { id: 'l4', text: 'A fourth line is never shown.', sources: [] },
    ],
    offers: ['Draft Friday’s newsletter', 'Plan the spring hire', 'Price the new menu', 'A fourth offer'],
    empty: false,
    ...overrides,
  }
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

const lineOf = (container: HTMLElement, id: string) => container.querySelector(`li[data-line-id="${id}"]`) as HTMLElement

/** An answer the test settles: the last one sent resolves when the test says so. */
function deferredAnswer() {
  const pending: { resolve?: () => void } = {}
  const onAnswer = vi.fn(
    () =>
      new Promise<void>((resolve) => {
        pending.resolve = resolve
      }),
  )
  return { onAnswer, resolve: () => pending.resolve?.() }
}

/** Lets the card's answer chain run to its end. */
const settle = () => act(async () => new Promise<void>((resolve) => setTimeout(resolve, 0)))

/** The card over the hook while the server offers none (K15): every answer is refused, and nothing is sent. */
function InertCalibration() {
  const answering = useAnswerCalibration()
  return <CalibrationCard calibration={calibration()} onAnswer={answering.mutate} />
}

function drawInert() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <InertCalibration />
    </QueryClientProvider>,
  )
}

describe('CalibrationCard parts', () => {
  it('shows up to three lines, each with its source count, and three offers', () => {
    const { container } = render(<CalibrationCard calibration={calibration()} />)
    expect(screen.getByRole('article', { name: CALIBRATION_TITLE })).toHaveAttribute('data-state', 'open')
    expect(container.querySelectorAll('li[data-line-id]')).toHaveLength(3)
    expect(within(lineOf(container, 'l1')).getByText('from 6 Deliverables and 2 Conversations')).toBeInTheDocument()
    expect(within(lineOf(container, 'l2')).getByText('from 1 document')).toBeInTheDocument()
    expect(screen.queryByText('A fourth line is never shown.')).toBeNull()
    const offers = screen.getByRole('list', { name: OFFERS_LABEL })
    expect(within(offers).getAllByRole('listitem').map((item) => item.textContent)).toEqual([
      'Draft Friday’s newsletter',
      'Plan the spring hire',
      'Price the new menu',
    ])
  })

  it('shows no percentage anywhere', () => {
    const { container } = render(<CalibrationCard calibration={calibration()} />)
    expect(container.textContent).not.toMatch(/\d\s*%/)
  })

  it('names each Reject by its line, and makes Sounds right the one ink primary', () => {
    render(<CalibrationCard calibration={calibration()} onAnswer={vi.fn()} onSomethingOff={vi.fn()} />)
    expect(screen.getByRole('button', { name: `Reject: ${BAKERY}` })).toBeEnabled()
    expect(screen.getByRole('button', { name: `Reject: ${NEWSLETTER}` })).toBeEnabled()
    const primaries = screen.getAllByRole('button').filter((button) => button.classList.contains('bg-primary'))
    expect(primaries).toEqual([screen.getByRole('button', { name: SOUNDS_RIGHT })])
    expect(screen.getByRole('button', { name: SOMETHING_OFF })).toBeEnabled()
  })

  it("marks the lines and the offers as Auto's words, and no control; the AI caption once", () => {
    render(<CalibrationCard calibration={calibration()} onAnswer={vi.fn()} onSomethingOff={vi.fn()} />)
    expect(screen.getByText(BAKERY)).toHaveAttribute('data-ai-generated', 'true')
    expect(screen.getByText('Plan the spring hire')).toHaveAttribute('data-ai-generated', 'true')
    screen.getAllByRole('button').forEach((button) => expect(button.closest('[data-ai-generated]')).toBeNull())
    const captions = screen.getAllByRole('note', { name: AI_CAPTION })
    expect(captions).toHaveLength(1)
    expect(captions[0].closest('[data-ai-generated]')).toBeNull()
  })

  it("draws the model's words as text, never markup", () => {
    const text = '<img src=x onerror=alert(1)>'
    const { container } = render(<CalibrationCard calibration={calibration({ lines: [{ id: 'x', text, sources: [] }] })} />)
    expect(container.querySelector('img')).toBeNull()
    expect(screen.getByText(text)).toBeInTheDocument()
  })

  it('counts each source kind in the order it first appears, and nothing for no sources', () => {
    expect(sourceCountOf(deliverables(1))).toBe('from 1 Deliverable')
    expect(sourceCountOf([{ type: 'skill', id: 's', label: 'x' }, ...deliverables(2), { type: 'skill', id: 't', label: 'y' }])).toBe(
      'from 2 skills and 2 Deliverables',
    )
    expect(sourceCountOf([])).toBeNull()
  })
})

describe('CalibrationCard, empty', () => {
  it('says there is nothing to go on and offers three starters, with no answers to give', () => {
    render(<CalibrationCard calibration={calibration({ lines: [], empty: true })} onAnswer={vi.fn()} />)
    expect(screen.getByRole('article', { name: CALIBRATION_TITLE })).toHaveAttribute('data-state', 'empty')
    expect(screen.getByText(CALIBRATION_EMPTY)).toBeInTheDocument()
    expect(within(screen.getByRole('list', { name: OFFERS_LABEL })).getAllByRole('listitem')).toHaveLength(3)
    expect(screen.queryByRole('button', { name: SOUNDS_RIGHT })).toBeNull()
  })
})

describe('CalibrationCard answers', () => {
  it('sends a line’s Reject, then strikes the line through with "I\'ll forget that"', async () => {
    const answer = deferredAnswer()
    const { container } = render(<CalibrationCard calibration={calibration()} onAnswer={answer.onAnswer} />)
    fireEvent.click(screen.getByRole('button', { name: `Reject: ${BAKERY}` }))
    await settle()
    expect(answer.onAnswer).toHaveBeenCalledWith({ calibration_id: 'cal-1', answer: 'reject', line_id: 'l1' })
    expect(lineOf(container, 'l1')).toHaveAttribute('data-state', 'open')
    answer.resolve()
    await settle()
    const line = lineOf(container, 'l1')
    expect(line).toHaveAttribute('data-state', 'rejected')
    expect(within(line).getByText(BAKERY)).toHaveClass('line-through')
    expect(within(line).getByText(FORGET_THAT)).toBeInTheDocument()
    expect(within(line).queryByRole('button')).toBeNull()
  })

  it('sends Sounds right as accept, then says "✓ Sounds right" in place', async () => {
    const answer = deferredAnswer()
    render(<CalibrationCard calibration={calibration()} onAnswer={answer.onAnswer} />)
    fireEvent.click(screen.getByRole('button', { name: SOUNDS_RIGHT }))
    await settle()
    expect(answer.onAnswer).toHaveBeenCalledWith({ calibration_id: 'cal-1', answer: 'accept' })
    expect(screen.getByRole('button', { name: SOUNDS_RIGHT })).toBeDisabled()
    answer.resolve()
    await settle()
    expect(screen.getByRole('article', { name: CALIBRATION_TITLE })).toHaveAttribute('data-state', 'accepted')
    expect(screen.getByRole('status')).toHaveTextContent(`✓ ${SOUNDS_RIGHT}`)
    expect(screen.queryByRole('button', { name: /^Reject:/ })).toBeNull()
  })

  it('leaves the card as it was, with one line, when the answer is refused', async () => {
    const { container } = drawInert()
    fireEvent.click(screen.getByRole('button', { name: `Reject: ${BAKERY}` }))
    await settle()
    expect(screen.getByRole('alert')).toHaveTextContent(REFUSED_FALLBACK)
    expect(screen.queryByText(CALIBRATION_UNAVAILABLE)).toBeNull()
    expect(lineOf(container, 'l1')).toHaveAttribute('data-state', 'open')
    expect(screen.getByRole('button', { name: `Reject: ${BAKERY}` })).toBeEnabled()
    expect(console.warn).toHaveBeenCalled()
    expect(postMock).not.toHaveBeenCalled()
  })

  it('starts a new calibration with none of the last one’s answers', async () => {
    const answer = deferredAnswer()
    const { container, rerender } = render(<CalibrationCard calibration={calibration()} onAnswer={answer.onAnswer} />)
    fireEvent.click(screen.getByRole('button', { name: `Reject: ${BAKERY}` }))
    await settle()
    answer.resolve()
    await settle()
    expect(lineOf(container, 'l1')).toHaveAttribute('data-state', 'rejected')
    rerender(<CalibrationCard calibration={calibration({ id: 'cal-2' })} onAnswer={answer.onAnswer} />)
    expect(lineOf(container, 'l1')).toHaveAttribute('data-state', 'open')
    expect(screen.getByRole('button', { name: `Reject: ${BAKERY}` })).toBeEnabled()
    expect(screen.getByRole('button', { name: SOUNDS_RIGHT })).toBeEnabled()
  })

  it('calls Something’s off on its press', () => {
    const onSomethingOff = vi.fn()
    render(<CalibrationCard calibration={calibration()} onSomethingOff={onSomethingOff} />)
    fireEvent.click(screen.getByRole('button', { name: SOMETHING_OFF }))
    expect(onSomethingOff).toHaveBeenCalledTimes(1)
  })

  it('holds the answers when the surface gives no way to send them', () => {
    render(<CalibrationCard calibration={calibration()} />)
    expect(screen.getByRole('button', { name: SOUNDS_RIGHT })).toBeDisabled()
    expect(screen.getByRole('button', { name: SOMETHING_OFF })).toBeDisabled()
    expect(screen.queryByRole('button', { name: /^Reject:/ })).toBeNull()
  })
})

const LOOKS = [
  { name: 'Studio light', ground: 'studio' },
  { name: 'Studio dark', ground: 'studio dark' },
  { name: 'Classic', ground: '' },
] as const

describe.each(LOOKS)('CalibrationCard in $name', ({ ground }) => {
  it('draws every part from the same token classes (each look\'s token file paints them)', () => {
    render(
      <div className={ground}>
        <CalibrationCard calibration={calibration()} onAnswer={vi.fn()} onSomethingOff={vi.fn()} />
      </div>,
    )
    expect(screen.getByText(CALIBRATION_TITLE)).toHaveClass('studio-eyebrow')
    expect(screen.getByRole('article', { name: CALIBRATION_TITLE })).toHaveClass('bg-card', 'border-border')
    expect(screen.getByRole('button', { name: SOUNDS_RIGHT })).toHaveClass('bg-primary', 'text-primary-foreground')
  })
})

describe('calibration card source', () => {
  const files = ['calibration-card.tsx', 'calibration-parts.tsx']

  it.each(files)('%s uses tokens only: no hex, no orange, no rounded-3xl', (file) => {
    const source = proactiveSource(file)
    expect(source).not.toMatch(HEX_COLOUR)
    expect(source).not.toMatch(ORANGE_CLASS)
    expect(source).not.toMatch(ROUNDED_3XL)
  })

  it(`keeps each component at most ${MAX_COMPONENT_LINES} lines`, () => {
    const spans = files.flatMap((file) => componentSpans(proactiveSource(file)))
    expect(spans.map((span) => span.name)).toEqual(
      expect.arrayContaining(['CalibrationCard', 'CalibrationAnswers', 'CalibrationLineItem', 'CalibrationOffers']),
    )
    spans.forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
  })
})
