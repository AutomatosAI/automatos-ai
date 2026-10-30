/**
 * PRD-268 §6.2.5, §6.12.3, K17 (US-C21c) — SuggestionCard: "Auto suggests", the text in Auto's
 * voice, what Start does, its credits when known, Start and Hide (named by the suggestion) with the
 * reason menu; every state (open, starting, started for 4 s, hidden, refused, expired); the pinned
 * one-line variant; data-ai-generated on Auto's words and not on the controls; the AI caption once.
 */
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

type Mutation = { mutate: ReturnType<typeof vi.fn>; reset: ReturnType<typeof vi.fn>; isLoading: boolean; isError: boolean; error: unknown }
const hooks = vi.hoisted(() => ({ start: null as unknown as Mutation, hide: null as unknown as Mutation }))
vi.mock('@/hooks/use-suggestions', () => ({
  useStartSuggestion: () => hooks.start,
  useHideSuggestion: () => hooks.hide,
}))

import { SuggestionCard } from '@/components/proactive/suggestion-card'
import { REFUSED_FALLBACK, STARTED_LINE, STARTED_MS } from '@/components/proactive/use-suggestion-card'
import type { Suggestion } from '@/lib/proactive/types'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, ROUNDED_3XL, componentSpans, proactiveSource } from './source-rules'

const ID = '7d1c0a52-3e0b-4c55-9a1e-2f6f0c1d9b10'
const TITLE = 'Look at the Lisbon flights'
const AI_CAPTION = 'Auto is an AI'

function suggestion(overrides: Partial<Suggestion> = {}): Suggestion {
  return {
    id: ID, workspace_id: 'ws-1', user_id: 1, source_type: 'check_in', source_agent_id: 4, run_id: null,
    title: TITLE, detail: 'Three runs failed since Monday.', evidence: [{ type: 'ticket', id: '0419' }],
    proposed_action: { kind: 'board_task', payload: {} }, subject_key: 'ticket:0419', severity: 'today',
    status: 'open', occurrences: 1, last_seen_at: null, started_ref: null, decided_via: null, hide_reason: null,
    decided_at: null, est_credits: null, created_at: null, delivered_at: null, expires_at: null,
    ...overrides,
  }
}

const mutation = (): Mutation => ({ mutate: vi.fn(), reset: vi.fn(), isLoading: false, isError: false, error: null })
const startButton = () => screen.getByRole('button', { name: `Start: ${TITLE}` })
const hideButton = () => screen.getByRole('button', { name: `Hide: ${TITLE}` })

/** Answer the last Start or Hide as the server would have. */
function succeed(target: Mutation, answer: unknown = {}) {
  const calls = target.mutate.mock.calls
  const options = calls[calls.length - 1][1] as { onSuccess: (answer: unknown) => void }
  act(() => options.onSuccess(answer))
}

beforeEach(() => {
  hooks.start = mutation()
  hooks.hide = mutation()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('SuggestionCard, open', () => {
  it('shows Auto suggests, the text, what Start does, Start and Hide, and the AI caption once', () => {
    render(<SuggestionCard suggestion={suggestion()} />)
    const card = screen.getByRole('article', { name: `Auto suggests: ${TITLE}` })
    expect(card).toHaveAttribute('data-state', 'open')
    expect(within(card).getByText('Auto suggests')).toHaveClass('studio-eyebrow')
    expect(screen.getByText(TITLE).closest('[data-ai-generated="true"]')).not.toBeNull()
    expect(screen.getByText('Three runs failed since Monday.').closest('[data-ai-generated="true"]')).not.toBeNull()
    expect(screen.getByText('Start does: files a ticket on the Board for an agent to work.')).toBeInTheDocument()
    expect(startButton()).toBeEnabled()
    expect(hideButton()).toBeEnabled()
    expect(screen.getAllByRole('note', { name: AI_CAPTION })).toHaveLength(1)
  })

  it('makes Start the one ink primary', () => {
    render(<SuggestionCard suggestion={suggestion()} />)
    const primaries = screen.getAllByRole('button').filter((button) => button.classList.contains('bg-primary'))
    expect(primaries).toHaveLength(1)
    expect(primaries[0]).toBe(startButton())
  })

  it('says what Start does for a chat', () => {
    render(<SuggestionCard suggestion={suggestion({ proposed_action: { kind: 'chat', payload: null } })} />)
    expect(screen.getByText('Start does: opens a chat with Auto about this.')).toBeInTheDocument()
  })

  it('shows about N credits once est_credits is known, and nothing about credits before', () => {
    const { rerender } = render(<SuggestionCard suggestion={suggestion()} />)
    expect(screen.queryByText(/credits/)).toBeNull()
    rerender(<SuggestionCard suggestion={suggestion({ est_credits: 12 })} />)
    expect(screen.getByText('(about 12 credits)')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: `Start: ${TITLE} (about 12 credits)` })).toBeInTheDocument()
  })

  it('draws what a surface adds under the text', () => {
    render(
      <SuggestionCard suggestion={suggestion()}>
        <a href="/activity?task=0419">Ticket 0419</a>
      </SuggestionCard>,
    )
    expect(screen.getByRole('link', { name: 'Ticket 0419' })).toHaveAttribute('href', '/activity?task=0419')
  })

  it("draws the model's words as text, never markup", () => {
    const title = '<img src=x onerror=alert(1)>'
    const { container } = render(<SuggestionCard suggestion={suggestion({ title })} />)
    expect(container.querySelector('img')).toBeNull()
    expect(screen.getByText(title)).toBeInTheDocument()
  })

  it("keeps data-ai-generated on Auto's words and off every control", () => {
    render(<SuggestionCard suggestion={suggestion({ est_credits: 3 })} />)
    fireEvent.click(hideButton())
    screen.getAllByRole('button').forEach((button) => {
      expect(button).not.toHaveAttribute('data-ai-generated')
      expect(button.closest('[data-ai-generated]')).toBeNull()
    })
    expect(screen.getByRole('note', { name: AI_CAPTION }).closest('[data-ai-generated]')).toBeNull()
  })
})

describe('SuggestionCard, Start', () => {
  it('posts Start with the suggestion and where it was pressed', () => {
    render(<SuggestionCard suggestion={suggestion()} via="chat" />)
    fireEvent.click(startButton())
    expect(hooks.start.mutate).toHaveBeenCalledWith({ id: ID, via: 'chat' }, expect.objectContaining({ onSuccess: expect.any(Function) }))
  })

  it('holds Start and Hide while starting', () => {
    hooks.start.isLoading = true
    render(<SuggestionCard suggestion={suggestion()} />)
    expect(screen.getByRole('article')).toHaveAttribute('data-state', 'starting')
    expect(startButton()).toBeDisabled()
    expect(startButton()).toHaveAttribute('aria-busy', 'true')
    expect(hideButton()).toBeDisabled()
  })

  it('says "Started. Auto is on it." in place for 4 s, hands over what Start made, then settles on one line', () => {
    vi.useFakeTimers()
    const onStarted = vi.fn()
    const answer = { suggestion: suggestion({ status: 'started' }), opens: { type: 'board_task', board_task_id: 419 } }
    render(<SuggestionCard suggestion={suggestion()} onStarted={onStarted} />)
    fireEvent.click(startButton())
    succeed(hooks.start, answer)
    expect(screen.getByRole('status')).toHaveTextContent(STARTED_LINE)
    expect(screen.getByRole('article')).toHaveAttribute('data-state', 'just_started')
    expect(screen.queryByRole('button', { name: `Start: ${TITLE}` })).toBeNull()
    expect(onStarted).toHaveBeenCalledWith(answer)
    act(() => vi.advanceTimersByTime(STARTED_MS - 1))
    expect(screen.getByRole('status')).toHaveTextContent(STARTED_LINE)
    act(() => vi.advanceTimersByTime(1))
    expect(screen.queryByText(STARTED_LINE)).toBeNull()
    expect(screen.getByText('Started').parentElement).toHaveAttribute('data-state', 'started')
  })

  it("keeps the card put with the server's one line when Start is refused", () => {
    hooks.start.isError = true
    hooks.start.error = Object.assign(new Error("Today's spend ceiling is reached."), {
      status: 402,
      body: { code: 'spend_ceiling', detail: "Today's spend ceiling is reached." },
    })
    render(<SuggestionCard suggestion={suggestion()} />)
    expect(screen.getByRole('alert')).toHaveTextContent("Today's spend ceiling is reached.")
    expect(screen.getByRole('article')).toHaveAttribute('data-state', 'open')
    expect(startButton()).toBeEnabled()
  })

  it("says the server's sentence for any refusal below 500, with or without a code", () => {
    hooks.start.isError = true
    hooks.start.error = Object.assign(new Error('That suggestion was already decided.'), {
      status: 409,
      body: { detail: 'That suggestion was already decided.' },
    })
    render(<SuggestionCard suggestion={suggestion()} />)
    expect(screen.getByRole('alert')).toHaveTextContent('That suggestion was already decided.')
  })

  it.each([
    ['a transport failure', new Error('Failed to fetch'), 'Failed to fetch'],
    ['a server error', Object.assign(new Error('Internal Server Error'), { status: 500, body: { detail: 'Internal Server Error' } }), 'Internal Server Error'],
  ])("says the card's own line, not the words of %s", (_case, error, words) => {
    hooks.start.isError = true
    hooks.start.error = error
    render(<SuggestionCard suggestion={suggestion()} />)
    expect(screen.getByRole('alert')).toHaveTextContent(REFUSED_FALLBACK)
    expect(screen.queryByText(words)).toBeNull()
  })

  it("clears the other's old refusal: Start resets Hide, and Hide resets Start", () => {
    render(<SuggestionCard suggestion={suggestion()} />)
    fireEvent.click(startButton())
    expect(hooks.hide.reset).toHaveBeenCalledTimes(1)
    fireEvent.click(hideButton())
    fireEvent.click(screen.getByRole('button', { name: 'Just hide' }))
    expect(hooks.start.reset).toHaveBeenCalledTimes(1)
  })
})

describe('SuggestionCard, Hide', () => {
  it('opens the reason menu and hides with the reason picked', () => {
    render(<SuggestionCard suggestion={suggestion()} />)
    expect(hideButton()).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(hideButton())
    expect(hideButton()).toHaveAttribute('aria-expanded', 'true')
    const menu = screen.getByRole('group', { name: 'Why hide it?' })
    expect(within(menu).getAllByRole('button').map((button) => button.textContent)).toEqual(['Not now', 'Not useful', 'Wrong agent', 'Just hide'])
    fireEvent.click(within(menu).getByRole('button', { name: 'Not useful' }))
    expect(hooks.hide.mutate).toHaveBeenCalledWith({ id: ID, via: 'web', reason: 'not_useful' }, expect.anything())
    succeed(hooks.hide)
    expect(screen.queryByRole('article')).toBeNull()
    expect(screen.getByText('Hidden').parentElement).toHaveAttribute('data-state', 'hidden')
  })

  it('moves focus into the menu, Escape closes it back onto Hide, and Just hide hides with no reason', () => {
    render(<SuggestionCard suggestion={suggestion()} />)
    fireEvent.click(hideButton())
    expect(screen.getByRole('button', { name: 'Not now' })).toHaveFocus()
    fireEvent.keyDown(screen.getByRole('group', { name: 'Why hide it?' }), { key: 'Escape' })
    expect(screen.queryByRole('group', { name: 'Why hide it?' })).toBeNull()
    expect(hideButton()).toHaveFocus()
    fireEvent.click(hideButton())
    fireEvent.click(screen.getByRole('button', { name: 'Just hide' }))
    expect(hooks.hide.mutate.mock.calls[0][0]).toEqual({ id: ID, via: 'web' })
  })
})

describe('SuggestionCard, decided on the server', () => {
  it.each([
    ['started', 'Started'],
    ['hidden', 'Hidden'],
    ['expired', 'Expired'],
  ] as const)('draws a %s suggestion as one line, with the title as Auto\'s words and the caption', (status, word) => {
    render(<SuggestionCard suggestion={suggestion({ status })} />)
    expect(screen.queryByRole('article')).toBeNull()
    expect(screen.queryByRole('button')).toBeNull()
    expect(screen.getByText(word).parentElement).toHaveAttribute('data-state', status)
    expect(screen.getByText(TITLE)).toHaveAttribute('data-ai-generated', 'true')
    expect(screen.getAllByRole('note', { name: AI_CAPTION })).toHaveLength(1)
  })
})

describe('SuggestionCard, pinned line', () => {
  it('keeps Start and Hide on one line, named by the suggestion, with no Start does', () => {
    render(<SuggestionCard suggestion={suggestion()} variant="pinned" />)
    const line = screen.getByRole('group', { name: `Auto suggests: ${TITLE}` })
    expect(line).toHaveAttribute('data-variant', 'pinned')
    expect(within(line).getByText(TITLE)).toHaveAttribute('data-ai-generated', 'true')
    expect(startButton()).toBeInTheDocument()
    expect(hideButton()).toBeInTheDocument()
    expect(screen.queryByText(/Start does/)).toBeNull()
    expect(screen.getAllByRole('note', { name: AI_CAPTION })).toHaveLength(1)
  })

  it('opens the card from its title when the surface can', () => {
    const onOpen = vi.fn()
    render(<SuggestionCard suggestion={suggestion()} variant="pinned" onOpen={onOpen} />)
    fireEvent.click(screen.getByRole('button', { name: `Open: ${TITLE}` }))
    expect(onOpen).toHaveBeenCalledTimes(1)
  })

  it('says Started in place, then leaves once the suggestion is decided', () => {
    vi.useFakeTimers()
    const { container } = render(<SuggestionCard suggestion={suggestion()} variant="pinned" />)
    fireEvent.click(startButton())
    succeed(hooks.start)
    expect(screen.getByRole('status')).toHaveTextContent(STARTED_LINE)
    act(() => vi.advanceTimersByTime(STARTED_MS))
    expect(container).toBeEmptyDOMElement()
  })

  it.each(['started', 'hidden', 'expired'] as const)('draws nothing for a %s suggestion', (status) => {
    const { container } = render(<SuggestionCard suggestion={suggestion({ status })} variant="pinned" />)
    expect(container).toBeEmptyDOMElement()
  })
})

describe('suggestion card source', () => {
  const files = ['suggestion-card.tsx', 'suggestion-parts.tsx', 'use-suggestion-card.ts', 'card-parts.tsx']

  it.each(files)('%s uses tokens only: no hex, no orange, no rounded-3xl', (file) => {
    const source = proactiveSource(file)
    expect(source).not.toMatch(HEX_COLOUR)
    expect(source).not.toMatch(ORANGE_CLASS)
    expect(source).not.toMatch(ROUNDED_3XL)
  })

  it(`keeps each component at most ${MAX_COMPONENT_LINES} lines`, () => {
    const spans = files.flatMap((file) => componentSpans(proactiveSource(file)))
    expect(spans.map((span) => span.name)).toEqual(expect.arrayContaining(['SuggestionCard', 'SuggestionActions', 'ClosedLine']))
    spans.forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
  })
})
