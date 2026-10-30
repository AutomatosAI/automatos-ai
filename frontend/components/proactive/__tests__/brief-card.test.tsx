/**
 * PRD-268 §6.4, §6.11.2, §6.12.3, K17 (US-C21c) — BriefCard: the date, the greeting in Newsreader
 * (the only serif), the prose inside the card, the five sections as title + why rows with their
 * actions as text, the fold ("Show N more" from the server's counts), the empty sentence, a pressed
 * row's "✓ {done_label}" in place, chevrons only where a row opens something, thumbs posting the
 * digest feedback with the brief's state_hash, the Listen slot, and data-ai-generated on Auto's words
 * and not on the controls.
 */
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const feedback = vi.hoisted(() => ({ mutate: vi.fn() }))
vi.mock('@/hooks/use-digest-api', () => ({ useSubmitDigestFeedback: () => feedback }))

import { BriefCard, briefDateLabel } from '@/components/proactive/brief-card'
import { BRIEF_SECTIONS, type Brief, type BriefRow } from '@/lib/proactive/types'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, ROUNDED_3XL, componentSpans, proactiveSource } from './source-rules'

const AI_CAPTION = 'Auto is an AI'

function row(id: string, title: string, extra: Partial<BriefRow> = {}): BriefRow {
  return { id, title, why: `Why: ${title}.`, ...extra }
}

const NEEDS_YOU = [0, 1, 2, 3, 4].map((n) =>
  row(`n${n}`, `Question ${n}`, { opens: { type: 'question', id: `q${n}` }, action: { verb: 'answer', label: 'Answer' }, display_state: 'needs_you' }),
)
const LISBON = row('s1', 'Lisbon flights', {
  opens: { type: 'suggestion', id: 's1' },
  action: { verb: 'start', label: 'Start', done_label: 'Started' },
})
const REPORT = row('t1', 'Weekly report', { at: '2026-10-30T15:10:00Z' })

function brief(overrides: Partial<Brief> = {}): Brief {
  return {
    v: 1,
    date: '2026-10-30',
    greeting: 'Morning, Sam.',
    prose: 'Two things moved overnight.',
    spoken: 'Morning, Sam. One: Question 0.',
    sections: {
      needs_you: NEEDS_YOU,
      suggestions: [LISBON],
      today: [REPORT],
      since: [row('d1', 'Invoice export', { display_state: 'done' })],
      unchecked: [row('u1', 'Calendar', { opens: { type: 'settings', id: 'calendar' }, action: { verb: 'connect', label: 'Set up' } })],
    },
    counts: { needs_you: 8, suggestions: 1, today: 1, since: 1 },
    labels: {
      sections: { needs_you: 'Needs you', suggestions: 'Suggested', today: 'Today', since: 'Since your last brief', unchecked: "Couldn't check" },
      more: { needs_you: 'Show 3 more' },
      empty: null,
    },
    state_hash: 'hash-7',
    generated_at: '2026-10-30T06:00:00Z',
    ...overrides,
  }
}

const sectionOf = (container: HTMLElement, name: string) => container.querySelector(`section[data-section="${name}"]`) as HTMLElement
const rowOf = (container: HTMLElement, id: string) => container.querySelector(`li[data-row-id="${id}"]`) as HTMLElement

beforeEach(() => {
  feedback.mutate.mockReset()
})

describe('BriefCard parts', () => {
  it('shows the date and the greeting in Newsreader, the only serif', () => {
    const { container } = render(<BriefCard brief={brief()} />)
    expect(container.querySelector('time[datetime="2026-10-30"]')).toHaveTextContent(briefDateLabel('2026-10-30'))
    const greeting = screen.getByRole('heading', { level: 2, name: 'Morning, Sam.' })
    expect(greeting.className).toContain('--font-newsreader')
    expect(container.querySelectorAll('[class*="newsreader"], .font-serif, [data-studio-serif]')).toHaveLength(1)
  })

  it('draws the prose inside the card, once', () => {
    render(<BriefCard brief={brief()} />)
    expect(screen.getAllByText('Two things moved overnight.')).toHaveLength(1)
    expect(screen.getByRole('article', { name: 'Morning, Sam.' })).toContainElement(screen.getByText('Two things moved overnight.'))
  })

  it("draws the five sections in Brief@1's order, with the server's labels and exact counts", () => {
    const { container } = render(<BriefCard brief={brief()} />)
    const order = Array.from(container.querySelectorAll('section[data-section]')).map((section) => section.getAttribute('data-section'))
    expect(order).toEqual([...BRIEF_SECTIONS])
    expect(within(sectionOf(container, 'needs_you')).getByText('Needs you', { selector: 'p' })).toHaveTextContent('Needs you · 8')
    expect(within(sectionOf(container, 'unchecked')).getByText("Couldn't check")).toBeInTheDocument()
    expect(sectionOf(container, 'unchecked').querySelector('[data-section-count]')).toBeNull()
  })

  it('draws each row as its title and why, with its time and its status', () => {
    const { container } = render(<BriefCard brief={brief()} />)
    expect(within(rowOf(container, 's1')).getByText('Why: Lisbon flights.')).toBeInTheDocument()
    expect(rowOf(container, 't1').querySelector('time')).toHaveAttribute('datetime', REPORT.at)
    expect(rowOf(container, 'd1').querySelector('[data-display-state="done"]')).toHaveTextContent('Done')
  })

  it("folds at the server's rows and says Show N more from its counts", () => {
    const onShowMore = vi.fn()
    const { container } = render(<BriefCard brief={brief()} onShowMore={onShowMore} />)
    expect(sectionOf(container, 'needs_you').querySelectorAll('li')).toHaveLength(5)
    fireEvent.click(screen.getByRole('button', { name: 'Show 3 more' }))
    expect(onShowMore).toHaveBeenCalledWith('needs_you')
    expect(screen.getAllByText(/Show \d+ more/)).toHaveLength(1)
  })

  it('draws Show N more as text when the surface has nowhere to show them, and not at all once the count is shown', () => {
    const { rerender } = render(<BriefCard brief={brief()} />)
    expect(screen.getByText('Show 3 more').tagName).toBe('P')
    rerender(<BriefCard brief={brief({ counts: { needs_you: 5, suggestions: 1, today: 1, since: 1 } })} />)
    expect(screen.queryByText('Show 3 more')).toBeNull()
  })

  it("says the server's empty sentence when no section has a row", () => {
    const empty = { needs_you: [], suggestions: [], today: [], since: [], unchecked: [] }
    const sentence = 'Nothing new needs you, and nothing is scheduled today.'
    const { container } = render(
      <BriefCard brief={brief({ sections: empty, counts: { needs_you: 0, suggestions: 0, today: 0, since: 0 }, labels: { sections: {}, more: {}, empty: sentence } })} />,
    )
    expect(screen.getByText(sentence)).toBeInTheDocument()
    expect(container.querySelector('section[data-section]')).toBeNull()
  })

  it('shows the AI caption once, outside the fence, and puts Listen in its slot', () => {
    render(<BriefCard brief={brief()} listen={<button type="button">Listen</button>} />)
    const captions = screen.getAllByRole('note', { name: AI_CAPTION })
    expect(captions).toHaveLength(1)
    expect(captions[0].closest('[data-ai-generated]')).toBeNull()
    expect(screen.getByRole('button', { name: 'Listen' }).closest('[data-brief-listen]')).not.toBeNull()
  })
})

describe('BriefCard rows', () => {
  it('draws a row action as text, named by its row, outside the fence', () => {
    render(<BriefCard brief={brief()} onAction={vi.fn()} />)
    const start = screen.getByRole('button', { name: 'Start: Lisbon flights' })
    expect(start).toHaveTextContent('Start →')
    expect(start.closest('[data-ai-generated]')).toBeNull()
  })

  it('draws no action and no chevron when the surface has nothing to run them', () => {
    const { container } = render(<BriefCard brief={brief()} />)
    expect(screen.queryByRole('button', { name: 'Start: Lisbon flights' })).toBeNull()
    expect(screen.queryByText('Start →')).toBeNull()
    expect(container.querySelector('[data-row-chevron]')).toBeNull()
    expect(within(rowOf(container, 's1')).getByText('Lisbon flights')).toBeInTheDocument()
  })

  it('shows ✓ {done_label} in place once a press is done', async () => {
    const onAction = vi.fn().mockResolvedValue(undefined)
    const { container } = render(<BriefCard brief={brief()} onAction={onAction} />)
    fireEvent.click(screen.getByRole('button', { name: 'Start: Lisbon flights' }))
    expect(await within(rowOf(container, 's1')).findByRole('status')).toHaveTextContent('✓ Started')
    expect(onAction).toHaveBeenCalledWith(LISBON, 'suggestions')
    expect(screen.queryByRole('button', { name: 'Start: Lisbon flights' })).toBeNull()
  })

  it('leaves a row as it was when the surface refuses the press', async () => {
    const onAction = vi.fn().mockRejectedValue(new Error('spend ceiling'))
    render(<BriefCard brief={brief()} onAction={onAction} />)
    const start = screen.getByRole('button', { name: 'Start: Lisbon flights' })
    fireEvent.click(start)
    await waitFor(() => expect(start).toBeEnabled())
    expect(start).toHaveTextContent('Start →')
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('draws a chevron only on a row that opens something, and opens it', () => {
    const onOpen = vi.fn()
    const { container } = render(<BriefCard brief={brief()} onOpen={onOpen} />)
    expect(rowOf(container, 't1').querySelector('[data-row-chevron]')).toBeNull()
    expect(rowOf(container, 'd1').querySelector('[data-row-chevron]')).toBeNull()
    expect(rowOf(container, 's1').querySelector('[data-row-chevron]')).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Open: Lisbon flights' }))
    expect(onOpen).toHaveBeenCalledWith(LISBON, 'suggestions')
  })

  it("keeps data-ai-generated on Auto's words and off every control", () => {
    const { container } = render(<BriefCard brief={brief()} onAction={vi.fn()} onOpen={vi.fn()} onShowMore={vi.fn()} />)
    expect(screen.getByRole('heading', { level: 2 })).toHaveAttribute('data-ai-generated', 'true')
    expect(screen.getByText('Two things moved overnight.')).toHaveAttribute('data-ai-generated', 'true')
    expect(within(rowOf(container, 's1')).getByText('Lisbon flights').closest('[data-ai-generated="true"]')).not.toBeNull()
    expect(within(rowOf(container, 's1')).getByText('Why: Lisbon flights.').closest('[data-ai-generated="true"]')).not.toBeNull()
    screen.getAllByRole('button').forEach((button) => {
      expect(button).not.toHaveAttribute('data-ai-generated')
      expect(button.closest('[data-ai-generated]')).toBeNull()
    })
  })
})

describe('BriefCard thumbs', () => {
  it("posts the digest feedback with the brief's state_hash, once", () => {
    render(<BriefCard brief={brief()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Helpful' }))
    expect(feedback.mutate).toHaveBeenCalledWith({ state_hash: 'hash-7', rating: 1 }, expect.anything())
    expect(screen.getByRole('button', { name: 'Helpful' })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: 'Not helpful' }))
    expect(feedback.mutate).toHaveBeenCalledTimes(1)
  })

  it('posts a thumbs down, and lets the person rate again when the post fails', () => {
    render(<BriefCard brief={brief()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Not helpful' }))
    expect(feedback.mutate).toHaveBeenCalledWith({ state_hash: 'hash-7', rating: -1 }, expect.anything())
    const options = feedback.mutate.mock.calls[0][1] as { onError: () => void }
    act(() => options.onError())
    expect(screen.getByRole('button', { name: 'Helpful' })).toBeEnabled()
  })
})

describe('briefDateLabel', () => {
  it("keeps a date it can't read as it is", () => {
    expect(briefDateLabel('soon')).toBe('soon')
  })
})

describe('brief card source', () => {
  const files = ['brief-card.tsx', 'brief-sections.tsx', 'brief-row.tsx']

  it.each(files)('%s uses tokens only: no hex, no orange, no rounded-3xl', (file) => {
    const source = proactiveSource(file)
    expect(source).not.toMatch(HEX_COLOUR)
    expect(source).not.toMatch(ORANGE_CLASS)
    expect(source).not.toMatch(ROUNDED_3XL)
  })

  it(`keeps each component at most ${MAX_COMPONENT_LINES} lines`, () => {
    const spans = files.flatMap((file) => componentSpans(proactiveSource(file)))
    expect(spans.map((span) => span.name)).toEqual(expect.arrayContaining(['BriefCard', 'BriefSections', 'BriefRowItem']))
    spans.forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
  })
})
