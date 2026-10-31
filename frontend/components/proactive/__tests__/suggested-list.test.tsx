/**
 * PRD-268 §6.5.1 (US-W01a) — the Command Center's Suggested list: one SuggestionCard per open suggestion
 * with its evidence links, what Start does, its credits, Start and Hide with a reason; the one ink
 * primary; a just-started card kept in place; the bell's ?suggestion=<id> scrolling to its card and
 * outlining it; nothing when there is none; and both shells mounting it beside Auto's Read.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

type Mutation = { mutate: ReturnType<typeof vi.fn>; reset: ReturnType<typeof vi.fn>; isLoading: boolean; isError: boolean; error: unknown }
const state = vi.hoisted(() => ({
  list: undefined as unknown,
  params: new URLSearchParams(),
  start: null as unknown as Mutation,
  hide: null as unknown as Mutation,
  push: vi.fn(),
  request: vi.fn(),
  toastError: vi.fn(),
}))
vi.mock('@/hooks/use-suggestions', () => ({
  useSuggestions: () => ({ data: state.list }),
  useStartSuggestion: () => state.start,
  useHideSuggestion: () => state.hide,
}))
vi.mock('next/navigation', () => ({
  useSearchParams: () => state.params,
  useRouter: () => ({ push: state.push }),
}))
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: unknown; children: ReactNode }) => (
    <a href={String(href)} {...rest}>
      {children}
    </a>
  ),
}))
vi.mock('@/lib/api-client', () => ({ apiClient: { request: (...args: unknown[]) => state.request(...args) } }))
vi.mock('sonner', () => ({ toast: { error: (...args: unknown[]) => state.toastError(...args), success: vi.fn() } }))
vi.mock('@/components/activity/autos-read', () => ({ AutosRead: ({ period }: { period?: string }) => <div data-testid="autos-read">{period}</div> }))

import { CommandCenterRead } from '@/components/proactive/command-center-read'
import { EVIDENCE_LIST_LABEL } from '@/components/proactive/evidence-links'
import { FOCUSED_CLASS, SUGGESTED_LABEL, SuggestedList } from '@/components/proactive/suggested-list'
import { STARTED_LINE, STARTED_MS } from '@/components/proactive/use-suggestion-card'
import { linkFor } from '@/components/notifications/notification-bell'
import type { NotificationRow } from '@/hooks/use-notifications-api'
import type { Suggestion } from '@/lib/proactive/types'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, ROUNDED_3XL, componentSpans, proactiveSource } from './source-rules'

const FIRST = '7d1c0a52-3e0b-4c55-9a1e-2f6f0c1d9b10'
const SECOND = '0b6f3e21-94c7-4d0a-8e15-6a2d9c7f4b33'

function suggestion(overrides: Partial<Suggestion> = {}): Suggestion {
  return {
    id: FIRST, workspace_id: 'ws-1', user_id: 1, source_type: 'check_in', source_agent_id: 4, run_id: null,
    title: 'Look at the Lisbon flights', detail: 'Three runs failed since Monday.',
    evidence: [{ type: 'ticket', id: '0419' }, { type: 'report', id: '12' }],
    proposed_action: { kind: 'board_task', payload: {} }, subject_key: 'ticket:0419', severity: 'today',
    status: 'open', occurrences: 1, last_seen_at: null, started_ref: null, decided_via: null, hide_reason: null,
    decided_at: null, est_credits: null, created_at: null, delivered_at: null, expires_at: null,
    ...overrides,
  }
}

const second = () => suggestion({ id: SECOND, title: 'Renew the domain', detail: 'It lapses on Friday.', evidence: [] })
const mutation = (): Mutation => ({ mutate: vi.fn(), reset: vi.fn(), isLoading: false, isError: false, error: null })
const card = (title: string) => screen.getByRole('article', { name: `Auto suggests: ${title}` })

function bellRow(link_type: string, link_id: string | null): NotificationRow {
  return {
    id: 'n1', workspace_id: 'ws1', user_id: 1, event_type: 'suggestion_ready', title: 't', message: null, link_type, link_id,
    agent_id: null, agent_name: 'Auto', status: 'ok', read_at: null, dismissed_at: null, created_at: '2026-10-30T00:00:00Z',
  }
}

/** Answer the last Start or Hide as the server would have. */
function succeed(target: Mutation, answer: unknown = {}) {
  const calls = target.mutate.mock.calls
  const options = calls[calls.length - 1][1] as { onSuccess: (answer: unknown) => void }
  act(() => options.onSuccess(answer))
}

const scrollIntoView = vi.fn()

beforeEach(() => {
  state.list = [suggestion(), second()]
  state.params = new URLSearchParams()
  state.start = mutation()
  state.hide = mutation()
  state.push.mockReset()
  state.request.mockReset()
  state.toastError.mockReset()
  scrollIntoView.mockReset()
  Element.prototype.scrollIntoView = scrollIntoView
})

afterEach(() => {
  vi.useRealTimers()
})

describe('SuggestedList, the cards', () => {
  it('draws one card per open suggestion under "Suggested", each with its words, what Start does, Start and Hide', () => {
    render(<SuggestedList />)
    const section = screen.getByRole('region', { name: SUGGESTED_LABEL })
    expect(within(section).getAllByRole('article')).toHaveLength(2)
    const first = card('Look at the Lisbon flights')
    expect(within(first).getByText('Three runs failed since Monday.').closest('[data-ai-generated="true"]')).not.toBeNull()
    expect(within(first).getByText('Start does: files a ticket on the Board for an agent to work.')).toBeInTheDocument()
    expect(within(first).getByRole('button', { name: 'Start: Look at the Lisbon flights' })).toBeEnabled()
    expect(within(first).getByRole('button', { name: 'Hide: Look at the Lisbon flights' })).toBeEnabled()
    expect(within(card('Renew the domain')).getByText('It lapses on Friday.')).toBeInTheDocument()
  })

  it('says "about N credits" on Start when the cost is known, and nothing when it is not', () => {
    state.list = [suggestion({ est_credits: 12 }), second()]
    render(<SuggestedList />)
    expect(within(card('Look at the Lisbon flights')).getByRole('button', { name: 'Start: Look at the Lisbon flights (about 12 credits)' })).toBeInTheDocument()
    expect(within(card('Renew the domain')).queryByText(/credits/)).toBeNull()
  })

  it('keeps one ink primary in the view: Start on the first card only', () => {
    render(<SuggestedList />)
    const primaries = screen.getAllByRole('button').filter((button) => button.classList.contains('bg-primary'))
    expect(primaries).toEqual([screen.getByRole('button', { name: 'Start: Look at the Lisbon flights' })])
  })

  it('renders nothing when there is no open suggestion, or before the list has answered', () => {
    state.list = []
    const { container, rerender } = render(<SuggestedList />)
    expect(container).toBeEmptyDOMElement()
    state.list = undefined
    rerender(<SuggestedList />)
    expect(container).toBeEmptyDOMElement()
  })
})

describe('SuggestedList, Start and Hide', () => {
  it('Start posts the host route from the web, and the started card keeps its place for "Started. Auto is on it."', () => {
    vi.useFakeTimers()
    const { rerender } = render(<SuggestedList />)
    fireEvent.click(screen.getByRole('button', { name: 'Start: Look at the Lisbon flights' }))
    expect(state.start.mutate).toHaveBeenCalledWith({ id: FIRST, via: 'web' }, expect.anything())
    succeed(state.start, { suggestion: { ...suggestion(), status: 'started' }, opens: { type: 'board_task', board_task_id: 2680 } })
    // The list is asked again and no longer holds the started suggestion.
    state.list = [second()]
    rerender(<SuggestedList />)
    // The second card cites nothing, so the list's only items are the two cards, the started one first.
    expect(screen.getAllByRole('listitem')[0]).toHaveTextContent(STARTED_LINE)
    act(() => vi.advanceTimersByTime(STARTED_MS))
    expect(screen.queryByText(STARTED_LINE)).toBeNull()
    expect(screen.queryByText('Look at the Lisbon flights')).toBeNull()
    expect(card('Renew the domain')).toBeInTheDocument()
  })

  it('Hide opens the reason menu and hides with the reason picked', () => {
    render(<SuggestedList />)
    fireEvent.click(screen.getByRole('button', { name: 'Hide: Look at the Lisbon flights' }))
    fireEvent.click(within(screen.getByRole('group', { name: 'Why hide it?' })).getByRole('button', { name: 'Not useful' }))
    expect(state.hide.mutate).toHaveBeenCalledWith({ id: FIRST, via: 'web', reason: 'not_useful' }, expect.anything())
    expect(state.start.mutate).not.toHaveBeenCalled()
  })
})

describe('SuggestedList, the evidence links', () => {
  it('links each ref to what it names in the app, and quotes a kind the app has no page for', () => {
    state.list = [suggestion({
      evidence: [
        { type: 'report', id: '12' },
        { type: 'run', id: '5d0c2b1a-8e7f-4a3b-9c6d-1e2f3a4b5c6d' },
        { type: 'document', id: '88' },
        { type: 'task', id: '2680' },
        { type: 'mission', id: 'm-1' },
        { type: 'calendar_event', id: '9' },
        { type: 'report', id: '../../admin' },
      ],
    })]
    render(<SuggestedList />)
    const links = within(screen.getByRole('list', { name: EVIDENCE_LIST_LABEL }))
    expect(links.getByRole('link', { name: 'Report 12' })).toHaveAttribute('href', '/command-center?tab=feed&report=12')
    expect(links.getByRole('link', { name: 'Check-in run 5d0c2b1a' })).toHaveAttribute('href', '/command-center?tab=feed')
    expect(links.getByRole('link', { name: 'Document 88' })).toHaveAttribute('href', '/documents')
    // A ticket's row id opens it on the board, and is never shown as its number.
    expect(links.getByRole('link', { name: 'Ticket' })).toHaveAttribute('href', '/command-center?tab=board&task_id=2680')
    expect(links.getByRole('link', { name: 'Mission m-1' })).toHaveAttribute('href', '/missions/m-1')
    expect(links.getByText('Calendar event 9').tagName).toBe('SPAN')
    expect(links.getByText('Report ../../admin').tagName).toBe('SPAN')
  })

  it('a ticket cited by its board number finds that ticket on the board and opens it', async () => {
    state.request.mockResolvedValue({ tasks: [
      { id: 77, title: 'Mentions 0419 in its title', workspace_seq: 4190 },
      { id: 2680, title: 'Lisbon flights', workspace_seq: 419 },
    ] })
    render(<SuggestedList />)
    fireEvent.click(screen.getByRole('button', { name: 'Ticket #0419' }))
    await waitFor(() => expect(state.push).toHaveBeenCalledWith('/command-center?tab=board&task_id=2680'))
    expect(state.request).toHaveBeenCalledWith('/api/v1/tasks?search=0419&finished_limit=200')
  })

  it("a mission step's number names the step, and opens its mission card when the step has no row of its own", async () => {
    state.list = [suggestion({ evidence: [{ type: 'ticket', id: '#0051.3' }] })]
    state.request.mockResolvedValue({ tasks: [{ id: 900, title: 'Launch plan', workspace_seq: 51, number: '#0051' }] })
    render(<SuggestedList />)
    fireEvent.click(screen.getByRole('button', { name: 'Ticket #0051.3' }))
    await waitFor(() => expect(state.push).toHaveBeenCalledWith('/command-center?tab=board&task_id=900'))
    expect(state.request).toHaveBeenCalledWith('/api/v1/tasks?search=0051&finished_limit=200')
  })

  it('a board number the board no longer has says so and opens nothing', async () => {
    state.request.mockResolvedValue({ tasks: [] })
    render(<SuggestedList />)
    fireEvent.click(screen.getByRole('button', { name: 'Ticket #0419' }))
    await waitFor(() => expect(state.toastError).toHaveBeenCalledWith("Ticket #0419 isn't on the board any more."))
    expect(state.push).not.toHaveBeenCalled()
  })
})

describe("the bell's suggestion param", () => {
  it('a suggestion_ready row links to its card in the Suggested list', () => {
    expect(linkFor(bellRow('suggestion', SECOND))).toBe(`/command-center?tab=summary&suggestion=${SECOND}`)
    expect(linkFor(bellRow('suggestion', null))).toBe('/command-center?tab=summary')
  })

  it('scrolls to the card the bell opened, once, outlines and focuses it, and makes its Start the primary', () => {
    state.params = new URLSearchParams({ tab: 'summary', suggestion: SECOND })
    const { rerender } = render(<SuggestedList />)
    const focused = card('Renew the domain')
    expect(focused.closest('li')).toHaveAttribute('data-focused', 'true')
    for (const outline of FOCUSED_CLASS.split(' ')) expect(focused).toHaveClass(outline)
    expect(card('Look at the Lisbon flights').closest('li')).not.toHaveAttribute('data-focused')
    expect(scrollIntoView).toHaveBeenCalledTimes(1)
    expect(scrollIntoView.mock.contexts[0]).toBe(focused.closest('li'))
    expect(scrollIntoView).toHaveBeenCalledWith(expect.objectContaining({ block: 'center' }))
    expect(document.activeElement).toBe(focused.closest('li'))
    const primaries = screen.getAllByRole('button').filter((button) => button.classList.contains('bg-primary'))
    expect(primaries).toEqual([screen.getByRole('button', { name: 'Start: Renew the domain' })])
    rerender(<SuggestedList />)
    expect(scrollIntoView).toHaveBeenCalledTimes(1)
  })

  it('ignores a param that is not a suggestion id', () => {
    state.params = new URLSearchParams({ suggestion: '../../x' })
    render(<SuggestedList />)
    expect(document.querySelector('[data-focused]')).toBeNull()
    expect(scrollIntoView).not.toHaveBeenCalled()
  })
})

describe('CommandCenterRead, in both shells', () => {
  const read = (file: string) => readFileSync(path.resolve(__dirname, '..', '..', '..', file), 'utf8')
  const IMPORT = "import { CommandCenterRead as AutosRead } from '@/components/proactive/command-center-read'"

  it("renders Auto's Read, then the Suggested list", () => {
    render(<CommandCenterRead period="7d" />)
    const autos = screen.getByTestId('autos-read')
    expect(autos).toHaveTextContent('7d')
    expect(autos.compareDocumentPosition(screen.getByRole('region', { name: SUGGESTED_LABEL })) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it("is what Studio's SummaryTab and Classic's ActivityPage mount in Auto's Read's place", () => {
    for (const file of ['components/command-center/summary-tab.tsx', 'components/activity/activity-page.tsx']) {
      const source = read(file)
      expect(source).toContain(IMPORT)
      expect(source).toContain('<AutosRead period={period} />')
    }
  })

  it('keeps the new files on tokens, with every component at most 150 lines', () => {
    for (const file of ['suggested-list.tsx', 'evidence-links.tsx', 'command-center-read.tsx']) {
      const source = proactiveSource(file)
      expect(source).not.toMatch(HEX_COLOUR)
      expect(source).not.toMatch(ORANGE_CLASS)
      expect(source).not.toMatch(ROUNDED_3XL)
      for (const span of componentSpans(source)) expect(span.lines).toBeLessThanOrEqual(MAX_COMPONENT_LINES)
    }
  })
})
