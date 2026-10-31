import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

// PRD-221 S13 — Auto's Read panel renders the digest, the needs-attention
// count, and posts a thumbs rating keyed by state_hash.
// PRD-268 US-W02 — when today's brief is held (useBrief), its rows sit under the
// prose; without one, the card is exactly as before.

const mutate = vi.fn()
let digestData: any = {
  text: 'Your workspace is healthy. One agent is active.',
  generated_at: '2026-07-17T06:00:00Z',
  state_hash: 'hash-1',
  needs_attention_count: 1,
}

vi.mock('@/hooks/use-digest-api', () => ({
  useWorkspaceDigest: () => ({ data: digestData, isLoading: false, isError: false }),
  useSubmitDigestFeedback: () => ({ mutate }),
}))

let briefData: any = null
vi.mock('@/hooks/use-brief', () => ({ useBrief: () => ({ data: briefData }) }))

const BRIEF = {
  v: 1,
  date: '2026-10-30',
  greeting: 'Morning, Sam.',
  prose: 'Your workspace is healthy. One agent is active.',
  spoken: 'Morning, Sam. 1. Send the Airbnb split.',
  sections: {
    needs_you: [{ id: 'grant:12', title: 'Send the Airbnb split', why: 'Waiting for your approval', display_state: 'needs_you' }],
    suggestions: [],
    today: [{ id: 'mission:4', title: 'Weekly metrics report', why: 'Runs at 14:00' }],
    since: [],
    unchecked: [],
  },
  counts: { needs_you: 1, suggestions: 0, today: 1, since: 0 },
  labels: { sections: { needs_you: 'Needs you', today: 'Today' }, more: {} },
  state_hash: 'brief-hash',
  generated_at: '2026-10-30T07:00:00Z',
}

import { AutosRead } from '@/components/activity/autos-read'

describe('AutosRead', () => {
  beforeEach(() => {
    mutate.mockClear()
    window.localStorage.clear()
    digestData = { ...digestData, state_hash: 'hash-1' }
    briefData = null
  })

  it('renders the digest text and the needs-attention badge', () => {
    render(<AutosRead />)
    expect(
      screen.getByText('Your workspace is healthy. One agent is active.'),
    ).toBeInTheDocument()
    expect(screen.getByText(/1 need.*attention/i)).toBeInTheDocument()
  })

  it('posts a thumbs-up keyed by state_hash', () => {
    render(<AutosRead />)
    fireEvent.click(screen.getByLabelText('Helpful'))
    expect(mutate).toHaveBeenCalledWith({ state_hash: 'hash-1', rating: 1 })
  })

  it('a thumbs-up collapses the card to one line; the chevron reopens it', () => {
    render(<AutosRead />)
    fireEvent.click(screen.getByLabelText('Helpful'))
    expect(screen.getByTestId('autos-read')).toHaveAttribute('data-collapsed', 'true')
    expect(screen.getByText(/· Your workspace is healthy\./)).toBeInTheDocument()
    fireEvent.click(screen.getByLabelText("Show Auto's read"))
    expect(screen.getByTestId('autos-read')).toHaveAttribute('data-collapsed', 'false')
  })

  it('a collapsed read stays collapsed for the same state_hash and reappears for a new one', () => {
    const { unmount } = render(<AutosRead />)
    fireEvent.click(screen.getByLabelText("Hide Auto's read"))
    unmount()
    render(<AutosRead />)
    expect(screen.getByTestId('autos-read')).toHaveAttribute('data-collapsed', 'true')
    digestData = { ...digestData, state_hash: 'hash-2' }
    const second = render(<AutosRead />)
    expect(second.getAllByTestId('autos-read').at(-1)).toHaveAttribute('data-collapsed', 'false')
  })

  it('posts a thumbs-down and then locks further rating', () => {
    render(<AutosRead />)
    fireEvent.click(screen.getByLabelText('Not helpful'))
    expect(mutate).toHaveBeenCalledWith({ state_hash: 'hash-1', rating: -1 })
    // second click is a no-op (buttons disabled after rating)
    fireEvent.click(screen.getByLabelText('Helpful'))
    expect(mutate).toHaveBeenCalledTimes(1)
  })

  it("without today's brief, draws no brief rows: the card is exactly as before", () => {
    render(<AutosRead />)
    expect(screen.queryByTestId('autos-read-brief')).toBeNull()
    expect(screen.queryByText('Send the Airbnb split')).toBeNull()
  })

  it("with today's brief, shows its rows under the prose, marked as Auto's words", () => {
    briefData = BRIEF
    render(<AutosRead />)
    const rows = screen.getByTestId('autos-read-brief')
    const prose = screen.getByText('Your workspace is healthy. One agent is active.')
    expect(prose.compareDocumentPosition(rows) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(rows).toHaveTextContent('Needs you')
    expect(rows).toHaveTextContent('Send the Airbnb split')
    expect(rows).toHaveTextContent('Weekly metrics report')
    expect(screen.getByText('Send the Airbnb split').closest('[data-ai-generated="true"]')).not.toBeNull()
    // The read's own thumbs still rate the read, not the brief.
    fireEvent.click(screen.getByLabelText('Helpful'))
    expect(mutate).toHaveBeenCalledWith({ state_hash: 'hash-1', rating: 1 })
  })

  it("folds the brief's rows away with the read, and brings them back with it", () => {
    briefData = BRIEF
    render(<AutosRead />)
    fireEvent.click(screen.getByLabelText("Hide Auto's read"))
    expect(screen.queryByTestId('autos-read-brief')).toBeNull()
    fireEvent.click(screen.getByLabelText("Show Auto's read"))
    expect(screen.getByTestId('autos-read-brief')).toHaveTextContent('Send the Airbnb split')
  })
})
