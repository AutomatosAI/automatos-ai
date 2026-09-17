import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

// PRD-221 S13 — Auto's Read panel renders the digest, the needs-attention
// count, and posts a thumbs rating keyed by state_hash.

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

import { AutosRead } from '@/components/activity/autos-read'

describe('AutosRead', () => {
  beforeEach(() => {
    mutate.mockClear()
    window.localStorage.clear()
    digestData = { ...digestData, state_hash: 'hash-1' }
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
})
