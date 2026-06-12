import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ErrorState } from '../error-state'
import { LoadingState } from '../loading-state'
import { DeleteConfirmation } from '../delete-confirmation'

// PRD-169 S2 — the canonical UI-state primitives. EmptyState + Skeleton already
// shipped; these three (Error, Loading, DeleteConfirmation) close the set.

describe('ErrorState', () => {
  it('renders the default title and derives a message from an Error', () => {
    render(<ErrorState error={new Error('boom')} />)
    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText('boom')).toBeInTheDocument()
  })

  it('prefers an explicit description over the error message', () => {
    render(<ErrorState description="explicit message" error={new Error('boom')} />)
    expect(screen.getByText('explicit message')).toBeInTheDocument()
    expect(screen.queryByText('boom')).not.toBeInTheDocument()
  })

  it('shows a retry button only when onRetry is provided, and fires it', async () => {
    const onRetry = vi.fn()
    const { rerender } = render(<ErrorState />)
    expect(screen.queryByRole('button')).not.toBeInTheDocument()

    rerender(<ErrorState onRetry={onRetry} retryLabel="Retry" />)
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })
})

describe('LoadingState', () => {
  it('exposes an accessible status with its label', () => {
    render(<LoadingState label="Loading agents…" />)
    expect(screen.getByRole('status', { name: 'Loading agents…' })).toBeInTheDocument()
  })

  it('renders the requested number of skeletons for the cards variant', () => {
    const { container } = render(<LoadingState variant="cards" count={4} />)
    expect(container.querySelectorAll('.animate-pulse')).toHaveLength(4)
  })

  it('renders a spinner (no skeletons) for the spinner variant', () => {
    const { container } = render(<LoadingState variant="spinner" />)
    expect(container.querySelectorAll('.animate-pulse')).toHaveLength(0)
    expect(screen.getByRole('status')).toBeInTheDocument()
  })
})

describe('DeleteConfirmation', () => {
  it('renders nothing while closed', () => {
    render(<DeleteConfirmation open={false} onOpenChange={() => {}} onConfirm={() => {}} />)
    expect(screen.queryByText('Delete this item?')).not.toBeInTheDocument()
  })

  it('builds a default body from itemName, confirms, then closes', async () => {
    const onConfirm = vi.fn()
    const onOpenChange = vi.fn()
    render(
      <DeleteConfirmation open onOpenChange={onOpenChange} itemName="the agent" onConfirm={onConfirm} />,
    )
    expect(screen.getByText(/This permanently deletes the agent/)).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Delete' }))
    await waitFor(() => expect(onConfirm).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false))
  })

  it('cancel closes without confirming', async () => {
    const onConfirm = vi.fn()
    const onOpenChange = vi.fn()
    render(<DeleteConfirmation open onOpenChange={onOpenChange} onConfirm={onConfirm} />)

    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onConfirm).not.toHaveBeenCalled()
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })
})
