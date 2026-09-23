/**
 * F091-C2 — a "no" in chat offers to withdraw Auto's request still waiting on a yes.
 * Withdraw denies the grant; Keep it leaves it pending. Nothing happens without a click.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

const denyMutate = vi.hoisted(() => vi.fn().mockResolvedValue({}))

vi.mock('@/hooks/use-approval-grants', () => ({
  useDenyApproval: () => ({ isLoading: false, mutateAsync: denyMutate }),
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import { WithdrawOfferCard } from '../withdraw-offer-card'

const OFFER = {
  message: 'You said no — withdraw the request still waiting for your yes?',
  requests: [{
    grant_id: 600,
    action: 'platform_delete_document',
    reason: "Confirmation required before running platform_delete_document on 'christmas-box-2026.csv' (document #716): Delete a document",
  }],
}

describe('WithdrawOfferCard (F091-C2)', () => {
  beforeEach(() => denyMutate.mockClear().mockResolvedValue({}))

  it('shows what is pending and withdraws it on the click', async () => {
    render(<WithdrawOfferCard offer={OFFER} />)
    expect(screen.getByText(OFFER.message)).toBeInTheDocument()
    expect(screen.getByText(/christmas-box-2026\.csv/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Withdraw' }))
    await waitFor(() => expect(denyMutate).toHaveBeenCalledWith(600))
    expect(await screen.findByText('Withdrawn.')).toBeInTheDocument()
  })

  it('keeps the request when the owner says keep it', () => {
    render(<WithdrawOfferCard offer={OFFER} />)
    fireEvent.click(screen.getByRole('button', { name: 'Keep it' }))
    expect(denyMutate).not.toHaveBeenCalled()
    expect(screen.getByText('Kept — it still waits for your yes.')).toBeInTheDocument()
  })
})
