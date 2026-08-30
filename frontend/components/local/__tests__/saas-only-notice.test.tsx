/**
 * PRD-233 S7 — the shared notice a SaaS-only route renders in the local edition.
 */
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { SaasOnlyNotice, SAAS_ONLY_NOTICE_COPY } from '../saas-only-notice'

describe('SaasOnlyNotice', () => {
  it('names the surface and carries the owner\'s copy verbatim', () => {
    render(<SaasOnlyNotice surface="Team" />)

    expect(screen.getByTestId('saas-only-notice')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Team' })).toBeInTheDocument()
    expect(SAAS_ONLY_NOTICE_COPY).toBe(
      'This area is part of the hosted edition; the local edition has no accounts, teams or plans.',
    )
    expect(screen.getByText(SAAS_ONLY_NOTICE_COPY)).toBeInTheDocument()
  })

  it('offers a way back into the product (chat)', () => {
    render(<SaasOnlyNotice surface="Workspace Admin" />)
    const back = screen.getByRole('link', { name: /back to chat/i })
    expect(back).toHaveAttribute('href', '/chat')
  })
})
