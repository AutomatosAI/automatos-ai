/**
 * F298 (night 8, #0394.2) — the link generate_document gives the owner,
 * /deliverables?tab=outputs&deliverable=<id>, opens that Deliverable's preview,
 * which fetches the file through apiClient in both editions. Closing it keeps
 * the rest of the address.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

const nav = vi.hoisted(() => ({ query: '', replace: vi.fn() }))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(nav.query),
}))
vi.mock('@/components/workspace/gallery-view/deliverable-preview', () => ({
  DeliverablePreview: ({ deliverableId, open, onOpenChange }: {
    deliverableId: string | null
    open: boolean
    onOpenChange: (open: boolean) => void
  }) => (
    <div data-testid="preview" data-id={deliverableId ?? ''} data-open={String(open)}>
      <button onClick={() => onOpenChange(false)}>close</button>
    </div>
  ),
}))

import { DeliverableDeepLink } from '@/components/deliverables/deliverable-deep-link'

const ID = '5f0c2a8e-2b7d-4c55-9a51-0d6f1e2b2980'

afterEach(() => {
  cleanup()
  nav.query = ''
  nav.replace.mockReset()
})

describe('DeliverableDeepLink', () => {
  it('opens the Deliverable the link names', () => {
    nav.query = `tab=outputs&deliverable=${ID}`
    render(<DeliverableDeepLink />)
    const preview = screen.getByTestId('preview')
    expect(preview.dataset.id).toBe(ID)
    expect(preview.dataset.open).toBe('true')
  })

  it('opens nothing when the link names no Deliverable', () => {
    nav.query = 'tab=outputs'
    render(<DeliverableDeepLink />)
    expect(screen.getByTestId('preview').dataset.open).toBe('false')
  })

  it('closing it drops the Deliverable from the address and keeps the tab', () => {
    nav.query = `tab=outputs&deliverable=${ID}`
    render(<DeliverableDeepLink />)
    fireEvent.click(screen.getByText('close'))
    expect(nav.replace).toHaveBeenCalledWith('/deliverables?tab=outputs')
  })
})
