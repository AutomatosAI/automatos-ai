/**
 * PRD-265 US-007 / W2 — the operator console's "Offer Business".
 *
 * The button POSTs /api/admin/workspaces/{id}/business-invite, which offers Business to that
 * workspace in-app for 14 days, and toasts "Business offered to <name> until <date>". There is
 * no link: nothing is copied to the clipboard. A refusal is toasted in the API's words. Hidden
 * for a workspace already on Business. The plan column (WorkspacePlanCell) mounts it beside the
 * plan dropdown.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { formatAbsoluteDate } from '@/lib/format-relative'

const { postMock, toastMock } = vi.hoisted(() => ({
  postMock: vi.fn(),
  toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

vi.mock('@/lib/api-client', () => ({ apiClient: { post: postMock } }))
vi.mock('sonner', () => ({ toast: toastMock }))
vi.mock('../workspace-plan-select', () => ({
  WorkspacePlanSelect: ({ plan }: { plan: string | null }) => <span data-testid="plan-select">{plan}</span>,
}))

import { BusinessInviteButton, OFFER_FAILED, offeredToast } from '../business-invite-button'
import { WorkspacePlanCell } from '../workspace-plan-cell'

const OFFER = { workspace_id: 'ws-1', business_offer_until: '2026-11-06T10:00:00+00:00' }
const originalClipboard = Object.getOwnPropertyDescriptor(window.navigator, 'clipboard')
const writeText = vi.fn()

const offerButton = () => screen.getByRole('button', { name: 'Offer Business to Acme' })

beforeEach(() => {
  postMock.mockReset()
  writeText.mockReset()
  Object.values(toastMock).forEach((mock) => mock.mockReset())
  Object.defineProperty(window.navigator, 'clipboard', { configurable: true, value: { writeText } })
})

afterEach(() => {
  cleanup()
  if (originalClipboard) Object.defineProperty(window.navigator, 'clipboard', originalClipboard)
  else delete (window.navigator as { clipboard?: unknown }).clipboard
})

describe('Offer Business (PRD-265 US-007 / W2)', () => {
  it('offers Business in-app and says until when, with no link to copy', async () => {
    postMock.mockResolvedValue(OFFER)
    render(<BusinessInviteButton workspaceId="ws-1" workspaceName="Acme" plan="pro" />)

    fireEvent.click(offerButton())

    await waitFor(() =>
      expect(toastMock.success).toHaveBeenCalledWith(`Business offered to Acme until ${formatAbsoluteDate(OFFER.business_offer_until)}`),
    )
    expect(postMock).toHaveBeenCalledWith('/api/admin/workspaces/ws-1/business-invite')
    expect(offeredToast('Acme', OFFER.business_offer_until)).toBe(`Business offered to Acme until ${formatAbsoluteDate(OFFER.business_offer_until)}`)
    expect(writeText).not.toHaveBeenCalled()
  })

  it("toasts the API's words when Business can't be offered, and the button works again", async () => {
    postMock.mockRejectedValue(Object.assign(new Error('Workspace not found'), { status: 404, body: { detail: 'Workspace not found.' } }))
    render(<BusinessInviteButton workspaceId="ws-1" workspaceName="Acme" plan="basic" />)

    fireEvent.click(offerButton())

    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith('Workspace not found.'))
    expect(toastMock.success).not.toHaveBeenCalled()
    await waitFor(() => expect(offerButton()).toBeEnabled())
  })

  it('falls back to its own sentence for a server failure', async () => {
    postMock.mockRejectedValue(Object.assign(new Error('boom'), { status: 500, body: { detail: 'Traceback …' } }))
    render(<BusinessInviteButton workspaceId="ws-1" workspaceName="Acme" plan={null} />)

    fireEvent.click(offerButton())

    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith(OFFER_FAILED))
  })

  it('is hidden for a workspace already on Business', () => {
    render(<BusinessInviteButton workspaceId="ws-1" workspaceName="Acme" plan="business" />)

    expect(screen.queryByRole('button')).toBeNull()
  })

  it('sits beside the plan dropdown in the console plan column', () => {
    render(
      <WorkspacePlanCell
        workspaceId="ws-1"
        workspaceName="Acme"
        plan="pro"
        membersCount={1}
        tiers={[]}
        onChanged={() => undefined}
        onError={() => undefined}
      />,
    )

    expect(screen.getByTestId('plan-select')).toHaveTextContent('pro')
    expect(offerButton()).toBeEnabled()
  })
})
