/**
 * PRD-265 US-007 — the operator console's "Business checkout link".
 *
 * The button POSTs /api/admin/workspaces/{id}/business-invite, copies the answered link and
 * toasts until when it works; where the clipboard is refused the toast carries the link to copy
 * by hand; a refusal is toasted in the API's words. Hidden for a workspace already on Business.
 * The plan column (WorkspacePlanCell) mounts it beside the plan dropdown.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

const { postMock, toastMock } = vi.hoisted(() => ({
  postMock: vi.fn(),
  toastMock: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))

vi.mock('@/lib/api-client', () => ({ apiClient: { post: postMock } }))
vi.mock('sonner', () => ({ toast: toastMock }))
vi.mock('../workspace-plan-select', () => ({
  WorkspacePlanSelect: ({ plan }: { plan: string | null }) => <span data-testid="plan-select">{plan}</span>,
}))

import { BusinessInviteButton, INVITE_FAILED, inviteExpiry } from '../business-invite-button'
import { WorkspacePlanCell } from '../workspace-plan-cell'

const INVITE = { url: 'https://checkout.stripe.com/c/pay/cs_test_business', expires_at: '2026-10-24T10:00:00+00:00' }
const originalClipboard = Object.getOwnPropertyDescriptor(window.navigator, 'clipboard')
const writeText = vi.fn()

function setClipboard(clipboard: { writeText: (text: string) => Promise<void> } | undefined): void {
  Object.defineProperty(window.navigator, 'clipboard', { configurable: true, value: clipboard })
}

const inviteButton = () => screen.getByRole('button', { name: 'Business checkout link for Acme' })

beforeEach(() => {
  postMock.mockReset()
  writeText.mockReset()
  writeText.mockResolvedValue(undefined)
  toastMock.success.mockReset()
  toastMock.error.mockReset()
  toastMock.info.mockReset()
  setClipboard({ writeText })
})

afterEach(() => {
  cleanup()
  if (originalClipboard) Object.defineProperty(window.navigator, 'clipboard', originalClipboard)
  else delete (window.navigator as { clipboard?: unknown }).clipboard
  vi.restoreAllMocks()
})

describe('Business checkout link (PRD-265 US-007)', () => {
  it('asks for the link, copies it and says until when it works', async () => {
    postMock.mockResolvedValue(INVITE)
    render(<BusinessInviteButton workspaceId="ws-1" workspaceName="Acme" plan="pro" />)

    fireEvent.click(inviteButton())

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(INVITE.url))
    expect(postMock).toHaveBeenCalledWith('/api/admin/workspaces/ws-1/business-invite')
    await waitFor(() =>
      expect(toastMock.success).toHaveBeenCalledWith('Business checkout link for Acme copied', {
        description: `Send it to the workspace owner. ${inviteExpiry(INVITE.expires_at)}`,
      }),
    )
    expect(inviteExpiry(INVITE.expires_at)).toMatch(/^It works until .+\.$/)
  })

  it('shows the link to copy by hand when the browser refuses the clipboard', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    writeText.mockRejectedValue(new Error('denied'))
    postMock.mockResolvedValue(INVITE)
    render(<BusinessInviteButton workspaceId="ws-1" workspaceName="Acme" plan={null} />)

    fireEvent.click(inviteButton())

    await waitFor(() => expect(toastMock.info).toHaveBeenCalledTimes(1))
    const [, options] = toastMock.info.mock.calls[0] as [string, { description: string }]
    expect(options.description).toContain(INVITE.url)
    expect(toastMock.success).not.toHaveBeenCalled()
  })

  it("toasts the API's words when the link cannot be made", async () => {
    postMock.mockRejectedValue(Object.assign(new Error("Billing isn't configured."), { status: 503, body: { detail: "Billing isn't configured." } }))
    render(<BusinessInviteButton workspaceId="ws-1" workspaceName="Acme" plan="basic" />)

    fireEvent.click(inviteButton())

    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith("Billing isn't configured."))
    expect(writeText).not.toHaveBeenCalled()
    await waitFor(() => expect(inviteButton()).toBeEnabled())
  })

  it('never copies an answer that is not an https address', async () => {
    postMock.mockResolvedValue({ url: 'http://example.test/x', expires_at: INVITE.expires_at })
    render(<BusinessInviteButton workspaceId="ws-1" workspaceName="Acme" plan="basic" />)

    fireEvent.click(inviteButton())

    await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith(INVITE_FAILED))
    expect(writeText).not.toHaveBeenCalled()
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
    expect(inviteButton()).toBeEnabled()
  })
})
