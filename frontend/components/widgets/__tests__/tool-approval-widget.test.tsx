/**
 * PRD-193 S3 (P2-12) — the in-chat tool-approval card (frontend half).
 *
 * The backend `tool_approval` payload carries the gated action, a params
 * digest, and the AI-Act oversight fields. This pins the widget: it renders
 * the oversight tier + rationale and the params digest, Approve fires the
 * grant mutation (the PRD-181 grant API's first frontend consumer), Deny
 * fires the deny mutation and renders the refusal, and a failed resume is
 * reported honestly — never as a fake success.
 *
 * Mirrors mission-approval-oversight.test.tsx (the card this one clones).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const grantMutateAsync = vi.fn()
const denyMutateAsync = vi.fn()

// PRD-225: the duplicate use-approval-grants-api was merged into
// use-approval-grants; the widget now imports the consolidated module.
vi.mock('@/hooks/use-approval-grants', () => ({
  useGrantApproval: () => ({ isLoading: false, mutateAsync: grantMutateAsync }),
  useDenyApproval: () => ({ isLoading: false, mutateAsync: denyMutateAsync }),
}))
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn(), info: vi.fn() }))
vi.mock('sonner', () => ({ toast }))

import { ToolApprovalWidget, executedOutcomeLine } from '../ToolApprovalWidget'
import { oversightTierLabel } from '../oversight-note'
import type { ToolApprovalWidgetData, WidgetMetadata } from '../types'

const metadata: WidgetMetadata = {
  source: { type: 'tool', name: 'platform_delete_document' },
  createdAt: new Date(),
}

function data(overrides?: Partial<ToolApprovalWidgetData>): ToolApprovalWidgetData {
  return {
    grant_id: 42,
    action: 'platform_delete_document',
    message: 'This action (destructive) requires confirmation.',
    permission_level: 'destructive',
    params: { document_id: 7 },
    risk_tier: 'human_in_the_loop',
    risk_class: 'destructive',
    oversight_rationale:
      'Destructive: deletes data. A human must approve before it runs.',
    requires_approval: true,
    ...overrides,
  }
}

beforeEach(() => {
  grantMutateAsync.mockReset()
  denyMutateAsync.mockReset()
  toast.error.mockReset()
  toast.success.mockReset()
})

describe('ToolApprovalWidget — PRD-193 S3', () => {
  it('renders the action, oversight tier + rationale, and the params digest', () => {
    render(<ToolApprovalWidget id="w1" title="Action approval needed" data={data()} metadata={metadata} />)

    // The action name appears in the card body (WidgetBase also badges the
    // widget's source name with the same text — hence getAllByText).
    expect(screen.getAllByText('platform_delete_document').length).toBeGreaterThan(0)
    // Oversight banner (clone of the mission card presentation).
    expect(screen.getByText(/Human approval required/)).toBeInTheDocument()
    expect(screen.getByText(/deletes data/)).toBeInTheDocument()
    expect(screen.getByRole('note', { name: /human oversight/i })).toBeInTheDocument()
    // The human sees exactly what they are approving.
    expect(screen.getByText('document_id')).toBeInTheDocument()
    expect(screen.getByText('7')).toBeInTheDocument()
  })

  it('approve fires the grant mutation and reflects the executed result', async () => {
    grantMutateAsync.mockResolvedValue({
      grant: {
        id: 42,
        status: 'revoked',
        details: { executed_result: { success: true } },
      },
    })
    const user = userEvent.setup()
    render(<ToolApprovalWidget id="w2" title="Action approval needed" data={data()} metadata={metadata} />)

    await user.click(screen.getByRole('button', { name: /approve & run/i }))

    expect(grantMutateAsync).toHaveBeenCalledWith(42)
    await waitFor(() =>
      expect(screen.getByRole('status')).toHaveTextContent(/the action ran/i),
    )
  })

  it('a failed resume is reported as a failure — never a fake success', async () => {
    grantMutateAsync.mockResolvedValue({
      grant: {
        id: 42,
        status: 'revoked',
        details: { executed_result: { success: false, error: 'boom' } },
      },
    })
    const user = userEvent.setup()
    render(<ToolApprovalWidget id="w3" title="Action approval needed" data={data()} metadata={metadata} />)

    await user.click(screen.getByRole('button', { name: /approve & run/i }))

    await waitFor(() =>
      expect(screen.getByRole('status')).toHaveTextContent(/execution failed: boom/i),
    )
  })

  it('deny fires the deny mutation and renders the refusal', async () => {
    denyMutateAsync.mockResolvedValue({ grant: { id: 42, status: 'denied' } })
    const user = userEvent.setup()
    render(<ToolApprovalWidget id="w4" title="Action approval needed" data={data()} metadata={metadata} />)

    await user.click(screen.getByRole('button', { name: /deny/i }))

    expect(denyMutateAsync).toHaveBeenCalledWith(42)
    expect(grantMutateAsync).not.toHaveBeenCalled()
    await waitFor(() =>
      expect(screen.getByRole('status')).toHaveTextContent(/will not run/i),
    )
  })

  it('renders no oversight banner when no risk fields are present', () => {
    render(
      <ToolApprovalWidget
        id="w5"
        title="Action approval needed"
        data={data({ risk_tier: undefined, risk_class: undefined, oversight_rationale: undefined })}
        metadata={metadata}
      />,
    )
    expect(screen.queryByRole('note', { name: /human oversight/i })).not.toBeInTheDocument()
  })

  it('maps oversight tiers and executed outcomes to honest copy', () => {
    expect(oversightTierLabel('human_in_the_loop')).toBe('Human approval required')
    expect(oversightTierLabel(undefined)).toBe('Human approval required')

    expect(executedOutcomeLine({ success: true })).toMatch(/action ran/i)
    expect(executedOutcomeLine({ success: false, error: 'nope' })).toMatch(/failed: nope/i)
    expect(executedOutcomeLine({ resumed_via: 'board_task_requeue' })).toMatch(/re-queued/i)
    expect(executedOutcomeLine({ requires_confirmation: true })).toMatch(/asked for confirmation again/i)
  })

  // PRD-256 FX-008: night 12's cards said 'change an agent' and CLUB DESK's brief was wiped.
  it('renders the question — the subject and each change, from → to — in place of the bare message', () => {
    const question_md =
      "Change an agent 'CLUB DESK' (agent #12):\n- description: Runs the club desk → (empty)"
    render(
      <ToolApprovalWidget
        id="w1"
        title="Action approval needed"
        data={data({ action: 'platform_update_agent', message: 'Waiting for the owner', question_md })}
        metadata={metadata}
      />,
    )
    expect(screen.getByText(/description: Runs the club desk → \(empty\)/)).toBeInTheDocument()
    expect(screen.getByText(/Change an agent 'CLUB DESK'/)).toBeInTheDocument()
    expect(screen.queryByText('Waiting for the owner')).not.toBeInTheDocument()
  })

  // PRD-260 T-005c: Play's chat renders this card and stops waiting once it is answered.
  it('tells onDecided once an approve or a deny went through, and never after a failed call', async () => {
    const user = userEvent.setup()
    const onDecided = vi.fn()
    grantMutateAsync.mockResolvedValue({ grant: { id: 42, status: 'revoked', details: {} } })
    const approved = render(
      <ToolApprovalWidget id="w6" title="Action approval needed" data={data()} metadata={metadata} onDecided={onDecided} />,
    )
    await user.click(screen.getByRole('button', { name: /approve & run/i }))
    await waitFor(() => expect(onDecided).toHaveBeenCalledWith('approved'))
    approved.unmount()

    denyMutateAsync.mockResolvedValue({ grant: { id: 42, status: 'denied' } })
    const denied = render(
      <ToolApprovalWidget id="w7" title="Action approval needed" data={data()} metadata={metadata} onDecided={onDecided} />,
    )
    await user.click(screen.getByRole('button', { name: /deny/i }))
    await waitFor(() => expect(onDecided).toHaveBeenCalledWith('denied'))
    denied.unmount()

    onDecided.mockReset()
    grantMutateAsync.mockRejectedValue(new Error('403'))
    render(<ToolApprovalWidget id="w8" title="Action approval needed" data={data()} metadata={metadata} onDecided={onDecided} />)
    await user.click(screen.getByRole('button', { name: /approve & run/i }))
    await waitFor(() => expect(grantMutateAsync).toHaveBeenCalledTimes(2))
    expect(onDecided).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /approve & run/i })).toBeInTheDocument()
  })
})

// P256-FIX-N17-3 (F439, night 17): a click on a card past its expiry (or one a newer card replaced)
// is refused by /grant with a 422 that says why; the chat's card toasts those words.
describe('ToolApprovalWidget — a refused click says why (P256-FIX-N17-3)', () => {
  const EXPIRED = 'This card expired at 05:12 on 15 Oct; nothing ran. Ask Auto again for a fresh card.'

  it("an expired card's 422 is toasted in the server's words", async () => {
    grantMutateAsync.mockRejectedValue(Object.assign(new Error(EXPIRED), { status: 422 }))
    const user = userEvent.setup()
    render(<ToolApprovalWidget id="w6" title="Action approval needed" data={data()} metadata={metadata} />)

    await user.click(screen.getByRole('button', { name: /approve & run/i }))

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(EXPIRED))
    expect(toast.success).not.toHaveBeenCalled()
  })

  it('a failure that is not a refusal keeps the card\'s own line', async () => {
    grantMutateAsync.mockRejectedValue(new Error('Network down'))
    const user = userEvent.setup()
    render(<ToolApprovalWidget id="w7" title="Action approval needed" data={data()} metadata={metadata} />)

    await user.click(screen.getByRole('button', { name: /approve & run/i }))

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith('Failed to approve the action'))
  })
})
