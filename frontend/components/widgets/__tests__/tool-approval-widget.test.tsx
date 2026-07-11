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

vi.mock('@/hooks/use-approval-grants-api', () => ({
  useGrantApproval: () => ({ isLoading: false, mutateAsync: grantMutateAsync }),
  useDenyApproval: () => ({ isLoading: false, mutateAsync: denyMutateAsync }),
}))

import {
  ToolApprovalWidget,
  executedOutcomeLine,
  oversightTierLabel,
} from '../ToolApprovalWidget'
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

    expect(grantMutateAsync).toHaveBeenCalledWith({ id: 42 })
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

    expect(denyMutateAsync).toHaveBeenCalledWith({ id: 42 })
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
})
