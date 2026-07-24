/**
 * PRD-196 S4 (P2-15, governance I.3) — the policy/budget editor.
 *
 * The panel seeds from the current document, round-trips a posture change (PUT
 * carries the edited posture + overrides), and — honest UI over silent placebo —
 * shows the "plane is OFF, settings inert until enforcing" banner when the S3
 * status says the plane is not enforcing.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

// The mocked hook data MUST be reference-stable across renders (the sibling
// tests' mockReturnValue idiom). An inline `() => ({ data: {...} })` mints a
// NEW object every render, so PolicyPane's `useEffect(..., [policy])` re-runs
// each pass and its `setOverrides(policy.route_overrides ?? {})` never bails
// out (fresh object identity) -> an act()-flushed infinite render loop that
// starves the event loop, hangs the whole vitest worker, and timed the
// Frontend CI job out at 20m from PRD-196's merge onward.
const { updatePolicy, updateBudget, statusMock, policyData, budgetData } = vi.hoisted(() => ({
  updatePolicy: vi.fn().mockResolvedValue({}),
  updateBudget: vi.fn().mockResolvedValue({}),
  statusMock: vi.fn(),
  policyData: { posture: 'balanced', agents_inherit_admin: false, route_overrides: {} },
  budgetData: {},
}))

vi.mock('@/hooks/use-governance', () => ({
  usePolicy: () => ({ data: policyData }),
  useBudget: () => ({ data: budgetData }),
  useGovernanceStatus: () => statusMock(),
  useUpdatePolicy: () => ({ isLoading: false, mutateAsync: updatePolicy }),
  useUpdateBudget: () => ({ isLoading: false, mutateAsync: updateBudget }),
}))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import { PolicyPane } from '../governance/policy-pane'

describe('PolicyPane — PRD-196 S4', () => {
  beforeEach(() => {
    updatePolicy.mockClear()
    updateBudget.mockClear()
    statusMock.mockReturnValue({ data: { policy_plane: { enforcing: false } } })
  })

  it('warns that settings are inert while the plane is OFF', () => {
    render(<PolicyPane />)
    expect(screen.getByText(/take effect only/i)).toBeInTheDocument()
  })

  it('round-trips a posture change through PUT /policy', () => {
    render(<PolicyPane />)
    fireEvent.click(screen.getByRole('radio', { name: /Strict/ }))
    fireEvent.click(screen.getByRole('button', { name: /Save policy/ }))
    expect(updatePolicy).toHaveBeenCalledWith({
      posture: 'strict',
      agents_inherit_admin: false,
      route_overrides: {},
    })
  })

  it('sends a per-risk override with the saved policy', () => {
    render(<PolicyPane />)
    fireEvent.change(screen.getByLabelText(/Override for External side-effect/), {
      target: { value: 'auto' },
    })
    fireEvent.click(screen.getByRole('button', { name: /Save policy/ }))
    expect(updatePolicy).toHaveBeenCalledWith({
      posture: 'balanced',
      agents_inherit_admin: false,
      route_overrides: { external_side_effect: 'auto' },
    })
  })

  it('does not show the OFF banner when the plane is enforcing', () => {
    statusMock.mockReturnValue({ data: { policy_plane: { enforcing: true } } })
    render(<PolicyPane />)
    expect(screen.queryByText(/take effect only/i)).not.toBeInTheDocument()
  })
})
