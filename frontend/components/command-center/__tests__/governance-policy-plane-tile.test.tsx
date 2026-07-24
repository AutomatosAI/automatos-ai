/**
 * PRD-196 S3 (P2-15, governance I.5) — the honest policy-plane tile.
 *
 * The is-it-working strip shows whether governance is ENFORCING. An OFF plane
 * (the shipped default until PRD-192 flips it) renders "OFF / not enforcing yet"
 * loudly but NOT as an error — the operator is never falsely reassured, and
 * never falsely alarmed. A non-admin (no status) shows a muted "—".
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

// The strip pulls many analytics tiles; stub them all to empty so the test is
// about the governance tile alone.
vi.mock('@/hooks/use-analytics-api', () => ({
  useWorkspaceActivation: () => ({ data: undefined }),
  useMissionSuccessRate: () => ({ data: undefined }),
  useErrorsBySubsystem: () => ({ data: undefined }),
  useWidgetEngagement: () => ({ data: undefined }),
  usePrimitiveHealth: () => ({ data: undefined }),
  useSLOs: () => ({ data: undefined }),
  useSubstrateHealth: () => ({ data: undefined }),
  useDeliverableFreshness: () => ({ data: undefined }),
  useCommerceIntegrity: () => ({ data: undefined }),
}))

const { statusMock } = vi.hoisted(() => ({ statusMock: vi.fn() }))
vi.mock('@/hooks/use-governance', () => ({
  useGovernanceStatus: () => statusMock(),
}))

import { IsItWorkingStrip } from '../is-it-working-strip'

describe('IsItWorkingStrip — policy-plane tile (PRD-196 S3)', () => {
  it('renders OFF loudly but not as an error when the plane is not enforcing', () => {
    statusMock.mockReturnValue({ data: { policy_plane: { enforcing: false } } })
    render(<IsItWorkingStrip />)
    expect(screen.getByText('POLICY PLANE')).toBeInTheDocument()
    const value = screen.getByText('OFF')
    expect(value).toBeInTheDocument()
    expect(screen.getByText(/not enforcing/)).toBeInTheDocument()
    // OFF is a warn tone, never the error tone — honest, not alarmist.
    expect(value.className).toContain('warn')
    expect(value.className).not.toContain('err')
  })

  it('renders ON when the plane is enforcing', () => {
    statusMock.mockReturnValue({ data: { policy_plane: { enforcing: true } } })
    render(<IsItWorkingStrip />)
    expect(screen.getByText('ON')).toBeInTheDocument()
    expect(screen.getByText('enforcing')).toBeInTheDocument()
  })

  it('shows a muted dash for a non-admin (no status)', () => {
    statusMock.mockReturnValue({ data: undefined })
    render(<IsItWorkingStrip />)
    expect(screen.getByText('POLICY PLANE')).toBeInTheDocument()
    expect(screen.getByText('admin only')).toBeInTheDocument()
  })
})
