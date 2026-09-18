/** PRD-244 (two styles) — the Governance sub-tabs render in the style of the page that mounts them. */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

vi.mock('@/components/command-center/governance/approvals-inbox', () => ({ ApprovalsInbox: () => <div data-testid="pane-approvals" /> }))
vi.mock('@/components/command-center/governance/audit-pane', () => ({ AuditPane: () => <div data-testid="pane-audit" /> }))
vi.mock('@/components/command-center/governance/policy-pane', () => ({ PolicyPane: () => <div data-testid="pane-policy" /> }))
vi.mock('@/components/command-center/governance/compliance-pane', () => ({ CompliancePane: () => <div data-testid="pane-compliance" /> }))

import { GovernanceTab } from '@/components/command-center/governance-tab'

afterEach(cleanup)

describe('GovernanceTab variants', () => {
  it('Studio (default) uses the shell tab strip and shows the lead pane', () => {
    const { container } = render(<GovernanceTab />)
    expect(container.querySelector('nav.cc-tabs')).not.toBeNull()
    expect(container.querySelectorAll('button.cc-tab')).toHaveLength(4)
    expect(screen.getByTestId('pane-approvals')).toBeInTheDocument()
  })

  it('Classic uses the shared FilterTabs primitive, no Studio class, and switches panes', () => {
    const { container } = render(<GovernanceTab variant="classic" />)
    expect(container.querySelector('.cc-tabs')).toBeNull()
    expect(screen.getAllByRole('tab')).toHaveLength(4)
    expect(screen.getByTestId('pane-approvals')).toBeInTheDocument()
    fireEvent.mouseDown(screen.getByRole('tab', { name: 'Audit' }))
    fireEvent.click(screen.getByRole('tab', { name: 'Audit' }))
    expect(screen.getByTestId('pane-audit')).toBeInTheDocument()
  })
})
