/**
 * PRD-266 US-016c: a content report reaches the owner in Studio.
 *
 * The backend files a report as one audit row (action 'content_reported',
 * details.reason) and notifies the owner with link_type 'content_report'
 * (services/content_reports/service.py). The bell opens the Governance tab; its
 * Audit pane filters the reports and shows each one's reason.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
import { render, screen, fireEvent } from '@testing-library/react'
import type { AuditLogRow } from '@/lib/api-client'
import type { NotificationRow } from '@/hooks/use-notifications-api'

const { useGovernanceAuditLog } = vi.hoisted(() => ({ useGovernanceAuditLog: vi.fn() }))

vi.mock('@/hooks/use-governance', () => ({ useGovernanceAuditLog }))

import { AuditPane } from '../audit-pane'
import { linkFor } from '@/components/notifications/notification-bell'

const REPORT_ROW: AuditLogRow = {
  id: 41,
  created_at: '2026-10-21T09:00:00Z',
  actor_type: 'user',
  user_id: 7,
  action: 'content_reported',
  resource_type: 'message',
  resource_id: '7c1e2a4e-0000-4000-8000-000000000001',
  resource_name: null,
  details: { reason: 'inaccurate', note: null, excerpt: 'The capital of Australia is Sydney.' },
}

function notice(link_type: string, link_id: string | null = null): NotificationRow {
  return {
    id: 'n1',
    workspace_id: 'ws1',
    user_id: 1,
    event_type: 'content_reported',
    title: 'Content reported',
    message: null,
    link_type,
    link_id,
    agent_id: null,
    agent_name: null,
    status: 'warning',
    read_at: null,
    dismissed_at: null,
    created_at: '2026-10-21T09:00:00Z',
  }
}

function lastFilters() {
  const calls = useGovernanceAuditLog.mock.calls
  return calls[calls.length - 1][0]
}

beforeEach(() => {
  useGovernanceAuditLog.mockReset()
  useGovernanceAuditLog.mockReturnValue({
    data: { rows: [REPORT_ROW], total: 1, limit: 25, offset: 0 },
    isLoading: false,
    isError: false,
  })
})

describe('PRD-266 US-016c — content reports in Governance', () => {
  it("choosing 'Content reports' asks the audit-log hook for action_prefix 'content_reported'", () => {
    render(<AuditPane />)
    expect(lastFilters().action_prefix).toBeUndefined()

    const select = screen.getByLabelText('Filter by action') as HTMLSelectElement
    const option = screen.getByRole('option', { name: 'Content reports' }) as HTMLOptionElement
    fireEvent.change(select, { target: { value: option.value } })

    expect(lastFilters()).toMatchObject({ action_prefix: 'content_reported', offset: 0 })
  })

  it('a content_reported row shows its reason in the Detail column', () => {
    render(<AuditPane />)
    expect(screen.getByText('content_reported')).toBeTruthy()
    expect(screen.getByText('inaccurate')).toBeTruthy()
  })

  it('linkFor routes link_type content_report to the Governance tab', () => {
    expect(linkFor(notice('content_report'))).toBe('/command-center?tab=governance')
    expect(linkFor(notice('content_report', '41'))).toBe('/command-center?tab=governance')
  })

  it('the Governance route names a tab the Command Center shell actually reads', () => {
    const shell = readFileSync(
      path.resolve(__dirname, '..', '..', 'command-center-shell.tsx'),
      'utf8',
    )
    const route = linkFor(notice('content_report'))!
    const tab = new URLSearchParams(route.split('?')[1]).get('tab')!
    expect(shell).toContain(`'${tab}'`)
  })
})
