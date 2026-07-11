/**
 * PRD-196 S7 (P2-15, governance I.4) — GDPR self-service, honesty in front.
 *
 * The erase-subject report renders exactly what was and was NOT deleted (the
 * gaps + untagged-history are the point, not a footnote); the erase-workspace
 * button refuses to submit until the exact workspace-id echo is typed.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import type { GdprSubjectErasureResult } from '@/lib/api-client'

const eraseSubjectMutate = vi.fn().mockResolvedValue({})
const eraseWorkspaceMutate = vi.fn().mockResolvedValue({})
const subjectHook = vi.fn()

vi.mock('@/hooks/use-gdpr', () => ({
  useGdprEraseSubject: () => subjectHook(),
  useGdprEraseWorkspace: () => ({ isLoading: false, mutateAsync: eraseWorkspaceMutate }),
}))

vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => ({ apiClient: { getGdprExport: vi.fn() } }))

import { CompliancePane } from '../governance/compliance-pane'

const report: GdprSubjectErasureResult = {
  workspace_id: 'ws-123',
  subject_id: 'user:42',
  erased_at: '2026-07-11T10:00:00Z',
  sql: { deleted: 0 },
  derived: { field_memory_deleted: 4, durable_memory_deleted: 2 },
  gaps: [{ store: 'sql', reason: 'SQL schema is workspace-scoped, not subject-scoped.' }],
  untagged_history: {
    stores: ['field_memory', 'durable_memory'],
    reason: 'Rows written before the tag existed carry no subject_id.',
  },
}

describe('CompliancePane — PRD-196 S7', () => {
  beforeEach(() => {
    eraseSubjectMutate.mockClear()
    eraseWorkspaceMutate.mockClear()
    subjectHook.mockReturnValue({ isLoading: false, mutateAsync: eraseSubjectMutate, data: undefined })
    localStorage.setItem('last_active_workspace', 'ws-123')
  })

  it('renders the honest erase report — deleted counts, gaps, and untagged history', () => {
    subjectHook.mockReturnValue({ isLoading: false, mutateAsync: eraseSubjectMutate, data: report })
    render(<CompliancePane />)
    // deleted counts
    expect(screen.getByText(/Field memory deleted:/)).toBeInTheDocument()
    expect(screen.getByText('4')).toBeInTheDocument()
    // the gap is shown, not hidden
    expect(screen.getByText(/Could not delete/)).toBeInTheDocument()
    expect(screen.getByText(/workspace-scoped, not subject-scoped/)).toBeInTheDocument()
    // the untagged-history caveat is shown
    expect(screen.getByText(/Untagged history/)).toBeInTheDocument()
    expect(screen.getByText(/before the tag existed/)).toBeInTheDocument()
  })

  it('disables workspace erase until the exact id echo is typed', () => {
    render(<CompliancePane />)
    const button = screen.getByRole('button', { name: /Erase workspace/ })
    expect(button).toBeDisabled()

    const input = screen.getByLabelText(/Confirm workspace id/)
    fireEvent.change(input, { target: { value: 'wrong-id' } })
    expect(button).toBeDisabled()

    fireEvent.change(input, { target: { value: 'ws-123' } })
    expect(button).not.toBeDisabled()

    fireEvent.click(button)
    expect(eraseWorkspaceMutate).toHaveBeenCalledWith('ws-123')
  })
})
