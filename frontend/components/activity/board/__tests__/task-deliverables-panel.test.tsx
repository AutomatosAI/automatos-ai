/**
 * PRD-234 S2 — the ticket's Deliverables tab lists what was registered against
 * this task and asks the shared hook with source_id=<task id>.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

const useDeliverablesMock = vi.hoisted(() => vi.fn())

vi.mock('@/hooks/use-deliverables-api', () => ({
  DEFAULT_FILTERS: { artifact_type: null, source_type: null, source_type_exclude: null, source_id: null, agent_id: null, date_range: 'all', search: '' },
  useDeliverables: useDeliverablesMock,
}))
vi.mock('next/link', () => ({ default: ({ children, href }: { children: React.ReactNode; href: string }) => <a href={href}>{children}</a> }))

import { TaskDeliverablesPanel } from '../task-deliverables-panel'

describe('TaskDeliverablesPanel', () => {
  it('lists the registered files for the task and scopes the query to it', () => {
    useDeliverablesMock.mockReturnValue({
      isLoading: false,
      isError: false,
      data: { pages: [{ deliverables: [
        { id: 'd1', title: 'hello.py', artifact_type: 'code', file_path: 'sessions/68/hello.py', preview_url: '/api/workspaces/ws/files/content?path=sessions%2F68%2Fhello.py' },
        { id: 'd2', title: 'README.md', artifact_type: 'document', file_path: 'sessions/68/README.md' },
      ], total: 2, offset: 0 }] },
    })
    render(<TaskDeliverablesPanel taskId={68} />)
    expect(useDeliverablesMock).toHaveBeenCalledWith(expect.objectContaining({ source_id: '68' }))
    expect(screen.getAllByTestId('task-deliverable-row')).toHaveLength(2)
    expect(screen.getByText('sessions/68/hello.py')).toBeInTheDocument()
    expect(screen.getByLabelText('Open hello.py')).toHaveAttribute('href', expect.stringContaining('/files/content'))
  })

  it('says so when nothing is registered', () => {
    useDeliverablesMock.mockReturnValue({ isLoading: false, isError: false, data: { pages: [{ deliverables: [], total: 0, offset: 0 }] } })
    render(<TaskDeliverablesPanel taskId="69" />)
    expect(screen.getByText(/Nothing registered yet/)).toBeInTheDocument()
  })
})
