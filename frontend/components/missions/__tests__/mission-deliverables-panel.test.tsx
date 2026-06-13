/**
 * PRD-164 S3 — mission page Deliverables tab (vitest proxy for the
 * browser AC; the loop is headless).
 *
 * Pins: the panel lists this mission's deliverables (scoped by
 * source_id=<missionId> through useDeliverables), renders title/agent/type
 * rows, and shows an honest empty state.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const useDeliverablesMock = vi.fn()

vi.mock('@/hooks/use-deliverables-api', async () => {
  const actual = await vi.importActual<
    typeof import('@/hooks/use-deliverables-api')
  >('@/hooks/use-deliverables-api')
  return {
    ...actual,
    useDeliverables: (filters: unknown) => useDeliverablesMock(filters),
  }
})

import { MissionDeliverablesPanel } from '../mission-deliverables-panel'

function deliverable(overrides: Record<string, unknown> = {}) {
  return {
    id: 'd-1',
    workspace_id: 'ws-1',
    source_type: 'mission',
    source_id: 'mission-42',
    agent_id: 7,
    agent_name: 'Auto',
    artifact_type: 'report',
    title: 'Competitor pricing synthesis',
    summary: 'Key findings on pricing.',
    storage_type: 'workspace',
    file_path: 'reports/auto/pricing.md',
    file_name: 'pricing.md',
    file_type: 'md',
    file_size_bytes: 1024,
    preview_url: '/api/workspaces/ws-1/files/content?path=reports/auto/pricing.md',
    preview_type: 'markdown',
    extra: {},
    status: 'ready',
    created_at: '2026-06-13T09:00:00Z',
    updated_at: '2026-06-13T09:00:00Z',
    ...overrides,
  }
}

function queryResult(deliverables: unknown[], extra: Record<string, unknown> = {}) {
  return {
    data: {
      pages: [
        { success: true, deliverables, total: deliverables.length, limit: 24, offset: 0 },
      ],
    },
    isLoading: false,
    isError: false,
    hasNextPage: false,
    isFetchingNextPage: false,
    fetchNextPage: vi.fn(),
    ...extra,
  }
}

beforeEach(() => {
  useDeliverablesMock.mockReset()
})

describe('MissionDeliverablesPanel (PRD-164 S3)', () => {
  it('scopes the query to this mission via source_id', () => {
    useDeliverablesMock.mockReturnValue(queryResult([]))
    render(<MissionDeliverablesPanel missionId="mission-42" />)
    expect(useDeliverablesMock).toHaveBeenCalledWith(
      expect.objectContaining({ source_id: 'mission-42' }),
    )
  })

  it('renders deliverable rows with title, agent and artifact type', () => {
    useDeliverablesMock.mockReturnValue(
      queryResult([
        deliverable(),
        deliverable({
          id: 'd-2',
          title: 'Generated PDF report',
          artifact_type: 'document',
          agent_name: 'Scribe',
        }),
      ]),
    )
    render(<MissionDeliverablesPanel missionId="mission-42" />)

    const rows = screen.getAllByTestId('mission-deliverable-row')
    expect(rows).toHaveLength(2)
    expect(screen.getByText('Competitor pricing synthesis')).toBeInTheDocument()
    expect(screen.getByText('Generated PDF report')).toBeInTheDocument()
    expect(screen.getByText('report')).toBeInTheDocument()
    expect(screen.getByText('document')).toBeInTheDocument()
    expect(screen.getByText(/Scribe/)).toBeInTheDocument()
    // open link uses the preview URL
    expect(
      screen.getByLabelText('Open Competitor pricing synthesis'),
    ).toHaveAttribute('href', expect.stringContaining('pricing.md'))
  })

  it('shows an honest empty state when the mission has no deliverables', () => {
    useDeliverablesMock.mockReturnValue(queryResult([]))
    render(<MissionDeliverablesPanel missionId="mission-42" />)
    expect(screen.getByTestId('mission-deliverables-empty')).toBeInTheDocument()
    expect(screen.getByText(/No deliverables yet/)).toBeInTheDocument()
  })

  it('shows skeletons while loading', () => {
    useDeliverablesMock.mockReturnValue(
      queryResult([], { isLoading: true, data: undefined }),
    )
    render(<MissionDeliverablesPanel missionId="mission-42" />)
    expect(screen.getByTestId('mission-deliverables-loading')).toBeInTheDocument()
  })
})
