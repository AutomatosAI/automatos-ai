import { describe, it, expect, vi, afterEach } from 'vitest'
import { render } from '@testing-library/react'
import type { SubstrateHealthMetric } from '@/lib/api-client'

// PRD-197 S4 — the RETRIEVAL substrate tile. Per-seam health (documents /
// memory / field) rolled into the Command Center "is-it-working" strip:
// healthy renders green with the roll-up count, a degraded seam is named,
// and no-data renders the honest "awaiting searches" muted state — never a
// fabricated green.

const holder: { substrate?: SubstrateHealthMetric } = {}

vi.mock('@/hooks/use-analytics-api', () => ({
  useWorkspaceActivation: () => ({ data: undefined }),
  useMissionSuccessRate: () => ({ data: undefined }),
  useErrorsBySubsystem: () => ({ data: undefined }),
  useWidgetEngagement: () => ({ data: undefined }),
  usePrimitiveHealth: () => ({ data: undefined }),
  useSLOs: () => ({ data: undefined }),
  useSubstrateHealth: () => ({ data: holder.substrate }),
  useDeliverableFreshness: () => ({ data: undefined }),
  useCommerceIntegrity: () => ({ data: undefined }),
}))

vi.mock('@/hooks/use-governance', () => ({
  useGovernanceStatus: () => ({ data: undefined }),
}))

function seam(
  name: string,
  status: 'green' | 'degraded' | 'down' | 'unknown',
  searches = 25,
) {
  return {
    seam: name,
    searches,
    error_rate: status === 'down' ? 0.5 : status === 'degraded' ? 0.1 : 0,
    empty_rate: 0.2,
    avg_latency_ms: 40.0,
    p95_latency_ms: 120.0,
    status,
  }
}

async function renderStrip() {
  const mod = await import('@/components/command-center/is-it-working-strip')
  return render(<mod.IsItWorkingStrip />)
}

afterEach(() => {
  holder.substrate = undefined
})

describe('PRD-197 S4 — RETRIEVAL substrate tile', () => {
  it('renders healthy: all seams green with the roll-up count', async () => {
    holder.substrate = {
      generated_at: '2026-07-16T21:00:00Z',
      window_seconds: 86400,
      seams: [seam('documents', 'green'), seam('memory', 'green'), seam('field', 'green')],
    }
    const { getByText } = await renderStrip()
    expect(getByText('RETRIEVAL')).toBeTruthy()
    expect(getByText('3/3')).toBeTruthy()
    expect(getByText('all seams green · 24h')).toBeTruthy()
  })

  it('renders degraded: the degraded seam is named', async () => {
    holder.substrate = {
      generated_at: '2026-07-16T21:00:00Z',
      window_seconds: 86400,
      seams: [seam('documents', 'green'), seam('memory', 'degraded'), seam('field', 'green')],
    }
    const { getByText } = await renderStrip()
    expect(getByText('2/3')).toBeTruthy()
    expect(getByText('memory degraded')).toBeTruthy()
  })

  it('renders a down seam as the loudest signal', async () => {
    holder.substrate = {
      generated_at: '2026-07-16T21:00:00Z',
      window_seconds: 86400,
      seams: [seam('documents', 'down'), seam('memory', 'degraded'), seam('field', 'green')],
    }
    const { getByText } = await renderStrip()
    expect(getByText('documents down')).toBeTruthy()
  })

  it('renders the honest no-data state, never a fake green', async () => {
    holder.substrate = {
      generated_at: '2026-07-16T21:00:00Z',
      window_seconds: 86400,
      seams: [
        seam('documents', 'unknown', 0),
        seam('memory', 'unknown', 0),
        seam('field', 'unknown', 0),
      ],
    }
    const { getByText } = await renderStrip()
    expect(getByText('awaiting searches')).toBeTruthy()
  })
})
