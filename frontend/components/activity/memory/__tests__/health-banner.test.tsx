import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { MemoryHealthResponse } from '@/hooks/use-memory-explorer-api'

// Memory on by default (the local stack runs Qdrant): a fresh install still has no
// embeddings key, so Knowledge → Memory is unavailable. The status tile said only
// "Unavailable"; it now shows the reason /api/v1/memory/health gives, and nothing more
// while memory works.

let healthData: MemoryHealthResponse | undefined

vi.mock('@/hooks/use-memory-explorer-api', () => ({
  useMemoryHealth: () => ({ data: healthData }),
}))

import { HealthBanner } from '../health-banner'

const NO_KEY =
  'Memory needs an embeddings key. Add the key for your embedding provider (OpenRouter, unless ' +
  'Settings → System Settings → Embeddings names another) in Settings → API Keys, then restart ' +
  'the backend so memory picks it up.'

function health(overrides: Partial<MemoryHealthResponse>): MemoryHealthResponse {
  return {
    success: true,
    mem0_available: true,
    total_memories: 0,
    oldest_memory: null,
    newest_memory: null,
    search_effectiveness: { total_searches: 0, hits: 0, hit_rate: 0 },
    health_status: 'degraded',
    reason: null,
    ...overrides,
  }
}

describe('HealthBanner', () => {
  beforeEach(() => {
    healthData = undefined
  })

  it('says why memory is unavailable', () => {
    healthData = health({ health_status: 'unavailable', reason: NO_KEY })
    render(<HealthBanner />)
    expect(screen.getByText('unavailable')).toBeTruthy()
    expect(screen.getByRole('status').textContent).toBe(NO_KEY)
  })

  it('shows no reason while memory works', () => {
    healthData = health({ health_status: 'healthy', total_memories: 3 })
    render(<HealthBanner />)
    expect(screen.getByText('healthy')).toBeTruthy()
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('shows no reason when an older backend sends none', () => {
    healthData = { ...health({ health_status: 'unavailable', mem0_available: false }), reason: undefined }
    render(<HealthBanner />)
    expect(screen.queryByRole('status')).toBeNull()
  })
})
