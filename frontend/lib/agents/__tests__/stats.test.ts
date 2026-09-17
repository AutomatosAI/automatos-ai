import { describe, it, expect } from 'vitest'
import { summariseAgents } from '@/lib/agents/stats'

describe('summariseAgents', () => {
  it('counts from the roster when the stats read is absent, and never invents a success rate', () => {
    const s = summariseAgents([{ status: 'active' }, { status: 'active' }, { status: 'failed' }, { status: 'inactive' }])
    expect(s).toEqual({ total: 4, active: 2, attention: 2, failing: 1, avgSuccess: null })
  })
  it('prefers the stats read for totals and carries its average', () => {
    const s = summariseAgents([{ status: 'active' }], { total_agents: 8, active_agents: 8, average_performance: 92.5 })
    expect(s.total).toBe(8)
    expect(s.active).toBe(8)
    expect(s.avgSuccess).toBe(92.5)
  })
})
