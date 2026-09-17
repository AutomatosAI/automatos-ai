/**
 * PRD-244 W5a — the four numbers the Agent Management head shows, computed
 * once from the roster and the stats read. Pure, so both the Studio page and
 * a test can call it; every value is honest (no fabricated percentages).
 */
export interface AgentLike {
  status?: string | null
}

export interface AgentStatsLike {
  total_agents?: number | null
  active_agents?: number | null
  average_performance?: number | null
}

export interface AgentSummary {
  total: number
  active: number
  /** Agents whose status is error, failed or inactive. */
  attention: number
  /** The error/failed subset of `attention`. */
  failing: number
  /** Average success rate in percent, or null when the read has none. */
  avgSuccess: number | null
}

const FAILING = new Set(['error', 'failed'])

export function summariseAgents(agents: readonly AgentLike[], stats?: AgentStatsLike | null): AgentSummary {
  const failing = agents.filter((a) => FAILING.has(a.status ?? '')).length
  const inactive = agents.filter((a) => a.status === 'inactive').length
  const activeFromRoster = agents.filter((a) => a.status === 'active').length
  const avg = stats?.average_performance
  return {
    total: stats?.total_agents || agents.length,
    active: stats?.active_agents || activeFromRoster,
    attention: failing + inactive,
    failing,
    avgSuccess: typeof avg === 'number' && Number.isFinite(avg) ? avg : null,
  }
}
