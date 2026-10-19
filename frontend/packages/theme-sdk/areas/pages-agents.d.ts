/**
 * PRD-261 A-12d — SDK area `pages/agents` (frontend/lib/theme-sdk/pages/agents.ts): the data the core /agents
 * page reads, for a theme's own /agents page (registerPage('/agents', …)). The hooks are the host's own, called
 * as the core page calls them, so they read through the host's backend client and share the core page's cache;
 * each hands back the same small, frozen view as the other page areas: `{ data, isLoading, isFetching, error,
 * refetch }`, null data until it answers, the error as its message, and the data a deep, frozen copy.
 */

/** One agent of the workspace's roster: the fields of the backend's agent a theme can rely on. */
export interface RosterAgent {
  readonly id: number
  /** The id the agent is known by outside the app (widgets, the public API). */
  readonly public_id?: string | null
  readonly name: string
  readonly description: string | null
  readonly job_title?: string | null
  readonly team?: string | null
  readonly agent_type: string
  /** 'active', 'inactive', 'error', … as the backend reports it. */
  readonly status: string
  readonly tags?: readonly string[]
  readonly priority_level?: string
  readonly is_system_agent?: boolean
  readonly slug?: string | null
  readonly marketplace_category?: string | null
  /** The host's icon name for the agent's category, or null when the host maps none. */
  readonly premium_icon: string | null
  /** Why the agent cannot run now (no AI credit, no CLI host online); absent or null when it can. */
  readonly unavailable?: string | null
  readonly created_at: string
  readonly updated_at: string
}

/** The workspace's agent counts, as the stats strip reads them. */
export interface AgentStatistics {
  readonly total_agents: number
  readonly active_agents: number
  readonly inactive_agents: number
  /** The number of agents of each agent type. */
  readonly agents_by_type: Readonly<Record<string, number>>
  /** When the backend counted (ISO 8601). */
  readonly timestamp: string
}

/** The roster: null until it answers; `isLoading` is the first load only. */
export interface RosterAgentsQuery {
  readonly data: readonly RosterAgent[] | null
  readonly isLoading: boolean
  readonly isFetching: boolean
  readonly error: string | null
  readonly refetch: () => void
}

/** The agent counts: null until they answer; `isLoading` is the first load only. */
export interface AgentStatisticsQuery {
  readonly data: AgentStatistics | null
  readonly isLoading: boolean
  readonly isFetching: boolean
  readonly error: string | null
  readonly refetch: () => void
}

/** The workspace's agents, as the core roster lists them (the workspace's hidden system agents left out). */
export declare function useAgents(): RosterAgentsQuery

/** The workspace's agent counts (total, active, inactive, by type), refreshed every 30 s as the core page's are. */
export declare function useAgentStats(): AgentStatisticsQuery
