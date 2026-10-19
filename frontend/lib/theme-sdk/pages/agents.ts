/**
 * PRD-261 A-12d — SDK area `pages/agents`: the data the core /agents page reads (the roster and the
 * stats strip, components/agents/studio/agent-management-studio.tsx), for a theme's own /agents page
 * (docs/theme-host/sdk-surface-map.md, A-10c).
 *
 * The hooks are the host's own (hooks/use-agent-api.ts), called exactly as the core page calls them,
 * so every read goes through the backend client with the workspace and the token, which a theme
 * never holds, and shares the core page's cache. Each hands back a frozen, read-only view of
 * react-query's result. The page's other hook, usePageAPI, is host-only (it tags requests with the
 * page name; the map's row says why), so this area wraps no other core hook and adds none. The types
 * are declared again for themes in packages/theme-sdk/areas/pages-agents.d.ts.
 */
import { useAgentStats as useHostAgentStats, useAgents as useHostAgents } from '@/hooks/use-agent-api'

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

/** The roster: react-query's result, read only. */
export interface RosterAgentsQuery {
  readonly data: readonly RosterAgent[] | undefined
  readonly isLoading: boolean
  readonly isFetching: boolean
  readonly isError: boolean
  readonly error: unknown
  readonly refetch: () => Promise<unknown>
}

/** The agent counts: react-query's result, read only. */
export interface AgentStatisticsQuery {
  readonly data: AgentStatistics | undefined
  readonly isLoading: boolean
  readonly isFetching: boolean
  readonly isError: boolean
  readonly error: unknown
  readonly refetch: () => Promise<unknown>
}

/** The fields of a host query result this area hands on. */
interface HostQueryResult {
  readonly data: unknown
  readonly isLoading: boolean
  readonly isFetching: boolean
  readonly isError: boolean
  readonly error: unknown
  readonly refetch: () => Promise<unknown>
}

/** What a theme gets: the same fields, with `data` typed as the backend's answer. */
interface QueryView<T> extends Omit<HostQueryResult, 'data'> {
  readonly data: T | undefined
}

/** A frozen copy of the fields a theme reads, with `data` as the backend's answer. */
function queryView<T>(result: HostQueryResult): QueryView<T> {
  return Object.freeze({
    data: result.data as T | undefined,
    isLoading: result.isLoading,
    isFetching: result.isFetching,
    isError: result.isError,
    error: result.error,
    refetch: result.refetch,
  })
}

/** The workspace's agents, as the core roster lists them (the workspace's hidden system agents left out). */
export function useAgents(): RosterAgentsQuery {
  return queryView<readonly RosterAgent[]>(useHostAgents())
}

/** The workspace's agent counts (total, active, inactive, by type), refreshed every 30 s as the core page's are. */
export function useAgentStats(): AgentStatisticsQuery {
  return queryView<AgentStatistics>(useHostAgentStats())
}
