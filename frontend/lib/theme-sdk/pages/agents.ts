/**
 * PRD-261 A-12d — SDK area `pages/agents`: the data the core /agents page reads (the roster and the
 * stats strip, components/agents/studio/agent-management-studio.tsx), for a theme's own /agents page
 * (docs/theme-host/sdk-surface-map.md, A-10c).
 *
 * The hooks are the host's own (hooks/use-agent-api.ts), called exactly as the core page calls them,
 * so every read goes through the backend client with the workspace and the token, which a theme
 * never holds, and shares the core page's cache. Each hands back the same small, frozen view as
 * the other page areas (internal/query-view.ts), never react-query's own result:
 * `{ data, isLoading, isFetching, error, refetch }`, null data until it answers, the error as its
 * message, and the data a deep, frozen copy, so a theme never changes the roster the core
 * page's agent pickers read. The page's other hook, usePageAPI, is host-only (it tags requests with the
 * page name; the map's row says why), so this area wraps no other core hook and adds none. The types
 * are declared again for themes in packages/theme-sdk/areas/pages-agents.d.ts.
 */
import { useAgentStats as useHostAgentStats, useAgents as useHostAgents } from '@/hooks/use-agent-api'

import { useQueryView } from '../internal/query-view'

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
export function useAgents(): RosterAgentsQuery {
  return useQueryView<readonly RosterAgent[]>(useHostAgents())
}

/** The workspace's agent counts (total, active, inactive, by type), refreshed every 30 s as the core page's are. */
export function useAgentStats(): AgentStatisticsQuery {
  return useQueryView<AgentStatistics>(useHostAgentStats())
}
