/**
 * Activity API hooks (PRD-72 US-006)
 * React Query integration for the unified activity feed and stats endpoints.
 */

import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

// ============= TYPES =============

export interface ActivityAgent {
  id: number | null
  name: string
  avatar_url: string | null
}

export interface ActivityStepProgress {
  current: number
  total: number
  steps: Array<{
    name: string
    status: 'pending' | 'running' | 'completed' | 'failed'
  }>
}

export interface ActivityChannel {
  type: 'telegram' | 'whatsapp' | 'slack' | 'email' | 'webchat'
  name: string
}

export interface ActivityFeedItem {
  id: string
  type: 'chat' | 'routine' | 'recipe' | 'mission' | 'task'
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused'
  started_at: string | null
  completed_at: string | null
  duration_seconds: number | null
  agent: ActivityAgent | null
  agents: ActivityAgent[]
  summary: string
  source_id: string | null
  source_url: string | null
  trigger: 'manual' | 'scheduled' | 'event' | 'webhook' | 'heartbeat' | string | null
  channel?: ActivityChannel | null
  step_progress?: ActivityStepProgress | null
  error_message: string | null
  tokens_used?: number | null
  total_tokens?: number | null
  total_duration_ms?: number | null
}

export interface ActivityFeedResponse {
  items: ActivityFeedItem[]
  total: number
  limit: number
  offset: number
}

export interface ActivityStats {
  working_now: number
  channels_live: number
  completed_today: number
  needs_attention: number
  agents_active: number
  tasks_in_queue: number
  period: string
}

export interface ActivityFeedFilters {
  types?: string[]
  status?: string | null
  period?: string
  limit?: number
  offset?: number
}

export interface ScheduleRecurrence {
  cron_expression: string | null
  interval_minutes: number | null
  timezone: string
  active_hours: Record<string, unknown> | null
}

export interface ScheduleItem {
  id: string
  name: string
  type: 'routine' | 'recipe' | 'task' | 'mission'
  next_run_at: string | null
  frequency: string
  agent_name: string | null
  agent_id: number | null
  recurrence?: ScheduleRecurrence
}

export interface ScheduleResponse {
  scheduled: ScheduleItem[]
  scheduler_active?: boolean
}

export interface AgentReport {
  agent_id: number
  agent_name: string
  agent_icon: string | null
  status: 'completed' | 'failed' | 'no_data' | 'error'
  summary: string
  last_run: string | null
  execution_id: string | null
  latest_file_path: string | null
}

export interface AgentReportsResponse {
  reports: AgentReport[]
}

// ============= QUERY KEYS =============

export const activityQueryKeys = {
  all: ['activity'] as const,
  feed: (filters?: ActivityFeedFilters) => ['activity', 'feed', filters] as const,
  stats: (period?: string) => ['activity', 'stats', period] as const,
  schedule: (range?: string) => ['activity', 'schedule', range] as const,
  agentReports: (agentIds?: number[]) => ['activity', 'agent-reports', agentIds] as const,
}

// ============= QUERY HOOKS =============

/**
 * Fetch the unified activity feed with polling.
 * Merges chats, routines, and recipes sorted by started_at DESC.
 */
export function useActivityFeed(filters?: ActivityFeedFilters) {
  const params = new URLSearchParams()

  if (filters?.types && filters.types.length > 0) {
    params.set('type', filters.types.join(','))
  }
  if (filters?.status) {
    params.set('status', filters.status)
  }
  if (filters?.period) {
    params.set('period', filters.period)
  }
  if (filters?.limit) {
    params.set('limit', String(filters.limit))
  }
  if (filters?.offset) {
    params.set('offset', String(filters.offset))
  }

  const queryString = params.toString()
  const endpoint = `/api/activity/feed${queryString ? `?${queryString}` : ''}`

  return useQuery<ActivityFeedResponse>({
    queryKey: activityQueryKeys.feed(filters),
    queryFn: () => apiClient.request<ActivityFeedResponse>(endpoint),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

/**
 * Fetch hero-card stats with polling.
 * Returns working_now, agents_active, tasks_in_queue, needs_attention.
 */
export function useActivityStats(period: string = '1d') {
  return useQuery<ActivityStats>({
    queryKey: activityQueryKeys.stats(period),
    queryFn: () =>
      apiClient.request<ActivityStats>(`/api/activity/stats?period=${period}`),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

/**
 * Fetch the workspace schedule for the calendar (PRD-162).
 *
 * The endpoint is now a stateless DB read — identical on every worker — so the
 * PRD-154 S11 stopgap (throw-on-inactive + aggressive worker-hop retry) is gone.
 * This is a plain cached query: a 30–60s `staleTime` serves instantly from cache
 * while the background `refetchInterval` keeps it fresh, and `keepPreviousData`
 * means a refetch never blanks the calendar.
 */
export function useActivitySchedule(range: string = '7d') {
  return useQuery<ScheduleResponse>({
    queryKey: activityQueryKeys.schedule(range),
    queryFn: () =>
      apiClient.request<ScheduleResponse>(`/api/activity/schedule?range=${range}`),
    refetchInterval: 60000,
    staleTime: 45000,
    keepPreviousData: true,
  })
}

export interface SchedulerHealth {
  healthy: boolean | null
  last_fired_at: string | null
}

/**
 * Advisory scheduler health for the calendar banner (PRD-162 S2). Separate from
 * the schedule query and NON-BLOCKING — the calendar renders configured
 * schedules regardless. `healthy: null` means "can't tell" → the banner stays
 * quiet.
 */
export function useSchedulerHealth() {
  return useQuery<SchedulerHealth>({
    queryKey: ['activity', 'scheduler-health'],
    queryFn: () =>
      apiClient.request<SchedulerHealth>('/api/activity/scheduler-health'),
    refetchInterval: 60000,
    staleTime: 45000,
  })
}

/**
 * Fetch latest reports from pinned agents.
 */
export function useAgentReports(agentIds: number[]) {
  const idsParam = agentIds.join(',')
  return useQuery<AgentReportsResponse>({
    queryKey: activityQueryKeys.agentReports(agentIds),
    queryFn: () =>
      apiClient.request<AgentReportsResponse>(`/api/activity/agent-reports?agent_ids=${idsParam}`),
    enabled: agentIds.length > 0,
    refetchInterval: 60000,
    staleTime: 30000,
  })
}

// ============= BOARD STATS =============

export interface BoardColumnCount {
  status: string
  count: number
}

export interface BoardPriorityCount {
  priority: string
  count: number
}

export interface BoardTypeCount {
  type: string
  count: number
  percentage: number
}

export interface BoardAgentWorkload {
  agent_id: number
  agent_name: string
  agent_icon: string | null
  task_count: number
}

export interface BoardStats {
  columns: BoardColumnCount[]
  total_tasks: number
  priorities: BoardPriorityCount[]
  types: BoardTypeCount[]
  workload: BoardAgentWorkload[]
}

/**
 * Fetch board-level stats for Summary widgets (donut, priority, types, workload).
 */
export function useBoardStats(period: string = '1d') {
  return useQuery<BoardStats>({
    queryKey: ['activity', 'board-stats', period],
    queryFn: () =>
      apiClient.request<BoardStats>(`/api/activity/board/stats?period=${period}`),
    refetchInterval: 60000,
    staleTime: 30000,
  })
}
