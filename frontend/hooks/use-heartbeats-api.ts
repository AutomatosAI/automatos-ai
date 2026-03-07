/**
 * Heartbeat API hooks (PRD-72 US-007)
 * React Query integration for heartbeat list, toggle, and execution history.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from '@/lib/api-client'

// ============= TYPES =============

export interface HeartbeatLastRun {
  status: string
  created_at: string | null
  tokens_used: number | null
}

export interface HeartbeatConfig {
  id: number
  agent_id: number
  agent_name: string
  agent_description: string | null
  enabled: boolean
  interval_minutes: number
  prompt: string
  auto_act: boolean
  timezone: string
  active_hours_start: string
  active_hours_end: string
  last_run: HeartbeatLastRun | null
  next_run_at: string | null
}

export interface HeartbeatListResponse {
  heartbeats: HeartbeatConfig[]
  total: number
}

export interface HeartbeatToggleResponse {
  id: number
  agent_id: number
  agent_name: string
  enabled: boolean
  interval_minutes: number
  message: string
}

export interface HeartbeatExecution {
  id: number
  status: string
  started_at: string | null
  completed_at: string | null
  duration_seconds: number | null
  tokens_used: number | null
  cost: number | null
  error_message: string | null
  findings_count: number
}

export interface HeartbeatExecutionsResponse {
  agent_id: number
  agent_name: string
  executions: HeartbeatExecution[]
  total: number
}

// ============= QUERY KEYS =============

export const heartbeatQueryKeys = {
  all: ['heartbeats'] as const,
  detail: (id: number) => ['heartbeats', id] as const,
  executions: (id: number) => ['heartbeats', id, 'executions'] as const,
}

// ============= QUERY HOOKS =============

/**
 * Fetch all heartbeat configs for the current workspace.
 * Used by the Routines tab to render routine cards.
 */
export function useHeartbeats() {
  return useQuery<HeartbeatListResponse>({
    queryKey: heartbeatQueryKeys.all,
    queryFn: () => apiClient.request<HeartbeatListResponse>('/api/heartbeat/workspace'),
    staleTime: 15000,
    refetchInterval: 30000,
  })
}

/**
 * Fetch execution history for a specific heartbeat (agent).
 * Only fetches when enabled (e.g. when routine card is expanded).
 */
export function useHeartbeatExecutions(
  heartbeatId: number | null,
  options?: { enabled?: boolean }
) {
  return useQuery<HeartbeatExecutionsResponse>({
    queryKey: heartbeatQueryKeys.executions(heartbeatId!),
    queryFn: () =>
      apiClient.request<HeartbeatExecutionsResponse>(
        `/api/heartbeat/${heartbeatId}/executions`
      ),
    enabled: !!heartbeatId && (options?.enabled !== false),
    staleTime: 15000,
  })
}

// ============= MUTATION HOOKS =============

/**
 * Toggle a heartbeat's enabled state (pause/resume).
 * Shows toast feedback and invalidates heartbeat queries.
 */
export function useToggleHeartbeat() {
  const queryClient = useQueryClient()

  return useMutation<HeartbeatToggleResponse, Error, number>({
    mutationFn: (heartbeatId: number) =>
      apiClient.request<HeartbeatToggleResponse>(
        `/api/heartbeat/${heartbeatId}/toggle`,
        { method: 'PATCH' }
      ),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: heartbeatQueryKeys.all })
      queryClient.invalidateQueries({ queryKey: heartbeatQueryKeys.detail(data.agent_id) })
      toast.success(data.message)
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to toggle heartbeat')
    },
  })
}
