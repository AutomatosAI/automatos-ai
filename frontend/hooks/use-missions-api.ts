/**
 * Mission Control React Query Hooks — PRD-82A
 *
 * Data fetching for missions API. Uses same patterns as
 * use-reports-api.ts and use-board-tasks-api.ts.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type {
  MissionListResponse,
  MissionDetailResponse,
  MissionApproveRequest,
  MissionRejectRequest,
  MissionReviewRequest,
  MissionCreateRequest,
  MissionResponse,
  RunState,
  SaveAsRoutineRequest,
  SaveAsRoutineResponse,
} from '@/types/missions'

// ── Query Keys ────────────────────────────────────────────────

export const missionQueryKeys = {
  all: ['missions'] as const,
  list: (filters: MissionFilters) => ['missions', 'list', filters] as const,
  detail: (id: string) => ['missions', id] as const,
  field: (id: string) => ['missions', id, 'field'] as const,
}

// ── Filter Types ──────────────────────────────────────────────

export interface MissionFilters {
  state?: RunState
  limit?: number
  offset?: number
}

// ── List Missions ─────────────────────────────────────────────

export function useMissions(filters: MissionFilters = {}) {
  const params = new URLSearchParams()
  if (filters.state) params.set('state', filters.state)
  if (filters.limit) params.set('limit', String(filters.limit))
  if (filters.offset) params.set('offset', String(filters.offset))

  const queryString = params.toString()
  const endpoint = queryString ? `/api/missions?${queryString}` : '/api/missions'

  return useQuery<MissionListResponse>({
    queryKey: missionQueryKeys.list(filters),
    queryFn: () => apiClient.request<MissionListResponse>(endpoint),
    staleTime: 15_000,
    refetchInterval: 30_000,
  })
}

// ── Mission Detail ────────────────────────────────────────────

export function useMission(id: string | null) {
  return useQuery<MissionDetailResponse>({
    queryKey: missionQueryKeys.detail(id!),
    queryFn: () => apiClient.request<MissionDetailResponse>(`/api/missions/${id}`),
    enabled: !!id,
    staleTime: 5_000,
    refetchInterval: 10_000,
  })
}

// ── Mission Field (PRD-108 Visualizer) ───────────────────────

export interface FieldPattern {
  id: string
  key: string
  value: string
  agent_id: number
  strength: number
  decayed_strength: number
  access_count: number
  created_at: string
  last_accessed: string
  is_archived: boolean
}

export interface FieldStability {
  stability: number
  pattern_count: number
  avg_strength: number
  organization?: number
  active_patterns?: number
  decayed_patterns?: number
}

export interface FieldMetrics {
  total_injections: number
  total_queries: number
  avg_query_latency_ms: number
  avg_results_per_query: number
  injections_by_agent: Record<number, number>
  queries_by_agent: Record<number, number>
}

export interface MissionFieldResponse {
  field_id: string | null
  backend: string | null
  patterns: FieldPattern[]
  stability: FieldStability
  metrics: FieldMetrics | null
}

export function useMissionField(id: string | null, enabled = true) {
  return useQuery<MissionFieldResponse>({
    queryKey: missionQueryKeys.field(id!),
    queryFn: () => apiClient.request<MissionFieldResponse>(`/api/missions/${id}/field`),
    enabled: !!id && enabled,
    staleTime: 5_000,
    refetchInterval: 8_000,
  })
}

// ── Create Mission ────────────────────────────────────────────

export function useCreateMission() {
  const queryClient = useQueryClient()

  return useMutation<MissionResponse, Error, MissionCreateRequest>({
    mutationFn: (body) =>
      apiClient.request<MissionResponse>('/api/missions', {
        method: 'POST',
        body: body as unknown as BodyInit,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.all })
    },
  })
}

// ── Approve Plan ──────────────────────────────────────────────

export function useApproveMission() {
  const queryClient = useQueryClient()

  return useMutation<MissionResponse, Error, { id: string; body?: MissionApproveRequest }>({
    mutationFn: ({ id, body }) =>
      apiClient.request<MissionResponse>(`/api/missions/${id}/approve`, {
        method: 'POST',
        body: (body ?? {}) as unknown as BodyInit,
      }),
    onSuccess: (_data, { id }) => {
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.all })
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.detail(id) })
    },
  })
}

// ── Reject Plan ───────────────────────────────────────────────

export function useRejectMission() {
  const queryClient = useQueryClient()

  return useMutation<MissionResponse, Error, { id: string; body: MissionRejectRequest }>({
    mutationFn: ({ id, body }) =>
      apiClient.request<MissionResponse>(`/api/missions/${id}/reject`, {
        method: 'POST',
        body: body as unknown as BodyInit,
      }),
    onSuccess: (_data, { id }) => {
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.all })
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.detail(id) })
    },
  })
}

// ── Human Review ──────────────────────────────────────────────

export function useReviewMission() {
  const queryClient = useQueryClient()

  return useMutation<MissionResponse, Error, { id: string; body: MissionReviewRequest }>({
    mutationFn: ({ id, body }) =>
      apiClient.request<MissionResponse>(`/api/missions/${id}/review`, {
        method: 'POST',
        body: body as unknown as BodyInit,
      }),
    onSuccess: (_data, { id }) => {
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.all })
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.detail(id) })
    },
  })
}

// ── Pause ─────────────────────────────────────────────────────

export function usePauseMission() {
  const queryClient = useQueryClient()

  return useMutation<MissionResponse, Error, string>({
    mutationFn: (id) =>
      apiClient.request<MissionResponse>(`/api/missions/${id}/pause`, {
        method: 'POST',
      }),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.all })
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.detail(id) })
    },
  })
}

// ── Resume ────────────────────────────────────────────────────

export function useResumeMission() {
  const queryClient = useQueryClient()

  return useMutation<MissionResponse, Error, string>({
    mutationFn: (id) =>
      apiClient.request<MissionResponse>(`/api/missions/${id}/resume`, {
        method: 'POST',
      }),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.all })
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.detail(id) })
    },
  })
}

// ── Cancel ────────────────────────────────────────────────────

export function useCancelMission() {
  const queryClient = useQueryClient()

  return useMutation<MissionResponse, Error, string>({
    mutationFn: (id) =>
      apiClient.request<MissionResponse>(`/api/missions/${id}/cancel`, {
        method: 'POST',
      }),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.all })
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.detail(id) })
    },
  })
}

// ── Replan Failed Mission (PRD-82B US-005) ───────────────────

export function useReplanMission() {
  const queryClient = useQueryClient()

  return useMutation<MissionResponse, Error, { id: string; notes?: string }>({
    mutationFn: ({ id, notes }) =>
      apiClient.request<MissionResponse>(`/api/missions/${id}/replan`, {
        method: 'POST',
        body: (notes ? { notes } : {}) as unknown as BodyInit,
      }),
    onSuccess: (_data, { id }) => {
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.all })
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.detail(id) })
    },
  })
}

// ── Re-run Mission (clone goal + config into new mission) ────

export function useRerunMission() {
  const queryClient = useQueryClient()

  return useMutation<MissionResponse, Error, MissionCreateRequest>({
    mutationFn: (body) =>
      apiClient.request<MissionResponse>('/api/missions', {
        method: 'POST',
        body: body as unknown as BodyInit,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.all })
    },
  })
}

// ── Save as Routine (PRD-82B US-008) ─────────────────────────

export function useSaveAsRoutine() {
  const queryClient = useQueryClient()

  return useMutation<SaveAsRoutineResponse, Error, { id: string; body: SaveAsRoutineRequest }>({
    mutationFn: ({ id, body }) =>
      apiClient.request<SaveAsRoutineResponse>(`/api/missions/${id}/save-as-routine`, {
        method: 'POST',
        body: body as unknown as BodyInit,
      }),
    onSuccess: (_data, { id }) => {
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.all })
      queryClient.invalidateQueries({ queryKey: missionQueryKeys.detail(id) })
    },
  })
}
