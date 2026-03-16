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
