/**
 * Board task CRUD & planning API hooks (PRD-72 v2).
 * Wraps /api/v1/tasks endpoints with React Query v4.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { boardQueryKeys } from './use-board-tasks'

// ============= TYPES =============

export interface CreateTaskPayload {
  title: string
  description: string
  priority: 'urgent' | 'high' | 'medium' | 'low'
  assigned_agent_id?: number
  tags?: string[]
  review_mode?: 'human' | 'llm' | 'auto'
  raw_prompt?: string
  planning_data?: any
}

export interface UpdateTaskPayload {
  title?: string
  description?: string
  priority?: 'urgent' | 'high' | 'medium' | 'low'
  assigned_agent_id?: number
  tags?: string[]
  review_mode?: 'human' | 'llm' | 'auto'
}

export interface PlanQuestion {
  id: string
  question: string
  options: string[]
  default: number
}

export interface PlanResponse {
  questions: PlanQuestion[]
  suggested_title: string
  suggested_priority: string
}

export interface RefineResponse {
  title: string
  description: string
  priority: string
  suggested_tags: string[]
}

export interface BoardTaskResponse {
  tasks: any[]
  total: number
}

// ============= CRUD HOOKS =============

export function useBoardTasksList(filters?: Record<string, string>) {
  const params = new URLSearchParams(filters)
  const endpoint = `/api/v1/tasks?${params.toString()}`

  return useQuery<BoardTaskResponse>({
    queryKey: boardQueryKeys.tasks(filters),
    queryFn: () => apiClient.request<BoardTaskResponse>(endpoint),
    staleTime: 30000,
    refetchInterval: 60000,
  })
}

export function useCreateTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (payload: CreateTaskPayload) =>
      apiClient.request('/api/v1/tasks', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: boardQueryKeys.all })
    },
  })
}

export function useUpdateTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ taskId, payload }: { taskId: string; payload: UpdateTaskPayload }) =>
      apiClient.request(`/api/v1/tasks/${taskId}`, {
        method: 'PATCH',
        body: JSON.stringify(payload),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: boardQueryKeys.all })
    },
  })
}

export function useDeleteTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (taskId: string) =>
      apiClient.request(`/api/v1/tasks/${taskId}`, { method: 'DELETE' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: boardQueryKeys.all })
    },
  })
}

// ============= PLANNING HOOKS =============

export function usePlanTask() {
  return useMutation<PlanResponse, Error, { raw_prompt: string }>({
    mutationFn: async (payload) => {
      const res = await apiClient.request<any>('/api/v1/tasks/plan', {
        method: 'POST',
        body: JSON.stringify(payload),
      })
      // Backend wraps in { planning: { ... }, raw_prompt }
      const planning = res?.planning ?? res
      return {
        questions: planning.questions ?? [],
        suggested_title: planning.suggested_title ?? '',
        suggested_priority: planning.suggested_priority ?? 'medium',
      }
    },
  })
}

export function useRefineTask() {
  return useMutation<RefineResponse, Error, { raw_prompt: string; answers: Record<string, number> }>({
    mutationFn: async (payload) => {
      const res = await apiClient.request<any>('/api/v1/tasks/plan/refine', {
        method: 'POST',
        body: JSON.stringify(payload),
      })
      // Backend wraps in { refined: { ... }, raw_prompt }
      const refined = res?.refined ?? res
      return {
        title: refined.title ?? '',
        description: refined.description ?? '',
        priority: refined.priority ?? 'medium',
        suggested_tags: refined.suggested_tags ?? [],
      }
    },
  })
}
