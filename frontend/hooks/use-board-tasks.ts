/**
 * Board task management hooks (PRD-72 v2).
 * Fetches tasks from /api/v1/tasks, groups by board status,
 * with optimistic drag-and-drop updates.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { BoardTask, BoardStatus, BoardColumn } from '@/types/board'
import { BOARD_COLUMNS } from '@/types/board'

// ============= TYPES =============

export interface BoardFilters {
  agent_id?: number | null
  priority?: string | null
  type?: string | null
  search?: string | null
  period?: string
}

interface BoardResponse {
  tasks: BoardTask[]
  total: number
}

// ============= QUERY KEYS =============

export const boardQueryKeys = {
  all: ['board'] as const,
  tasks: (filters?: any) => ['board', 'tasks', filters] as const,
}

// ============= HOOKS =============

/**
 * Fetch all board tasks from /api/v1/tasks and group into columns.
 */
export function useBoardTasks(filters?: BoardFilters) {
  const params = new URLSearchParams()
  if (filters?.agent_id) params.set('agent_id', String(filters.agent_id))
  if (filters?.priority) params.set('priority', filters.priority)
  if (filters?.search) params.set('search', filters.search)
  params.set('limit', '200')

  const endpoint = `/api/v1/tasks?${params.toString()}`

  const query = useQuery<BoardResponse>({
    queryKey: boardQueryKeys.tasks(filters),
    queryFn: async () => {
      const response = await apiClient.request<any>(endpoint)
      const tasks = (response.tasks ?? []).map((t: any) => mapTaskToBoardTask(t))
      return { tasks, total: response.total ?? tasks.length }
    },
    refetchInterval: 60000,
    staleTime: 30000,
  })

  // Group into columns
  const columns: BoardColumn[] = BOARD_COLUMNS.map((col) => {
    const tasks = (query.data?.tasks ?? []).filter((t) => {
      if (col.status === 'done') {
        return t.status === 'done' || t.status === ('completed' as any) || t.status === ('failed' as any)
      }
      if (col.status === 'in_progress') {
        return t.status === 'in_progress' || t.status === ('running' as any)
      }
      return t.status === col.status
    })
    return { ...col, count: tasks.length, tasks }
  })

  return { ...query, columns }
}

/**
 * Optimistic status update for drag-and-drop.
 */
export function useUpdateTaskStatus() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ taskId, status }: { taskId: string; status: BoardStatus }) => {
      return apiClient.request(`/api/v1/tasks/${taskId}/status`, {
        method: 'PATCH',
        body: JSON.stringify({ status }),
      })
    },
    onMutate: async ({ taskId, status }) => {
      await queryClient.cancelQueries({ queryKey: boardQueryKeys.all })
      const previous = queryClient.getQueriesData({ queryKey: boardQueryKeys.all })

      queryClient.setQueriesData<BoardResponse>(
        { queryKey: boardQueryKeys.all },
        (old) => {
          if (!old) return old
          return {
            ...old,
            tasks: old.tasks.map((t) =>
              t.id === taskId ? { ...t, status } : t
            ),
          }
        }
      )

      return { previous }
    },
    onError: (_err, _vars, context) => {
      if (context?.previous) {
        for (const [key, data] of context.previous) {
          queryClient.setQueryData(key, data)
        }
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: boardQueryKeys.all })
    },
  })
}

// ============= HELPERS =============

/**
 * Map a /api/v1/tasks response item into a BoardTask.
 */
function mapTaskToBoardTask(item: any): BoardTask {
  return {
    id: String(item.id),
    type: (item.source_type === 'recipe' || item.source_type === 'playbook') ? 'playbook' : (item.type ?? 'task'),
    name: item.title ?? 'Untitled',
    description: item.description ?? undefined,
    status: (item.status as BoardStatus) ?? 'inbox',
    priority: item.priority ?? 'medium',
    tags: item.tags ?? [],
    assignee: item.agent
      ? {
          agent_id: item.agent.id,
          agent_name: item.agent.name,
          agent_icon: item.agent.agent_icon ?? null,
        }
      : undefined,
    review_mode: item.review_mode ?? 'auto',
    started_at: item.started_at ?? undefined,
    completed_at: item.completed_at ?? undefined,
    error_message: item.error_message ?? undefined,
    source_id: item.source_id ?? String(item.id),
    step_progress: item.planning_data?.step_progress ?? undefined,
    planning_data: item.planning_data ? {
      // Normalize recipe_id (legacy) to playbook_id
      playbook_id: item.planning_data.playbook_id ?? item.planning_data.recipe_id,
      execution_id: item.planning_data.execution_id,
      step_progress: item.planning_data.step_progress,
    } : undefined,
    result: item.result,
  }
}
