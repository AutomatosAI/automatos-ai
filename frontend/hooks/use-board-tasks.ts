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

  // Build child count map and filter to top-level tasks
  const allTasks = query.data?.tasks ?? []
  const childCountMap = new Map<string, number>()
  for (const t of allTasks) {
    if (t.parent_task_id) {
      childCountMap.set(t.parent_task_id, (childCountMap.get(t.parent_task_id) ?? 0) + 1)
    }
  }
  // Annotate parent tasks with child_count
  const annotatedTasks = allTasks.map((t) => ({
    ...t,
    child_count: childCountMap.get(t.id) ?? 0,
  }))
  // Top-level = tasks without a parent (children nest under their parent card)
  const topLevelTasks = annotatedTasks.filter((t) => !t.parent_task_id)

  // Group into columns (with client-side type filter)
  const typeFilter = filters?.type ?? null
  const columns: BoardColumn[] = BOARD_COLUMNS.map((col) => {
    const tasks = topLevelTasks.filter((t) => {
      // Type filter (client-side — mission, playbook, task)
      if (typeFilter && t.type !== typeFilter) return false

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

/**
 * Approve a task in review status — executes approval_action (e.g., publish blog).
 */
export function useApproveTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ taskId }: { taskId: string }) => {
      return apiClient.request(`/api/v1/tasks/${taskId}/approve`, {
        method: 'POST',
        body: JSON.stringify({}),
      })
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: boardQueryKeys.all })
    },
  })
}

/**
 * Reject a task in review status with optional feedback.
 */
export function useRejectTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ taskId, feedback }: { taskId: string; feedback?: string }) => {
      return apiClient.request(`/api/v1/tasks/${taskId}/reject`, {
        method: 'POST',
        body: JSON.stringify({ feedback }),
      })
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
  const tags: string[] = item.tags ?? []
  const missionTag = tags.find((t: string) => t.startsWith('mission:'))
  const missionName = missionTag ? missionTag.slice(8) : undefined

  const type = (item.source_type === 'recipe' || item.source_type === 'playbook')
    ? 'playbook' as const
    : (item.source_type === 'orchestration' || item.source_type === 'orchestration_task')
      ? 'mission' as const
      : (item.type ?? 'task') as 'task'

  return {
    id: String(item.id),
    type,
    name: item.title ?? 'Untitled',
    description: item.description ?? undefined,
    status: (item.status as BoardStatus) ?? 'inbox',
    priority: item.priority ?? 'medium',
    tags: tags.filter((t: string) => !t.startsWith('mission:')),
    mission_name: missionName,
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
    project_id: item.orchestration_run_id
      ? item.orchestration_run_id.slice(0, 8)
      : undefined,
    step_progress: item.planning_data?.step_progress ?? undefined,
    planning_data: item.planning_data ? {
      // Normalize recipe_id (legacy) to playbook_id
      playbook_id: item.planning_data.playbook_id ?? item.planning_data.recipe_id,
      execution_id: item.planning_data.execution_id,
      step_progress: item.planning_data.step_progress,
      approval_action: item.planning_data.approval_action,
    } : undefined,
    parent_task_id: item.parent_task_id ? String(item.parent_task_id) : undefined,
    sla_deadline: item.sla_deadline ?? undefined,
    blocked_at: item.blocked_at ?? undefined,
    blocked_reason: item.blocked_reason ?? undefined,
    result: item.result,
  }
}
