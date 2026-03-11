/**
 * Board task management hooks (PRD-72 v2).
 * Fetches tasks grouped by board status, with optimistic drag-and-drop updates.
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
  tasks: (filters?: BoardFilters) => ['board', 'tasks', filters] as const,
}

// ============= HOOKS =============

/**
 * Fetch all board tasks and group into columns.
 * Falls back to activity feed endpoint with board-specific status filters.
 */
export function useBoardTasks(filters?: BoardFilters) {
  const queryClient = useQueryClient()

  const params = new URLSearchParams()
  params.set('status', 'inbox,assigned,in_progress,review,done,completed,failed')
  if (filters?.agent_id) params.set('agent_id', String(filters.agent_id))
  if (filters?.priority) params.set('priority', filters.priority)
  if (filters?.type) params.set('type', filters.type)
  if (filters?.search) params.set('search', filters.search)
  if (filters?.period) params.set('period', filters.period ?? '30d')
  params.set('limit', '200')

  const endpoint = `/api/activity/feed?${params.toString()}`

  const query = useQuery<BoardResponse>({
    queryKey: boardQueryKeys.tasks(filters),
    queryFn: async () => {
      const response = await apiClient.request<any>(endpoint)
      // Transform ActivityFeedItems into BoardTasks
      const items = response.items ?? response ?? []
      const tasks: BoardTask[] = items.map((item: any) => mapFeedItemToBoardTask(item))
      return { tasks, total: tasks.length }
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
      return apiClient.request(`/api/activity/tasks/${taskId}/status`, {
        method: 'PATCH',
        body: JSON.stringify({ status }),
      })
    },
    onMutate: async ({ taskId, status }) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: boardQueryKeys.all })

      // Snapshot previous state
      const previous = queryClient.getQueriesData({ queryKey: boardQueryKeys.all })

      // Optimistic update: move task to new status
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
      // Revert on failure
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
 * Map an ActivityFeedItem (from existing API) into a BoardTask.
 * This bridges the gap until a dedicated board endpoint exists.
 */
function mapFeedItemToBoardTask(item: any): BoardTask {
  let status: BoardStatus = 'inbox'

  // Map existing statuses to board columns
  const rawStatus = item.status ?? 'pending'
  switch (rawStatus) {
    case 'pending':
      status = 'inbox'
      break
    case 'running':
    case 'in_progress':
      status = 'in_progress'
      break
    case 'review':
      status = 'review'
      break
    case 'completed':
    case 'done':
      status = 'done'
      break
    case 'failed':
    case 'cancelled':
      status = 'done'
      break
    case 'assigned':
      status = 'assigned'
      break
    case 'inbox':
      status = 'inbox'
      break
    default:
      status = 'inbox'
  }

  return {
    id: item.id ?? '',
    type: item.type ?? 'recipe',
    name: item.name ?? 'Untitled',
    description: item.summary ?? undefined,
    status,
    priority: item.priority ?? 'medium',
    tags: item.tags ?? [],
    assignee: item.agent
      ? {
          agent_id: item.agent.id,
          agent_name: item.agent.name,
          agent_icon: item.agent.avatar_url ?? null,
        }
      : undefined,
    creator: undefined,
    due_date: item.due_date ?? undefined,
    review_mode: item.review_mode ?? 'human',
    started_at: item.started_at ?? undefined,
    completed_at: item.completed_at ?? undefined,
    duration_ms: item.duration_seconds ? item.duration_seconds * 1000 : undefined,
    step_progress: item.step_progress
      ? { current: item.step_progress.current, total: item.step_progress.total }
      : undefined,
    error_message: item.error_message ?? undefined,
    report_id: undefined,
    source_id: item.source_id ?? '',
    project_id: undefined,
  }
}
