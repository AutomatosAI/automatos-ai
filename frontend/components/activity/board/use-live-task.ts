/**
 * The viewer's ticket, live while it runs (moved out of board-task-viewer.tsx, which is over 800 lines).
 *
 * Polls the backend for the ticket while it is in_progress and falls back to the prop ticket until
 * the first answer lands. Ticket PR fields: a pull request the session reports mid-run shows too.
 */
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { pullRequestOf } from '@/lib/ticket-pull-request'
import type { BoardTask } from '@/types/board'

/** How often a running ticket is read again, and how long an answer stays fresh. */
export const LIVE_TASK_REFETCH_MS = 5000
export const LIVE_TASK_STALE_MS = 3000

export function useLiveTask(task: BoardTask | null) {
  const isActive = task?.status === 'in_progress'

  const { data } = useQuery({
    queryKey: ['board-task-live', task?.id],
    queryFn: () => apiClient.request<any>(`/api/v1/tasks/${task?.source_id || task?.id}`),
    enabled: !!task && isActive,
    refetchInterval: isActive ? LIVE_TASK_REFETCH_MS : false,
    staleTime: LIVE_TASK_STALE_MS,
  })

  if (!task) return null
  if (!data) return task

  return {
    ...task,
    status: data.status ?? task.status,
    result: data.result ?? task.result,
    error_message: data.error_message ?? task.error_message,
    started_at: data.started_at ?? task.started_at,
    completed_at: data.completed_at ?? task.completed_at,
    step_progress: data.step_progress ?? task.step_progress,
    runtime_ref: data.runtime_ref ?? task.runtime_ref,
    pull_request: pullRequestOf(data.pull_request) ?? task.pull_request,
  } as BoardTask
}
