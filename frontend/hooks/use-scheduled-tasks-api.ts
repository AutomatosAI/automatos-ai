/**
 * Scheduled-task API hooks — the PRD-77 `agent_scheduled_tasks` rows the
 * Command Centre calendar surfaces.
 *
 * PATCH /api/v1/scheduled-tasks/{id}/status pauses, resumes or cancels a task
 * (all three are reversible: a cancelled task can be set active again).
 * POST /api/v1/scheduled-tasks schedules a BOARD ticket for later (the Create
 * Task dialog's "When"): it waits on the calendar and is filed on the board
 * when it fires. The calendar reads these rows through the schedule feed, so a
 * successful change invalidates that feed rather than a list of its own.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import { activityQueryKeys } from './use-activity-api'

export type ScheduledTaskStatus = 'active' | 'paused' | 'cancelled'

export interface ScheduledTaskStatusResponse {
  success: boolean
  task_id: number
  status: ScheduledTaskStatus
}

const STATUS_VERB: Record<ScheduledTaskStatus, string> = {
  active: 'resumed',
  paused: 'paused',
  cancelled: 'cancelled',
}

/** The query-key prefix every schedule range shares (`['activity', 'schedule', range]`). */
export const SCHEDULE_FEED_KEY = [...activityQueryKeys.all, 'schedule'] as const

export function useUpdateScheduledTaskStatus() {
  const queryClient = useQueryClient()

  return useMutation<
    ScheduledTaskStatusResponse,
    Error,
    { taskId: number; status: ScheduledTaskStatus }
  >({
    mutationFn: ({ taskId, status }) =>
      apiClient.request<ScheduledTaskStatusResponse>(
        `/api/v1/scheduled-tasks/${taskId}/status`,
        { method: 'PATCH', body: JSON.stringify({ status }) },
      ),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: SCHEDULE_FEED_KEY })
      toast.success(`Scheduled task #${data.task_id} ${STATUS_VERB[data.status]}`)
    },
    onError: (error) => {
      toast.error(error.message || 'Could not update the scheduled task')
    },
  })
}

export interface CreateScheduledBoardTaskPayload {
  title: string
  description?: string
  /** ISO datetime for one_shot, 5-field cron (UTC) for recurring */
  schedule: string
  task_type: 'one_shot' | 'recurring'
  priority: 'urgent' | 'high' | 'medium' | 'low'
  assigned_agent_id?: number
  review_mode?: 'auto' | 'human' | 'llm'
  tags?: string[]
  max_runs?: number
}

export interface CreateScheduledTaskResponse {
  success: boolean
  task_id: number
  task_type: 'one_shot' | 'recurring'
  deliver_as: 'chat' | 'board_task'
  title?: string | null
  next_run_at: string | null
  message?: string
}

export function useCreateScheduledBoardTask() {
  const queryClient = useQueryClient()

  return useMutation<CreateScheduledTaskResponse, Error, CreateScheduledBoardTaskPayload>({
    mutationFn: (payload) =>
      apiClient.request<CreateScheduledTaskResponse>('/api/v1/scheduled-tasks', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: SCHEDULE_FEED_KEY })
    },
  })
}
