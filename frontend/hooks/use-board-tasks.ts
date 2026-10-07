/**
 * Board task management hooks (PRD-72 v2).
 * Fetches tasks from /api/v1/tasks, groups by board status,
 * with optimistic drag-and-drop updates.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
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

/** What POST /api/v1/tasks/{id}/approve returns; `action_result` is null without an approval_action. */
export interface ApproveResult {
  task_id?: number | string
  status?: string
  action_result?: { type?: string; title?: string; topic?: string; warning?: string } | null
}

// ============= QUERY KEYS =============

export const boardQueryKeys = {
  all: ['board'] as const,
  tasks: (filters?: any) => ['board', 'tasks', filters] as const,
  task: (taskId: string) => ['board', 'task', taskId] as const,
}

// ============= HOOKS =============

/**
 * One board task by id (GET /api/v1/tasks/{id}) — for deep links such as the
 * calendar's "Open on board" (?task_id=), which must open the card even when
 * the loaded columns are filtered and don't contain it. Disabled without an id.
 */
export function useBoardTask(taskId: string | null) {
  return useQuery<BoardTask>({
    queryKey: boardQueryKeys.task(taskId ?? ''),
    queryFn: async () => mapTaskToBoardTask(await apiClient.request<any>(`/api/v1/tasks/${taskId}`)),
    enabled: Boolean(taskId),
    staleTime: 30000,
  })
}

/**
 * Fetch all board tasks from /api/v1/tasks and group into columns.
 */
export function useBoardTasks(filters?: BoardFilters) {
  const params = new URLSearchParams()
  if (filters?.agent_id) params.set('agent_id', String(filters.agent_id))
  if (filters?.priority) params.set('priority', filters.priority)
  if (filters?.search) params.set('search', filters.search)
  // F225: every open ticket, whatever its age (an old one in Review fell off the
  // newest 200); only Done and Cancelled are windowed, to the newest 200.
  params.set('finished_limit', '200')

  const endpoint = `/api/v1/tasks?${params.toString()}`

  const query = useQuery<BoardResponse>({
    queryKey: boardQueryKeys.tasks(filters),
    queryFn: async () => {
      const response = await apiClient.request<any>(endpoint)
      const tasks = (response.tasks ?? []).map((t: any) => mapTaskToBoardTask(t))
      return { tasks, total: response.total ?? tasks.length }
    },
    // PRD-180 S1 (F090): no interval poll — the board LISTEN/NOTIFY SSE
    // (useBoardEventStream) invalidates this query on real pushed events, so
    // the board refetches the moment state changes, not on a 60s tick.
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
        return t.status === 'done' || t.status === ('completed' as any)
      }
      if (col.status === 'in_progress') {
        return t.status === 'in_progress' || t.status === ('running' as any)
      }
      if (col.status === 'cancelled') {
        return t.status === 'cancelled' || t.status === 'closed'   // PRD-252 R7: one stage
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
    onError: (err, _vars, context) => {
      if (context?.previous) {
        for (const [key, data] of context.previous) {
          queryClient.setQueryData(key, data)
        }
      }
      // #1094: a refused move says why ("Assign an agent first…"), not just snaps back.
      toast.error(err instanceof Error ? err.message : 'Could not change the status')
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: boardQueryKeys.all })
    },
  })
}

/**
 * Approve a task in review status — executes approval_action (e.g., publish blog).
 * PRD-252 R2: an optional note, kept on the ticket (F038: it was thrown away).
 */
export function useApproveTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ taskId, note }: { taskId: string; note?: string }) => {
      return apiClient.request<ApproveResult>(`/api/v1/tasks/${taskId}/approve`, {
        method: 'POST',
        body: JSON.stringify(note ? { note } : {}),
      })
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: boardQueryKeys.all })
    },
  })
}

/**
 * Reject a task in review status. PRD-252 R2: the owner's words are required
 * here; they lead the redo's brief (services/ticket_redo.py).
 */
export function useRejectTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ taskId, feedback }: { taskId: string; feedback: string }) => {
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

/**
 * PRD-252 R2: Discuss's "Update ticket and re-queue" — the brief agreed in the
 * chat becomes the ticket's brief, and the ticket goes back to its agent.
 */
export function useRebriefTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ taskId, brief }: { taskId: string; brief: string }) => {
      return apiClient.request<{ task_id: number; status: string }>(`/api/v1/tasks/${taskId}/rebrief`, {
        method: 'POST',
        body: JSON.stringify({ brief }),
      })
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: boardQueryKeys.all })
    },
  })
}

/**
 * PRD-161 S5: Run a task now — re-dispatch it immediately through the board loop.
 */
export function useRunTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ taskId }: { taskId: string }) => {
      return apiClient.request(`/api/v1/tasks/${taskId}/run-now`, {
        method: 'POST',
        body: JSON.stringify({}),
      })
    },
    // PRD-234: a refused Run Now (409 "Task is already running", 422 "Assign an
    // agent…") must never look like nothing happened.
    onError: (err) => {
      toast.error(err instanceof Error ? err.message : 'Run Now was refused')
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

  return {
    id: String(item.id),
    type: boardType(item),
    name: item.title ?? 'Untitled',
    description: item.description ?? undefined,
    status: (item.status as BoardStatus) ?? 'inbox',
    priority: item.priority ?? 'medium',
    tags: tags.filter((t: string) => !t.startsWith('mission:')),
    mission_name: missionName,
    mission_id: item.orchestration_run_id ? String(item.orchestration_run_id) : undefined,
    assignee: assigneeOf(item.agent),
    review_mode: item.review_mode ?? 'auto',
    started_at: item.started_at ?? undefined,
    completed_at: item.completed_at ?? undefined,
    error_message: item.error_message ?? undefined,
    attempts: item.attempts ?? 0,
    source_id: item.source_id ?? String(item.id),
    project_id: item.orchestration_run_id
      ? item.orchestration_run_id.slice(0, 8)
      : undefined,
    step_progress: item.planning_data?.step_progress ?? undefined,
    planning_data: planningDataOf(item.planning_data),
    parent_task_id: item.parent_task_id ? String(item.parent_task_id) : undefined,
    sla_deadline: item.sla_deadline ?? undefined,
    blocked_at: item.blocked_at ?? undefined,
    blocked_reason: item.blocked_reason ?? undefined,
    result: item.result,
    runtime_ref: item.runtime_ref ?? undefined,  // PRD-234
    review_reason: item.review_reason ?? null,  // PRD-252 R3
    blocked_code: item.blocked_code ?? null,
    source_type: item.source_type ?? undefined,
    number: item.number ?? null,  // PRD-252 R4
    times_sent_back: item.times_sent_back ?? 0,  // PRD-252 D3
    kept_draft: item.kept_draft ?? null,  // F243: a failed redo keeps the draft it corrected
    knowledge_document_id: item.knowledge_document_id ?? null,  // PRE-11: its answer is in Knowledge
  }
}

/** The board's type from what filed the ticket: a playbook's run, a mission's card or step, else a task. */
function boardType(item: any): BoardTask['type'] {
  if (item.source_type === 'recipe' || item.source_type === 'playbook') return 'playbook'
  if (item.source_type === 'orchestration' || item.source_type === 'orchestration_task') return 'mission'
  return (item.type ?? 'task') as 'task'
}

function assigneeOf(agent: any): BoardTask['assignee'] {
  return agent ? { agent_id: agent.id, agent_name: agent.name, agent_icon: agent.agent_icon ?? null } : undefined
}

function planningDataOf(data: any): BoardTask['planning_data'] {
  if (!data) return undefined
  return {
    // Normalize recipe_id (legacy) to playbook_id
    playbook_id: data.playbook_id ?? data.recipe_id,
    execution_id: data.execution_id,
    step_progress: data.step_progress,
    approval_action: data.approval_action,
  }
}
