/**
 * PRD-268 §6.11.1 (US-W04) — what Try again runs on a failed row, through the routes that already do it.
 *
 * - `retry_ticket`: the board's Run now (POST /api/v1/tasks/{id}/run-now, useRunTask, which says why when refused).
 * - `replan_mission`: the Mission's Replan (POST /api/missions/{id}/replan, useReplanMission), which keeps its
 *   finished steps. A failed Mission's own board card and needs-you row are replanned too: its decision is the
 *   Mission's.
 * - `rerun_mission`: a new Mission from the same goal and config (POST /api/missions, useRerunMission), as the
 *   Mission page's Re-run does.
 *
 * A row offers Try again only when it names what its retry needs (`canRetry`): a ticket's id, a Mission's id, or the
 * Mission's goal. While one runs, `busy` is true, so the row's Try again can't send it twice. Each says how it went
 * in a toast, and the Needs-you number is read again, as the failed row may have left it.
 */
import { useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { useRunTask } from '@/hooks/use-board-tasks'
import { useReplanMission, useRerunMission } from '@/hooks/use-missions-api'
import { needsYouQueryKeys } from '@/hooks/use-needs-you'
import type { Retry } from '@/lib/proactive/types'

/** What a row's Try again acts on. */
export interface RetryTarget {
  readonly ticketId?: string | number | null
  readonly missionId?: string | null
  /** The Mission's goal and config, for a rerun. */
  readonly mission?: { readonly goal: string; readonly config?: Record<string, unknown> | null } | null
}

export interface RowRetry {
  readonly run: (retry: Retry, target: RetryTarget) => void
  /** A retry is on its way. */
  readonly busy: boolean
}

const RAN = 'Sent to run again'
const REPLANNED = 'Mission replanned — new tasks generated'
const RERUN = 'New mission created from same goal'
const REFUSED = 'Try again was refused'

/** Whether `target` names what `retry` needs. */
export function canRetry(retry: Retry, target: RetryTarget): boolean {
  if (retry === 'retry_ticket') return target.ticketId !== null && target.ticketId !== undefined && String(target.ticketId) !== ''
  if (retry === 'replan_mission') return Boolean(target.missionId)
  return Boolean(target.mission?.goal)
}

/** A Mission row's target: its id, and its goal and config for a rerun. */
export function missionRetryTarget(mission: { readonly id: string; readonly goal: string; readonly config?: Record<string, unknown> | null }): RetryTarget {
  return { missionId: mission.id, mission: { goal: mission.goal, config: mission.config ?? null } }
}

function refused(error: unknown): void {
  toast.error(error instanceof Error && error.message ? error.message : REFUSED)
}

interface RetryRoutes {
  readonly runTask: ReturnType<typeof useRunTask>
  readonly replan: ReturnType<typeof useReplanMission>
  readonly rerun: ReturnType<typeof useRerunMission>
}

/** Sends `retry` for `target` through its route, and says how it went. */
function sendRetry({ runTask, replan, rerun }: RetryRoutes, retry: Retry, target: RetryTarget): Promise<void> {
  if (retry === 'retry_ticket') {
    // A refused Run now is said by useRunTask's own onError toast, so it isn't said twice here.
    return runTask.mutateAsync({ taskId: String(target.ticketId) }).then(() => void toast.success(RAN), () => undefined)
  }
  if (retry === 'replan_mission') {
    return replan.mutateAsync({ id: String(target.missionId) }).then(() => void toast.success(REPLANNED), refused)
  }
  const mission = target.mission ?? { goal: '', config: null }
  return rerun.mutateAsync({ goal: mission.goal, config: mission.config ?? undefined }).then(() => void toast.success(RERUN), refused)
}

/** Runs a row's retry against its target; a target that lacks what the retry needs runs nothing. */
export function useRowRetry(): RowRetry {
  const queryClient = useQueryClient()
  const runTask = useRunTask()
  const replan = useReplanMission()
  const rerun = useRerunMission()
  const run = useCallback(
    (retry: Retry, target: RetryTarget) => {
      if (!canRetry(retry, target)) return
      void sendRetry({ runTask, replan, rerun }, retry, target).then(() =>
        queryClient.invalidateQueries({ queryKey: needsYouQueryKeys.all }),
      )
    },
    [queryClient, runTask, replan, rerun],
  )
  return { run, busy: Boolean(runTask.isLoading || replan.isLoading || rerun.isLoading) }
}
