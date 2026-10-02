'use client'

/**
 * Filing a task from the create dialog: now (onto the board) or later (a
 * scheduled task on the calendar). Split out of CreateTaskDialog (PRD-252).
 *
 * PRD-252 R1: "View on Board" opens the ticket just filed. It opened the whole
 * board, where the new card was one of many.
 */

import { useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { toast } from 'sonner'
import { useCreateTask, type CreateTaskPayload } from '@/hooks/use-board-tasks-api'
import { useCreateScheduledBoardTask } from '@/hooks/use-scheduled-tasks-api'
import { BOARD_HREF, ticketHref } from '@/lib/ticket-links'
import { buildSchedulePayload, describeSchedule, isScheduleInFuture, type ScheduleMode } from './schedule-choice'

const CALENDAR_HREF = '/command-center?tab=calendar'

interface SubmitOptions {
  buildPayload: () => CreateTaskPayload
  scheduleMode: ScheduleMode
  scheduleAt: string
  attachmentCount: number
  /** The assigned agent's name, for the toast; null when unassigned. */
  agentName: string | null
  onDone: () => void
}

type Navigate = (href: string) => void
type CreateNow = ReturnType<typeof useCreateTask>['mutateAsync']
type CreateLater = ReturnType<typeof useCreateScheduledBoardTask>['mutateAsync']

/** The board, opened at the ticket the create returned (the board alone when it returned no id). */
export function filedTicketHref(created: unknown): string {
  const id = (created as { id?: number | string } | null | undefined)?.id
  return id != null ? ticketHref(id) : BOARD_HREF
}

async function fileNow(create: CreateNow, payload: CreateTaskPayload, agentName: string | null, navigate: Navigate) {
  try {
    const created = await create(payload)
    toast.success(agentName ? `Task assigned to ${agentName}.` : 'Task created.', {
      duration: 5000,
      action: { label: 'View on Board', onClick: () => navigate(filedTicketHref(created)) },
    })
    return true
  } catch {
    toast.error('Failed to create task')
    return false
  }
}

async function fileLater(create: CreateLater, payload: CreateTaskPayload, opts: SubmitOptions, navigate: Navigate) {
  const scheduled = buildSchedulePayload(opts.scheduleMode, opts.scheduleAt)
  if (!scheduled) return refuse('Pick a date and time')
  if (scheduled.task_type === 'one_shot' && !isScheduleInFuture(opts.scheduleAt)) {
    return refuse('The time must be in the future')
  }
  // PRD-127 attachments are ephemeral; a ticket filed later cannot carry them.
  if (opts.attachmentCount > 0) return refuse('Attachments can’t be scheduled — create the task now instead')
  try {
    await create({
      title: payload.title,
      description: payload.description,
      priority: payload.priority,
      assigned_agent_id: payload.assigned_agent_id,
      review_mode: payload.review_mode,
      tags: payload.tags,
      ...scheduled,
    })
    toast.success(`Scheduled ${describeSchedule(opts.scheduleMode, opts.scheduleAt)}. It’s filed on the board when it fires.`, {
      duration: 6000,
      action: { label: 'View calendar', onClick: () => navigate(CALENDAR_HREF) },
    })
    return true
  } catch (err) {
    toast.error(err instanceof Error ? err.message : 'Failed to schedule the task')
    return false
  }
}

function refuse(message: string): false {
  toast.error(message)
  return false
}

export function useCreateTaskSubmit(opts: SubmitOptions) {
  const router = useRouter()
  const createTask = useCreateTask()
  const createScheduledTask = useCreateScheduledBoardTask()
  const navigate = useCallback((href: string) => router.push(href as any), [router])

  const submit = useCallback(async () => {
    const payload = opts.buildPayload()
    if (!payload.title.trim()) return void refuse('Title is required')
    const filed = opts.scheduleMode === 'now'
      ? await fileNow(createTask.mutateAsync, payload, opts.agentName, navigate)
      : await fileLater(createScheduledTask.mutateAsync, payload, opts, navigate)
    if (filed) opts.onDone()
  }, [opts, createTask.mutateAsync, createScheduledTask.mutateAsync, navigate])

  return { submit, isSubmitting: createTask.isLoading || createScheduledTask.isLoading }
}
