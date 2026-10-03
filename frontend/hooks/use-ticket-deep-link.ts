'use client'

/**
 * PRD-252 R1 — the board's deep link, shared by the Studio board (BoardTab) and
 * the classic one (BoardView), which each carried a copy of it.
 *
 * `?task_id=<id>` opens that ticket's viewer. The ticket is fetched by id, so one
 * the loaded columns don't hold (done long ago, a mission or playbook step, a
 * filter) opens all the same. `&question=<grant id>` opens it at that question.
 * A question link without a ticket (a notification carries only the question)
 * is matched to its ticket among the open questions; one that matches no ticket
 * opens the Questions tab. The returned `clear` drops both parameters when the
 * viewer closes, so a link opens its ticket once and a tab switch never reopens it.
 */
import { useCallback, useEffect, useRef } from 'react'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { toast } from 'sonner'
import { useBoardTask } from '@/hooks/use-board-tasks'
import { useQuestions } from '@/hooks/use-approval-grants'
import { QUESTIONS_HREF, questionTicketId } from '@/lib/ticket-links'
import type { BoardTask } from '@/types/board'

export type OpenTicket = (task: BoardTask, questionId: number | null) => void

const DEEP_LINK_PARAMS = ['task_id', 'question'] as const

export function useTicketDeepLink(onOpen: OpenTicket): () => void {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const taskParam = searchParams?.get('task_id') || null
  const questionParam = searchParams?.get('question') || null
  const questions = useQuestions()
  const asked = questionParam ? questions.data?.grants?.find((q) => String(q.id) === questionParam) : undefined
  const taskId = taskParam ?? (asked ? questionTicketId(asked) : null)
  const ticket = useBoardTask(taskId)
  const key = taskId ? `${taskId}:${questionParam ?? ''}` : null
  const opened = useRef<string | null>(null)

  useEffect(() => {
    if (!key) opened.current = null // the link is gone: the same link opens again
  }, [key])

  useEffect(() => {
    if (!key || opened.current === key) return
    if (ticket.data) {
      opened.current = key
      onOpen(ticket.data, questionParam ? Number(questionParam) : null)
    } else if (ticket.isError) {
      opened.current = key
      toast.error(`Ticket ${taskId} could not be opened. It may have been deleted.`)  // PRD-252 R4: '#' means a number
    }
  }, [key, ticket.data, ticket.isError, taskId, questionParam, onOpen])

  // A question that is no longer open, or that was not asked on a ticket.
  const unmatched = !taskParam && !!questionParam && !!questions.data && !taskId
  useEffect(() => {
    if (unmatched) router.replace(QUESTIONS_HREF as any)
  }, [unmatched, router])

  return useCallback(() => {
    if (!DEEP_LINK_PARAMS.some((name) => searchParams?.has(name))) return
    const params = new URLSearchParams(searchParams?.toString() ?? '')
    DEEP_LINK_PARAMS.forEach((name) => params.delete(name))
    router.replace(`${pathname}?${params.toString()}` as any, { scroll: false })
  }, [router, pathname, searchParams])
}
