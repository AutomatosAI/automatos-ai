'use client'

/**
 * PRD-238 S6 — a ticket, live, inside the reply that filed or checked it.
 *
 * Status, who has it, how long it has run, the session's last tool and the
 * files it touched. Refreshes itself on the board's SSE lane
 * (`automatos:board-changed`) so the card moves as the ticket moves.
 */
import { useCallback, useEffect, useState } from 'react'
import { ClipboardList, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api-client'
import { useRouter } from 'next/navigation'
import type { TaskCardData } from '@/types'
import { STATUS_CONFIG, type BoardStatus } from '@/types/board'

export interface TaskCardProps {
  card: TaskCardData
}

const TERMINAL = new Set(['done', 'failed', 'cancelled'])

function elapsedLabel(startedAt?: string | null, completedAt?: string | null, now = Date.now()): string {
  if (!startedAt) return ''
  const start = Date.parse(startedAt)
  if (!Number.isFinite(start)) return ''
  const end = completedAt ? Date.parse(completedAt) : now
  const seconds = Math.max(0, Math.round((end - start) / 1000))
  if (seconds < 60) return `${seconds} s`
  const minutes = Math.floor(seconds / 60)
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`
}

/** Map a task-read payload (API) onto the card shape the stream sends. */
function cardFromTask(task: Record<string, any>, previous: TaskCardData): TaskCardData {
  const ref = (task.runtime_ref && typeof task.runtime_ref === 'object' ? task.runtime_ref : {}) as Record<string, any>
  const tools = Array.isArray(ref.recent_tools) ? ref.recent_tools : []
  const last = tools[tools.length - 1]
  return {
    ...previous,
    title: task.title ?? previous.title,
    status: task.status ?? previous.status,
    assigned_agent: task.assigned_agent_name ?? task.assigned_agent ?? previous.assigned_agent,
    last_tool: last ? (typeof last === 'object' ? last.name : String(last)) : previous.last_tool,
    files_touched: Array.isArray(ref.files_touched) ? ref.files_touched.length : previous.files_touched,
    exit_reason: ref.exit_reason ?? previous.exit_reason,
    started_at: task.started_at ?? previous.started_at,
    completed_at: task.completed_at ?? previous.completed_at,
  }
}

export function TaskCard({ card: initial }: TaskCardProps) {
  const router = useRouter()
  const [card, setCard] = useState<TaskCardData>(initial)

  useEffect(() => {
    setCard((current) => ({ ...current, ...initial }))
  }, [initial])

  const refresh = useCallback(async () => {
    try {
      const task = await apiClient.request<Record<string, any>>(`/api/v1/tasks/${card.id}`)
      setCard((current) => cardFromTask(task, current))
    } catch {
      // The stream's snapshot stays on screen; the next board event retries.
    }
  }, [card.id])

  // Move with the board: refresh when this ticket changes anywhere.
  useEffect(() => {
    if (typeof window === 'undefined' || TERMINAL.has(card.status)) return
    const onBoardChanged = (event: Event) => {
      const detail = (event as CustomEvent).detail as { task_id?: number | string } | undefined
      if (detail && String(detail.task_id) === String(card.id)) void refresh()
    }
    window.addEventListener('automatos:board-changed', onBoardChanged)
    return () => window.removeEventListener('automatos:board-changed', onBoardChanged)
  }, [card.id, card.status, refresh])

  const status = STATUS_CONFIG[card.status as BoardStatus]
  const elapsed = elapsedLabel(card.started_at, card.completed_at)
  const facts = [
    card.assigned_agent && card.assigned_agent !== 'unassigned' ? card.assigned_agent : null,
    elapsed ? `${elapsed}${TERMINAL.has(card.status) ? '' : ' so far'}` : null,
    card.last_tool ? `last tool: ${card.last_tool}` : null,
    card.files_touched ? `${card.files_touched} file${card.files_touched !== 1 ? 's' : ''} touched` : null,
    card.exit_reason && card.exit_reason !== card.status ? `exit: ${card.exit_reason}` : null,
  ].filter(Boolean)

  return (
    <div className="max-w-md space-y-2 rounded-xl border border-border bg-card/50 p-3 backdrop-blur" data-testid="task-card">
      <div className="flex items-center gap-2">
        <ClipboardList className="h-4 w-4 text-primary" />
        <span className="text-xs font-medium text-primary">Ticket #{card.id}</span>
        <span
          className="ml-auto inline-flex items-center gap-1.5 rounded-full border border-border/60 px-2 py-0.5 text-[11px] text-muted-foreground"
          data-testid="task-card-status"
        >
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: status ? `hsl(var(${status.cssVar}))` : 'currentColor' }} aria-hidden />
          {status?.label ?? card.status}
        </span>
      </div>
      <p className="text-sm leading-snug text-foreground/80">{card.title}</p>
      {facts.length > 0 && <p className="text-xs text-muted-foreground">{facts.join(' · ')}</p>}
      <Button
        variant="outline"
        size="sm"
        className="border-primary/30 text-primary hover:bg-primary/10"
        onClick={() => router.push(`/command-center?tab=board&task_id=${card.id}` as any)}
      >
        Open on the board
        <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
      </Button>
    </div>
  )
}
