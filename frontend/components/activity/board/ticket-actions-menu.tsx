'use client'

/**
 * PRD-252 R7 — Assign and Cancel from the board, from a card's menu or the
 * ticket's viewer, without leaving the board. A mission's tickets open the
 * mission instead: the mission engine runs them (D6).
 */

import Link from 'next/link'
import { toast } from 'sonner'
import { MoreHorizontal, Target, UserPlus, XCircle } from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useAssignableAgents } from '@/hooks/use-agent-api'
import { useCancelTask, useUpdateTask } from '@/hooks/use-board-tasks-api'
import { missionHref } from '@/lib/ticket-links'
import type { BoardTask } from '@/types/board'
import { canAssign, canCancel, isMissionTicket } from './ticket-actions'

export interface AgentChoice {
  id: number
  name: string
}

function failure(err: unknown, fallback: string): string {
  return err instanceof Error && err.message ? err.message : fallback
}

/** The agents a ticket can go to, as id and name. */
export function useAgentChoices(): AgentChoice[] {
  const { data } = useAssignableAgents()
  return Array.isArray(data) ? data.map((a: any) => ({ id: Number(a.id), name: String(a.name) })) : []
}

/** Assign and Cancel for one ticket, each saying what happened. */
export function useTicketActions(task: BoardTask) {
  const update = useUpdateTask()
  const cancel = useCancelTask()
  const assignTo = (agent: AgentChoice) =>
    update.mutate({ taskId: task.id, payload: { assigned_agent_id: agent.id } }, {
      onSuccess: () => toast.success(`Ticket ${task.number ?? task.id} is assigned to ${agent.name}.`),
      onError: (err) => toast.error(failure(err, 'Could not assign the ticket')),
    })
  const cancelIt = () =>
    cancel.mutate(task.id, {
      onSuccess: () => toast.success(`Ticket ${task.number ?? task.id} is cancelled.`),
      onError: (err) => toast.error(failure(err, 'Could not cancel the ticket')),
    })
  return { assignTo, cancelIt, busy: update.isLoading || cancel.isLoading }
}

/** The ⋯ menu on a board card. Clicks stay in the menu: the card opens the viewer. */
export function TicketActionsMenu({ task, className = 'cc-kb-menu' }: { task: BoardTask; className?: string }) {
  const agents = useAgentChoices()
  const actions = useTicketActions(task)
  const mission = isMissionTicket(task)
  if (mission ? !task.mission_id : !canAssign(task) && !canCancel(task)) return null
  const stop = (e: React.SyntheticEvent) => e.stopPropagation()
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button type="button" className={className} aria-label={`Actions for ticket ${task.id}`} onClick={stop}>
          <MoreHorizontal style={{ width: 13, height: 13 }} />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" onClick={stop}>
        {mission && task.mission_id && (
          <DropdownMenuItem asChild>
            <Link href={missionHref(task.mission_id) as any}><Target className="w-3.5 h-3.5 mr-2" /> Open the mission</Link>
          </DropdownMenuItem>
        )}
        {canAssign(task) && (
          <DropdownMenuSub>
            <DropdownMenuSubTrigger><UserPlus className="w-3.5 h-3.5 mr-2" /> Assign to</DropdownMenuSubTrigger>
            <DropdownMenuSubContent className="max-h-72 overflow-y-auto">
              {agents.map((agent) => (
                <DropdownMenuItem key={agent.id} disabled={actions.busy} onSelect={() => actions.assignTo(agent)}>
                  {agent.name}
                </DropdownMenuItem>
              ))}
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        )}
        {canCancel(task) && (
          <DropdownMenuItem disabled={actions.busy} onSelect={actions.cancelIt}>
            <XCircle className="w-3.5 h-3.5 mr-2" /> Cancel ticket
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
