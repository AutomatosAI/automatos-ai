'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Target, Clock, Coins, MoreHorizontal, Pause, Play, X, Trash2 } from 'lucide-react'
import { motion, useReducedMotion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { DeleteConfirmation } from '@/components/shared'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { MissionStatusBadge } from './mission-status-badge'
import { cn } from '@/lib/utils'
import type { MissionResponse, RunState } from '@/types/missions'
import { DONE_TASK_STATES, TERMINAL_RUN_STATES } from '@/types/missions'
import { usePauseMission, useResumeMission, useCancelMission, useDeleteMission } from '@/hooks/use-missions-api'
import { formatDistanceToNow } from 'date-fns'

interface MissionCardProps {
  mission: MissionResponse
  index: number
}

export function MissionCard({ mission, index }: MissionCardProps) {
  const router = useRouter()
  const prefersReducedMotion = useReducedMotion()
  const pauseMutation = usePauseMission()
  const resumeMutation = useResumeMission()
  const cancelMutation = useCancelMission()
  const deleteMutation = useDeleteMission()
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false)

  // Parse plan for task stats (plan is JSONB with tasks array)
  const planTasks = (mission.plan as { tasks?: Array<Record<string, unknown>> })?.tasks ?? []
  const taskCount = planTasks.length
  const elapsed = mission.started_at
    ? formatDistanceToNow(new Date(mission.started_at), { addSuffix: false })
    : null

  const isTerminal = (TERMINAL_RUN_STATES as readonly string[]).includes(mission.state)
  const needsAction = mission.state === 'awaiting_approval' || mission.state === 'awaiting_human'

  const handleNavigate = () => {
    if (mission.state === 'awaiting_human') {
      router.push(`/missions/${mission.id}?tab=review` as any)
    } else {
      router.push(`/missions/${mission.id}` as any)
    }
  }

  const ctaLabel = getCTALabel(mission.state)

  return (
    <motion.div
      initial={prefersReducedMotion ? false : { opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
      className={cn(
        'glass-card card-glow rounded-xl p-4 flex flex-col gap-3 cursor-pointer',
        'hover:border-primary/20 transition-colors',
        needsAction && 'ring-1 ring-[hsl(var(--warning))]/20',
      )}
      onClick={handleNavigate}
    >
      {/* Header: status + menu */}
      <div className="flex items-center justify-between">
        <MissionStatusBadge state={mission.state} size="sm" />

        <DropdownMenu>
          <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0">
              <MoreHorizontal className="w-4 h-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
            {mission.state === 'running' && (
              <DropdownMenuItem onClick={() => pauseMutation.mutate(mission.id)}>
                <Pause className="w-4 h-4 mr-2" /> Pause
              </DropdownMenuItem>
            )}
            {mission.state === 'paused' && (
              <DropdownMenuItem onClick={() => resumeMutation.mutate(mission.id)}>
                <Play className="w-4 h-4 mr-2" /> Resume
              </DropdownMenuItem>
            )}
            {!isTerminal && (
              <DropdownMenuItem
                className="text-destructive"
                onClick={() => cancelMutation.mutate(mission.id)}
              >
                <X className="w-4 h-4 mr-2" /> Cancel
              </DropdownMenuItem>
            )}
            {isTerminal && (
              <DropdownMenuItem
                className="text-destructive"
                onSelect={(e) => {
                  e.preventDefault()
                  setConfirmDeleteOpen(true)
                }}
              >
                <Trash2 className="w-4 h-4 mr-2" /> Delete
              </DropdownMenuItem>
            )}
          </DropdownMenuContent>
        </DropdownMenu>

        <DeleteConfirmation
          open={confirmDeleteOpen}
          onOpenChange={setConfirmDeleteOpen}
          title="Delete this mission?"
          description="This permanently deletes the mission and its tasks. This cannot be undone."
          onConfirm={() => deleteMutation.mutateAsync(mission.id)}
        />
      </div>

      {/* Title + description */}
      <div>
        <h3 className="text-sm font-semibold line-clamp-2 leading-snug">
          {String((mission.config as Record<string, unknown>)?.name || '') || mission.goal.split(':')[0].slice(0, 80)}
        </h3>
      </div>

      {/* Progress bar (only if we have plan tasks) */}
      {taskCount > 0 && (
        <div>
          <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
            <span>{taskCount} tasks</span>
            <span>{mission.tokens_used.toLocaleString()} tokens</span>
          </div>
          <div className="h-1.5 bg-secondary/50 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, (mission.tokens_used / Math.max(mission.token_budget_estimate ?? 1, 1)) * 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Meta row */}
      <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
        {elapsed && (
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {elapsed}
          </span>
        )}
        {mission.tokens_used > 0 && (
          <span className="flex items-center gap-1">
            <Coins className="w-3 h-3" />
            {mission.tokens_used.toLocaleString()} tok
          </span>
        )}
      </div>

      {/* CTA */}
      {ctaLabel && (
        <Button
          variant={needsAction ? 'default' : 'outline'}
          size="sm"
          className={cn(
            'w-full mt-auto',
            needsAction && 'bg-[hsl(var(--warning))] hover:bg-[hsl(var(--warning))]/90 text-black',
          )}
          onClick={(e) => {
            e.stopPropagation()
            handleNavigate()
          }}
        >
          <Target className="w-3.5 h-3.5 mr-1.5" />
          {ctaLabel}
        </Button>
      )}
    </motion.div>
  )
}

function getCTALabel(state: RunState): string | null {
  switch (state) {
    case 'awaiting_approval':
      return 'Review Plan'
    case 'awaiting_human':
      return 'Review Now'
    case 'running':
    case 'verifying':
      return 'View Progress'
    case 'completed':
      return 'View Results'
    case 'failed':
      return 'View Details'
    case 'paused':
      return 'View Mission'
    default:
      return null
  }
}
