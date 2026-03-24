'use client'

import { memo } from 'react'
import { Handle, Position, type NodeProps } from 'reactflow'
import {
  Circle, Clock, UserCheck, Loader2, CheckCircle2, Search,
  ShieldCheck, XCircle, SkipForward, AlertTriangle, RefreshCw,
  Merge, Coins,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { TaskState } from '@/types/missions'
import { TASK_STATE_CONFIG } from '@/types/missions'

const ICON_MAP: Record<string, typeof Circle> = {
  Circle,
  Clock,
  UserCheck,
  Loader2,
  CheckCircle2,
  Search,
  ShieldCheck,
  XCircle,
  SkipForward,
  AlertTriangle,
  RefreshCw,
}

export interface MissionTaskNodeData {
  id: string
  title: string
  agentName: string | null
  agentRole: string | null
  sequenceNumber: number
  state: TaskState
  taskType?: string | null
  tokensUsed?: number
  estimatedTokens?: number
  isSelected: boolean
  mode: 'plan' | 'execution' | 'review'
  outputExcerpt?: string | null
  attemptNumber?: number
}

function MissionTaskNodeInner({ data }: NodeProps<MissionTaskNodeData>) {
  const config = TASK_STATE_CONFIG[data.state]
  const IconComponent = ICON_MAP[config.icon] ?? Circle
  const isSynthesis = data.taskType === 'synthesis'

  // In plan mode, all nodes look neutral (not yet running)
  const isPlanMode = data.mode === 'plan'

  return (
    <div
      className={cn(
        'rounded-lg border-2 px-3 py-2.5 min-w-[200px] max-w-[240px] backdrop-blur-sm transition-all',
        isPlanMode
          ? 'border-muted-foreground/20 bg-muted/5'
          : `${config.borderClass} ${config.bgClass}`,
        isSynthesis && 'border-purple-500/40 bg-purple-500/5',
        data.isSelected && 'ring-2 ring-primary shadow-[0_0_12px_rgba(249,115,22,0.15)]',
      )}
    >
      <Handle type="target" position={Position.Top} className="!bg-muted-foreground/30 !w-2 !h-2" />

      {/* Header: icon + sequence + title */}
      <div className="flex items-start gap-2">
        <div className={cn('mt-0.5 shrink-0', isSynthesis ? 'text-purple-400' : isPlanMode ? 'text-muted-foreground' : config.color)}>
          {isSynthesis ? (
            <Merge className="w-4 h-4" />
          ) : (
            <IconComponent className={cn('w-4 h-4', config.animate)} />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold leading-tight line-clamp-2">{data.title}</p>
        </div>
        <span className="text-[10px] text-muted-foreground shrink-0 font-mono">
          #{data.sequenceNumber}
        </span>
      </div>

      {/* Agent assignment */}
      <div className="mt-1.5 flex items-center gap-1.5">
        {data.agentName ? (
          <>
            <div className="w-4 h-4 rounded-full bg-[hsl(var(--agent))]/20 flex items-center justify-center">
              <span className="text-[8px] font-bold text-[hsl(var(--agent))]">
                {data.agentName[0]?.toUpperCase()}
              </span>
            </div>
            <span className="text-[10px] text-muted-foreground truncate">{data.agentName}</span>
          </>
        ) : data.agentRole ? (
          <span className="text-[10px] text-muted-foreground italic">Role: {data.agentRole}</span>
        ) : (
          <span className="text-[10px] text-muted-foreground/50">Unassigned</span>
        )}
      </div>

      {/* Retry indicator */}
      {data.attemptNumber != null && data.attemptNumber > 1 && (
        <div className="mt-1 text-[10px] text-[hsl(var(--warning))]">
          Attempt {data.attemptNumber}
        </div>
      )}

      {/* Token usage (execution/review mode) */}
      {!isPlanMode && (data.tokensUsed != null && data.tokensUsed > 0 || data.estimatedTokens != null && data.estimatedTokens > 0) && (
        <div className="mt-1 flex items-center gap-1 text-[10px] text-muted-foreground">
          <Coins className="w-3 h-3" />
          <span className="font-mono">
            {(data.tokensUsed ?? 0).toLocaleString()}
            {data.estimatedTokens ? ` / ${data.estimatedTokens.toLocaleString()}` : ''}
          </span>
        </div>
      )}

      {/* Status label (execution/review mode) */}
      {!isPlanMode && (
        <div className={cn('mt-1 text-[10px] font-medium', isSynthesis ? 'text-purple-400' : config.color)}>
          {isSynthesis && 'Synthesis · '}{config.label}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground/30 !w-2 !h-2" />
    </div>
  )
}

export const MissionTaskNode = memo(MissionTaskNodeInner)
