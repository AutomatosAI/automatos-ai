'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import {
  RefreshCw,
  Pause,
  Play,
  Settings,
  Clock,
  ChevronDown,
  CheckCircle2,
  XCircle,
  Loader2,
} from 'lucide-react'
import { formatDistanceToNow, format } from 'date-fns'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { ItemCard } from '@/components/shared/item-card'
import { StatusBadge } from '@/components/shared/status-badge'
import { useToggleHeartbeat, useHeartbeatExecutions } from '@/hooks/use-heartbeats-api'
import type { HeartbeatConfig, HeartbeatExecution } from '@/hooks/use-heartbeats-api'

// ─── Helpers ──────────────────────────────────────────────

function formatInterval(minutes: number): string {
  if (minutes < 60) return `Every ${minutes}m`
  const hours = Math.floor(minutes / 60)
  const remaining = minutes % 60
  if (remaining === 0) return `Every ${hours}h`
  return `Every ${hours}h ${remaining}m`
}

function formatRelativeTime(iso: string | null): string {
  if (!iso) return 'Never'
  try {
    return formatDistanceToNow(new Date(iso), { addSuffix: true })
  } catch {
    return 'Unknown'
  }
}

function formatNextRun(iso: string | null): string {
  if (!iso) return 'Not scheduled'
  try {
    return format(new Date(iso), 'HH:mm')
  } catch {
    return 'Unknown'
  }
}

// ─── Component ────────────────────────────────────────────

interface RoutineCardProps {
  heartbeat: HeartbeatConfig
  animationDelay?: number
}

export function RoutineCard({ heartbeat, animationDelay = 0 }: RoutineCardProps) {
  const router = useRouter()
  const toggleMutation = useToggleHeartbeat()
  const [expanded, setExpanded] = useState(false)

  const { data: historyData, isLoading: historyLoading } =
    useHeartbeatExecutions(heartbeat.agent_id, { enabled: expanded })

  const isActive = heartbeat.enabled

  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation()
    toggleMutation.mutate(heartbeat.agent_id)
  }

  const handleEdit = (e: React.MouseEvent) => {
    e.stopPropagation()
    router.push(`/agents?agent=${heartbeat.agent_id}`)
  }

  const handleExpandToggle = (e: React.MouseEvent) => {
    e.stopPropagation()
    setExpanded((prev) => !prev)
  }

  return (
    <ItemCard
      animationDelay={animationDelay}
      icon={
        <div className="w-10 h-10 rounded-lg bg-[hsl(var(--agent))]/10 flex items-center justify-center">
          <RefreshCw className="w-5 h-5 text-[hsl(var(--agent))]" />
        </div>
      }
      title={heartbeat.agent_name}
      subtitle={`Agent routine`}
      titleBadges={
        <StatusBadge
          status={isActive ? 'success' : 'warning'}
          dot
          size="sm"
        >
          {isActive ? 'Active' : 'Paused'}
        </StatusBadge>
      }
      description={heartbeat.prompt}
      meta={
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {formatInterval(heartbeat.interval_minutes)}
          </span>
          <span>
            Last ran {formatRelativeTime(heartbeat.last_run?.created_at ?? null)}
          </span>
          <span>
            Next: {formatNextRun(heartbeat.next_run_at)}
          </span>
        </div>
      }
      actions={
        <div className="flex items-center gap-2 w-full justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleExpandToggle}
            className="text-xs text-muted-foreground"
          >
            <ChevronDown
              className={`w-3.5 h-3.5 mr-1 transition-transform duration-200 ${
                expanded ? 'rotate-180' : ''
              }`}
            />
            History
          </Button>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleToggle}
              disabled={toggleMutation.isLoading}
              className="text-xs"
            >
              {isActive ? (
                <>
                  <Pause className="w-3.5 h-3.5 mr-1" />
                  Pause
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 mr-1" />
                  Resume
                </>
              )}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleEdit}
              className="text-xs"
            >
              <Settings className="w-3.5 h-3.5 mr-1" />
              Edit
            </Button>
          </div>
        </div>
      }
    >
      {/* Expandable execution history */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="pt-2 space-y-1.5">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                Execution History
              </div>
              {historyLoading ? (
                <ExecutionHistorySkeleton />
              ) : historyData?.executions?.length ? (
                historyData.executions.map((exec) => (
                  <ExecutionLogEntry key={exec.id} execution={exec} />
                ))
              ) : (
                <p className="text-xs text-muted-foreground py-2">
                  No executions recorded yet
                </p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </ItemCard>
  )
}

// ─── Execution Log Entry ─────────────────────────────────

function ExecutionLogEntry({ execution }: { execution: HeartbeatExecution }) {
  const isSuccess = execution.status === 'completed' || execution.status === 'success'
  const isFailed = execution.status === 'failed' || execution.status === 'error'
  const isRunning = execution.status === 'running'

  const entryClass = isSuccess
    ? 'log-entry log-entry-success'
    : isFailed
      ? 'log-entry log-entry-error'
      : isRunning
        ? 'log-entry log-entry-info'
        : 'log-entry'

  const StatusIcon = isSuccess
    ? CheckCircle2
    : isFailed
      ? XCircle
      : isRunning
        ? Loader2
        : Clock

  const statusLabel = isSuccess
    ? 'Completed'
    : isFailed
      ? 'Failed'
      : isRunning
        ? 'Running'
        : execution.status

  const durationText =
    execution.duration_seconds != null
      ? execution.duration_seconds < 60
        ? `${execution.duration_seconds.toFixed(1)}s`
        : `${Math.floor(execution.duration_seconds / 60)}m ${Math.round(execution.duration_seconds % 60)}s`
      : null

  const timeText = execution.started_at
    ? format(new Date(execution.started_at), 'MMM d, HH:mm')
    : 'Unknown'

  return (
    <div className={`${entryClass} px-3 py-2 text-xs`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <StatusIcon
            className={`w-3.5 h-3.5 shrink-0 ${
              isSuccess
                ? 'text-[hsl(var(--success))]'
                : isFailed
                  ? 'text-destructive'
                  : isRunning
                    ? 'text-[hsl(var(--info))] animate-spin'
                    : 'text-muted-foreground'
            }`}
          />
          <span className="font-medium">{statusLabel}</span>
          <span className="text-muted-foreground">·</span>
          <span className="text-muted-foreground">{timeText}</span>
          {durationText && (
            <>
              <span className="text-muted-foreground">·</span>
              <span className="text-muted-foreground">{durationText}</span>
            </>
          )}
        </div>
        {execution.findings_count > 0 && (
          <span className="text-muted-foreground shrink-0">
            {execution.findings_count} finding{execution.findings_count !== 1 ? 's' : ''}
          </span>
        )}
      </div>
      {isFailed && execution.error_message && (
        <p className="mt-1 text-destructive/80 truncate pl-5.5">
          {execution.error_message}
        </p>
      )}
    </div>
  )
}

// ─── Skeleton ────────────────────────────────────────────

function ExecutionHistorySkeleton() {
  return (
    <div className="space-y-1.5">
      {[0, 1, 2].map((i) => (
        <div key={i} className="log-entry px-3 py-2">
          <div className="flex items-center gap-2">
            <Skeleton className="w-3.5 h-3.5 rounded-full" />
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-3 w-24" />
            <Skeleton className="h-3 w-10" />
          </div>
        </div>
      ))}
    </div>
  )
}
