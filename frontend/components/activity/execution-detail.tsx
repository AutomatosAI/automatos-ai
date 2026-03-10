'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion'
import {
  ArrowLeft,
  ChefHat,
  RefreshCw,
  MessageCircle,
  Rocket,
  Play,
  Pencil,
  FileText,
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  Timer,
  User,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import { format, formatDistanceToNow } from 'date-fns'
import { toast } from 'react-hot-toast'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/shared/status-badge'
import type { StatusVariant } from '@/components/shared/status-badge'
import { useExecuteRecipe } from '@/hooks/use-recipe-api'
import type { ActivityFeedItem, ActivityStepProgress } from '@/hooks/use-activity-api'

// ─── Type Config ─────────────────────────────────────────

const TYPE_CONFIG = {
  chat: {
    icon: MessageCircle,
    label: 'Chat',
    accentColor: 'text-primary',
    borderColor: 'border-l-[hsl(var(--primary))]',
  },
  routine: {
    icon: RefreshCw,
    label: 'Routine',
    accentColor: 'text-[hsl(var(--agent))]',
    borderColor: 'border-l-[hsl(var(--agent))]',
  },
  recipe: {
    icon: ChefHat,
    label: 'Recipe',
    accentColor: 'text-[hsl(var(--info))]',
    borderColor: 'border-l-[hsl(var(--info))]',
  },
  mission: {
    icon: Rocket,
    label: 'Mission',
    accentColor: 'text-[hsl(var(--success))]',
    borderColor: 'border-l-[hsl(var(--success))]',
  },
} as const

const STATUS_MAP: Record<ActivityFeedItem['status'], { label: string; variant: StatusVariant; dot: boolean }> = {
  pending: { label: 'Waiting', variant: 'neutral', dot: false },
  running: { label: 'Working...', variant: 'active', dot: true },
  completed: { label: 'Done', variant: 'success', dot: false },
  failed: { label: 'Needs Attention', variant: 'error', dot: false },
  cancelled: { label: 'Cancelled', variant: 'neutral', dot: false },
  paused: { label: 'Paused', variant: 'warning', dot: false },
}

// ─── Helpers ─────────────────────────────────────────────

function formatDuration(seconds: number | null): string {
  if (seconds == null) return '—'
  if (seconds < 60) return `${Math.round(seconds)}s`
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  if (mins < 60) return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`
  const hours = Math.floor(mins / 60)
  const remainingMins = mins % 60
  return remainingMins > 0 ? `${hours}h ${remainingMins}m` : `${hours}h`
}

function formatTimestamp(iso: string | null): string {
  if (!iso) return '—'
  try {
    return format(new Date(iso), 'MMM d, HH:mm')
  } catch {
    return '—'
  }
}

function formatRelativeTime(iso: string | null): string {
  if (!iso) return ''
  try {
    return formatDistanceToNow(new Date(iso), { addSuffix: true })
  } catch {
    return ''
  }
}

// ─── Step Status Icon ────────────────────────────────────

function StepStatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'completed':
      return <CheckCircle2 className="w-5 h-5 text-[hsl(var(--success))]" />
    case 'running':
      return <Loader2 className="w-5 h-5 text-[hsl(var(--info))] animate-spin" />
    case 'failed':
      return <XCircle className="w-5 h-5 text-destructive" />
    case 'pending':
    default:
      return <Clock className="w-5 h-5 text-muted-foreground/50" />
  }
}

function stepCssClass(status: string): string {
  switch (status) {
    case 'completed':
      return 'stage-completed'
    case 'running':
      return 'stage-active'
    case 'failed':
      return 'stage-completed' // use completed styling with error icon
    case 'pending':
    default:
      return 'stage-pending'
  }
}

// ─── Step Pipeline ───────────────────────────────────────

function StepPipeline({ progress }: { progress: ActivityStepProgress }) {
  return (
    <div className="space-y-0">
      {progress.steps.map((step, i) => (
        <div key={i}>
          {/* Connector between steps */}
          {i > 0 && (
            <div className="flex items-center pl-6 py-0">
              <div
                className={`stage-connector w-0.5 h-6 mx-auto ${
                  step.status === 'completed' || step.status === 'running'
                    ? 'stage-connector-completed'
                    : ''
                }`}
                style={{ width: '2px', height: '24px' }}
              />
            </div>
          )}

          {/* Step card */}
          <div
            className={`${stepCssClass(step.status)} rounded-lg border p-3 flex items-center gap-3`}
          >
            <StepStatusIcon status={step.status} />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{step.name}</p>
            </div>
            <span className="text-xs text-muted-foreground shrink-0">
              Step {i + 1} of {progress.total}
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}

// ─── Execution Log ───────────────────────────────────────

interface LogEntry {
  timestamp: string
  message: string
  level: 'info' | 'success' | 'error' | 'warning'
}

function generateLogEntries(item: ActivityFeedItem): LogEntry[] {
  const entries: LogEntry[] = []

  if (item.started_at) {
    entries.push({
      timestamp: item.started_at,
      message: `Started execution${item.trigger ? ` (triggered: ${item.trigger})` : ''}`,
      level: 'info',
    })
  }

  if (item.step_progress) {
    for (const step of item.step_progress.steps) {
      if (step.status === 'completed') {
        entries.push({
          timestamp: item.started_at ?? '',
          message: `${step.name}: Complete`,
          level: 'success',
        })
      } else if (step.status === 'running') {
        entries.push({
          timestamp: item.started_at ?? '',
          message: `${step.name}: In progress...`,
          level: 'info',
        })
      } else if (step.status === 'failed') {
        entries.push({
          timestamp: item.started_at ?? '',
          message: `${step.name}: Failed`,
          level: 'error',
        })
      }
    }
  }

  if (item.error_message) {
    entries.push({
      timestamp: item.completed_at ?? item.started_at ?? '',
      message: item.error_message,
      level: 'error',
    })
  }

  if (item.status === 'completed' && item.completed_at) {
    entries.push({
      timestamp: item.completed_at,
      message: `Execution completed in ${formatDuration(item.duration_seconds)}`,
      level: 'success',
    })
  }

  return entries
}

const LOG_LEVEL_CLASS: Record<LogEntry['level'], string> = {
  info: 'log-entry-info',
  success: 'log-entry-success',
  error: 'log-entry-error',
  warning: 'log-entry-warning',
}

function ExecutionLog({ entries }: { entries: LogEntry[] }) {
  if (entries.length === 0) {
    return (
      <p className="text-sm text-muted-foreground/60 text-center py-4">
        No log entries available
      </p>
    )
  }

  return (
    <div className="space-y-1">
      {entries.map((entry, i) => (
        <div
          key={i}
          className={`log-entry ${LOG_LEVEL_CLASS[entry.level]} rounded-lg px-3 py-2 flex items-start gap-3 text-sm`}
        >
          <span className="text-xs text-muted-foreground shrink-0 pt-0.5 font-mono">
            {entry.timestamp ? format(new Date(entry.timestamp), 'HH:mm:ss') : '—'}
          </span>
          <span className="flex-1 min-w-0 break-words">{entry.message}</span>
        </div>
      ))}
    </div>
  )
}

// ─── Main Component ──────────────────────────────────────

interface ExecutionDetailProps {
  item: ActivityFeedItem
  onClose?: () => void
}

export function ExecutionDetail({ item, onClose }: ExecutionDetailProps) {
  const router = useRouter()
  const prefersReducedMotion = useReducedMotion()
  const [logsExpanded, setLogsExpanded] = useState(true)

  const typeConfig = TYPE_CONFIG[item.type]
  const statusConfig = STATUS_MAP[item.status]
  const TypeIcon = typeConfig.icon
  const logEntries = generateLogEntries(item)

  const executeRecipe = useExecuteRecipe()

  const handleBack = () => {
    if (onClose) {
      onClose()
    } else {
      router.push('/activity')
    }
  }

  const handleReRun = () => {
    if (item.type !== 'recipe' || !item.source_id) return
    executeRecipe.mutate(
      { recipeId: item.source_id },
      {
        onSuccess: () => {
          toast.success('Recipe execution started')
        },
        onError: () => {
          toast.error('Failed to start recipe execution')
        },
      }
    )
  }

  const handleEdit = () => {
    if (item.type === 'recipe' && item.source_id) {
      router.push('/activity')
    } else if (item.type === 'routine' && item.agent?.id) {
      router.push(`/agents?agent=${item.agent.id}&tab=heartbeat`)
    }
  }

  const handleViewSource = () => {
    if (item.type === 'chat' && item.source_id) {
      router.push(`/chat/${item.source_id}`)
    } else if (item.source_url) {
      router.push(item.source_url)
    }
  }

  return (
    <motion.div
      initial={prefersReducedMotion ? false : { opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={prefersReducedMotion ? undefined : { opacity: 0, y: 20 }}
      transition={{ duration: prefersReducedMotion ? 0 : 0.3 }}
    >
      <div className={`glass-panel rounded-xl border-l-[3px] ${typeConfig.borderColor} p-4 sm:p-6 space-y-5 sm:space-y-6`}>
        {/* Back link */}
        <button
          onClick={handleBack}
          className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors min-h-[44px] sm:min-h-0"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Activity
        </button>

        {/* Header */}
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <TypeIcon className={`w-6 h-6 ${typeConfig.accentColor}`} />
            <h2 className="text-2xl md:text-3xl font-bold truncate">
              {item.name}
            </h2>
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
            <span className="text-muted-foreground">{typeConfig.label}</span>
            <span className="text-muted-foreground/40">·</span>
            <span className="text-muted-foreground">
              Started {formatTimestamp(item.started_at)}
            </span>
            {item.started_at && (
              <>
                <span className="text-muted-foreground/40">·</span>
                <span className="text-xs text-muted-foreground">
                  {formatRelativeTime(item.started_at)}
                </span>
              </>
            )}
            <span className="text-muted-foreground/40">·</span>
            <StatusBadge status={statusConfig.variant} dot={statusConfig.dot}>
              {statusConfig.label}
            </StatusBadge>
            <span className="text-muted-foreground/40">·</span>
            <span className="flex items-center gap-1 text-muted-foreground">
              <Timer className="w-3.5 h-3.5" />
              {formatDuration(item.duration_seconds)}
            </span>
          </div>

          {/* Agent info */}
          {item.agent && (
            <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
              <User className="w-3.5 h-3.5" />
              <span>Agent: {item.agent.name}</span>
            </div>
          )}
          {item.agents && item.agents.length > 1 && (
            <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
              <User className="w-3.5 h-3.5" />
              <span>Agents: {item.agents.map((a) => a.name).join(', ')}</span>
            </div>
          )}

          {/* Channel badge */}
          {item.channel && (
            <span className="text-xs text-muted-foreground">
              via {item.channel.name || item.channel.type}
            </span>
          )}
        </div>

        {/* Step Pipeline (recipes/missions with steps) */}
        {item.step_progress && item.step_progress.steps.length > 0 && (
          <section className="space-y-3">
            <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
              Step Pipeline
            </h3>
            <StepPipeline progress={item.step_progress} />
          </section>
        )}

        {/* Output */}
        <section className="space-y-3">
          <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
            Output
          </h3>
          <div className="glass-card rounded-lg p-4">
            {item.summary ? (
              <p className="text-sm whitespace-pre-wrap">{item.summary}</p>
            ) : (
              <p className="text-sm text-muted-foreground/60 text-center">
                No output captured
              </p>
            )}
          </div>
        </section>

        {/* Error message */}
        {item.error_message && (
          <section className="space-y-3">
            <h3 className="text-sm font-medium text-destructive uppercase tracking-wider">
              Error
            </h3>
            <div className="log-entry log-entry-error rounded-lg px-4 py-3">
              <p className="text-sm text-destructive">{item.error_message}</p>
            </div>
          </section>
        )}

        {/* Execution Log */}
        <section className="space-y-3">
          <button
            onClick={() => setLogsExpanded(!logsExpanded)}
            className="flex items-center gap-2 text-sm font-medium text-muted-foreground uppercase tracking-wider hover:text-foreground transition-colors w-full"
          >
            <span>Execution Log</span>
            {logsExpanded ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
            <span className="text-xs font-normal normal-case">
              ({logEntries.length} {logEntries.length === 1 ? 'entry' : 'entries'})
            </span>
          </button>
          <AnimatePresence>
            {logsExpanded && (
              <motion.div
                initial={prefersReducedMotion ? false : { height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={prefersReducedMotion ? undefined : { height: 0, opacity: 0 }}
                transition={{ duration: prefersReducedMotion ? 0 : 0.2 }}
                className="overflow-hidden"
              >
                <ExecutionLog entries={logEntries} />
              </motion.div>
            )}
          </AnimatePresence>
        </section>

        {/* Actions */}
        <div className="flex flex-col sm:flex-row sm:flex-wrap items-stretch sm:items-center sm:justify-end gap-2 pt-2 border-t border-border/30">
          {item.type === 'recipe' && item.source_id && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleReRun}
              disabled={executeRecipe.isLoading}
              className="text-xs min-h-[44px] sm:min-h-0 justify-center"
            >
              {executeRecipe.isLoading ? (
                <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
              ) : (
                <Play className="w-3.5 h-3.5 mr-1.5" />
              )}
              Re-run
            </Button>
          )}
          {(item.type === 'recipe' || item.type === 'routine') && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleEdit}
              className="text-xs min-h-[44px] sm:min-h-0 justify-center"
            >
              <Pencil className="w-3.5 h-3.5 mr-1.5" />
              {item.type === 'recipe' ? 'Edit Recipe' : 'Edit Routine'}
            </Button>
          )}
          {item.type === 'routine' && item.agent?.id && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => router.push(`/activity?tab=reports&agent_id=${item.agent?.id}`)}
              className="text-xs min-h-[44px] sm:min-h-0 justify-center"
            >
              <FileText className="w-3.5 h-3.5 mr-1.5" />
              View Report
            </Button>
          )}
          {item.source_url && item.type !== 'routine' && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleViewSource}
              className="text-xs min-h-[44px] sm:min-h-0 justify-center"
            >
              <FileText className="w-3.5 h-3.5 mr-1.5" />
              View Logs
            </Button>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// ─── Skeleton ────────────────────────────────────────────

export function ExecutionDetailSkeleton() {
  return (
    <div className="glass-panel rounded-xl border-l-[3px] border-border/30 p-4 sm:p-6 space-y-5 sm:space-y-6">
      {/* Back link */}
      <div className="h-4 w-28 bg-secondary/50 rounded animate-pulse" />

      {/* Header */}
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 rounded bg-secondary/50 animate-pulse" />
          <div className="h-7 w-64 bg-secondary/50 rounded animate-pulse" />
        </div>
        <div className="flex items-center gap-3">
          <div className="h-4 w-16 bg-secondary/50 rounded animate-pulse" />
          <div className="h-4 w-32 bg-secondary/50 rounded animate-pulse" />
          <div className="h-5 w-20 bg-secondary/50 rounded-full animate-pulse" />
          <div className="h-4 w-12 bg-secondary/50 rounded animate-pulse" />
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-3">
        <div className="h-4 w-24 bg-secondary/50 rounded animate-pulse" />
        {[0, 1, 2].map((i) => (
          <div key={i} className="stage-pending rounded-lg border p-3 flex items-center gap-3">
            <div className="w-5 h-5 rounded-full bg-secondary/50 animate-pulse" />
            <div className="h-4 w-40 bg-secondary/50 rounded animate-pulse" />
          </div>
        ))}
      </div>

      {/* Output */}
      <div className="space-y-3">
        <div className="h-4 w-16 bg-secondary/50 rounded animate-pulse" />
        <div className="glass-card rounded-lg p-4">
          <div className="h-4 w-3/4 bg-secondary/50 rounded animate-pulse" />
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-end gap-2 pt-2 border-t border-border/30">
        <div className="h-8 w-20 bg-secondary/50 rounded animate-pulse" />
        <div className="h-8 w-24 bg-secondary/50 rounded animate-pulse" />
      </div>
    </div>
  )
}
