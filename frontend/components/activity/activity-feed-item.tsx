'use client'

import { useRouter } from 'next/navigation'
import { motion, useReducedMotion } from 'framer-motion'
import {
  MessageCircle,
  RefreshCw,
  ChefHat,
  Rocket,
  Clock,
  Eye,
  Settings,
  CheckCircle2,
  XCircle,
  Loader2,
  Pause,
  Timer,
  User,
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/shared/status-badge'
import type { StatusVariant } from '@/components/shared/status-badge'
import { PremiumIcon } from '@/components/shared'
import { cn } from '@/lib/utils'
import type { ActivityFeedItem } from '@/hooks/use-activity-api'
import { useAgents } from '@/hooks/use-agent-api'

// ─── Type Config ─────────────────────────────────────────

const TYPE_CONFIG = {
  chat: {
    icon: MessageCircle,
    label: 'Chat',
    borderColor: 'border-l-[hsl(var(--primary))]',
    iconBg: 'bg-primary/10',
    iconColor: 'text-primary',
  },
  routine: {
    icon: RefreshCw,
    label: 'Routine',
    borderColor: 'border-l-[hsl(var(--agent))]',
    iconBg: 'bg-[hsl(var(--agent))]/10',
    iconColor: 'text-[hsl(var(--agent))]',
  },
  recipe: {
    icon: ChefHat,
    label: 'Recipe',
    borderColor: 'border-l-[hsl(var(--info))]',
    iconBg: 'bg-[hsl(var(--info))]/10',
    iconColor: 'text-[hsl(var(--info))]',
  },
  mission: {
    icon: Rocket,
    label: 'Mission',
    borderColor: 'border-l-[hsl(var(--success))]',
    iconBg: 'bg-[hsl(var(--success))]/10',
    iconColor: 'text-[hsl(var(--success))]',
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

function formatRelativeTime(iso: string | null): string {
  if (!iso) return ''
  try {
    return formatDistanceToNow(new Date(iso), { addSuffix: true })
  } catch {
    return ''
  }
}

function formatDuration(seconds: number | null): string | null {
  if (seconds == null) return null
  if (seconds < 60) return `${Math.round(seconds)}s`
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  if (mins < 60) return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`
  const hours = Math.floor(mins / 60)
  const remainingMins = mins % 60
  return remainingMins > 0 ? `${hours}h ${remainingMins}m` : `${hours}h`
}

function getViewUrl(item: ActivityFeedItem): string | null {
  switch (item.type) {
    case 'chat':
      return item.source_id ? `/chat/${item.source_id}` : null
    case 'recipe':
      return item.source_url ?? null
    case 'routine':
      return null // Routines expand inline on the Routines tab
    case 'mission':
      return null // Future
    default:
      return null
  }
}

function getConfigureUrl(item: ActivityFeedItem): string | null {
  switch (item.type) {
    case 'chat':
      return null
    case 'routine':
      return item.agent?.id ? `/agents?agent=${item.agent.id}` : null
    case 'recipe':
      return item.source_id ? `/workflows?recipe=${item.source_id}` : null
    case 'mission':
      return null
    default:
      return null
  }
}

// ─── Context Line ────────────────────────────────────────

function ContextLine({ item }: { item: ActivityFeedItem }) {
  switch (item.type) {
    case 'chat':
      if (!item.summary) return null
      return (
        <p className="text-sm text-muted-foreground truncate">
          {item.summary.slice(0, 100)}
        </p>
      )

    case 'routine':
      return (
        <p className="text-xs text-muted-foreground">
          {item.summary || 'Recurring check'}
        </p>
      )

    case 'recipe':
      if (item.step_progress) {
        return (
          <div className="flex items-center gap-1 text-xs text-muted-foreground overflow-hidden">
            {item.step_progress.steps.map((step, i) => (
              <span key={i} className="flex items-center gap-0.5 shrink-0">
                {i > 0 && <span className="mx-0.5">→</span>}
                {step.status === 'completed' && (
                  <CheckCircle2 className="w-3 h-3 text-[hsl(var(--success))]" />
                )}
                {step.status === 'running' && (
                  <Loader2 className="w-3 h-3 text-[hsl(var(--info))] animate-spin" />
                )}
                {step.status === 'failed' && (
                  <XCircle className="w-3 h-3 text-destructive" />
                )}
                {step.status === 'pending' && (
                  <Clock className="w-3 h-3 text-muted-foreground/50" />
                )}
                <span className="truncate max-w-[100px]">{step.name}</span>
              </span>
            ))}
          </div>
        )
      }
      if (item.summary) {
        return (
          <p className="text-sm text-muted-foreground truncate">
            {item.summary}
          </p>
        )
      }
      return null

    case 'mission':
      if (item.step_progress) {
        const pct = Math.round((item.step_progress.current / item.step_progress.total) * 100)
        return (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <div className="flex-1 h-1.5 bg-secondary/50 rounded-full overflow-hidden">
              <div
                className="h-full bg-[hsl(var(--success))] rounded-full transition-all"
                style={{ width: `${pct}%` }}
              />
            </div>
            <span>
              {pct}% · {item.step_progress.current} of {item.step_progress.total} tasks
            </span>
          </div>
        )
      }
      return null

    default:
      return null
  }
}

// ─── Component ───────────────────────────────────────────

interface ActivityFeedItemCardProps {
  item: ActivityFeedItem
  animationDelay?: number
  isNew?: boolean
  onViewItem?: (item: ActivityFeedItem) => void
}

export function ActivityFeedItemCard({ item, animationDelay = 0, isNew = false, onViewItem }: ActivityFeedItemCardProps) {
  const router = useRouter()
  const prefersReducedMotion = useReducedMotion()
  const typeConfig = TYPE_CONFIG[item.type]
  const statusConfig = STATUS_MAP[item.status]
  const TypeIcon = typeConfig.icon
  const duration = formatDuration(item.duration_seconds)
  const relativeTime = formatRelativeTime(item.started_at)
  const configureUrl = getConfigureUrl(item)
  const isRunning = item.status === 'running'

  // Cross-reference with standard agents to get premium_icon mapping
  const { data: agents = [] } = useAgents()
  const agentDetails = item.agent?.id ? agents.find((a: any) => a.id === item.agent?.id) : null
  const premiumIconName = agentDetails?.premium_icon || null

  // Recipes navigate to ExecutionKitchen via /workflows URL params.
  // Routines drill down inline. Chats navigate to /chat.
  const handleView = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (item.type === 'recipe' && item.source_url) {
      // Deep-link to ExecutionKitchen on the workflows page
      router.push(item.source_url)
    } else if (onViewItem && item.type === 'routine') {
      onViewItem(item)
    } else {
      const url = getViewUrl(item)
      if (url) router.push(url)
    }
  }

  const handleConfigure = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (configureUrl) router.push(configureUrl)
  }

  const handleCardClick = () => {
    if (item.type === 'recipe' && item.source_url) {
      router.push(item.source_url)
    } else if (onViewItem && item.type === 'routine') {
      onViewItem(item)
    }
  }

  return (
    <motion.div
      initial={prefersReducedMotion ? false : { opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={prefersReducedMotion ? undefined : { opacity: 0, y: 20 }}
      transition={{ duration: 0.4, delay: animationDelay }}
      className={cn(isNew && !prefersReducedMotion && 'log-entry-new')}
    >
      <div
        onClick={handleCardClick}
        className={`glass-card card-glow border-l-[3px] ${typeConfig.borderColor} hover:border-primary/20 transition-all duration-300 p-3 sm:p-4 space-y-2 ${onViewItem && (item.type === 'recipe' || item.type === 'routine') ? 'cursor-pointer' : ''
          }`}
      >
        {/* Row 1: Type + Name + Time */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <div className={`w-7 h-7 rounded-md ${typeConfig.iconBg} flex items-center justify-center shrink-0`}>
              <TypeIcon className={`w-3.5 h-3.5 ${typeConfig.iconColor}`} />
            </div>
            <span className="text-xs font-medium text-muted-foreground shrink-0">
              {typeConfig.label}
            </span>
            <span className="text-muted-foreground/50 shrink-0">·</span>
            <span className="text-sm font-medium truncate">
              {item.name}
            </span>
          </div>
          {relativeTime && (
            <span className="text-xs text-muted-foreground shrink-0">
              {relativeTime}
            </span>
          )}
        </div>

        {/* Row 2: Agent + Status + Duration + Channel */}
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
          {item.agent && (
            <span className="flex items-center gap-1 text-muted-foreground">
              {premiumIconName ? (
                <PremiumIcon name={premiumIconName} size={12} className="shrink-0 grayscale opacity-70" />
              ) : (
                <User className="w-3 h-3 shrink-0" />
              )}
              {item.agent.name}
            </span>
          )}
          <StatusBadge
            status={statusConfig.variant}
            dot={statusConfig.dot}
            size="sm"
            className={cn(isRunning && 'shadow-[0_0_12px_hsla(var(--info)/0.3)] stage-active-badge')}
          >
            {statusConfig.label}
          </StatusBadge>
          {duration && (
            <span className="flex items-center gap-1 text-muted-foreground">
              <Timer className="w-3 h-3" />
              {duration}
            </span>
          )}
          {item.channel && (
            <span className="text-xs text-muted-foreground">
              via {item.channel.name || item.channel.type}
            </span>
          )}
        </div>

        {/* Context Line */}
        <ContextLine item={item} />

        {/* Error message */}
        {item.status === 'failed' && item.error_message && (
          <p className="text-xs text-destructive/80 truncate">
            {item.error_message}
          </p>
        )}

        {/* Actions */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center sm:justify-end gap-2 pt-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleView}
            className="text-xs min-h-[44px] sm:min-h-0 sm:h-7 justify-center"
          >
            <Eye className="w-3.5 h-3.5 mr-1" />
            View
          </Button>
          {configureUrl && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleConfigure}
              className="text-xs min-h-[44px] sm:min-h-0 sm:h-7 justify-center"
            >
              <Settings className="w-3.5 h-3.5 mr-1" />
              Configure
            </Button>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// ─── Skeleton ────────────────────────────────────────────

export function ActivityFeedItemSkeleton() {
  return (
    <div className="glass-card border-l-[3px] border-border/30 p-4 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-md bg-secondary/50 animate-pulse" />
          <div className="h-3 w-14 bg-secondary/50 rounded animate-pulse" />
          <div className="h-4 w-32 bg-secondary/50 rounded animate-pulse" />
        </div>
        <div className="h-3 w-16 bg-secondary/50 rounded animate-pulse" />
      </div>
      <div className="flex items-center gap-2">
        <div className="h-3 w-20 bg-secondary/50 rounded animate-pulse" />
        <div className="h-5 w-16 bg-secondary/50 rounded-full animate-pulse" />
        <div className="h-3 w-10 bg-secondary/50 rounded animate-pulse" />
      </div>
      <div className="h-3 w-3/4 bg-secondary/50 rounded animate-pulse" />
    </div>
  )
}
