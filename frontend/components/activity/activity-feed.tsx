'use client'

import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Activity,
  RefreshCw,
  ChefHat,
  Rocket,
  Loader2,
  CheckCircle,
  AlertTriangle,
  Clock,
  Zap,
  Eye,
  MessageCircle,
} from 'lucide-react'
import { useRouter } from 'next/navigation'
import { formatDistanceToNow } from 'date-fns'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useActivityFeed } from '@/hooks/use-activity-api'
import type { ActivityFeedFilters, ActivityFeedItem } from '@/hooks/use-activity-api'
import { ExecutionDetail } from './execution-detail'
import dynamic from 'next/dynamic'
import { cn } from '@/lib/utils'

// Lazy-load ExecutionKitchen (heavy component with streaming, theater sub-components)
const ExecutionKitchen = dynamic(
  () => import('@/components/workflows/execution-kitchen').then((m) => m.ExecutionKitchen),
  { ssr: false, loading: () => <KitchenSkeleton /> }
)

function KitchenSkeleton() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="h-8 w-32 bg-secondary/30 rounded" />
      <div className="h-[400px] bg-secondary/20 rounded-xl" />
    </div>
  )
}

// ─── Filter Chip Config ─────────────────────────────────

const TYPE_CHIPS = [
  { value: 'chat', label: 'Chats', icon: MessageCircle },
  { value: 'routine', label: 'Routines', icon: RefreshCw },
  { value: 'recipe', label: 'Recipes', icon: ChefHat },
  { value: 'mission', label: 'Missions', icon: Rocket, comingSoon: true },
] as const

const STATUS_OPTIONS = [
  { value: 'all', label: 'All Statuses' },
  { value: 'running', label: 'Working' },
  { value: 'completed', label: 'Done' },
  { value: 'failed', label: 'Needs Attention' },
  { value: 'pending', label: 'Upcoming' },
] as const

const PAGE_SIZE = 20

// ─── Status Helpers ─────────────────────────────────────

function getStatusIcon(status: string) {
  switch (status) {
    case 'completed':
      return <CheckCircle className="w-5 h-5 text-green-400" />
    case 'failed':
      return <AlertTriangle className="w-5 h-5 text-red-400" />
    case 'running':
      return <Activity className="w-5 h-5 text-blue-400 animate-pulse" />
    default:
      return <Clock className="w-5 h-5 text-gray-400" />
  }
}

function getStatusBg(status: string) {
  switch (status) {
    case 'completed':
      return 'bg-green-500/20'
    case 'failed':
      return 'bg-red-500/20'
    case 'running':
      return 'bg-blue-500/20'
    default:
      return 'bg-gray-500/20'
  }
}

function getStatusBadge(status: string) {
  switch (status) {
    case 'completed':
      return { label: 'cooked', className: 'bg-green-500/20 text-green-400 border-green-500/30' }
    case 'failed':
      return { label: 'burnt', className: 'bg-red-500/20 text-red-400 border-red-500/30' }
    case 'running':
      return { label: 'cooking', className: 'bg-blue-500/20 text-blue-400 border-blue-500/30' }
    default:
      return { label: status, className: 'bg-gray-500/20 text-gray-400 border-gray-500/30' }
  }
}

function formatTimestamp(dateStr: string | null) {
  if (!dateStr) return ''
  try {
    return formatDistanceToNow(new Date(dateStr), { addSuffix: true })
  } catch {
    return ''
  }
}

// ─── Component ──────────────────────────────────────────

interface ActivityFeedProps {
  period?: string
  openExecution?: string | null
  deepLinkRecipeId?: string | null
}

export function ActivityFeed({ period = '1d', openExecution, deepLinkRecipeId }: ActivityFeedProps) {
  const router = useRouter()
  const [activeTypes, setActiveTypes] = useState<string[]>([])
  const [statusFilter, setStatusFilter] = useState('all')
  const [limit, setLimit] = useState(PAGE_SIZE)
  const [selectedItem, setSelectedItem] = useState<ActivityFeedItem | null>(null)
  const deepLinkHandled = useRef(false)

  const filters = useMemo<ActivityFeedFilters>(
    () => ({
      types: activeTypes.length > 0 ? activeTypes : undefined,
      status: statusFilter !== 'all' ? statusFilter : null,
      period,
      limit,
      offset: 0,
    }),
    [activeTypes, statusFilter, period, limit]
  )

  const { data, isLoading, isFetching } = useActivityFeed(filters)

  const items = data?.items ?? []
  const total = data?.total ?? 0
  const hasMore = items.length < total

  // Track known IDs to detect items arriving via polling (for log-slide-in animation)
  const knownIds = useRef<Set<string>>(new Set())
  const isInitialLoad = useRef(true)

  const newItemIds = useMemo(() => {
    if (isInitialLoad.current || items.length === 0) return new Set<string>()
    const result = new Set<string>()
    for (const item of items) {
      if (!knownIds.current.has(item.id)) result.add(item.id)
    }
    return result
  }, [items])

  useEffect(() => {
    if (items.length > 0) {
      for (const item of items) {
        knownIds.current.add(item.id)
      }
      isInitialLoad.current = false
    }
  }, [items])

  // Deep-link: /activity?openExecution=X&recipeId=Y → auto-select matching item
  useEffect(() => {
    if (!openExecution || deepLinkHandled.current || items.length === 0) return
    // Find matching item by execution ID in the source_url or id
    const match = items.find(
      (item) =>
        item.source_url?.includes(`openExecution=${openExecution}`) ||
        item.id === `recipe-${openExecution}`
    )
    if (match) {
      setSelectedItem(match)
      deepLinkHandled.current = true
    } else if (!isLoading) {
      // Item not in current page — build a minimal stub so ExecutionDetail can render
      setSelectedItem({
        id: `recipe-${openExecution}`,
        type: 'recipe',
        name: 'Recipe Execution',
        status: 'completed',
        started_at: null,
        completed_at: null,
        duration_seconds: null,
        agent: null,
        agents: [],
        summary: '',
        source_id: deepLinkRecipeId || null,
        source_url: `/activity?openExecution=${openExecution}&recipeId=${deepLinkRecipeId}`,
        trigger: null,
        error_message: null,
      })
      deepLinkHandled.current = true
    }
  }, [openExecution, deepLinkRecipeId, items, isLoading])

  const toggleType = useCallback((type: string) => {
    setActiveTypes((prev) => {
      if (type === '__all__') return []
      // Exclusive selection: clicking active chip deselects, clicking another selects only that one
      return prev.length === 1 && prev[0] === type ? [] : [type]
    })
    setLimit(PAGE_SIZE)
  }, [])

  const handleStatusChange = useCallback((value: string) => {
    setStatusFilter(value)
    setLimit(PAGE_SIZE)
  }, [])

  const handleLoadMore = useCallback(() => {
    setLimit((prev) => prev + PAGE_SIZE)
  }, [])

  const handleViewItem = useCallback((item: ActivityFeedItem) => {
    // Navigate to ExecutionKitchen for recipes (detailed live view)
    if (item.type === 'recipe' && item.source_id) {
      const execId = item.id.replace('recipe-', '')
      router.push(`/activity/execution?id=${execId}&recipeId=${item.source_id}`)
      return
    }
    // For chats, navigate to the chat
    if (item.type === 'chat' && item.source_id) {
      router.push(`/chat/${item.source_id}`)
      return
    }
    // Fallback to inline detail for routines
    setSelectedItem(item)
  }, [router])

  const handleCloseDetail = useCallback(() => {
    setSelectedItem(null)
  }, [])

  // ─── Execution Detail Drill-Down ──────────────────

  if (selectedItem) {
    // Recipes → full ExecutionKitchen (cooking theater with streaming logs, self-learning)
    if (selectedItem.type === 'recipe') {
      // Extract execution ID: source_url has ?openExecution=X or id is recipe-X
      const execId =
        selectedItem.source_url?.match(/openExecution=([^&]+)/)?.[1] ??
        selectedItem.id.replace('recipe-', '')
      const recipeId = selectedItem.source_id ?? selectedItem.source_url?.match(/recipeId=([^&]+)/)?.[1] ?? ''

      return (
        <ExecutionKitchen
          workflowId={0}
          onBack={handleCloseDetail}
          autoStart={false}
          executionType="recipe"
          recipeExecutionId={execId}
          recipeId={recipeId}
          recipeSteps={[]}
        />
      )
    }

    // Non-recipes → standard detail view
    return <ExecutionDetail item={selectedItem} onClose={handleCloseDetail} />
  }

  // ─── Empty State ────────────────────────────────

  if (!isLoading && items.length === 0) {
    return (
      <div className="space-y-4">
        <FilterBar
          activeTypes={activeTypes}
          statusFilter={statusFilter}
          onToggleType={toggleType}
          onStatusChange={handleStatusChange}
          isFetching={isFetching}
        />
        <div className="glass-card p-12 text-center text-muted-foreground">
          <Activity className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p className="font-medium">No activity yet</p>
          <p className="text-sm mt-1 max-w-sm mx-auto">
            Create a routine or run a recipe to see your workforce in action
          </p>
        </div>
      </div>
    )
  }

  // ─── Loading State ──────────────────────────────

  if (isLoading) {
    return (
      <div className="space-y-4">
        <FilterBar
          activeTypes={activeTypes}
          statusFilter={statusFilter}
          onToggleType={toggleType}
          onStatusChange={handleStatusChange}
          isFetching={false}
        />
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="glass-card p-4 animate-pulse">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-lg bg-secondary/30" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-secondary/30 rounded w-1/3" />
                  <div className="h-3 bg-secondary/20 rounded w-1/4" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  // ─── Feed List ──────────────────────────────────

  return (
    <div className="space-y-4">
      <FilterBar
        activeTypes={activeTypes}
        statusFilter={statusFilter}
        onToggleType={toggleType}
        onStatusChange={handleStatusChange}
        isFetching={isFetching}
      />

      <div className="space-y-3">
        <AnimatePresence mode="popLayout">
          {items.map((item, index) => (
            <FeedRow
              key={item.id}
              item={item}
              index={index}
              isNew={newItemIds.has(item.id)}
              onView={handleViewItem}
            />
          ))}
        </AnimatePresence>
      </div>

      {hasMore && (
        <div className="flex justify-center pt-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleLoadMore}
            disabled={isFetching}
            className="min-w-[120px] w-full sm:w-auto min-h-[44px] sm:min-h-0"
          >
            {isFetching ? (
              <>
                <Loader2 className="w-3.5 h-3.5 mr-2 animate-spin" />
                Loading...
              </>
            ) : (
              'Load More'
            )}
          </Button>
        </div>
      )}
    </div>
  )
}

// ─── Feed Row (Cooking-Style Slim Row) ──────────────────

interface FeedRowProps {
  item: ActivityFeedItem
  index: number
  isNew: boolean
  onView: (item: ActivityFeedItem) => void
}

function FeedRow({ item, index, isNew, onView }: FeedRowProps) {
  const isRecipe = item.type === 'recipe'
  const isRoutine = item.type === 'routine'
  const statusBadge = getStatusBadge(item.status)

  return (
    <motion.div
      className="glass-card p-4 hover:border-primary/20 transition-all duration-300 cursor-pointer"
      initial={{ opacity: 0, x: isNew ? -20 : 0 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
      layout
      onClick={() => onView(item)}
    >
      <div className="flex items-center gap-4">
        {/* Status Indicator */}
        <div className={cn('flex items-center justify-center w-10 h-10 rounded-lg shrink-0', getStatusBg(item.status))}>
          {getStatusIcon(item.status)}
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <h4 className="font-semibold text-sm sm:text-base truncate">{item.name}</h4>
            {item.type === 'chat' && (
              <Badge className="text-xs bg-blue-500/20 text-blue-300 border-blue-500/30 shrink-0">
                <MessageCircle className="w-3 h-3 mr-1" />
                Chat
              </Badge>
            )}
            {isRoutine && (
              <Badge className="text-xs bg-purple-500/20 text-purple-300 border-purple-500/30 shrink-0">
                <RefreshCw className="w-3 h-3 mr-1" />
                Routine
              </Badge>
            )}
            {isRecipe && (
              <Badge className="text-xs bg-orange-500/20 text-orange-300 border-orange-500/30 shrink-0">
                <Zap className="w-3 h-3 mr-1" />
                Recipe
              </Badge>
            )}
            {item.type === 'mission' && (
              <Badge className="text-xs bg-cyan-500/20 text-cyan-300 border-cyan-500/30 shrink-0">
                <Rocket className="w-3 h-3 mr-1" />
                Mission
              </Badge>
            )}
            <Badge className={cn('text-xs shrink-0', statusBadge.className)}>
              {statusBadge.label}
            </Badge>
          </div>

          {/* Sub-line: step progress or summary */}
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            {isRecipe && item.step_progress ? (
              <>
                <span>Step {item.step_progress.current} of {item.step_progress.total}</span>
                {item.error_message && (
                  <span className="text-red-400 truncate max-w-[200px]">{item.error_message}</span>
                )}
              </>
            ) : (
              <span className="truncate max-w-[300px]">{item.summary}</span>
            )}
          </div>

          {/* Agent pills */}
          {item.agents && item.agents.length > 0 && (
            <div className="flex gap-2 flex-wrap mt-1.5">
              {item.agents.map((agent, agentIdx) => (
                <div
                  key={agent.id ?? agentIdx}
                  className={cn(
                    'flex items-center gap-1 text-xs px-2 py-0.5 rounded-md',
                    item.status === 'completed'
                      ? 'bg-green-500/10 text-green-400'
                      : item.status === 'failed'
                        ? 'bg-red-500/10 text-red-400'
                        : item.status === 'running'
                          ? 'bg-blue-500/10 text-blue-400'
                          : 'bg-gray-500/10 text-gray-400'
                  )}
                >
                  {item.status === 'completed' ? (
                    <CheckCircle className="w-3 h-3" />
                  ) : item.status === 'failed' ? (
                    <AlertTriangle className="w-3 h-3" />
                  ) : item.status === 'running' ? (
                    <Activity className="w-3 h-3 animate-pulse" />
                  ) : (
                    <Clock className="w-3 h-3" />
                  )}
                  <span>{agent.name}</span>
                  {item.duration_seconds != null && item.duration_seconds > 0 && (
                    <span className="opacity-60">({item.duration_seconds.toFixed(1)}s)</span>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Recipe step agent pills (from step_progress) */}
          {isRecipe && item.step_progress?.steps && item.step_progress.steps.length > 0 && item.agents.length === 0 && (
            <div className="flex gap-2 flex-wrap mt-1.5">
              {item.step_progress.steps.map((step, stepIdx) => (
                <div
                  key={stepIdx}
                  className={cn(
                    'flex items-center gap-1 text-xs px-2 py-0.5 rounded-md',
                    step.status === 'completed'
                      ? 'bg-green-500/10 text-green-400'
                      : step.status === 'failed'
                        ? 'bg-red-500/10 text-red-400'
                        : step.status === 'running'
                          ? 'bg-blue-500/10 text-blue-400'
                          : 'bg-gray-500/10 text-gray-400'
                  )}
                >
                  {step.status === 'completed' ? (
                    <CheckCircle className="w-3 h-3" />
                  ) : step.status === 'failed' ? (
                    <AlertTriangle className="w-3 h-3" />
                  ) : step.status === 'running' ? (
                    <Activity className="w-3 h-3 animate-pulse" />
                  ) : (
                    <Clock className="w-3 h-3" />
                  )}
                  <span>{step.name}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Metrics (hidden on mobile) */}
        <div className="hidden lg:flex items-center gap-6 text-sm shrink-0">
          {item.started_at && (
            <div className="text-left max-w-[200px]">
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="w-3 h-3" />
                <span>{formatTimestamp(item.started_at)}</span>
              </div>
            </div>
          )}

          {isRecipe && item.step_progress && (
            <div className="text-center">
              <p className="text-muted-foreground text-xs">Steps</p>
              <p className="font-medium">{item.step_progress.total}</p>
            </div>
          )}

          {/* Tokens for recipes */}
          {isRecipe && item.total_tokens != null && item.total_tokens > 0 && (
            <div className="text-center">
              <p className="text-muted-foreground text-xs">Tokens</p>
              <p className="font-medium">{item.total_tokens.toLocaleString()}</p>
            </div>
          )}

          {/* Tokens for routines */}
          {isRoutine && item.tokens_used != null && item.tokens_used > 0 && (
            <div className="text-center">
              <p className="text-muted-foreground text-xs">Tokens</p>
              <p className="font-medium">{item.tokens_used.toLocaleString()}</p>
            </div>
          )}

          {/* Duration for recipes */}
          {isRecipe && item.total_duration_ms != null && item.total_duration_ms > 0 && (
            <div className="text-center">
              <p className="text-muted-foreground text-xs">Duration</p>
              <p className="font-medium">{(item.total_duration_ms / 1000).toFixed(1)}s</p>
            </div>
          )}

          {/* Duration fallback from duration_seconds */}
          {!isRecipe && item.duration_seconds != null && item.duration_seconds > 0 && (
            <div className="text-center">
              <p className="text-muted-foreground text-xs">Duration</p>
              <p className="font-medium">{item.duration_seconds.toFixed(1)}s</p>
            </div>
          )}
        </div>

        {/* View button */}
        <Button
          size="sm"
          variant="outline"
          className="shrink-0"
          onClick={(e) => {
            e.stopPropagation()
            onView(item)
          }}
        >
          <Eye className="w-4 h-4" />
          <span className="hidden xl:inline ml-1">View</span>
        </Button>
      </div>
    </motion.div>
  )
}

// ─── Filter Bar ──────────────────────────────────────────

interface FilterBarProps {
  activeTypes: string[]
  statusFilter: string
  onToggleType: (type: string) => void
  onStatusChange: (value: string) => void
  isFetching: boolean
}

function FilterBar({
  activeTypes,
  statusFilter,
  onToggleType,
  onStatusChange,
  isFetching,
}: FilterBarProps) {
  return (
    <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-thin -mx-1 px-1">
      {/* Type chips */}
      <Button
        variant={activeTypes.length === 0 ? 'secondary' : 'ghost'}
        size="sm"
        onClick={() => onToggleType('__all__')}
        className={cn(
          'text-xs min-h-[44px] sm:min-h-0 sm:h-7 shrink-0',
          activeTypes.length === 0 && 'bg-secondary/80'
        )}
      >
        All
      </Button>

      {TYPE_CHIPS.map((chip) => {
        const isActive = activeTypes.includes(chip.value)
        const Icon = chip.icon
        const isDisabled = 'comingSoon' in chip && chip.comingSoon
        return (
          <Button
            key={chip.value}
            variant={isActive ? 'secondary' : 'ghost'}
            size="sm"
            onClick={() => !isDisabled && onToggleType(chip.value)}
            disabled={isDisabled}
            className={cn(
              'text-xs min-h-[44px] sm:min-h-0 sm:h-7 shrink-0 gap-1.5',
              isActive && 'bg-secondary/80',
              isDisabled && 'opacity-40 cursor-not-allowed'
            )}
          >
            <Icon className="w-3.5 h-3.5" />
            <span className="hidden xs:inline">
              {chip.label}{isDisabled ? ' (soon)' : ''}
            </span>
          </Button>
        )
      })}

      {/* Divider */}
      <div className="w-px h-5 bg-border/50 shrink-0 mx-1" />

      {/* Status dropdown */}
      <Select value={statusFilter} onValueChange={onStatusChange}>
        <SelectTrigger className="w-[140px] min-h-[44px] sm:min-h-0 sm:h-7 text-xs bg-secondary/50 shrink-0">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {STATUS_OPTIONS.map((opt) => (
            <SelectItem key={opt.value} value={opt.value} className="text-xs">
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Fetching indicator */}
      {isFetching && (
        <Loader2 className="w-3.5 h-3.5 text-muted-foreground animate-spin shrink-0" />
      )}
    </div>
  )
}
