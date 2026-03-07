'use client'

import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
import { AnimatePresence } from 'framer-motion'
import {
  Activity,
  RefreshCw,
  ChefHat,
  Rocket,
  Loader2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useActivityFeed } from '@/hooks/use-activity-api'
import type { ActivityFeedFilters, ActivityFeedItem } from '@/hooks/use-activity-api'
import { ActivityFeedItemCard, ActivityFeedItemSkeleton } from './activity-feed-item'
import { ExecutionDetail } from './execution-detail'
import { cn } from '@/lib/utils'

// ─── Filter Chip Config ─────────────────────────────────

const TYPE_CHIPS = [
  { value: 'routine', label: 'Routines', icon: RefreshCw },
  { value: 'recipe', label: 'Recipes', icon: ChefHat },
  { value: 'mission', label: 'Missions', icon: Rocket },
] as const

const STATUS_OPTIONS = [
  { value: 'all', label: 'All Statuses' },
  { value: 'running', label: 'Working' },
  { value: 'completed', label: 'Done' },
  { value: 'failed', label: 'Needs Attention' },
  { value: 'pending', label: 'Upcoming' },
] as const

const PAGE_SIZE = 20

// ─── Component ──────────────────────────────────────────

interface ActivityFeedProps {
  period?: string
}

export function ActivityFeed({ period = '1d' }: ActivityFeedProps) {
  const [activeTypes, setActiveTypes] = useState<string[]>([])
  const [statusFilter, setStatusFilter] = useState('all')
  const [limit, setLimit] = useState(PAGE_SIZE)
  const [selectedItem, setSelectedItem] = useState<ActivityFeedItem | null>(null)

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

  const toggleType = useCallback((type: string) => {
    setActiveTypes((prev) => {
      if (type === '__all__') return []
      return prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    })
    setLimit(PAGE_SIZE) // reset pagination on filter change
  }, [])

  const handleStatusChange = useCallback((value: string) => {
    setStatusFilter(value)
    setLimit(PAGE_SIZE) // reset pagination on filter change
  }, [])

  const handleLoadMore = useCallback(() => {
    setLimit((prev) => prev + PAGE_SIZE)
  }, [])

  const handleViewItem = useCallback((item: ActivityFeedItem) => {
    setSelectedItem(item)
  }, [])

  const handleCloseDetail = useCallback(() => {
    setSelectedItem(null)
  }, [])

  // ─── Execution Detail Drill-Down ──────────────────

  if (selectedItem) {
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
            <ActivityFeedItemSkeleton key={i} />
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
            <ActivityFeedItemCard
              key={item.id}
              item={item}
              animationDelay={index * 0.05}
              isNew={newItemIds.has(item.id)}
              onViewItem={handleViewItem}
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
        return (
          <Button
            key={chip.value}
            variant={isActive ? 'secondary' : 'ghost'}
            size="sm"
            onClick={() => onToggleType(chip.value)}
            className={cn(
              'text-xs min-h-[44px] sm:min-h-0 sm:h-7 shrink-0 gap-1.5',
              isActive && 'bg-secondary/80'
            )}
          >
            <Icon className="w-3.5 h-3.5" />
            <span className="hidden xs:inline">{chip.label}</span>
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
