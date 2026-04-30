'use client'

import { useState, useMemo, useCallback } from 'react'
import {
  History,
  ChefHat,
  Rocket,
  ListTodo,
  CalendarClock,
  Loader2,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import { format, formatDistanceToNow } from 'date-fns'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet'
import { StatusBadge } from '@/components/shared/status-badge'
import type { StatusVariant } from '@/components/shared/status-badge'
import { ExecutionDetail } from './execution-detail'
import { useActivityFeed } from '@/hooks/use-activity-api'
import type { ActivityFeedItem, ActivityFeedFilters } from '@/hooks/use-activity-api'
import { cn } from '@/lib/utils'

// ─── Constants ──────────────────────────────────────────

const PAGE_SIZE = 50

const TYPE_CHIPS = [
  { value: '__all__', label: 'All', icon: History },
  { value: 'recipe', label: 'Playbooks', icon: ChefHat },
  { value: 'mission', label: 'Missions', icon: Rocket },
  { value: 'routine', label: 'Tasks', icon: ListTodo },
  { value: 'scheduled', label: 'Schedules', icon: CalendarClock },
] as const

const STATUS_OPTIONS = [
  { value: 'all', label: 'All Statuses' },
  { value: 'completed', label: 'Success' },
  { value: 'failed', label: 'Failed' },
  { value: 'cancelled', label: 'Cancelled' },
  { value: 'paused', label: 'Partial' },
] as const

const STATUS_VARIANT_MAP: Record<string, { label: string; variant: StatusVariant }> = {
  completed: { label: 'Success', variant: 'success' },
  failed: { label: 'Failed', variant: 'error' },
  cancelled: { label: 'Cancelled', variant: 'neutral' },
  paused: { label: 'Partial', variant: 'warning' },
  running: { label: 'Running', variant: 'active' },
  pending: { label: 'Pending', variant: 'neutral' },
}

const SOURCE_LABEL: Record<string, string> = {
  recipe: 'Playbook',
  mission: 'Mission',
  routine: 'Task',
  chat: 'Chat',
}

// ─── Helpers ────────────────────────────────────────────

function formatWhen(iso: string | null): string {
  if (!iso) return '\u2014'
  try {
    return format(new Date(iso), 'MMM d, HH:mm')
  } catch {
    return '\u2014'
  }
}

function formatWhenRelative(iso: string | null): string {
  if (!iso) return ''
  try {
    return formatDistanceToNow(new Date(iso), { addSuffix: true })
  } catch {
    return ''
  }
}

function formatDuration(item: ActivityFeedItem): string {
  const seconds = item.total_duration_ms
    ? item.total_duration_ms / 1000
    : item.duration_seconds
  if (seconds == null) return '\u2014'
  if (seconds < 60) return `${Math.round(seconds)}s`
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  if (mins < 60) return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`
  const hours = Math.floor(mins / 60)
  const remainingMins = mins % 60
  return remainingMins > 0 ? `${hours}h ${remainingMins}m` : `${hours}h`
}

function formatCost(item: ActivityFeedItem): string {
  const tokens = item.total_tokens ?? item.tokens_used
  if (tokens == null || tokens === 0) return '\u2014'
  return `${tokens.toLocaleString()} tok`
}

function agentNames(item: ActivityFeedItem): string {
  if (item.agents && item.agents.length > 0) {
    return item.agents.map((a) => a.name).join(', ')
  }
  if (item.agent) return item.agent.name
  return '\u2014'
}

// ─── Component ──────────────────────────────────────────

interface CommandCenterHistoryProps {
  period?: string
}

export function CommandCenterHistory({ period = '30d' }: CommandCenterHistoryProps) {
  const [activeType, setActiveType] = useState('__all__')
  const [statusFilter, setStatusFilter] = useState('all')
  const [page, setPage] = useState(0)
  const [selectedItem, setSelectedItem] = useState<ActivityFeedItem | null>(null)

  const filters = useMemo<ActivityFeedFilters>(() => {
    const types =
      activeType === '__all__' || activeType === 'scheduled'
        ? undefined
        : [activeType]
    return {
      types,
      status: statusFilter !== 'all' ? statusFilter : null,
      period,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    }
  }, [activeType, statusFilter, period, page])

  const { data, isLoading, isFetching } = useActivityFeed(filters)

  const items = data?.items ?? []
  const total = data?.total ?? 0
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  const handleTypeChange = useCallback((type: string) => {
    setActiveType(type)
    setPage(0)
  }, [])

  const handleStatusChange = useCallback((value: string) => {
    setStatusFilter(value)
    setPage(0)
  }, [])

  const handleRowClick = useCallback((item: ActivityFeedItem) => {
    setSelectedItem(item)
  }, [])

  const handleCloseDetail = useCallback(() => {
    setSelectedItem(null)
  }, [])

  // ─── Empty state ────────────────────────────────

  const hasFilters = activeType !== '__all__' || statusFilter !== 'all'

  return (
    <div className="space-y-4">
      {/* Filter Bar */}
      <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-thin -mx-1 px-1">
        {TYPE_CHIPS.map((chip) => {
          const isActive = activeType === chip.value
          const Icon = chip.icon
          return (
            <Button
              key={chip.value}
              variant={isActive ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => handleTypeChange(chip.value)}
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

        <div className="w-px h-5 bg-border/50 shrink-0 mx-1" />

        <Select value={statusFilter} onValueChange={handleStatusChange}>
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

        {isFetching && (
          <Loader2 className="w-3.5 h-3.5 text-muted-foreground animate-spin shrink-0" />
        )}
      </div>

      {/* Table */}
      {isLoading ? (
        <HistorySkeleton />
      ) : items.length === 0 ? (
        <div className="glass-card p-12 text-center text-muted-foreground">
          <History className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p className="font-medium">
            {hasFilters ? 'No runs match your filters' : 'No run history yet'}
          </p>
          <p className="text-sm mt-1 max-w-sm mx-auto">
            {hasFilters
              ? 'Try adjusting the type or status filters above'
              : 'Run a Playbook, Mission, or Task to see your history here'}
          </p>
        </div>
      ) : (
        <>
          <div className="glass-card rounded-xl overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="w-[140px]">When</TableHead>
                  <TableHead className="w-[100px]">Source</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead className="hidden md:table-cell">Agent(s)</TableHead>
                  <TableHead className="w-[100px]">Status</TableHead>
                  <TableHead className="w-[90px] hidden sm:table-cell">Duration</TableHead>
                  <TableHead className="w-[100px] hidden lg:table-cell">Cost</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.map((item) => {
                  const sv = STATUS_VARIANT_MAP[item.status] ?? {
                    label: item.status,
                    variant: 'neutral' as StatusVariant,
                  }
                  return (
                    <TableRow
                      key={item.id}
                      className="cursor-pointer"
                      onClick={() => handleRowClick(item)}
                    >
                      <TableCell className="text-xs text-muted-foreground">
                        <div>{formatWhen(item.started_at ?? item.completed_at)}</div>
                        <div className="text-[10px] opacity-70">
                          {formatWhenRelative(item.started_at ?? item.completed_at)}
                        </div>
                      </TableCell>
                      <TableCell className="text-xs">
                        {SOURCE_LABEL[item.type] ?? item.type}
                      </TableCell>
                      <TableCell className="font-medium text-sm truncate max-w-[200px]">
                        {item.name}
                      </TableCell>
                      <TableCell className="hidden md:table-cell text-xs text-muted-foreground truncate max-w-[160px]">
                        {agentNames(item)}
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={sv.variant} size="sm">
                          {sv.label}
                        </StatusBadge>
                      </TableCell>
                      <TableCell className="hidden sm:table-cell text-xs text-muted-foreground">
                        {formatDuration(item)}
                      </TableCell>
                      <TableCell className="hidden lg:table-cell text-xs text-muted-foreground">
                        {formatCost(item)}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              {total} {total === 1 ? 'run' : 'runs'} total
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                className="h-7 w-7 p-0"
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <span>
                Page {page + 1} of {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                disabled={page + 1 >= totalPages}
                onClick={() => setPage((p) => p + 1)}
                className="h-7 w-7 p-0"
              >
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </>
      )}

      {/* Detail slide-over */}
      <Sheet open={selectedItem !== null} onOpenChange={(open) => !open && handleCloseDetail()}>
        <SheetContent side="right" className="w-full sm:max-w-xl overflow-y-auto">
          <SheetHeader className="sr-only">
            <SheetTitle>Run Details</SheetTitle>
            <SheetDescription>Details for the selected run</SheetDescription>
          </SheetHeader>
          {selectedItem && (
            <ExecutionDetail item={selectedItem} onClose={handleCloseDetail} />
          )}
        </SheetContent>
      </Sheet>
    </div>
  )
}

// ─── Skeleton ───────────────────────────────────────────

function HistorySkeleton() {
  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className="w-[140px]">When</TableHead>
            <TableHead className="w-[100px]">Source</TableHead>
            <TableHead>Name</TableHead>
            <TableHead className="hidden md:table-cell">Agent(s)</TableHead>
            <TableHead className="w-[100px]">Status</TableHead>
            <TableHead className="w-[90px] hidden sm:table-cell">Duration</TableHead>
            <TableHead className="w-[100px] hidden lg:table-cell">Cost</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {Array.from({ length: 8 }).map((_, i) => (
            <TableRow key={i}>
              <TableCell><div className="h-4 w-20 bg-secondary/30 rounded animate-pulse" /></TableCell>
              <TableCell><div className="h-4 w-16 bg-secondary/30 rounded animate-pulse" /></TableCell>
              <TableCell><div className="h-4 w-32 bg-secondary/30 rounded animate-pulse" /></TableCell>
              <TableCell className="hidden md:table-cell"><div className="h-4 w-24 bg-secondary/30 rounded animate-pulse" /></TableCell>
              <TableCell><div className="h-5 w-16 bg-secondary/30 rounded-full animate-pulse" /></TableCell>
              <TableCell className="hidden sm:table-cell"><div className="h-4 w-12 bg-secondary/30 rounded animate-pulse" /></TableCell>
              <TableCell className="hidden lg:table-cell"><div className="h-4 w-16 bg-secondary/30 rounded animate-pulse" /></TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
