'use client'

import { useState, useCallback } from 'react'
import {
  Brain,
  Search,
  Trash2,
  Merge,
  Activity,
  Shield,
  AlertTriangle,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Skeleton } from '@/components/ui/skeleton'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { cn } from '@/lib/utils'
import {
  useMemoryBrowse,
  useMemoryHealth,
  useMemoryExplorerStats,
  useDeleteMemory,
  useConsolidateMemories,
} from '@/hooks/use-memory-explorer-api'
import type { MemoryFilters } from '@/hooks/use-memory-explorer-api'
import { MemoryCard } from './memory-card'

// ─── Skeleton ────────────────────────────────────────────

function MemoryCardSkeleton() {
  return (
    <div className="glass-card border-l-[3px] border-l-secondary p-4 space-y-2">
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-3/4" />
      <div className="flex gap-2">
        <Skeleton className="h-5 w-16 rounded-full" />
        <Skeleton className="h-5 w-20 rounded-full" />
        <Skeleton className="h-3 w-12 ml-auto" />
      </div>
    </div>
  )
}

// ─── Empty State ─────────────────────────────────────────

function MemoryEmptyState({ hasQuery }: { hasQuery: boolean }) {
  return (
    <div className="glass-card p-6 sm:p-8 text-center text-muted-foreground">
      <Brain className="w-12 h-12 mx-auto mb-3 opacity-30" />
      <p className="font-medium">{hasQuery ? 'No matching memories' : 'No memories yet'}</p>
      <p className="text-sm mt-1 max-w-xs mx-auto">
        {hasQuery
          ? 'Try a different search term or clear your search.'
          : 'Memories are created as your agents learn from conversations and tasks.'}
      </p>
    </div>
  )
}

// ─── Health Banner ───────────────────────────────────────

function HealthBanner() {
  const { data: health } = useMemoryHealth()

  if (!health) return null

  const statusColors: Record<string, string> = {
    healthy: 'text-[hsl(var(--success))]',
    degraded: 'text-[hsl(var(--warning))]',
    unavailable: 'text-destructive',
  }

  const statusIcons: Record<string, typeof Activity> = {
    healthy: Shield,
    degraded: AlertTriangle,
    unavailable: AlertTriangle,
  }

  const StatusIcon = statusIcons[health.health_status] || Activity

  return (
    <div className="flex flex-wrap gap-3 mb-4">
      <div className="glass-card px-3 py-2 text-center min-w-[80px]">
        <div className="text-lg font-bold leading-none">{health.total_memories}</div>
        <div className="text-[10px] text-muted-foreground mt-1">Total</div>
      </div>

      <div className="glass-card px-3 py-2 text-center min-w-[80px]">
        <div className={cn('text-lg font-bold leading-none flex items-center justify-center gap-1', statusColors[health.health_status])}>
          <StatusIcon className="w-4 h-4" />
          <span className="capitalize">{health.health_status}</span>
        </div>
        <div className="text-[10px] text-muted-foreground mt-1">Status</div>
      </div>

      <div className="glass-card px-3 py-2 text-center min-w-[80px]">
        <div className="text-lg font-bold leading-none">
          {Math.round((health.search_effectiveness?.hit_rate ?? 0) * 100)}%
        </div>
        <div className="text-[10px] text-muted-foreground mt-1">Hit Rate</div>
      </div>

      <div className="glass-card px-3 py-2 text-center min-w-[80px]">
        <div className="text-lg font-bold leading-none">
          {health.search_effectiveness?.total_searches ?? 0}
        </div>
        <div className="text-[10px] text-muted-foreground mt-1">Searches</div>
      </div>

      {health.newest_memory && (
        <div className="glass-card px-3 py-2 text-center min-w-[80px]">
          <div className="text-sm font-bold leading-none">
            {new Date(health.newest_memory).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}
          </div>
          <div className="text-[10px] text-muted-foreground mt-1">Latest</div>
        </div>
      )}
    </div>
  )
}

// ─── Main Component ──────────────────────────────────────

interface ActivityMemoryProps {
  period?: string
}

export function ActivityMemory({ period = '30d' }: ActivityMemoryProps) {
  const [filters, setFilters] = useState<MemoryFilters>({ limit: 30 })
  const [searchInput, setSearchInput] = useState('')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)
  const [showConsolidate, setShowConsolidate] = useState(false)

  const { data, isLoading } = useMemoryBrowse(filters)
  const deleteMemory = useDeleteMemory()
  const consolidate = useConsolidateMemories()

  const memories = data?.memories || []
  const total = data?.total || 0

  const handleSearch = useCallback(() => {
    setFilters((prev) => ({
      ...prev,
      query: searchInput.trim() || undefined,
    }))
  }, [searchInput])

  const handleClearSearch = useCallback(() => {
    setSearchInput('')
    setFilters((prev) => ({ ...prev, query: undefined }))
  }, [])

  const handleSelect = useCallback((id: string, checked: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (checked) next.add(id)
      else next.delete(id)
      return next
    })
  }, [])

  const handleDeleteConfirm = useCallback(() => {
    if (deleteTarget) {
      deleteMemory.mutate({ memoryId: deleteTarget })
      setDeleteTarget(null)
      setSelectedIds((prev) => {
        const next = new Set(prev)
        next.delete(deleteTarget)
        return next
      })
    }
  }, [deleteTarget, deleteMemory])

  const handleBulkDelete = useCallback(() => {
    selectedIds.forEach((id) => deleteMemory.mutate({ memoryId: id }))
    setSelectedIds(new Set())
  }, [selectedIds, deleteMemory])

  const handleConsolidate = useCallback(
    (strategy: 'merge' | 'summarise') => {
      consolidate.mutate({
        memory_ids: Array.from(selectedIds),
        strategy,
      })
      setSelectedIds(new Set())
      setShowConsolidate(false)
    },
    [selectedIds, consolidate]
  )

  const handleLoadMore = useCallback(() => {
    setFilters((prev) => ({ ...prev, limit: (prev.limit || 30) + 30 }))
  }, [])

  return (
    <div className="space-y-4">
      {/* Health stats */}
      <HealthBanner />

      {/* Search + actions bar */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search memories..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            className="pl-9 h-8 text-sm bg-secondary/40"
          />
        </div>

        <Button variant="outline" size="sm" className="h-8 text-xs" onClick={handleSearch}>
          Search
        </Button>

        {filters.query && (
          <Button variant="ghost" size="sm" className="h-8 text-xs" onClick={handleClearSearch}>
            Clear
          </Button>
        )}

        {/* Bulk actions */}
        {selectedIds.size > 0 && (
          <>
            <div className="h-4 w-px bg-border mx-1" />
            <Badge variant="secondary" className="text-xs">
              {selectedIds.size} selected
            </Badge>
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs"
              onClick={() => setShowConsolidate(true)}
              disabled={selectedIds.size < 2}
            >
              <Merge className="w-3 h-3 mr-1" />
              Merge
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs text-destructive hover:text-destructive"
              onClick={handleBulkDelete}
            >
              <Trash2 className="w-3 h-3 mr-1" />
              Delete
            </Button>
          </>
        )}

        <span className="text-xs text-muted-foreground ml-auto">
          {total} memor{total !== 1 ? 'ies' : 'y'}
          {filters.query && ` matching "${filters.query}"`}
        </span>
      </div>

      {/* Memory cards */}
      {isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <MemoryCardSkeleton key={i} />
          ))}
        </div>
      ) : memories.length === 0 ? (
        <MemoryEmptyState hasQuery={!!filters.query} />
      ) : (
        <div className="space-y-3">
          {memories.map((memory) => (
            <MemoryCard
              key={memory.id}
              memory={memory}
              selected={selectedIds.has(memory.id)}
              onSelect={handleSelect}
              onDelete={setDeleteTarget}
            />
          ))}
        </div>
      )}

      {/* Load more */}
      {memories.length > 0 && memories.length < total && (
        <div className="text-center">
          <Button variant="outline" size="sm" onClick={handleLoadMore}>
            Load More ({total - memories.length} remaining)
          </Button>
        </div>
      )}

      {/* Delete confirmation */}
      <AlertDialog open={!!deleteTarget} onOpenChange={() => setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Memory</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently remove this memory. Your agents will no longer recall this information.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteConfirm} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Consolidate dialog */}
      <AlertDialog open={showConsolidate} onOpenChange={setShowConsolidate}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Consolidate {selectedIds.size} Memories</AlertDialogTitle>
            <AlertDialogDescription>
              Choose how to combine these memories into a single entry.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="grid grid-cols-2 gap-3 py-4">
            <button
              onClick={() => handleConsolidate('merge')}
              className="glass-card p-4 text-left hover:bg-secondary/40 transition-colors"
            >
              <p className="font-medium text-sm">Merge</p>
              <p className="text-xs text-muted-foreground mt-1">
                Combine content into one memory, keeping the latest metadata.
              </p>
            </button>
            <button
              onClick={() => handleConsolidate('summarise')}
              className="glass-card p-4 text-left hover:bg-secondary/40 transition-colors"
            >
              <p className="font-medium text-sm">Summarise</p>
              <p className="text-xs text-muted-foreground mt-1">
                AI generates a concise summary replacing all selected memories.
              </p>
            </button>
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
