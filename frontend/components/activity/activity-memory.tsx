'use client'

import { useState, useCallback, useMemo } from 'react'
import { ArrowLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
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
import {
  useMemoryBrowse,
  useDeleteMemory,
  useConsolidateMemories,
} from '@/hooks/use-memory-explorer-api'
import type { MemoryFilters } from '@/hooks/use-memory-explorer-api'
import { HealthBanner, MemorySidebar, MemoryViewer } from './memory/index'

interface ActivityMemoryProps {
  period?: string
}

export function ActivityMemory({ period = '30d' }: ActivityMemoryProps) {
  const [filters, setFilters] = useState<MemoryFilters>({ limit: 50 })
  const [searchInput, setSearchInput] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)
  const [showConsolidate, setShowConsolidate] = useState(false)
  const [mobileShowViewer, setMobileShowViewer] = useState(false)

  const { data, isLoading } = useMemoryBrowse(filters)
  const deleteMemory = useDeleteMemory()
  const consolidate = useConsolidateMemories()

  const memories = data?.memories ?? []
  const total = data?.total ?? 0
  const activeMemory = useMemo(
    () => memories.find((m) => m.id === selectedId) ?? null,
    [memories, selectedId]
  )

  const handleSearch = useCallback(() => {
    setFilters((prev) => ({ ...prev, query: searchInput.trim() || undefined }))
  }, [searchInput])

  const handleClearSearch = useCallback(() => {
    setSearchInput('')
    setFilters((prev) => ({ ...prev, query: undefined }))
  }, [])

  const handleSelectMemory = useCallback((id: string) => {
    setSelectedId(id)
    setMobileShowViewer(true)
  }, [])

  const handleToggleSelect = useCallback((id: string, checked: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (checked) next.add(id)
      else next.delete(id)
      return next
    })
  }, [])

  const handleDeleteConfirm = useCallback(() => {
    if (!deleteTarget) return
    deleteMemory.mutate({ memoryId: deleteTarget })
    setDeleteTarget(null)
    setSelectedIds((prev) => {
      const next = new Set(prev)
      next.delete(deleteTarget)
      return next
    })
    if (selectedId === deleteTarget) setSelectedId(null)
  }, [deleteTarget, deleteMemory, selectedId])

  const handleBulkDelete = useCallback(() => {
    selectedIds.forEach((id) => deleteMemory.mutate({ memoryId: id }))
    if (selectedId && selectedIds.has(selectedId)) setSelectedId(null)
    setSelectedIds(new Set())
  }, [selectedIds, deleteMemory, selectedId])

  const handleConsolidate = useCallback(
    (strategy: 'merge' | 'summarise') => {
      consolidate.mutate({ memory_ids: Array.from(selectedIds), strategy })
      setSelectedIds(new Set())
      setShowConsolidate(false)
    },
    [selectedIds, consolidate]
  )

  return (
    <div className="space-y-4">
      <HealthBanner />

      {/* Two-panel layout */}
      <div className="glass-card overflow-hidden" style={{ height: 'calc(100vh - 260px)', minHeight: 480 }}>
        <div className="flex h-full">
          {/* Left sidebar */}
          <div className={cn(
            'w-full lg:w-[340px] lg:min-w-[280px] border-r border-border/40 flex-shrink-0',
            mobileShowViewer ? 'hidden lg:flex lg:flex-col' : 'flex flex-col'
          )}>
            <MemorySidebar
              memories={memories}
              selectedId={selectedId}
              onSelect={handleSelectMemory}
              searchInput={searchInput}
              onSearchChange={setSearchInput}
              onSearchSubmit={handleSearch}
              onClearSearch={handleClearSearch}
              hasActiveQuery={!!filters.query}
              total={total}
              isLoading={isLoading}
            />
          </div>

          {/* Right viewer */}
          <div className={cn(
            'flex-1 flex flex-col min-w-0',
            mobileShowViewer ? 'flex' : 'hidden lg:flex'
          )}>
            {mobileShowViewer && (
              <div className="lg:hidden px-3 py-2 border-b border-border/40">
                <Button variant="ghost" size="sm" className="h-7 text-xs gap-1" onClick={() => setMobileShowViewer(false)}>
                  <ArrowLeft className="w-3.5 h-3.5" />
                  Back to list
                </Button>
              </div>
            )}
            <MemoryViewer
              memory={activeMemory}
              isSelected={selectedId ? selectedIds.has(selectedId) : false}
              onToggleSelect={handleToggleSelect}
              onDelete={setDeleteTarget}
              selectedCount={selectedIds.size}
              onMerge={() => setShowConsolidate(true)}
              onBulkDelete={handleBulkDelete}
            />
          </div>
        </div>
      </div>

      {memories.length > 0 && memories.length < total && (
        <div className="text-center">
          <Button variant="outline" size="sm" onClick={() => setFilters((prev) => ({ ...prev, limit: (prev.limit || 50) + 50 }))}>
            Load More ({total - memories.length} remaining)
          </Button>
        </div>
      )}

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
            <AlertDialogDescription>Choose how to combine these memories into a single entry.</AlertDialogDescription>
          </AlertDialogHeader>
          <div className="grid grid-cols-2 gap-3 py-4">
            <button onClick={() => handleConsolidate('merge')} className="glass-card p-4 text-left hover:bg-secondary/40 transition-colors">
              <p className="font-medium text-sm">Merge</p>
              <p className="text-xs text-muted-foreground mt-1">Combine content into one memory, keeping the latest metadata.</p>
            </button>
            <button onClick={() => handleConsolidate('summarise')} className="glass-card p-4 text-left hover:bg-secondary/40 transition-colors">
              <p className="font-medium text-sm">Summarise</p>
              <p className="text-xs text-muted-foreground mt-1">AI generates a concise summary replacing all selected memories.</p>
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
