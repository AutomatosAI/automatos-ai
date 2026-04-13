'use client'

import { useState } from 'react'
import { Target, Search, Plus, RefreshCw, ExternalLink } from 'lucide-react'
import { motion, useReducedMotion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Skeleton } from '@/components/ui/skeleton'
import { MissionCard } from './mission-card'
import { CreateMissionModal } from './create-mission-modal'
import { cn } from '@/lib/utils'
import { useMissions, type MissionFilters } from '@/hooks/use-missions-api'
import type { RunState } from '@/types/missions'

// ── Filter config ─────────────────────────────────────────────

type FilterKey = 'all' | 'active' | 'needs_review' | 'completed'

const FILTERS: { value: FilterKey; label: string; states?: RunState[] }[] = [
  { value: 'all', label: 'All' },
  { value: 'active', label: 'Active', states: ['running', 'planning', 'verifying'] },
  { value: 'needs_review', label: 'Needs Review', states: ['awaiting_approval', 'awaiting_human'] },
  { value: 'completed', label: 'Completed', states: ['completed', 'failed', 'cancelled'] },
]

// ── Component ─────────────────────────────────────────────────

export function MissionList() {
  const prefersReducedMotion = useReducedMotion()
  const [activeFilter, setActiveFilter] = useState<FilterKey>('all')
  const [search, setSearch] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)

  // Build query filters
  const filters: MissionFilters = {}
  const selectedFilter = FILTERS.find((f) => f.value === activeFilter)
  // Note: Backend only supports single state filter, so we filter client-side for multi-state groups
  if (selectedFilter?.states?.length === 1) {
    filters.state = selectedFilter.states[0]
  }

  const { data, isLoading, refetch } = useMissions(filters)
  const missions = data?.missions ?? []

  // Client-side filtering for multi-state groups and search
  const filtered = missions.filter((m) => {
    // State filter
    if (selectedFilter?.states && selectedFilter.states.length > 1) {
      if (!selectedFilter.states.includes(m.state)) return false
    }
    // Search
    if (search) {
      const q = search.toLowerCase()
      return m.goal.toLowerCase().includes(q)
    }
    return true
  })

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
        <div className="flex items-center gap-1 p-1 rounded-lg bg-secondary/30">
          {FILTERS.map((f) => (
            <button
              key={f.value}
              onClick={() => setActiveFilter(f.value)}
              className={cn(
                'px-3 py-1.5 text-xs font-medium rounded-md transition-colors cursor-pointer',
                activeFilter === f.value
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50',
              )}
            >
              {f.label}
            </button>
          ))}
        </div>

        <div className="relative flex-1 w-full sm:max-w-xs">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
          <Input
            placeholder="Search missions..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-8 h-8 text-sm bg-secondary/20"
          />
        </div>

        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            className="shrink-0"
            onClick={() => refetch()}
          >
            <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
            <span className="hidden sm:inline">Refresh</span>
          </Button>
          <Button
            size="sm"
            className="shrink-0"
            onClick={() => setShowCreateModal(true)}
          >
            <Plus className="w-3.5 h-3.5 mr-1.5" />
            New Mission
          </Button>
        </div>
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-48 rounded-xl" />
          ))}
        </div>
      )}

      {/* Mission grid */}
      {!isLoading && filtered.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {filtered.map((mission, i) => (
            <MissionCard key={mission.id} mission={mission} index={i} />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!isLoading && filtered.length === 0 && (
        <motion.div
          initial={prefersReducedMotion ? false : { opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="glass-card p-8 md:p-12 max-w-lg mx-auto text-center space-y-4"
        >
          <Target className="w-10 h-10 mx-auto text-muted-foreground/30" />

          <div>
            <h3 className="text-lg font-semibold">
              {search || activeFilter !== 'all' ? 'No missions match' : 'No missions yet'}
            </h3>
            <p className="text-sm text-muted-foreground mt-1 max-w-sm mx-auto">
              {search || activeFilter !== 'all'
                ? 'Try adjusting your filters or search.'
                : 'Missions are multi-agent goals that your AI workforce decomposes into tasks, executes, and verifies. Start one from the chat.'}
            </p>
          </div>

          {!search && activeFilter === 'all' && (
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                className="border-primary/30 hover:border-primary/50"
                onClick={() => setShowCreateModal(true)}
              >
                <Target className="w-4 h-4 mr-2" />
                New Mission
              </Button>
              <a
                href="https://automatos.gitbook.io/automatos-ai/activity/missions"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-sm text-primary hover:text-primary/80 transition-colors"
              >
                Read the Guide
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          )}
        </motion.div>
      )}

      <CreateMissionModal open={showCreateModal} onOpenChange={setShowCreateModal} />
    </div>
  )
}
