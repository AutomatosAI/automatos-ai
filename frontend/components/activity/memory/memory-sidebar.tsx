'use client'

import { useMemo } from 'react'
import { Search, Brain, Calendar } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import type { MemoryItem } from '@/hooks/use-memory-explorer-api'

// ── Date grouping ───────────────────────────────────────

type DateGroup = 'Today' | 'Yesterday' | 'This Week' | 'This Month' | 'Older'

function getDateGroup(dateStr: string | null): DateGroup {
  if (!dateStr) return 'Older'
  const now = new Date()
  const d = new Date(dateStr)
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const diffMs = startOfToday.getTime() - new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
  const diffDays = Math.floor(diffMs / 86_400_000)

  if (diffDays <= 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays < 7) return 'This Week'
  if (diffDays < 30) return 'This Month'
  return 'Older'
}

const GROUP_ORDER: DateGroup[] = ['Today', 'Yesterday', 'This Week', 'This Month', 'Older']

function groupMemories(memories: MemoryItem[]): Map<DateGroup, MemoryItem[]> {
  const groups = new Map<DateGroup, MemoryItem[]>()
  for (const m of memories) {
    const group = getDateGroup(m.created_at)
    const list = groups.get(group) ?? []
    list.push(m)
    groups.set(group, list)
  }
  return groups
}

function formatTime(dateStr: string | null): string {
  if (!dateStr) return ''
  try {
    return new Date(dateStr).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text
  return text.slice(0, max).trimEnd() + '...'
}

// ── Component ───────────────────────────────────────────

interface MemorySidebarProps {
  memories: MemoryItem[]
  selectedId: string | null
  onSelect: (id: string) => void
  searchInput: string
  onSearchChange: (value: string) => void
  onSearchSubmit: () => void
  onClearSearch: () => void
  hasActiveQuery: boolean
  total: number
  isLoading: boolean
}

export function MemorySidebar({
  memories,
  selectedId,
  onSelect,
  searchInput,
  onSearchChange,
  onSearchSubmit,
  onClearSearch,
  hasActiveQuery,
  total,
  isLoading,
}: MemorySidebarProps) {
  const grouped = useMemo(() => groupMemories(memories), [memories])

  return (
    <div className="flex flex-col h-full">
      {/* Search */}
      <div className="p-3 border-b border-border/40 space-y-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search memories..."
            value={searchInput}
            onChange={(e) => onSearchChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') onSearchSubmit()
              if (e.key === 'Escape' && hasActiveQuery) onClearSearch()
            }}
            className="pl-9 h-8 text-sm bg-secondary/40"
          />
        </div>
        <div className="flex items-center justify-between text-[11px] text-muted-foreground px-0.5">
          <span>{total} memor{total !== 1 ? 'ies' : 'y'}</span>
          {hasActiveQuery && (
            <button onClick={onClearSearch} className="hover:text-foreground transition-colors">
              Clear search
            </button>
          )}
        </div>
      </div>

      {/* Grouped list */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="p-4 space-y-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="h-10 rounded bg-secondary/30 animate-pulse" />
            ))}
          </div>
        ) : memories.length === 0 ? (
          <div className="p-6 text-center text-muted-foreground">
            <Brain className="w-8 h-8 mx-auto mb-2 opacity-30" />
            <p className="text-sm">{hasActiveQuery ? 'No matches' : 'No memories yet'}</p>
          </div>
        ) : (
          GROUP_ORDER.map((group) => {
            const items = grouped.get(group)
            if (!items?.length) return null
            return (
              <div key={group}>
                <div className="sticky top-0 z-10 bg-background/80 backdrop-blur-sm px-3 py-1.5 flex items-center gap-1.5">
                  <Calendar className="w-3 h-3 text-muted-foreground" />
                  <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
                    {group}
                  </span>
                  <span className="text-[10px] text-muted-foreground/60 ml-auto">{items.length}</span>
                </div>
                {items.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => onSelect(m.id)}
                    className={cn(
                      'w-full text-left px-3 py-2 border-l-2 transition-colors',
                      'hover:bg-secondary/30',
                      selectedId === m.id
                        ? 'border-l-[hsl(var(--info))] bg-[hsl(var(--info))]/5'
                        : 'border-l-transparent'
                    )}
                  >
                    <p className="text-sm leading-snug line-clamp-2">{truncate(m.content, 120)}</p>
                    <div className="flex items-center gap-2 mt-1 text-[10px] text-muted-foreground">
                      <span>{formatTime(m.created_at)}</span>
                      {m.tier && <span className="capitalize">{m.tier}</span>}
                      {m.metadata?.agent_name && (
                        <span className="truncate max-w-[80px]">{m.metadata.agent_name}</span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
