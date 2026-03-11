'use client'

import { useState } from 'react'
import { Search, Filter } from 'lucide-react'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { BoardFilters } from '@/hooks/use-board-tasks'
import { cn } from '@/lib/utils'

interface BoardFiltersBarProps {
  filters: BoardFilters
  onFiltersChange: (filters: BoardFilters) => void
  agentCount: number
  taskCount: number
  className?: string
}

export function BoardFiltersBar({
  filters,
  onFiltersChange,
  agentCount,
  taskCount,
  className,
}: BoardFiltersBarProps) {
  const [searchValue, setSearchValue] = useState(filters.search ?? '')

  const handleSearch = (value: string) => {
    setSearchValue(value)
    // Debounce the actual filter update
    clearTimeout((window as any).__boardSearchTimeout)
    ;(window as any).__boardSearchTimeout = setTimeout(() => {
      onFiltersChange({ ...filters, search: value || null })
    }, 300)
  }

  return (
    <div className={cn('flex items-center justify-between gap-3 flex-wrap', className)}>
      <div className="flex items-center gap-2 flex-wrap">
        {/* Priority filter */}
        <Select
          value={filters.priority ?? 'all'}
          onValueChange={(v) => onFiltersChange({ ...filters, priority: v === 'all' ? null : v })}
        >
          <SelectTrigger className="w-32 h-8 text-xs bg-secondary/30">
            <SelectValue placeholder="Priority" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Priorities</SelectItem>
            <SelectItem value="urgent">Urgent</SelectItem>
            <SelectItem value="high">High</SelectItem>
            <SelectItem value="medium">Medium</SelectItem>
            <SelectItem value="low">Low</SelectItem>
          </SelectContent>
        </Select>

        {/* Type filter */}
        <Select
          value={filters.type ?? 'all'}
          onValueChange={(v) => onFiltersChange({ ...filters, type: v === 'all' ? null : v })}
        >
          <SelectTrigger className="w-28 h-8 text-xs bg-secondary/30">
            <SelectValue placeholder="Type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="routine">Routine</SelectItem>
            <SelectItem value="recipe">Task</SelectItem>
          </SelectContent>
        </Select>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
          <Input
            value={searchValue}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search tasks..."
            className="h-8 text-xs pl-7 w-48 bg-secondary/30"
          />
        </div>
      </div>

      {/* Stats */}
      <div className="flex items-center gap-4 text-xs text-muted-foreground">
        <span>
          <strong className="text-foreground">{agentCount}</strong> agents active
        </span>
        <span>
          <strong className="text-foreground">{taskCount}</strong> tasks in queue
        </span>
      </div>
    </div>
  )
}
