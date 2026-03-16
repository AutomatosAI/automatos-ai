'use client'

import { useState } from 'react'
import { Search, Filter, Bot } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { PremiumIcon } from '@/components/shared'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useAgents } from '@/hooks/use-agent-api'
import type { BoardFilters } from '@/hooks/use-board-tasks'
import { cn } from '@/lib/utils'

interface BoardFiltersBarProps {
  filters: BoardFilters
  onFiltersChange: (filters: BoardFilters) => void
  selectedAgentId: number | null
  onSelectAgent: (agentId: number | null) => void
  agentCount: number
  taskCount: number
  className?: string
}

export function BoardFiltersBar({
  filters,
  onFiltersChange,
  selectedAgentId,
  onSelectAgent,
  agentCount,
  taskCount,
  className,
}: BoardFiltersBarProps) {
  const [searchValue, setSearchValue] = useState(filters.search ?? '')
  const { data: rawAgents } = useAgents()
  const agents = (rawAgents as any[]) ?? []

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

        {/* Agent filter */}
        <Select
          value={selectedAgentId?.toString() ?? 'all'}
          onValueChange={(v) => onSelectAgent(v === 'all' ? null : Number(v))}
        >
          <SelectTrigger className="w-40 h-8 text-xs bg-secondary/30">
            <div className="flex items-center gap-1.5">
              <Bot className="w-3 h-3 text-muted-foreground shrink-0" />
              <SelectValue placeholder="Agent" />
            </div>
          </SelectTrigger>
          <SelectContent className="max-h-64">
            <SelectItem value="all">All Agents</SelectItem>
            {agents.map((agent: any) => (
              <SelectItem key={agent.id} value={agent.id.toString()}>
                <div className="flex items-center gap-1.5">
                  {agent.premium_icon ? (
                    <PremiumIcon name={agent.premium_icon} size={14} />
                  ) : (
                    <Bot className="w-3.5 h-3.5 text-primary" />
                  )}
                  <span className="truncate">{agent.name}</span>
                </div>
              </SelectItem>
            ))}
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
