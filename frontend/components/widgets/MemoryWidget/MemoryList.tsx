'use client'

/**
 * MemoryList Component for PRD-38.2 Extended Widgets
 *
 * List of memories with grouping and empty states
 */

import { ScrollArea } from '@/components/ui/scroll-area'
import { Brain } from 'lucide-react'
import { cn } from '@/lib/utils'
import { MemoryItem } from './MemoryItem'
import type { Memory } from '../types'

interface MemoryListProps {
  memories: Memory[]
  onDelete?: (id: string) => void
  emptyMessage?: string
  showRelevance?: boolean
  className?: string
}

export function MemoryList({
  memories,
  onDelete,
  emptyMessage = 'No memories found',
  showRelevance = true,
  className,
}: MemoryListProps) {
  if (memories.length === 0) {
    return (
      <div
        className={cn(
          'flex flex-col items-center justify-center h-32 text-muted-foreground',
          className
        )}
      >
        <Brain className="h-8 w-8 mb-2 opacity-50" />
        <p className="text-sm">{emptyMessage}</p>
      </div>
    )
  }

  return (
    <ScrollArea className={cn('flex-1', className)}>
      <div className="group">
        {memories.map((memory) => (
          <MemoryItem
            key={memory.id}
            memory={memory}
            onDelete={onDelete}
            showRelevance={showRelevance}
          />
        ))}
      </div>
    </ScrollArea>
  )
}
