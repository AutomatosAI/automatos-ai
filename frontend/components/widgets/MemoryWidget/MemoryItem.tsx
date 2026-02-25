'use client'

/**
 * MemoryItem Component for PRD-38.2 Extended Widgets
 *
 * Single memory display with type badge, content, and actions
 */

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Trash2,
  Info,
  Lightbulb,
  MessageSquare,
  FileText,
  Clock,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { formatDistanceToNow } from 'date-fns'
import type { Memory, MemoryType } from '../types'

interface MemoryItemProps {
  memory: Memory
  onDelete?: (id: string) => void
  showRelevance?: boolean
  className?: string
}

/**
 * Get memory type configuration
 */
function getTypeConfig(type: MemoryType) {
  switch (type) {
    case 'fact':
      return {
        icon: Info,
        label: 'fact',
        className: 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300',
      }
    case 'preference':
      return {
        icon: Lightbulb,
        label: 'preference',
        className: 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
      }
    case 'context':
      return {
        icon: MessageSquare,
        label: 'context',
        className: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
      }
    case 'instruction':
      return {
        icon: FileText,
        label: 'instruction',
        className: 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300',
      }
    default:
      return {
        icon: Info,
        label: type,
        className: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300',
      }
  }
}

/**
 * Format timestamp
 */
function formatTimestamp(timestamp: string): string {
  try {
    const date = new Date(timestamp)
    const now = new Date()
    const diffDays = Math.floor(
      (now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24)
    )

    if (diffDays === 0) {
      return 'Today'
    } else if (diffDays === 1) {
      return 'Yesterday'
    } else {
      return formatDistanceToNow(date, { addSuffix: true })
    }
  } catch {
    return timestamp
  }
}

export function MemoryItem({
  memory,
  onDelete,
  showRelevance = true,
  className,
}: MemoryItemProps) {
  const config = getTypeConfig(memory.type)
  const TypeIcon = config.icon

  return (
    <div
      className={cn(
        'group px-3 py-2.5 hover:bg-muted/30 transition-colors',
        'border-b border-border/30 last:border-0',
        className
      )}
    >
      <div className="flex items-start gap-2">
        {/* Type badge */}
        <Badge
          variant="secondary"
          className={cn('gap-1 flex-shrink-0 text-xs', config.className)}
        >
          <TypeIcon className="h-3 w-3" />
          {config.label}
        </Badge>

        {/* Relevance score */}
        {showRelevance && memory.relevance !== undefined && (
          <span className="text-xs text-muted-foreground flex-shrink-0">
            {Math.round(memory.relevance * 100)}% match
          </span>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Delete button */}
        {onDelete && (
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 flex-shrink-0 opacity-0 group-hover:opacity-100 hover:text-destructive"
            onClick={() => onDelete(memory.id)}
            title="Delete memory"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>

      {/* Content */}
      <p className="mt-1.5 text-sm">{memory.content}</p>

      {/* Timestamp */}
      <div className="flex items-center gap-1 mt-1.5 text-xs text-muted-foreground">
        <Clock className="h-3 w-3" />
        <span>Learned: {formatTimestamp(memory.source.timestamp)}</span>
        {memory.source.trigger && (
          <span className="ml-2 text-muted-foreground/60">
            via {memory.source.trigger}
          </span>
        )}
      </div>
    </div>
  )
}
