'use client'

import { useState } from 'react'
import { Trash2, Copy, Check, ChevronDown, ChevronUp } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { cn } from '@/lib/utils'
import type { MemoryItem } from '@/hooks/use-memory-explorer-api'

function formatDate(dateStr: string | null): string {
  if (!dateStr) return 'Unknown'
  try {
    const d = new Date(dateStr)
    const now = new Date()
    const diffMs = now.getTime() - d.getTime()
    const diffH = Math.floor(diffMs / 3600000)
    if (diffH < 1) return `${Math.max(1, Math.floor(diffMs / 60000))}m ago`
    if (diffH < 24) return `${diffH}h ago`
    const diffD = Math.floor(diffH / 24)
    if (diffD < 7) return `${diffD}d ago`
    return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })
  } catch {
    return dateStr
  }
}

interface MemoryCardProps {
  memory: MemoryItem
  selected?: boolean
  onSelect?: (id: string, checked: boolean) => void
  onDelete?: (id: string) => void
}

export function MemoryCard({ memory, selected, onSelect, onDelete }: MemoryCardProps) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const content = memory.content || ''
  const isLong = content.length > 200

  const handleCopy = () => {
    navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div
      className={cn(
        'glass-card border-l-[3px] p-4 space-y-2 transition-colors',
        selected ? 'border-l-[hsl(var(--info))] bg-[hsl(var(--info))]/5' : 'border-l-secondary',
        memory.score && memory.score > 0.8
          ? 'border-l-[hsl(var(--success))]'
          : memory.score && memory.score > 0.5
          ? 'border-l-[hsl(var(--warning))]'
          : ''
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          {onSelect && (
            <Checkbox
              checked={selected}
              onCheckedChange={(checked) => onSelect(memory.id, !!checked)}
              className="mt-0.5"
            />
          )}
          <div className="min-w-0">
            <p className={cn('text-sm leading-relaxed', !expanded && isLong && 'line-clamp-3')}>
              {content}
            </p>
            {isLong && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="text-xs text-muted-foreground hover:text-foreground mt-1 flex items-center gap-0.5"
              >
                {expanded ? (
                  <>Show less <ChevronUp className="w-3 h-3" /></>
                ) : (
                  <>Show more <ChevronDown className="w-3 h-3" /></>
                )}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Metadata row */}
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span>{formatDate(memory.created_at)}</span>

        {memory.score != null && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
            Score: {memory.score.toFixed(2)}
          </Badge>
        )}

        {memory.tier && (
          <Badge
            variant={memory.tier === 'global' ? 'default' : 'secondary'}
            className="text-[10px] px-1.5 py-0 capitalize"
          >
            {memory.tier}
          </Badge>
        )}

        {memory.metadata?.agent_name && (
          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
            {memory.metadata.agent_name}
          </Badge>
        )}

        {memory.metadata?.category && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0 capitalize">
            {memory.metadata.category}
          </Badge>
        )}

        <div className="ml-auto flex items-center gap-1">
          <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={handleCopy}>
            {copied ? (
              <Check className="w-3 h-3 text-[hsl(var(--success))]" />
            ) : (
              <Copy className="w-3 h-3" />
            )}
          </Button>
          {onDelete && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0 text-destructive hover:text-destructive"
              onClick={() => onDelete(memory.id)}
            >
              <Trash2 className="w-3 h-3" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
