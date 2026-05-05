'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Brain,
  Copy,
  Check,
  Trash2,
  Merge,
  Clock,
  Tag,
  User,
  BarChart3,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { cn } from '@/lib/utils'
import type { MemoryItem } from '@/hooks/use-memory-explorer-api'

// ── Helpers ─────────────────────────────────────────────

function formatFullDate(dateStr: string | null): string {
  if (!dateStr) return 'Unknown'
  try {
    return new Date(dateStr).toLocaleString('en-GB', {
      day: 'numeric', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  } catch {
    return dateStr
  }
}

function scoreColor(score: number | null): string {
  if (score == null) return 'text-muted-foreground'
  if (score > 0.8) return 'text-[hsl(var(--success))]'
  if (score > 0.5) return 'text-[hsl(var(--warning))]'
  return 'text-muted-foreground'
}

function MetaCell({ icon: Icon, iconClass, label, children }: {
  icon: React.ElementType; iconClass?: string; label: string; children: React.ReactNode
}) {
  return (
    <div className="glass-card p-3 flex items-center gap-2">
      <Icon className={cn('w-4 h-4 text-muted-foreground', iconClass)} />
      <div>
        {children}
        <div className="text-[10px] text-muted-foreground">{label}</div>
      </div>
    </div>
  )
}

// ── Component ───────────────────────────────────────────

interface MemoryViewerProps {
  memory: MemoryItem | null
  isSelected: boolean
  onToggleSelect: (id: string, checked: boolean) => void
  onDelete: (id: string) => void
  selectedCount: number
  onMerge: () => void
  onBulkDelete: () => void
}

export function MemoryViewer({
  memory,
  isSelected,
  onToggleSelect,
  onDelete,
  selectedCount,
  onMerge,
  onBulkDelete,
}: MemoryViewerProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    if (!memory) return
    navigator.clipboard.writeText(memory.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  if (!memory) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground">
        <div className="text-center">
          <Brain className="w-12 h-12 mx-auto mb-3 opacity-20" />
          <p className="text-sm">Select a memory to view details</p>
        </div>
      </div>
    )
  }

  const meta = memory.metadata ?? {}

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={memory.id}
        initial={{ opacity: 0, x: 12 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -12 }}
        transition={{ duration: 0.15 }}
        className="flex-1 flex flex-col h-full overflow-hidden"
      >
        {/* Toolbar */}
        <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border/40 shrink-0">
          <Checkbox
            checked={isSelected}
            onCheckedChange={(checked) => onToggleSelect(memory.id, !!checked)}
          />
          <span className="text-xs text-muted-foreground">
            {selectedCount > 0 ? `${selectedCount} selected` : 'Select'}
          </span>

          <div className="ml-auto flex items-center gap-1">
            {selectedCount >= 2 && (
              <Button variant="outline" size="sm" className="h-7 text-xs" onClick={onMerge}>
                <Merge className="w-3 h-3 mr-1" />
                Merge
              </Button>
            )}
            {selectedCount > 0 && (
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-xs text-destructive hover:text-destructive"
                onClick={onBulkDelete}
              >
                <Trash2 className="w-3 h-3 mr-1" />
                Delete {selectedCount > 1 ? `(${selectedCount})` : ''}
              </Button>
            )}
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={handleCopy}>
              {copied ? (
                <Check className="w-3.5 h-3.5 text-[hsl(var(--success))]" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0 text-destructive hover:text-destructive"
              onClick={() => onDelete(memory.id)}
            >
              <Trash2 className="w-3.5 h-3.5" />
            </Button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <p className="whitespace-pre-wrap leading-relaxed">{memory.content}</p>
          </div>

          {/* Metadata grid */}
          <div className="grid grid-cols-2 gap-3 text-sm">
            {memory.score != null && (
              <MetaCell icon={BarChart3} iconClass={scoreColor(memory.score)} label="Relevance Score">
                <span className={cn('font-medium', scoreColor(memory.score))}>{memory.score.toFixed(2)}</span>
              </MetaCell>
            )}
            {memory.tier && (
              <MetaCell icon={Tag} label="Tier">
                <Badge variant={memory.tier === 'global' ? 'default' : 'secondary'} className="text-[10px] capitalize">{memory.tier}</Badge>
              </MetaCell>
            )}
            <MetaCell icon={Clock} label="Created">
              <span className="text-xs">{formatFullDate(memory.created_at)}</span>
            </MetaCell>
            {memory.updated_at && memory.updated_at !== memory.created_at && (
              <MetaCell icon={Clock} label="Updated">
                <span className="text-xs">{formatFullDate(memory.updated_at)}</span>
              </MetaCell>
            )}
          </div>

          {/* Metadata badges */}
          {Object.keys(meta).length > 0 && (
            <div className="space-y-2">
              <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Metadata</p>
              <div className="flex flex-wrap gap-1.5">
                {meta.agent_name && <Badge variant="secondary" className="text-xs gap-1"><User className="w-3 h-3" />{meta.agent_name}</Badge>}
                {meta.category && <Badge variant="outline" className="text-xs capitalize">{meta.category}</Badge>}
                {meta.source && <Badge variant="outline" className="text-xs">{meta.source}</Badge>}
                {meta.agent_id && <Badge variant="outline" className="text-[10px] font-mono">agent:{meta.agent_id}</Badge>}
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  )
}
