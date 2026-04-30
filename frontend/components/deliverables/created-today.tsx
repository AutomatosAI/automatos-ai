/**
 * CreatedToday — "Created today" hero section for /deliverables
 * ================================================================
 *
 * Horizontal scrolling card row showing deliverables created today.
 * Uses the existing useDeliverables hook with date_range='today'.
 * Click a card to open the existing DeliverablePreview slide-over.
 */

'use client'

import { useCallback, useMemo, useRef, useState } from 'react'
import { formatDistanceToNow } from 'date-fns'
import {
  Bot,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Sparkles,
} from 'lucide-react'

import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import {
  useDeliverables,
  type Deliverable,
  type FilterState,
} from '@/hooks/use-deliverables-api'
import { DeliverablePreview } from '@/components/workspace/gallery-view/deliverable-preview'
import { DeliverableArtwork } from '@/components/deliverables/deliverable-artwork'

// ============= CONSTANTS =============

const TODAY_FILTERS: FilterState = {
  artifact_type: null,
  source_type: null,
  agent_id: null,
  date_range: 'today',
  search: '',
}

// ============= HELPERS =============

function formatTimeAgo(iso: string): string {
  try {
    return formatDistanceToNow(new Date(iso), { addSuffix: true })
  } catch {
    return ''
  }
}

function buildCaption(d: Deliverable): string {
  const agent = d.agent_name ?? 'An agent'
  const type = d.artifact_type ?? 'file'
  return `${agent} finished your ${type}`
}

// ============= SUB-COMPONENTS =============

interface TodayCardProps {
  deliverable: Deliverable
  onClick: (d: Deliverable) => void
}

function TodayCard({ deliverable, onClick }: TodayCardProps) {
  const {
    artifact_type,
    preview_url,
    title,
    file_name,
    agent_name,
    created_at,
  } = deliverable

  const isImage = artifact_type === 'image' && !!preview_url

  return (
    <button
      type="button"
      onClick={() => onClick(deliverable)}
      className={cn(
        'group flex w-56 shrink-0 flex-col overflow-hidden rounded-lg border border-border bg-card transition-all',
        'hover:border-primary/50 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-primary/40',
      )}
    >
      {/* Thumbnail */}
      <div className="relative flex h-28 items-center justify-center overflow-hidden bg-muted/30">
        {isImage ? (
          <img
            src={preview_url!}
            alt={title}
            loading="lazy"
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="h-16 w-16 overflow-hidden rounded-md">
            <DeliverableArtwork type={artifact_type} />
          </div>
        )}
        <Badge
          variant="secondary"
          className="absolute right-2 top-2 text-[10px] capitalize"
        >
          {artifact_type}
        </Badge>
      </div>

      {/* Details */}
      <div className="flex flex-col gap-1 p-3 text-left">
        <p className="line-clamp-1 text-sm font-medium text-foreground">
          {file_name ?? title}
        </p>
        <p className="line-clamp-1 text-xs text-muted-foreground">
          {buildCaption(deliverable)}
        </p>
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <Bot className="h-3 w-3 shrink-0" />
          <span className="truncate">{agent_name ?? 'Unknown'}</span>
          <span aria-hidden>·</span>
          <span className="shrink-0">{formatTimeAgo(created_at)}</span>
        </div>
      </div>
    </button>
  )
}

// ============= MAIN COMPONENT =============

export interface CreatedTodayProps {
  className?: string
  onBrowseRecent?: () => void
}

export function CreatedToday({ className, onBrowseRecent }: CreatedTodayProps) {
  const { data, isLoading } = useDeliverables(TODAY_FILTERS)
  const [previewId, setPreviewId] = useState<string | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  const deliverables = useMemo<Deliverable[]>(() => {
    if (!data?.pages) return []
    return data.pages.flatMap((page) => page.deliverables)
  }, [data])

  const handleCardClick = useCallback((d: Deliverable) => {
    setPreviewId(d.id)
    setPreviewOpen(true)
  }, [])

  const handlePreviewOpenChange = useCallback((open: boolean) => {
    setPreviewOpen(open)
    if (!open) {
      setTimeout(() => setPreviewId(null), 200)
    }
  }, [])

  const scroll = useCallback((direction: 'left' | 'right') => {
    const el = scrollRef.current
    if (!el) return
    const amount = direction === 'left' ? -280 : 280
    el.scrollBy({ left: amount, behavior: 'smooth' })
  }, [])

  // Don't render skeleton while loading — keep layout clean
  if (isLoading) {
    return (
      <div className={cn('flex items-center gap-2 px-1 py-4', className)}>
        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        <span className="text-sm text-muted-foreground">Loading today&apos;s deliverables...</span>
      </div>
    )
  }

  // Empty state
  if (deliverables.length === 0) {
    return (
      <div className={cn('rounded-lg border border-dashed border-border/50 bg-muted/10 px-6 py-5', className)}>
        <div className="flex items-center gap-3">
          <Sparkles className="h-5 w-5 text-muted-foreground" />
          <div>
            <p className="text-sm font-medium text-foreground">
              Your crew hasn&apos;t shipped anything yet today.
            </p>
            {onBrowseRecent && (
              <button
                type="button"
                onClick={onBrowseRecent}
                className="mt-0.5 text-sm text-primary hover:underline"
              >
                Browse recent
              </button>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={cn('space-y-2', className)}>
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          <h2 className="text-sm font-semibold text-foreground">Created today</h2>
          <span className="text-xs text-muted-foreground">
            {deliverables.length} {deliverables.length === 1 ? 'item' : 'items'}
          </span>
        </div>
        {deliverables.length > 3 && (
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => scroll('left')}
              className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              aria-label="Scroll left"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              type="button"
              onClick={() => scroll('right')}
              className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              aria-label="Scroll right"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>

      {/* Scrollable card row */}
      <div
        ref={scrollRef}
        className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border"
      >
        {deliverables.map((d) => (
          <TodayCard key={d.id} deliverable={d} onClick={handleCardClick} />
        ))}
      </div>

      <DeliverablePreview
        deliverableId={previewId}
        open={previewOpen}
        onOpenChange={handlePreviewOpenChange}
      />
    </div>
  )
}
