/**
 * TypeRow — Netflix-style horizontal scroller for one artifact type.
 *
 * Each type (report, image, document, code, slide, spreadsheet, blog_post)
 * gets its own row with a typed accent dot, count, scroll chevrons, and a
 * "See all →" hand-off to the Explorer tab pre-filtered to that type.
 */

'use client'

import { useCallback, useMemo, useRef, useState } from 'react'
import { ArrowRight, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react'

import { cn } from '@/lib/utils'
import {
  FEED_DEFAULT_FILTERS,
  useDeliverables,
  type Deliverable,
  type FilterState,
} from '@/hooks/use-deliverables-api'
import { DeliverableCard } from '@/components/workspace/gallery-view/deliverable-card'
import { DeliverablePreview } from '@/components/workspace/gallery-view/deliverable-preview'
import {
  DELIVERABLE_ACCENTS,
  deliverableLabel,
  type DeliverableType,
} from '@/components/icons/deliverable-icon'

interface TypeRowProps {
  type: DeliverableType
  onSeeAll?: (type: DeliverableType) => void
  /** Aspect-ratio variant — slides photograph well portrait. */
  portrait?: boolean
  className?: string
}

export function TypeRow({ type, onSeeAll, portrait = false, className }: TypeRowProps) {
  const filters = useMemo<FilterState>(
    () => ({ ...FEED_DEFAULT_FILTERS, artifact_type: type }),
    [type],
  )
  const { data, isLoading } = useDeliverables(filters)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [previewId, setPreviewId] = useState<string | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)

  const deliverables = useMemo<Deliverable[]>(() => {
    if (!data?.pages) return []
    return data.pages.flatMap((page) => page.deliverables)
  }, [data])

  const total = data?.pages?.[0]?.total ?? 0
  const accent = DELIVERABLE_ACCENTS[type]

  const handleCardClick = useCallback((d: Deliverable) => {
    setPreviewId(d.id)
    setPreviewOpen(true)
  }, [])

  const handlePreviewOpenChange = useCallback((open: boolean) => {
    setPreviewOpen(open)
    if (!open) setTimeout(() => setPreviewId(null), 200)
  }, [])

  const scroll = useCallback((direction: 'left' | 'right') => {
    const el = scrollRef.current
    if (!el) return
    const amount = direction === 'left' ? -360 : 360
    el.scrollBy({ left: amount, behavior: 'smooth' })
  }, [])

  // Hide rows with zero items — no value showing 8 empty Netflix slots.
  if (!isLoading && deliverables.length === 0) return null

  return (
    <section className={cn('space-y-2', className)}>
      <div className="flex items-end justify-between gap-3">
        <div className="flex items-baseline gap-2">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ backgroundColor: accent.hex }}
            aria-hidden
          />
          <h3 className="text-sm font-semibold text-foreground">{accent.label}</h3>
          <span className="text-xs text-muted-foreground">
            {total === 0 ? '' : `${total} ${total === 1 ? 'item' : 'items'}`}
          </span>
        </div>
        <div className="flex items-center gap-1">
          {deliverables.length > 4 && (
            <>
              <button
                type="button"
                onClick={() => scroll('left')}
                className="rounded-md p-1 text-muted-foreground transition hover:bg-muted hover:text-foreground"
                aria-label={`Scroll ${accent.label} left`}
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={() => scroll('right')}
                className="rounded-md p-1 text-muted-foreground transition hover:bg-muted hover:text-foreground"
                aria-label={`Scroll ${accent.label} right`}
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </>
          )}
          {onSeeAll && (
            <button
              type="button"
              onClick={() => onSeeAll(type)}
              className="ml-2 inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-muted-foreground transition hover:bg-muted hover:text-foreground"
            >
              See all
              <ArrowRight className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center gap-2 px-1 py-4 text-xs text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Loading {deliverableLabel(type).toLowerCase()}…
        </div>
      ) : (
        <div
          ref={scrollRef}
          className="flex snap-x snap-mandatory gap-3 overflow-x-auto pb-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border"
        >
          {deliverables.map((d) => (
            <div
              key={d.id}
              className={cn(
                'shrink-0 snap-start',
                portrait ? 'w-40 sm:w-44' : 'w-44 sm:w-52',
              )}
            >
              <DeliverableCard deliverable={d} onClick={handleCardClick} />
            </div>
          ))}
        </div>
      )}

      <DeliverablePreview
        deliverableId={previewId}
        open={previewOpen}
        onOpenChange={handlePreviewOpenChange}
      />
    </section>
  )
}
