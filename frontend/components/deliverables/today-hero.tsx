/**
 * TodayHero — top-of-page "what did the team make today?" panel.
 *
 * Replaces the older `created-today.tsx`. Wider card with a headline,
 * type-count pills, and a horizontal strip of thumbnails. Heartbeats
 * are excluded — they go to the System Strip.
 */

'use client'

import { useCallback, useMemo, useRef, useState } from 'react'
import { format } from 'date-fns'
import { ChevronLeft, ChevronRight, Loader2 } from 'lucide-react'

import { cn } from '@/lib/utils'
import {
  FEED_DEFAULT_FILTERS,
  useDeliverables,
  type Deliverable,
  type FilterState,
} from '@/hooks/use-deliverables-api'
import { DeliverablePreview } from '@/components/workspace/gallery-view/deliverable-preview'
import { DeliverableCard } from '@/components/workspace/gallery-view/deliverable-card'
import {
  DELIVERABLE_ACCENTS,
  isDeliverableType,
  type DeliverableType,
} from '@/components/icons/deliverable-icon'

const TODAY_FILTERS: FilterState = {
  ...FEED_DEFAULT_FILTERS,
  date_range: 'today',
}

const TYPE_PILL_ORDER: DeliverableType[] = [
  'image', 'slide', 'report', 'document', 'spreadsheet', 'code', 'blog_post',
]

const TYPE_PILL_LABEL_SINGULAR: Record<DeliverableType, string> = {
  report: 'report',
  image: 'image',
  document: 'doc',
  code: 'code file',
  slide: 'slide',
  spreadsheet: 'spreadsheet',
  blog_post: 'blog post',
}

const TYPE_PILL_LABEL_PLURAL: Record<DeliverableType, string> = {
  report: 'reports',
  image: 'images',
  document: 'docs',
  code: 'code files',
  slide: 'slides',
  spreadsheet: 'spreadsheets',
  blog_post: 'blog posts',
}

function pluralize(type: DeliverableType, count: number): string {
  return count === 1 ? TYPE_PILL_LABEL_SINGULAR[type] : TYPE_PILL_LABEL_PLURAL[type]
}

interface TypePillProps {
  type: DeliverableType
  count: number
}

function TypePill({ type, count }: TypePillProps) {
  const accent = DELIVERABLE_ACCENTS[type]
  return (
    <div className="inline-flex items-center gap-1.5 rounded-full border border-border bg-card/40 px-3 py-1 text-xs">
      <span
        className="inline-block h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: accent.hex }}
      />
      <span className="font-semibold tabular-nums text-foreground">{count}</span>
      <span className="text-muted-foreground">{pluralize(type, count)}</span>
    </div>
  )
}

export interface TodayHeroProps {
  className?: string
  onBrowseRecent?: () => void
}

export function TodayHero({ className, onBrowseRecent }: TodayHeroProps) {
  const { data, isLoading } = useDeliverables(TODAY_FILTERS)
  const [previewId, setPreviewId] = useState<string | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  const deliverables = useMemo<Deliverable[]>(() => {
    if (!data?.pages) return []
    return data.pages.flatMap((page) => page.deliverables)
  }, [data])

  const total = data?.pages?.[0]?.total ?? deliverables.length

  const typeCounts = useMemo<Record<string, number>>(() => {
    const counts: Record<string, number> = {}
    for (const d of deliverables) {
      counts[d.artifact_type] = (counts[d.artifact_type] ?? 0) + 1
    }
    return counts
  }, [deliverables])

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
    const amount = direction === 'left' ? -320 : 320
    el.scrollBy({ left: amount, behavior: 'smooth' })
  }, [])

  const todayLabel = format(new Date(), 'EEEE, MMM d')

  if (isLoading) {
    return (
      <div className={cn('rounded-xl border border-border bg-card/30 px-6 py-8', className)}>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          Loading today&apos;s deliverables…
        </div>
      </div>
    )
  }

  if (deliverables.length === 0) {
    return (
      <div
        className={cn(
          'rounded-xl border border-border bg-gradient-to-br from-card/50 to-card/20 px-6 py-7',
          className,
        )}
      >
        <h2 className="text-lg font-semibold tracking-tight text-foreground">
          Nothing shipped today — yet.
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {todayLabel}. When your agents create reports, slides, or images, they&apos;ll show up here.
        </p>
        {onBrowseRecent && (
          <button
            type="button"
            onClick={onBrowseRecent}
            className="mt-3 text-sm font-medium text-primary hover:underline"
          >
            Browse recent →
          </button>
        )}
      </div>
    )
  }

  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-xl border border-border bg-gradient-to-br from-card/60 via-card/30 to-card/10 px-6 py-6',
        className,
      )}
    >
      {/* Soft brand glow top-right */}
      <div
        aria-hidden
        className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-primary/10 blur-3xl"
      />

      {/* Headline + stats */}
      <div className="relative mb-5 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-foreground">
            Your team made{' '}
            <span className="text-primary">
              {total} {total === 1 ? 'thing' : 'things'}
            </span>{' '}
            today
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">{todayLabel}</p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {TYPE_PILL_ORDER.filter((t) => (typeCounts[t] ?? 0) > 0).map((type) => (
            <TypePill key={type} type={type} count={typeCounts[type] ?? 0} />
          ))}
          {Object.entries(typeCounts)
            .filter(([key]) => !isDeliverableType(key))
            .map(([key, count]) => (
              <div
                key={key}
                className="inline-flex items-center gap-1.5 rounded-full border border-border bg-card/40 px-3 py-1 text-xs"
              >
                <span className="font-semibold tabular-nums text-foreground">{count}</span>
                <span className="text-muted-foreground">{key}</span>
              </div>
            ))}
        </div>
      </div>

      {/* Strip of thumbnails */}
      <div className="relative">
        {deliverables.length > 4 && (
          <div className="pointer-events-none absolute right-0 top-0 z-10 hidden translate-y-[-2.25rem] gap-1 sm:flex">
            <button
              type="button"
              onClick={() => scroll('left')}
              className="pointer-events-auto rounded-md border border-border bg-card/80 p-1 text-muted-foreground backdrop-blur transition hover:border-primary/40 hover:text-foreground"
              aria-label="Scroll left"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              type="button"
              onClick={() => scroll('right')}
              className="pointer-events-auto rounded-md border border-border bg-card/80 p-1 text-muted-foreground backdrop-blur transition hover:border-primary/40 hover:text-foreground"
              aria-label="Scroll right"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        )}
        <div
          ref={scrollRef}
          className="flex snap-x snap-mandatory gap-3 overflow-x-auto pb-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border"
        >
          {deliverables.map((d) => (
            <div key={d.id} className="w-44 shrink-0 snap-start sm:w-52">
              <DeliverableCard deliverable={d} onClick={handleCardClick} />
            </div>
          ))}
        </div>
      </div>

      <DeliverablePreview
        deliverableId={previewId}
        open={previewOpen}
        onOpenChange={handlePreviewOpenChange}
      />
    </div>
  )
}
