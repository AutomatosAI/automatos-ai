/**
 * GalleryView (PRD-129: Workspace Outputs Hub)
 * ==============================================
 *
 * Consumer-facing container for agent deliverables. Renders a FilterBar,
 * a responsive card grid (using DeliverableCard), and a slide-over preview
 * (DeliverablePreview). Pagination uses the useDeliverables infinite query
 * with a Load More button at the bottom.
 */

'use client'

import { useCallback, useMemo, useState } from 'react'
import { FolderOpen, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  DEFAULT_FILTERS,
  useDeliverables,
  type Deliverable,
  type FilterState,
} from '@/hooks/use-deliverables-api'

import { DeliverableCard } from './deliverable-card'
import { DeliverablePreview } from './deliverable-preview'
import { FilterBar } from './filter-bar'

export interface GalleryViewProps {
  workspaceId?: string | null
  className?: string
}

export function GalleryView({ className }: GalleryViewProps) {
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)

  const {
    data,
    isLoading,
    isError,
    error,
    hasNextPage,
    fetchNextPage,
    isFetchingNextPage,
  } = useDeliverables(filters)

  // Flatten paginated results into a single list (immutable).
  const deliverables = useMemo<Deliverable[]>(() => {
    if (!data?.pages) return []
    return data.pages.flatMap((page) => page.deliverables)
  }, [data])

  const total = data?.pages?.[0]?.total ?? 0

  const handleCardClick = useCallback((deliverable: Deliverable) => {
    setSelectedId(deliverable.id)
    setPreviewOpen(true)
  }, [])

  const handlePreviewOpenChange = useCallback((open: boolean) => {
    setPreviewOpen(open)
    if (!open) {
      // Keep id briefly so the sheet can animate out cleanly.
      setTimeout(() => setSelectedId(null), 200)
    }
  }, [])

  return (
    <div className={className}>
      <FilterBar
        filters={filters}
        onFiltersChange={setFilters}
        total={total}
      />

      {/* Content */}
      <div className="mt-4">
        {isLoading ? (
          <div className="flex min-h-[300px] items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        ) : isError ? (
          <div className="flex min-h-[300px] items-center justify-center text-sm text-destructive">
            Failed to load outputs
            {error instanceof Error && error.message ? `: ${error.message}` : ''}
          </div>
        ) : deliverables.length === 0 ? (
          <div className="flex min-h-[300px] flex-col items-center justify-center gap-3 text-center text-muted-foreground">
            <FolderOpen className="h-12 w-12" strokeWidth={1.5} />
            <div className="text-base font-medium text-foreground">
              No outputs yet
            </div>
            <div className="max-w-sm text-sm">
              Reports, images, and documents your agents produce will show up
              here.
            </div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
              {deliverables.map((deliverable) => (
                <DeliverableCard
                  key={deliverable.id}
                  deliverable={deliverable}
                  onClick={handleCardClick}
                />
              ))}
            </div>

            {hasNextPage && (
              <div className="mt-6 flex justify-center">
                <Button
                  variant="outline"
                  onClick={() => fetchNextPage()}
                  disabled={isFetchingNextPage}
                >
                  {isFetchingNextPage ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Loading…
                    </>
                  ) : (
                    'Load more'
                  )}
                </Button>
              </div>
            )}
          </>
        )}
      </div>

      <DeliverablePreview
        deliverableId={selectedId}
        open={previewOpen}
        onOpenChange={handlePreviewOpenChange}
      />
    </div>
  )
}

export default GalleryView
