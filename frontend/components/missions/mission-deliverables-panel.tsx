'use client'

/**
 * MissionDeliverablesPanel (PRD-164 S3)
 * =====================================
 * The mission page's Deliverables tab: every output this mission produced
 * (reports, generated documents, images, code, …) scoped via the
 * deliverables API `source_id=<missionId>` filter. Reuses useDeliverables —
 * no parallel fetch path.
 */

import { useMemo } from 'react'
import {
  FileText,
  FileCode,
  FileSpreadsheet,
  Image as ImageIcon,
  Package,
  Presentation,
  ExternalLink,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import {
  DEFAULT_FILTERS,
  useDeliverables,
  type Deliverable,
} from '@/hooks/use-deliverables-api'

interface MissionDeliverablesPanelProps {
  missionId: string
  className?: string
}

const ARTIFACT_ICONS: Record<string, typeof FileText> = {
  report: FileText,
  document: FileText,
  blog_post: FileText,
  code: FileCode,
  spreadsheet: FileSpreadsheet,
  image: ImageIcon,
  slide: Presentation,
  archive: Package,
}

function artifactIcon(type: string | null | undefined) {
  return ARTIFACT_ICONS[type ?? ''] ?? FileText
}

function formatWhen(iso: string | null | undefined): string {
  if (!iso) return ''
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function DeliverableRow({ deliverable }: { deliverable: Deliverable }) {
  const Icon = artifactIcon(deliverable.artifact_type)
  const openHref = deliverable.preview_url ?? deliverable.content_url ?? null

  return (
    <li
      data-testid="mission-deliverable-row"
      className="flex items-start gap-3 px-3 py-2.5 rounded-md border border-border/60 bg-card/50"
    >
      <Icon className="w-4 h-4 mt-0.5 shrink-0 text-muted-foreground" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-foreground truncate">
            {deliverable.title}
          </span>
          <Badge variant="outline" className="text-[10px] shrink-0">
            {deliverable.artifact_type}
          </Badge>
        </div>
        <div className="mt-0.5 text-[11px] text-muted-foreground truncate">
          {[deliverable.agent_name, formatWhen(deliverable.created_at)]
            .filter(Boolean)
            .join(' · ')}
        </div>
        {deliverable.summary && (
          <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
            {deliverable.summary}
          </p>
        )}
      </div>
      {openHref && (
        <a
          href={openHref}
          target="_blank"
          rel="noreferrer"
          aria-label={`Open ${deliverable.title}`}
          className="shrink-0 text-muted-foreground hover:text-foreground transition-colors"
        >
          <ExternalLink className="w-3.5 h-3.5" />
        </a>
      )}
    </li>
  )
}

export function MissionDeliverablesPanel({
  missionId,
  className,
}: MissionDeliverablesPanelProps) {
  const filters = useMemo(
    () => ({ ...DEFAULT_FILTERS, source_id: missionId }),
    [missionId],
  )
  const { data, isLoading, isError, hasNextPage, fetchNextPage, isFetchingNextPage } =
    useDeliverables(filters)

  const deliverables: Deliverable[] = useMemo(
    () => data?.pages.flatMap((page) => page.deliverables) ?? [],
    [data],
  )

  if (isLoading) {
    return (
      <div className={cn('p-3 space-y-2', className)} data-testid="mission-deliverables-loading">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    )
  }

  if (isError) {
    return (
      <div className={cn('p-4 text-sm text-muted-foreground', className)}>
        Could not load deliverables for this mission.
      </div>
    )
  }

  if (deliverables.length === 0) {
    return (
      <div
        className={cn('p-4 text-sm text-muted-foreground', className)}
        data-testid="mission-deliverables-empty"
      >
        No deliverables yet — outputs this mission produces will appear here.
      </div>
    )
  }

  return (
    <div className={cn('p-3', className)}>
      <ul className="space-y-2" data-testid="mission-deliverables-list">
        {deliverables.map((deliverable) => (
          <DeliverableRow key={deliverable.id} deliverable={deliverable} />
        ))}
      </ul>
      {hasNextPage && (
        <div className="mt-3 flex justify-center">
          <Button
            variant="outline"
            size="sm"
            onClick={() => fetchNextPage()}
            disabled={isFetchingNextPage}
          >
            {isFetchingNextPage ? 'Loading…' : 'Load more'}
          </Button>
        </div>
      )}
    </div>
  )
}
