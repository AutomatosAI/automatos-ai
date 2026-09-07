'use client'

/**
 * PRD-234 S2 — a ticket's Deliverables: every file registered against this
 * task (`source_type=task`, `source_id=<task id>`), which is how a Claude Code
 * session's files reach the same place as any other agent's output. Reuses
 * useDeliverables — no parallel fetch path (mirrors the mission panel).
 */

import { useMemo } from 'react'
import Link from 'next/link'
import {
  FileText,
  FileCode,
  FileSpreadsheet,
  Image as ImageIcon,
  Package,
  Presentation,
  ExternalLink,
  FolderOpen,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { DEFAULT_FILTERS, useDeliverables, type Deliverable } from '@/hooks/use-deliverables-api'

interface TaskDeliverablesPanelProps {
  taskId: string | number
  className?: string
}

const ARTIFACT_ICONS: Record<string, typeof FileText> = {
  report: FileText,
  document: FileText,
  code: FileCode,
  spreadsheet: FileSpreadsheet,
  image: ImageIcon,
  slide: Presentation,
  archive: Package,
}

function artifactIcon(type: string | null | undefined) {
  return ARTIFACT_ICONS[type ?? ''] ?? FileText
}

function DeliverableRow({ deliverable }: { deliverable: Deliverable }) {
  const Icon = artifactIcon(deliverable.artifact_type)
  const openHref = (deliverable as { preview_url?: string | null }).preview_url
    ?? (deliverable as { content_url?: string | null }).content_url
    ?? null
  return (
    <li
      data-testid="task-deliverable-row"
      className="flex items-start gap-3 px-3 py-2 rounded-md border border-border/60 bg-card/50"
    >
      <Icon className="w-4 h-4 mt-0.5 shrink-0 text-muted-foreground" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-foreground truncate">{deliverable.title}</span>
          <Badge variant="outline" className="text-[10px] shrink-0">{deliverable.artifact_type}</Badge>
        </div>
        {(deliverable as { file_path?: string | null }).file_path && (
          <div className="mt-0.5 text-[11px] font-mono text-muted-foreground truncate">
            {(deliverable as { file_path?: string | null }).file_path}
          </div>
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

export function TaskDeliverablesPanel({ taskId, className }: TaskDeliverablesPanelProps) {
  const filters = useMemo(() => ({ ...DEFAULT_FILTERS, source_id: String(taskId) }), [taskId])
  const { data, isLoading, isError } = useDeliverables(filters)
  const deliverables: Deliverable[] = useMemo(
    () => data?.pages.flatMap((page) => page.deliverables) ?? [],
    [data],
  )

  return (
    <div className={cn('space-y-2', className)} data-testid="task-deliverables-panel">
      <div className="flex items-center justify-between">
        <p className="text-xs uppercase tracking-wide text-muted-foreground flex items-center gap-1.5">
          <FolderOpen className="w-3 h-3" /> Deliverables
        </p>
        <Link href="/deliverables/explorer" className="text-xs text-muted-foreground hover:text-foreground">
          Open the explorer
        </Link>
      </div>
      {isLoading && (
        <div className="space-y-2" data-testid="task-deliverables-loading">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      )}
      {isError && <p className="text-xs text-destructive">Could not load this ticket&apos;s deliverables.</p>}
      {!isLoading && !isError && deliverables.length === 0 && (
        <p className="text-xs text-muted-foreground">
          Nothing registered yet. Files a session writes under the workspace folder are registered here when the ticket finishes.
        </p>
      )}
      {deliverables.length > 0 && (
        <ul className="space-y-1.5">
          {deliverables.map((d) => <DeliverableRow key={d.id} deliverable={d} />)}
        </ul>
      )}
    </div>
  )
}

export default TaskDeliverablesPanel
