/**
 * DeliverableRow (PRD-129: Workspace Outputs Hub)
 * ================================================
 *
 * Compact row layout for the Gallery list view. Same data as DeliverableCard
 * but rendered horizontally: icon | title + agent | source | date | size.
 */

import { memo } from 'react'
import { formatDistanceToNow } from 'date-fns'
import {
  Archive,
  Bot,
  File,
  FileCode,
  FileText,
  Image as ImageIcon,
  LucideIcon,
  MessageSquare,
  Music,
  Presentation,
  Sheet,
  Upload,
  Video,
  Workflow,
  Wrench,
  Zap,
} from 'lucide-react'

import { cn } from '@/lib/utils'
import type { Deliverable } from '@/hooks/use-deliverables-api'

// ============= STYLE MAPS =============

const ARTIFACT_ICONS: Record<string, LucideIcon> = {
  report: FileText,
  document: FileText,
  image: ImageIcon,
  code: FileCode,
  slide: Presentation,
  spreadsheet: Sheet,
  archive: Archive,
  audio: Music,
  video: Video,
}

const ARTIFACT_COLORS: Record<string, string> = {
  report: 'text-blue-500',
  document: 'text-slate-500',
  image: 'text-purple-500',
  code: 'text-emerald-500',
  slide: 'text-orange-500',
  spreadsheet: 'text-green-500',
  archive: 'text-amber-500',
  audio: 'text-pink-500',
  video: 'text-red-500',
}

const SOURCE_LABELS: Record<string, string> = {
  chat: 'Chat',
  task: 'Task',
  mission: 'Mission',
  heartbeat: 'Heartbeat',
  playbook: 'Playbook',
  trigger: 'Trigger',
  upload: 'Upload',
}

const SOURCE_ICONS: Record<string, LucideIcon> = {
  chat: MessageSquare,
  task: Wrench,
  mission: Workflow,
  heartbeat: Zap,
  playbook: FileText,
  trigger: Zap,
  upload: Upload,
}

// ============= HELPERS =============

function formatFileSize(bytes: number | null): string {
  if (bytes === null || bytes === undefined) return ''
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  const size = bytes / Math.pow(1024, i)
  const rounded = size >= 10 || i === 0 ? Math.round(size) : Math.round(size * 10) / 10
  return `${rounded} ${units[i]}`
}

function formatTimeAgo(iso: string): string {
  try {
    return formatDistanceToNow(new Date(iso), { addSuffix: true })
  } catch {
    return ''
  }
}

// ============= COMPONENT =============

export interface DeliverableRowProps {
  deliverable: Deliverable
  onClick?: (deliverable: Deliverable) => void
  className?: string
}

function DeliverableRowImpl({ deliverable, onClick, className }: DeliverableRowProps) {
  const {
    artifact_type,
    source_type,
    title,
    agent_name,
    created_at,
    file_size_bytes,
  } = deliverable

  const ArtifactIcon = ARTIFACT_ICONS[artifact_type] ?? File
  const iconColor = ARTIFACT_COLORS[artifact_type] ?? 'text-muted-foreground'
  const SourceIcon = SOURCE_ICONS[source_type] ?? Zap
  const sourceLabel = SOURCE_LABELS[source_type] ?? source_type

  const handleClick = () => onClick?.(deliverable)
  const handleKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onClick?.(deliverable)
    }
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={handleClick}
      onKeyDown={handleKey}
      className={cn(
        'group flex cursor-pointer items-center gap-4 rounded-lg border border-border bg-card px-4 py-3 transition-all',
        'hover:border-primary/50 hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-primary/40',
        className,
      )}
    >
      {/* Icon */}
      <div className={cn('flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-muted/50', iconColor)}>
        <ArtifactIcon className="h-5 w-5" strokeWidth={1.5} />
      </div>

      {/* Title + agent */}
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="truncate text-sm font-medium text-foreground">{title}</span>
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <Bot className="h-3 w-3 shrink-0" />
          <span className="truncate">{agent_name ?? 'Unknown agent'}</span>
        </div>
      </div>

      {/* Source */}
      <div className="hidden items-center gap-1 text-xs text-muted-foreground sm:flex">
        <SourceIcon className="h-3 w-3" />
        <span>{sourceLabel}</span>
      </div>

      {/* Date */}
      <span className="hidden shrink-0 text-xs text-muted-foreground md:block">
        {formatTimeAgo(created_at)}
      </span>

      {/* Size */}
      <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
        {formatFileSize(file_size_bytes)}
      </span>
    </div>
  )
}

export const DeliverableRow = memo(DeliverableRowImpl)
