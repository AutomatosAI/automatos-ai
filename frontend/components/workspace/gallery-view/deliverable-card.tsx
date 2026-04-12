/**
 * DeliverableCard (PRD-129: Workspace Outputs Hub)
 * =================================================
 *
 * Visual card for a single agent deliverable in the Gallery grid.
 * Renders an image preview (for artifact_type='image') or a colored icon chip
 * (for every other type), with a source badge, title, agent/time, and size.
 */

import { memo } from 'react'
import { formatDistanceToNow } from 'date-fns'
import {
  Archive,
  Bot,
  Calendar,
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
  report: 'bg-blue-500/10 text-blue-500',
  document: 'bg-slate-500/10 text-slate-500',
  image: 'bg-purple-500/10 text-purple-500',
  code: 'bg-emerald-500/10 text-emerald-500',
  slide: 'bg-orange-500/10 text-orange-500',
  spreadsheet: 'bg-green-500/10 text-green-500',
  archive: 'bg-amber-500/10 text-amber-500',
  audio: 'bg-pink-500/10 text-pink-500',
  video: 'bg-red-500/10 text-red-500',
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

export interface DeliverableCardProps {
  deliverable: Deliverable
  onClick?: (deliverable: Deliverable) => void
  className?: string
}

function DeliverableCardImpl({ deliverable, onClick, className }: DeliverableCardProps) {
  const {
    artifact_type,
    source_type,
    preview_url,
    title,
    agent_name,
    created_at,
    file_size_bytes,
  } = deliverable

  const ArtifactIcon = ARTIFACT_ICONS[artifact_type] ?? File
  const artifactColor = ARTIFACT_COLORS[artifact_type] ?? 'bg-muted text-muted-foreground'
  const SourceIcon = SOURCE_ICONS[source_type] ?? Calendar

  const isImage = artifact_type === 'image' && !!preview_url
  const sizeLabel = formatFileSize(file_size_bytes)
  const timeAgo = formatTimeAgo(created_at)

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
        'group relative flex cursor-pointer flex-col overflow-hidden rounded-lg border border-border bg-card transition-all',
        'hover:border-primary/50 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-primary/40',
        className,
      )}
    >
      {/* Preview area */}
      <div className={cn('relative flex aspect-square items-center justify-center', !isImage && artifactColor)}>
        {isImage ? (
          <img
            src={preview_url!}
            alt={title}
            loading="lazy"
            className="h-full w-full object-cover"
          />
        ) : (
          <ArtifactIcon className="h-12 w-12" strokeWidth={1.5} />
        )}

        {/* Source badge */}
        <div
          className="absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full bg-background/90 text-foreground shadow-sm backdrop-blur-sm"
          title={source_type}
        >
          <SourceIcon className="h-3 w-3" />
        </div>
      </div>

      {/* Metadata */}
      <div className="flex flex-col gap-1 p-3">
        <h3 className="line-clamp-2 text-sm font-medium text-foreground">{title}</h3>
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <Bot className="h-3 w-3 shrink-0" />
          <span className="truncate">{agent_name ?? 'Unknown agent'}</span>
          <span aria-hidden>·</span>
          <span className="shrink-0">{timeAgo}</span>
        </div>
        {sizeLabel && (
          <div className="text-xs text-muted-foreground">{sizeLabel}</div>
        )}
      </div>
    </div>
  )
}

export const DeliverableCard = memo(DeliverableCardImpl)
