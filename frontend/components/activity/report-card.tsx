'use client'

import { formatDistanceToNow } from 'date-fns'
import {
  ClipboardCheck,
  Search,
  AlertTriangle,
  FileText,
  Package,
  Shield,
  Star,
  Download,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { AgentReport } from '@/hooks/use-reports-api'

// ─── Report Type Config ─────────────────────────────────

const REPORT_TYPE_CONFIG: Record<string, { label: string; icon: typeof FileText }> = {
  standup: { label: 'Standup', icon: ClipboardCheck },
  research: { label: 'Research', icon: Search },
  incident: { label: 'Incident', icon: AlertTriangle },
  summary: { label: 'Summary', icon: FileText },
  delivery: { label: 'Delivery', icon: Package },
  audit: { label: 'Audit', icon: Shield },
}

const STATUS_BORDER: Record<string, string> = {
  ok: 'border-l-[hsl(var(--success))]',
  warning: 'border-l-[hsl(var(--warning))]',
  critical: 'border-l-[hsl(var(--destructive))]',
  info: 'border-l-[hsl(var(--info))]',
}

const STATUS_BADGE: Record<string, { label: string; className: string }> = {
  ok: { label: 'OK', className: 'bg-[hsl(var(--success))]/10 text-[hsl(var(--success))]' },
  warning: { label: 'Warning', className: 'bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))]' },
  critical: { label: 'Critical', className: 'bg-destructive/10 text-destructive' },
  info: { label: 'Info', className: 'bg-[hsl(var(--info))]/10 text-[hsl(var(--info))]' },
}

interface ReportCardProps {
  report: AgentReport
  onView: (report: AgentReport) => void
  onDownload?: (report: AgentReport) => void
}

export function ReportCard({ report, onView, onDownload }: ReportCardProps) {
  const typeConfig = REPORT_TYPE_CONFIG[report.report_type] || REPORT_TYPE_CONFIG.standup
  const TypeIcon = typeConfig.icon
  const statusBorder = STATUS_BORDER[report.status] || STATUS_BORDER.info
  const statusBadge = STATUS_BADGE[report.status] || STATUS_BADGE.info

  const metrics = report.metrics || {}
  const metricEntries = Object.entries(metrics).slice(0, 4)

  const timeAgo = report.created_at
    ? formatDistanceToNow(new Date(report.created_at), { addSuffix: true })
    : 'unknown'

  return (
    <div
      className={cn(
        'glass-card border-l-[3px] p-4 space-y-3 cursor-pointer hover:bg-secondary/20 transition-colors',
        statusBorder
      )}
      onClick={() => onView(report)}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-medium truncate">{report.agent_name}</span>
            <Badge variant="outline" className="text-[10px] font-normal shrink-0">
              <TypeIcon className="w-3 h-3 mr-1" />
              {typeConfig.label}
            </Badge>
          </div>
          <h3 className="font-semibold mt-0.5 truncate">{report.title}</h3>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Badge className={cn('text-[10px]', statusBadge.className)}>
            {statusBadge.label}
          </Badge>
          <span className="text-xs text-muted-foreground">{timeAgo}</span>
        </div>
      </div>

      {/* Summary */}
      {report.summary && (
        <p className="text-sm text-muted-foreground line-clamp-2">{report.summary}</p>
      )}

      {/* Metrics bar */}
      {metricEntries.length > 0 && (
        <div className="flex flex-wrap gap-3">
          {metricEntries.map(([key, value]) => (
            <div
              key={key}
              className="text-xs px-2 py-1 rounded-md bg-secondary/30"
            >
              <span className="text-muted-foreground">
                {key.replace(/_/g, ' ')}:
              </span>{' '}
              <span className="font-medium">
                {typeof value === 'number' ? value.toLocaleString() : String(value)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Footer: grade + actions */}
      <div className="flex items-center justify-between pt-1">
        <div className="flex items-center gap-1">
          {report.grade ? (
            <div className="flex items-center gap-0.5">
              {[1, 2, 3, 4, 5].map((s) => (
                <Star
                  key={s}
                  className={cn(
                    'w-3.5 h-3.5',
                    s <= report.grade! ? 'text-[hsl(var(--warning))] fill-[hsl(var(--warning))]' : 'text-muted-foreground/30'
                  )}
                />
              ))}
              <span className="text-xs text-muted-foreground ml-1">
                graded {report.graded_at ? formatDistanceToNow(new Date(report.graded_at), { addSuffix: true }) : ''}
              </span>
            </div>
          ) : (
            <span className="text-xs text-muted-foreground/60">Not graded</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {onDownload && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              onClick={(e) => {
                e.stopPropagation()
                onDownload(report)
              }}
            >
              <Download className="w-3.5 h-3.5 mr-1" />
              Download
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-xs"
            onClick={(e) => {
              e.stopPropagation()
              onView(report)
            }}
          >
            View
          </Button>
        </div>
      </div>
    </div>
  )
}
