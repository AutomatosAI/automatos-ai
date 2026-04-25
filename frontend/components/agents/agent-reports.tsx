'use client'

import { useState, useCallback } from 'react'
import { formatDistanceToNow } from 'date-fns'
import {
  FileText,
  Star,
  ExternalLink,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import { useAgentReports, useReportStats } from '@/hooks/use-reports-api'
import type { AgentReport } from '@/hooks/use-reports-api'
import { ReportViewer } from '@/components/activity/report-viewer'

// ─── Status config ───────────────────────────────────────

const STATUS_DOT: Record<string, string> = {
  ok: 'bg-[hsl(var(--success))]',
  warning: 'bg-[hsl(var(--warning))]',
  critical: 'bg-destructive',
  info: 'bg-[hsl(var(--info))]',
}

// ─── Props ───────────────────────────────────────────────

interface AgentReportsProps {
  agentId: number
  agentName?: string
}

// ─── Component ───────────────────────────────────────────

export function AgentReports({ agentId, agentName }: AgentReportsProps) {
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null)

  const { data, isLoading } = useAgentReports(agentId, { period: '30d', limit: 10 })
  const reports = data?.reports || []
  const total = data?.total || 0

  // Calculate summary stats from reports
  const okCount = reports.filter((r) => r.status === 'ok').length
  const warnCount = reports.filter((r) => r.status === 'warning' || r.status === 'critical').length
  const gradedReports = reports.filter((r) => r.grade)
  const avgGrade = gradedReports.length > 0
    ? (gradedReports.reduce((sum, r) => sum + (r.grade || 0), 0) / gradedReports.length).toFixed(1)
    : null
  const ungradedCount = reports.filter((r) => !r.grade).length

  if (isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-20 w-full rounded-xl" />
        <Skeleton className="h-12 w-full rounded-lg" />
        <Skeleton className="h-12 w-full rounded-lg" />
        <Skeleton className="h-12 w-full rounded-lg" />
      </div>
    )
  }

  if (reports.length === 0) {
    return (
      <div className="glass-card p-6 text-center text-muted-foreground">
        <FileText className="w-10 h-10 mx-auto mb-2 opacity-30" />
        <p className="font-medium text-sm">No reports yet</p>
        <p className="text-xs mt-1">
          Reports will appear here when this agent runs heartbeat checks or submits deliverables.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Summary card */}
      <div className="glass-card p-4">
        <h4 className="text-sm font-medium text-muted-foreground mb-3">
          Last 30 days
        </h4>
        <div className="flex flex-wrap gap-4 text-sm">
          <div>
            <span className="text-lg font-bold">{total}</span>
            <span className="text-muted-foreground ml-1">reports</span>
          </div>
          <div className="flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5 text-[hsl(var(--success))]" />
            <span className="font-medium">{okCount}</span>
            <span className="text-muted-foreground">OK</span>
          </div>
          {warnCount > 0 && (
            <div className="flex items-center gap-1">
              <AlertTriangle className="w-3.5 h-3.5 text-[hsl(var(--warning))]" />
              <span className="font-medium">{warnCount}</span>
              <span className="text-muted-foreground">attention</span>
            </div>
          )}
          {avgGrade && (
            <div className="flex items-center gap-1">
              <Star className="w-3.5 h-3.5 text-[hsl(var(--warning))] fill-[hsl(var(--warning))]" />
              <span className="font-medium">{avgGrade}</span>
              <span className="text-muted-foreground">avg grade</span>
            </div>
          )}
          {ungradedCount > 0 && (
            <div className="text-muted-foreground">
              {ungradedCount} ungraded
            </div>
          )}
        </div>
      </div>

      {/* Recent reports list */}
      <div className="glass-card overflow-hidden">
        <div className="px-4 py-2 border-b border-border/50">
          <h4 className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
            Recent Reports
          </h4>
        </div>
        <div className="divide-y divide-border/30">
          {reports.map((report) => (
            <button
              key={report.id}
              className="w-full flex items-center gap-3 px-4 py-3 hover:bg-secondary/20 transition-colors text-left"
              onClick={() => setSelectedReportId(report.id)}
            >
              <div
                className={cn(
                  'w-2 h-2 rounded-full shrink-0',
                  STATUS_DOT[report.status] || STATUS_DOT.info
                )}
              />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">{report.title}</div>
                <div className="text-xs text-muted-foreground">
                  {report.created_at
                    ? formatDistanceToNow(new Date(report.created_at), { addSuffix: true })
                    : ''}
                </div>
              </div>
              <div className="shrink-0">
                {report.grade ? (
                  <div className="flex items-center gap-0.5">
                    {[1, 2, 3, 4, 5].map((s) => (
                      <Star
                        key={s}
                        className={cn(
                          'w-3 h-3',
                          s <= report.grade!
                            ? 'text-[hsl(var(--warning))] fill-[hsl(var(--warning))]'
                            : 'text-muted-foreground/20'
                        )}
                      />
                    ))}
                  </div>
                ) : (
                  <span className="text-[10px] text-muted-foreground">--</span>
                )}
              </div>
            </button>
          ))}
        </div>

        {total > reports.length && (
          <div className="px-4 py-2 border-t border-border/50 text-center">
            <Button
              variant="ghost"
              size="sm"
              className="text-xs"
              onClick={() => {
                window.location.href = `/command-center?tab=board&agent_id=${agentId}`
              }}
            >
              View All ({total}) <ExternalLink className="w-3 h-3 ml-1" />
            </Button>
          </div>
        )}
      </div>

      {/* Report viewer */}
      <ReportViewer
        reportId={selectedReportId}
        onClose={() => setSelectedReportId(null)}
      />
    </div>
  )
}
