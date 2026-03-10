'use client'

import { useState, useCallback } from 'react'
import {
  FileText,
  Star,
  Filter,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { cn } from '@/lib/utils'
import { useReports, useReportStats } from '@/hooks/use-reports-api'
import type { AgentReport, ReportFilters } from '@/hooks/use-reports-api'
import { ReportCard } from './report-card'
import { ReportViewer } from './report-viewer'

// ─── Skeleton ────────────────────────────────────────────

function ReportCardSkeleton() {
  return (
    <div className="glass-card border-l-[3px] border-l-secondary p-4 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div className="space-y-1.5">
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-5 w-56" />
        </div>
        <div className="flex items-center gap-2">
          <Skeleton className="h-5 w-12 rounded-full" />
          <Skeleton className="h-3 w-16" />
        </div>
      </div>
      <Skeleton className="h-4 w-full" />
      <div className="flex gap-3">
        <Skeleton className="h-6 w-24 rounded-md" />
        <Skeleton className="h-6 w-20 rounded-md" />
        <Skeleton className="h-6 w-16 rounded-md" />
      </div>
      <div className="flex justify-between">
        <Skeleton className="h-4 w-28" />
        <Skeleton className="h-7 w-14 rounded-md" />
      </div>
    </div>
  )
}

// ─── Empty State ─────────────────────────────────────────

function ReportsEmptyState() {
  return (
    <div className="glass-card p-6 sm:p-8 text-center text-muted-foreground">
      <FileText className="w-12 h-12 mx-auto mb-3 opacity-30" />
      <p className="font-medium">No reports yet</p>
      <p className="text-sm mt-1 max-w-xs mx-auto">
        Reports appear here when your agents complete heartbeat checks, research tasks, or deliveries.
      </p>
    </div>
  )
}

// ─── Stats Bar ───────────────────────────────────────────

function ReportStatsBar({ period }: { period: string }) {
  const { data: stats } = useReportStats(period)

  if (!stats) return null

  return (
    <div className="flex flex-wrap gap-3 mb-4">
      <div className="glass-card px-3 py-2 text-center min-w-[80px]">
        <div className="text-lg font-bold leading-none">{stats.total}</div>
        <div className="text-[10px] text-muted-foreground mt-1">Total</div>
      </div>
      <div className="glass-card px-3 py-2 text-center min-w-[80px]">
        <div className="text-lg font-bold leading-none">{stats.ungraded_count}</div>
        <div className="text-[10px] text-muted-foreground mt-1">Ungraded</div>
      </div>
      {stats.avg_grade && (
        <div className="glass-card px-3 py-2 text-center min-w-[80px]">
          <div className="text-lg font-bold leading-none flex items-center justify-center gap-1">
            {stats.avg_grade}
            <Star className="w-3.5 h-3.5 text-[hsl(var(--warning))] fill-[hsl(var(--warning))]" />
          </div>
          <div className="text-[10px] text-muted-foreground mt-1">Avg Grade</div>
        </div>
      )}
      {Object.entries(stats.by_status || {}).map(([status, count]) => (
        <div key={status} className="glass-card px-3 py-2 text-center min-w-[80px]">
          <div className="text-lg font-bold leading-none">{count}</div>
          <div className="text-[10px] text-muted-foreground mt-1 capitalize">{status}</div>
        </div>
      ))}
    </div>
  )
}

// ─── Main Component ──────────────────────────────────────

interface ActivityReportsProps {
  period?: string
}

export function ActivityReports({ period = '30d' }: ActivityReportsProps) {
  const [filters, setFilters] = useState<ReportFilters>({ period, limit: 20 })
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null)

  const { data, isLoading } = useReports(filters)
  const reports = data?.reports || []
  const total = data?.total || 0

  const handleView = useCallback((report: AgentReport) => {
    setSelectedReportId(report.id)
  }, [])

  const handleDownload = useCallback((report: AgentReport) => {
    window.open(`/api/reports/${report.id}/download`, '_blank')
  }, [])

  const handleLoadMore = useCallback(() => {
    setFilters((prev) => ({
      ...prev,
      limit: (prev.limit || 20) + 20,
    }))
  }, [])

  const handleFilterChange = useCallback(
    (key: keyof ReportFilters, value: string | boolean | undefined) => {
      setFilters((prev) => ({
        ...prev,
        [key]: value === 'all' ? undefined : value,
        offset: 0,
      }))
    },
    []
  )

  return (
    <div className="space-y-4">
      {/* Stats */}
      <ReportStatsBar period={period} />

      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-2">
        <Filter className="w-4 h-4 text-muted-foreground" />

        <Select
          value={filters.report_type || 'all'}
          onValueChange={(v) => handleFilterChange('report_type', v)}
        >
          <SelectTrigger className="w-32 h-8 text-xs bg-secondary/40">
            <SelectValue placeholder="All Types" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="standup">Standup</SelectItem>
            <SelectItem value="research">Research</SelectItem>
            <SelectItem value="incident">Incident</SelectItem>
            <SelectItem value="summary">Summary</SelectItem>
            <SelectItem value="delivery">Delivery</SelectItem>
            <SelectItem value="audit">Audit</SelectItem>
          </SelectContent>
        </Select>

        <Select
          value={filters.status || 'all'}
          onValueChange={(v) => handleFilterChange('status', v)}
        >
          <SelectTrigger className="w-28 h-8 text-xs bg-secondary/40">
            <SelectValue placeholder="All Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="ok">OK</SelectItem>
            <SelectItem value="warning">Warning</SelectItem>
            <SelectItem value="critical">Critical</SelectItem>
            <SelectItem value="info">Info</SelectItem>
          </SelectContent>
        </Select>

        <Button
          variant={filters.graded === false ? 'default' : 'outline'}
          size="sm"
          className="h-8 text-xs"
          onClick={() =>
            handleFilterChange(
              'graded',
              filters.graded === false ? undefined : false
            )
          }
        >
          <Star className="w-3 h-3 mr-1" />
          Ungraded
        </Button>

        <span className="text-xs text-muted-foreground ml-auto">
          {total} report{total !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Report cards */}
      {isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <ReportCardSkeleton key={i} />
          ))}
        </div>
      ) : reports.length === 0 ? (
        <ReportsEmptyState />
      ) : (
        <div className="space-y-3">
          {reports.map((report) => (
            <ReportCard
              key={report.id}
              report={report}
              onView={handleView}
              onDownload={handleDownload}
            />
          ))}
        </div>
      )}

      {/* Load more */}
      {reports.length < total && (
        <div className="text-center">
          <Button
            variant="outline"
            size="sm"
            onClick={handleLoadMore}
          >
            Load More ({total - reports.length} remaining)
          </Button>
        </div>
      )}

      {/* Report viewer slide-over */}
      <ReportViewer
        reportId={selectedReportId}
        onClose={() => setSelectedReportId(null)}
      />
    </div>
  )
}
