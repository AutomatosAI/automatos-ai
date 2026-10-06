'use client'

import { useEffect, useCallback } from 'react'
import { formatDistanceToNow, format } from 'date-fns'
import {
  Star,
  ClipboardCheck,
  Search,
  AlertTriangle,
  FileText,
  Package,
  Shield,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import { useReport, useGradeReport } from '@/hooks/use-reports-api'
import { apiClient } from '@/lib/api-client'
import type { AgentReport } from '@/hooks/use-reports-api'
import { ReportGradeForm } from './report-grade-form'
import { ReportAttachments, ReportMetrics, ReportViewerHeader } from './report-viewer-parts'

// ─── Config ─────────────────────────────────────────────

const TYPE_ICONS: Record<string, typeof FileText> = {
  standup: ClipboardCheck,
  research: Search,
  incident: AlertTriangle,
  summary: FileText,
  delivery: Package,
  audit: Shield,
}

const STATUS_CONFIG: Record<string, { label: string; className: string }> = {
  ok: { label: 'OK', className: 'bg-[hsl(var(--success))]/10 text-[hsl(var(--success))]' },
  warning: { label: 'Warning', className: 'bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))]' },
  critical: { label: 'Critical', className: 'bg-destructive/10 text-destructive' },
  info: { label: 'Info', className: 'bg-[hsl(var(--info))]/10 text-[hsl(var(--info))]' },
}

// ─── Props ──────────────────────────────────────────────

interface ReportViewerProps {
  reportId: string | null
  onClose: () => void
}

// ─── Component ──────────────────────────────────────────

export function ReportViewer({ reportId, onClose }: ReportViewerProps) {
  const { data, isLoading } = useReport(reportId, { enabled: !!reportId })
  const gradeReport = useGradeReport()

  const report = data?.report

  // Close on Escape
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    },
    [onClose]
  )

  useEffect(() => {
    if (reportId) {
      document.addEventListener('keydown', handleKeyDown)
      return () => document.removeEventListener('keydown', handleKeyDown)
    }
  }, [reportId, handleKeyDown])

  if (!reportId) return null

  const handleDownload = () => {
    if (!report) return
    const content = report.content || report.summary || 'No content'
    const blob = new Blob([content], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${report.title.replace(/\s+/g, '-').toLowerCase()}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleGrade = (grade: number, notes?: string) => {
    if (!report) return
    gradeReport.mutate({ reportId: report.id, grade, grade_notes: notes })
  }

  const TypeIcon = report ? (TYPE_ICONS[report.report_type] || FileText) : FileText
  const statusCfg = report ? (STATUS_CONFIG[report.status] || STATUS_CONFIG.info) : STATUS_CONFIG.info

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Slide-over panel */}
      <div className="fixed inset-y-0 right-0 z-50 w-full sm:w-[60%] md:w-[55%] lg:w-[50%] max-w-[800px] glass-panel border-l border-border/50 shadow-2xl overflow-y-auto">
        <ReportViewerHeader report={report} onClose={onClose} onDownload={handleDownload} />

        {isLoading ? (
          <div className="p-6 space-y-4">
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-64 w-full" />
          </div>
        ) : report ? (
          <div className="p-6 space-y-6">
            {/* Report header */}
            <div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
                <TypeIcon className="w-4 h-4" />
                <span className="capitalize">{report.report_type}</span>
                <span>&middot;</span>
                <span>
                  {report.created_at
                    ? format(new Date(report.created_at), 'MMM d, yyyy HH:mm')
                    : 'Unknown date'}
                </span>
                <span>&middot;</span>
                <Badge className={cn('text-[10px]', statusCfg.className)}>
                  {statusCfg.label}
                </Badge>
              </div>
              <h2 className="text-xl font-bold">{report.title}</h2>
              <p className="text-sm text-muted-foreground mt-0.5">
                Agent: <span className="font-medium text-foreground">{report.agent_name}</span>
              </p>
            </div>

            <ReportMetrics metrics={report.metrics} />

            {/* Report content */}
            <div className="glass-card p-4 sm:p-6">
              <h3 className="text-sm font-medium text-muted-foreground mb-3">Report Content</h3>
              {report.content ? (
                <div className="prose prose-sm prose-invert max-w-none whitespace-pre-wrap text-sm leading-relaxed">
                  {report.content}
                </div>
              ) : report.content_error ? (
                <div className="text-sm text-destructive">
                  Could not load report content: {report.content_error}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">No content available</div>
              )}
            </div>

            <ReportAttachments attachments={report.attachments} />

            {/* Grade section */}
            <div className="glass-card p-4">
              <h3 className="text-sm font-medium text-muted-foreground mb-3 flex items-center gap-2">
                <Star className="w-4 h-4" />
                Grade This Report
              </h3>
              <ReportGradeForm
                currentGrade={report.grade}
                currentNotes={report.grade_notes}
                onSubmit={handleGrade}
                isSubmitting={gradeReport.isLoading}
              />
            </div>
          </div>
        ) : (
          <div className="p-6 text-center text-muted-foreground">
            Report not found
          </div>
        )}
      </div>
    </>
  )
}
