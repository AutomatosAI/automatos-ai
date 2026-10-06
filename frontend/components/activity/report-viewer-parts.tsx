'use client'

/**
 * The report viewer's header, metrics and attachments (PRD-76), in their own file so
 * ReportViewer (report-viewer.tsx) stays within the 150-line component limit. PRE-11:
 * the header offers Add to Knowledge once the report has loaded.
 */

import { Download, Paperclip, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { AgentReport } from '@/hooks/use-reports-api'
import { ReportKnowledgeButton } from './report-knowledge-button'

interface ReportViewerHeaderProps {
  report?: AgentReport
  onClose: () => void
  onDownload: () => void
}

export function ReportViewerHeader({ report, onClose, onDownload }: ReportViewerHeaderProps) {
  return (
    <div className="sticky top-0 z-10 bg-background/80 backdrop-blur-xl border-b border-border/50 px-6 py-4">
      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X className="w-4 h-4 mr-1" /> Close
        </Button>
        <div className="flex items-center gap-2">
          {report && <ReportKnowledgeButton report={report} />}
          <Button variant="outline" size="sm" onClick={onDownload}>
            <Download className="w-4 h-4 mr-1" /> Download
          </Button>
        </div>
      </div>
    </div>
  )
}

export function ReportMetrics({ metrics }: { metrics: AgentReport['metrics'] }) {
  if (!metrics || Object.keys(metrics).length === 0) return null
  return (
    <div className="flex flex-wrap gap-3">
      {Object.entries(metrics).map(([key, value]) => (
        <div
          key={key}
          className="glass-card px-3 py-2 text-center min-w-[80px]"
        >
          <div className="text-lg font-bold leading-none">
            {typeof value === 'number' ? value.toLocaleString() : String(value)}
          </div>
          <div className="text-[10px] text-muted-foreground mt-1 capitalize">
            {key.replace(/_/g, ' ')}
          </div>
        </div>
      ))}
    </div>
  )
}

export function ReportAttachments({ attachments }: { attachments: AgentReport['attachments'] }) {
  if (!attachments || attachments.length === 0) return null
  return (
    <div className="glass-card p-4">
      <h3 className="text-sm font-medium text-muted-foreground mb-3 flex items-center gap-2">
        <Paperclip className="w-4 h-4" />
        Attachments
      </h3>
      <div className="space-y-2">
        {attachments.map((att, i) => (
          <div
            key={i}
            className="flex items-center justify-between py-2 px-3 rounded-lg bg-secondary/20"
          >
            <div className="flex items-center gap-2 text-sm">
              <Paperclip className="w-3.5 h-3.5 text-muted-foreground" />
              <span>{att.title}</span>
              <span className="text-xs text-muted-foreground">({att.file_type})</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              disabled={!att.url && !att.file_path}
              onClick={() => {
                const href = att.url || att.file_path
                if (href) window.open(href, '_blank')
              }}
            >
              <Download className="w-3 h-3 mr-1" /> Download
            </Button>
          </div>
        ))}
      </div>
    </div>
  )
}
