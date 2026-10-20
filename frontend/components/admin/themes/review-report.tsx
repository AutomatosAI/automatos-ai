'use client'

/**
 * PRD-261 C-03b — one theme version's report in the admin review queue
 * (`GET /api/admin/themes/{packageId}/report`): what it is, its scan report, the isolated review
 * note, the decision (pending), the signature (approved), take-down (approved) and its history.
 *
 * The report stays open after a decision: the version leaves the pending list, and its report,
 * read again, shows the new status and the signature request approve opened.
 */
import { Badge } from '@/components/ui/badge'
import type { ThemeReviewEvent, ThemeReviewReport } from '@/lib/api-client'
import { formatRelative } from '@/lib/format-relative'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { DeactivateForm } from './deactivate-form'
import { ReportNoteSection } from './report-note-form'
import { ReviewActions } from './review-actions'
import { STATUS_LABELS, labelOf } from './review-words'
import { ScanChecks } from './scan-checks'
import { SignaturePanel } from './signature-panel'
import { useThemeReport } from './use-theme-review'

function ReportHeader({ report }: { report: ThemeReviewReport }) {
  return (
    <header className="space-y-1">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-lg font-semibold">{report.name || report.themeId}</h2>
        <Badge variant="secondary" data-testid="theme-review-status">
          {labelOf(STATUS_LABELS, report.status)}
        </Badge>
      </div>
      <p className="text-xs text-muted-foreground">
        {report.themeId} · {report.version} · publisher {report.publisher}
        {report.createdAt && <> · submitted {formatRelative(report.createdAt)}</>}
      </p>
      <p className="text-xs">
        Digest <code className="break-all">{report.digest}</code>
      </p>
    </header>
  )
}

function History({ events }: { events: readonly ThemeReviewEvent[] }) {
  if (events.length === 0) return null
  return (
    <section aria-label="History" className="space-y-1">
      <p className="text-sm font-medium">History</p>
      <ol className="space-y-0.5 text-xs text-muted-foreground">
        {events.map((event, index) => (
          <li key={`${event.at ?? ''}-${index}`}>
            {formatRelative(event.at)} · {event.event}
            {event.to && event.from !== event.to && <> ({event.from ?? '—'} → {event.to})</>}
            {event.reason && <> · {event.reason}</>}
          </li>
        ))}
      </ol>
    </section>
  )
}

export function ReviewReport({ packageId, hosted }: { packageId: string; hosted: boolean }) {
  const query = useThemeReport(packageId)
  if (query.isLoading) return <p className="text-sm text-muted-foreground">Loading the report…</p>
  if (query.isError || !query.data) {
    return (
      <p role="alert" className="text-sm text-destructive">
        {themeErrorMessage(query.error)}
      </p>
    )
  }
  const report = query.data
  return (
    <article className="space-y-4" data-testid="theme-review-report">
      <ReportHeader report={report} />
      <ScanChecks report={report.scanReport} verdict={report.scanVerdict} />
      <ReportNoteSection report={report} hosted={hosted} />
      {report.status === 'pending' && <ReviewActions report={report} />}
      <SignaturePanel report={report} />
      <DeactivateForm report={report} />
      <History events={report.history} />
    </article>
  )
}
