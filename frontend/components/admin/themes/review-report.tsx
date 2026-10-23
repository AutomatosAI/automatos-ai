'use client'

/**
 * PRD-261 C-03b — one theme version's report in the admin review queue
 * (`GET /api/admin/themes/{packageId}/report`): what it is, its scan report, the isolated review
 * note, the decision (pending), the signature (approved), take-down (approved) and its history.
 *
 * The report stays open after a decision: the version leaves the pending list, and its report,
 * read again, shows the new status and the signature request approve opened.
 *
 * C-05j: on the hosted edition a pending version also shows the review job's inputs (C-03c
 * `GET …/review-package`): the kit's Theme review workflow takes `package_url` and
 * `expected_digest`. The link is asked for when the reviewer wants it, since each answer stores a
 * fresh copy of the package and lasts 15 minutes. The local edition has no isolated job.
 */
import { useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { ThemeReviewEvent, ThemeReviewReport } from '@/lib/api-client'
import { formatRelative } from '@/lib/format-relative'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { DeactivateForm } from './deactivate-form'
import { ReportNoteSection } from './report-note-form'
import { ReviewActions } from './review-actions'
import {
  GET_REVIEW_JOB_INPUTS,
  REVIEW_JOB_INPUTS,
  REVIEW_JOB_INPUTS_HINT,
  STATUS_LABELS,
  labelOf,
  linkExpiryWords,
} from './review-words'
import { ScanChecks } from './scan-checks'
import { SignaturePanel } from './signature-panel'
import { useReviewPackage, useThemeReport } from './use-theme-review'

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

function JobInput({ name, value }: { name: string; value: string }) {
  return (
    <div className="space-y-0.5 text-xs">
      <p className="font-medium"><code>{name}</code></p>
      <pre className="whitespace-pre-wrap break-all rounded bg-muted p-2" data-testid={`theme-review-input-${name}`}>{value}</pre>
    </div>
  )
}

export function ReviewJobInputs({ packageId }: { packageId: string }) {
  const [requested, setRequested] = useState(false)
  const link = useReviewPackage(packageId, requested)
  return (
    <section aria-label={REVIEW_JOB_INPUTS} className="space-y-2 rounded-lg border border-border/50 p-3">
      <p className="text-sm font-medium">{REVIEW_JOB_INPUTS}</p>
      <p className="text-xs text-muted-foreground">{REVIEW_JOB_INPUTS_HINT}</p>
      <Button
        type="button"
        size="sm"
        variant="outline"
        disabled={link.isFetching}
        onClick={() => (requested ? void link.refetch() : setRequested(true))}
      >
        {GET_REVIEW_JOB_INPUTS}
      </Button>
      {link.isError ? (
        <p role="alert" className="text-xs text-destructive">
          {themeErrorMessage(link.error)}
        </p>
      ) : null}
      {link.data && !link.isError ? (
        <div className="space-y-2" data-testid="theme-review-inputs">
          <JobInput name="package_url" value={link.data.url} />
          <JobInput name="expected_digest" value={link.data.digest} />
          <p className="text-xs text-muted-foreground">{linkExpiryWords(link.data.expiresAt)}</p>
        </div>
      ) : null}
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
      {hosted && report.status === 'pending' && <ReviewJobInputs packageId={report.id} />}
      <ReportNoteSection report={report} hosted={hosted} />
      {report.status === 'pending' && <ReviewActions report={report} />}
      <SignaturePanel report={report} />
      <DeactivateForm report={report} />
      <History events={report.history} />
    </article>
  )
}
