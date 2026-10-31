'use client'

/**
 * PRD-261 C-03b — Approve and Reject for a pending theme version.
 *
 * Approve is disabled for a blocked or not-run scan whatever else is true (`review-words.ts`
 * `scanRefusal`, the rule `review.approval_refusal` enforces on the server), and whenever the
 * server's own `approval` says no (on the hosted edition: no passing isolated-job note naming this
 * digest yet). The reason is shown under the button. Reject needs a reason the submitter will see.
 */
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import type { ThemeReviewReport } from '@/lib/api-client'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { REASON_MAX, approveRefusal } from './review-words'
import { useApproveTheme, useRejectTheme } from './use-theme-review'

export const APPROVE = 'Approve'
export const REJECT = 'Reject'

function ActionError({ error }: { error: unknown }) {
  if (!error) return null
  return (
    <p role="alert" className="text-xs text-destructive">
      {themeErrorMessage(error)}
    </p>
  )
}

function ApproveButton({ report }: { report: ThemeReviewReport }) {
  const approve = useApproveTheme()
  const refusal = approveRefusal(report.scanVerdict, report.approval)
  return (
    <div className="space-y-1">
      <Button
        onClick={() => approve.mutate(report.id)}
        disabled={refusal !== null || approve.isLoading}
        aria-describedby={refusal ? 'theme-approve-refusal' : undefined}
      >
        {APPROVE}
      </Button>
      {refusal && (
        <p id="theme-approve-refusal" className="text-xs text-muted-foreground" data-testid="theme-approve-refusal">
          {refusal}
        </p>
      )}
      <ActionError error={approve.error} />
    </div>
  )
}

function RejectForm({ packageId }: { packageId: string }) {
  const reject = useRejectTheme()
  const [reason, setReason] = useState('')
  const blank = reason.trim() === ''
  return (
    <form
      className="space-y-2"
      onSubmit={(event) => {
        event.preventDefault()
        if (!blank) reject.mutate({ packageId, reason: reason.trim() })
      }}
    >
      <label className="block text-xs font-medium" htmlFor="theme-reject-reason">
        Reason (the submitter sees it)
      </label>
      <Textarea
        id="theme-reject-reason"
        value={reason}
        maxLength={REASON_MAX}
        onChange={(event) => setReason(event.target.value)}
        rows={2}
      />
      <Button type="submit" variant="destructive" disabled={blank || reject.isLoading}>
        {REJECT}
      </Button>
      <ActionError error={reject.error} />
    </form>
  )
}

export function ReviewActions({ report }: { report: ThemeReviewReport }) {
  return (
    <section aria-label="Decision" className="space-y-4 rounded-lg border border-border/50 p-3">
      <ApproveButton report={report} />
      <RejectForm packageId={report.id} />
    </section>
  )
}
