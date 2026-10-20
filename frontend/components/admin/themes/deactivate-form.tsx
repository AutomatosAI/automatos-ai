'use client'

/**
 * PRD-261 C-03b — take an approved version off the Marketplace (admin + step-up).
 *
 * The server sets it `deactivated` and writes a `disable` revocation of its digest, so every
 * install falls back to its base on the next load, and withdraws the update offers its approval
 * made (`review.deactivate`). The reason goes into the audit trail.
 */
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import type { ThemeReviewReport } from '@/lib/api-client'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { REASON_MAX } from './review-words'
import { useDeactivateTheme } from './use-theme-review'

export const DEACTIVATE = 'Deactivate'

export function DeactivateForm({ report }: { report: ThemeReviewReport }) {
  const deactivate = useDeactivateTheme()
  const [reason, setReason] = useState('')
  if (report.status !== 'approved') return null
  const blank = reason.trim() === ''
  return (
    <form
      aria-label="Deactivate"
      className="space-y-2 rounded-lg border border-destructive/30 p-3"
      onSubmit={(event) => {
        event.preventDefault()
        if (!blank) deactivate.mutate({ packageId: report.id, reason: reason.trim() })
      }}
    >
      <p className="text-sm font-medium">Take this version down</p>
      <p className="text-xs text-muted-foreground">
        Every install of it falls back to its base theme on the next load.
      </p>
      <label className="block text-xs font-medium" htmlFor="theme-deactivate-reason">
        Reason
      </label>
      <Textarea
        id="theme-deactivate-reason"
        value={reason}
        maxLength={REASON_MAX}
        rows={2}
        onChange={(event) => setReason(event.target.value)}
      />
      <Button type="submit" variant="destructive" disabled={blank || deactivate.isLoading}>
        {DEACTIVATE}
      </Button>
      {deactivate.error ? (
        <p role="alert" className="text-xs text-destructive">
          {themeErrorMessage(deactivate.error)}
        </p>
      ) : null}
    </form>
  )
}
