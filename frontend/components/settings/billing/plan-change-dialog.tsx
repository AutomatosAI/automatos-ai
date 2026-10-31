'use client'

/**
 * PRD-265 W2 — the plan-change preview: what switching costs, before it is made.
 *
 * "You'll pay $13.50 now (prorated), then $29 a month." for an upgrade (charged now, for the
 * rest of the period); "Changes to Own keys on 23 Nov 2026; nothing to pay now." for a
 * downgrade (it waits for the period end). The confirm button makes the switch
 * (./use-plan-switch.ts); the dialog closes first, so a bank's check shows over the page.
 */
import { Loader2 } from 'lucide-react'
import type { UseQueryResult } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import type { ApiHttpError } from '@/lib/api-client'
import { billingErrorMessage } from '@/lib/billing-errors'
import type { BillingPlan } from '@/hooks/use-billing-plans'
import type { PlanChangePreview } from '@/hooks/use-billing-plan-change'
import { formatDay, formatMoney, formatUsd } from './billing-format'

export const PREVIEW_FAILED = "The price of this change couldn't be worked out. Try again in a moment."
export const PREVIEW_LOADING = 'Working out the price…'
const NOW = 'now'

/** An upgrade is charged now; a downgrade waits for the period end. */
export function chargesNow(preview: PlanChangePreview): boolean {
  return preview.effective_at === NOW || preview.direction === 'upgrade'
}

/** The preview in one sentence. */
export function previewSentence(preview: PlanChangePreview, plan: Pick<BillingPlan, 'name' | 'price_usd'>): string {
  if (!chargesNow(preview)) return `Changes to ${plan.name} on ${formatDay(preview.effective_at)}; nothing to pay now.`
  const tax = preview.tax_cents > 0 ? `, including ${formatMoney(preview.tax_cents, preview.currency)} tax` : ''
  const then = plan.price_usd === null ? '' : `, then ${formatUsd(plan.price_usd)} a month`
  return `You'll pay ${formatMoney(preview.amount_due_cents, preview.currency)} now (prorated${tax})${then}.`
}

/** "Pay $13.50 and switch" for an upgrade; "Switch to Own keys" for a downgrade. */
export function confirmLabel(preview: PlanChangePreview | undefined, plan: Pick<BillingPlan, 'name'>): string {
  if (preview && chargesNow(preview) && preview.amount_due_cents > 0) {
    return `Pay ${formatMoney(preview.amount_due_cents, preview.currency)} and switch`
  }
  return `Switch to ${plan.name}`
}

interface PlanChangeDialogProps {
  readonly target: BillingPlan | null
  readonly preview: Pick<UseQueryResult<PlanChangePreview, ApiHttpError>, 'data' | 'error' | 'isError' | 'isLoading'>
  readonly busy: boolean
  readonly onConfirm: () => void
  readonly onClose: () => void
}

function PreviewBody({ preview, plan }: { preview: PlanChangeDialogProps['preview']; plan: BillingPlan }) {
  if (preview.isError) {
    return (
      <p role="alert" className="text-sm text-destructive">
        {billingErrorMessage(preview.error, PREVIEW_FAILED)}
      </p>
    )
  }
  if (!preview.data) return <p className="text-sm text-muted-foreground">{PREVIEW_LOADING}</p>
  return (
    <p data-testid="plan-change-preview" className="text-sm text-foreground">
      {previewSentence(preview.data, plan)}
    </p>
  )
}

export function PlanChangeDialog({ target, preview, busy, onConfirm, onClose }: PlanChangeDialogProps) {
  const ready = Boolean(preview.data) && !preview.isError
  return (
    <Dialog open={target !== null} onOpenChange={(open) => !open && !busy && onClose()}>
      {target && (
        <DialogContent size="sm">
          <DialogHeader>
            <DialogTitle>Switch to {target.name}</DialogTitle>
            <DialogDescription>What the switch costs, before you make it.</DialogDescription>
          </DialogHeader>
          <PreviewBody preview={preview} plan={target} />
          <DialogFooter className="gap-2">
            <Button type="button" variant="outline" disabled={busy} onClick={onClose}>
              Not now
            </Button>
            <Button type="button" disabled={!ready || busy} onClick={onConfirm}>
              {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />}
              {confirmLabel(preview.data, target)}
            </Button>
          </DialogFooter>
        </DialogContent>
      )}
    </Dialog>
  )
}
