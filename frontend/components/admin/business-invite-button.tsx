'use client'

/**
 * PRD-265 US-007 / W2 — "Offer Business" in the operator console (admin/workspaces).
 *
 * Business is by invite. A platform admin offers it to one workspace
 * (`POST /api/admin/workspaces/{id}/business-invite`): for the next 14 days the owner sees
 * Business as buyable on their own Billing tab, and gets a notification. There is no link to
 * copy or send; the toast says until when the offer stands. Hidden for a workspace that is
 * already on Business. Errors are toasted in the API's plain words (lib/billing-errors.ts).
 */
import { useState } from 'react'
import { Gift, Loader2 } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api-client'
import { billingErrorMessage } from '@/lib/billing-errors'
import { formatAbsoluteDate } from '@/lib/format-relative'

export const BUSINESS_PLAN = 'business'
export const OFFER_FAILED = "Business couldn't be offered. Try again in a moment."
export const OFFER_LABEL = 'Offer Business'

export interface BusinessOffer {
  workspace_id: string
  business_offer_until: string
}

interface BusinessInviteButtonProps {
  readonly workspaceId: string
  readonly workspaceName: string
  readonly plan: string | null
  readonly disabled?: boolean
}

export function businessInvitePath(workspaceId: string): string {
  return `/api/admin/workspaces/${encodeURIComponent(workspaceId)}/business-invite`
}

/** "Business offered to Acme until 6 Nov 2026" (without the date when the API sent none). */
export function offeredToast(workspaceName: string, until: string | null | undefined): string {
  const date = formatAbsoluteDate(until)
  return date === '—' ? `Business offered to ${workspaceName}` : `Business offered to ${workspaceName} until ${date}`
}

export function BusinessInviteButton({ workspaceId, workspaceName, plan, disabled = false }: BusinessInviteButtonProps) {
  const [busy, setBusy] = useState(false)
  if (plan === BUSINESS_PLAN) return null

  async function offer() {
    setBusy(true)
    try {
      const answer = await apiClient.post<BusinessOffer>(businessInvitePath(workspaceId))
      toast.success(offeredToast(workspaceName, answer?.business_offer_until))
    } catch (error) {
      toast.error(billingErrorMessage(error, OFFER_FAILED))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Button
      type="button"
      size="sm"
      variant="outline"
      className="h-7 px-2"
      title={OFFER_LABEL}
      aria-label={`${OFFER_LABEL} to ${workspaceName}`}
      disabled={disabled || busy}
      onClick={() => void offer()}
    >
      {busy ? <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" /> : <Gift className="h-3 w-3" aria-hidden="true" />}
    </Button>
  )
}
