'use client'

/**
 * PRD-265 US-007 — "Business checkout link" in the operator console (admin/workspaces).
 *
 * Business is by invite: a platform admin asks
 * `POST /api/admin/workspaces/{id}/business-invite` for a Stripe Checkout link that works for
 * that one workspace (for up to 24 hours), copies it, and sends it to the workspace's owner.
 * The link is put on the clipboard and a toast says until when it works; where the browser
 * refuses the clipboard, the toast shows the link to copy by hand. Hidden for a workspace that
 * is already on Business. Errors are toasted in the API's plain words (lib/billing-errors.ts).
 */
import { useState } from 'react'
import { Link2, Loader2 } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api-client'
import { billingErrorMessage, isHttpsUrl } from '@/lib/billing-errors'
import { copyToClipboard } from '@/lib/utils'

export const BUSINESS_PLAN = 'business'
export const INVITE_FAILED = "The Business checkout link couldn't be made. Try again in a moment."
export const INVITE_LABEL = 'Business checkout link'
/** The by-hand toast stays until it is closed: the admin needs time to copy the link. */
const MANUAL_COPY_DURATION_MS = Number.POSITIVE_INFINITY

export interface BusinessInvite {
  url: string
  expires_at: string
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

/** "It works until 24 Oct 2026, 10:00." (empty when the API sent no usable time). */
export function inviteExpiry(expiresAt: string | null | undefined): string {
  const until = expiresAt ? new Date(expiresAt) : null
  if (!until || Number.isNaN(until.getTime())) return ''
  return `It works until ${until.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}.`
}

async function requestInvite(workspaceId: string): Promise<BusinessInvite> {
  const invite = await apiClient.post<BusinessInvite>(businessInvitePath(workspaceId))
  if (!isHttpsUrl(invite?.url)) throw new Error(INVITE_FAILED)
  return invite
}

async function handOver(invite: BusinessInvite, workspaceName: string): Promise<void> {
  const expiry = inviteExpiry(invite.expires_at)
  if (await copyToClipboard(invite.url)) {
    toast.success(`Business checkout link for ${workspaceName} copied`, {
      description: ['Send it to the workspace owner.', expiry].filter(Boolean).join(' '),
    })
    return
  }
  toast.info(`Copy the Business checkout link for ${workspaceName}`, {
    description: [invite.url, expiry].filter(Boolean).join(' '),
    duration: MANUAL_COPY_DURATION_MS,
  })
}

export function BusinessInviteButton({ workspaceId, workspaceName, plan, disabled = false }: BusinessInviteButtonProps) {
  const [busy, setBusy] = useState(false)
  if (plan === BUSINESS_PLAN) return null

  async function issue() {
    setBusy(true)
    try {
      await handOver(await requestInvite(workspaceId), workspaceName)
    } catch (error) {
      toast.error(billingErrorMessage(error, INVITE_FAILED))
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
      title={INVITE_LABEL}
      aria-label={`${INVITE_LABEL} for ${workspaceName}`}
      disabled={disabled || busy}
      onClick={() => void issue()}
    >
      {busy ? <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" /> : <Link2 className="h-3 w-3" aria-hidden="true" />}
    </Button>
  )
}
