'use client'

/**
 * PRD-251 S2.4 — "New campaign": a name and how its posts are approved, post by
 * post (the default) or as a series (D6). It creates the campaign through
 * POST /api/socials/campaigns; the campaigns refetch and it opens.
 */
import { useState, type FormEvent } from 'react'
import { Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { SocialCampaignApprovalMode, SocialCampaignWithPosts } from '@/lib/api-client'
import { useCreateSocialCampaign } from '@/hooks/use-socials-api'
import { APPROVAL_MODE_LABELS, SOCIAL_CAMPAIGN_NAME_MAX_CHARS } from './socials-series'

interface SocialsCampaignNewProps {
  /** Called with the created campaign, or null when the form is cancelled. */
  onDone: (campaign: SocialCampaignWithPosts | null) => void
}

export function SocialsCampaignNew({ onDone }: SocialsCampaignNewProps) {
  const create = useCreateSocialCampaign()
  const [name, setName] = useState('')
  const [mode, setMode] = useState<SocialCampaignApprovalMode>('per_post')
  const nameOk = name.trim().length > 0

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!nameOk || create.isLoading) return
    create.mutate({ name: name.trim(), approval_mode: mode }, { onSuccess: (campaign) => onDone(campaign) })
  }

  return (
    <form onSubmit={handleSubmit} aria-label="New campaign" className="space-y-3 rounded-xl border border-border bg-card/40 p-4">
      <div className="space-y-1.5">
        <Label htmlFor="socials-campaign-name">Name</Label>
        <Input
          id="socials-campaign-name"
          value={name}
          maxLength={SOCIAL_CAMPAIGN_NAME_MAX_CHARS}
          onChange={(event) => setName(event.target.value)}
          placeholder="Web Summit countdown"
          required
        />
      </div>
      <div className="space-y-1.5">
        <Label htmlFor="socials-campaign-mode">Approval</Label>
        <select
          id="socials-campaign-mode"
          value={mode}
          onChange={(event) => setMode(event.target.value as SocialCampaignApprovalMode)}
          className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm"
        >
          {(Object.keys(APPROVAL_MODE_LABELS) as SocialCampaignApprovalMode[]).map((value) => (
            <option key={value} value={value}>
              {APPROVAL_MODE_LABELS[value]}
            </option>
          ))}
        </select>
      </div>
      <div className="flex justify-end gap-2">
        <Button type="button" variant="ghost" size="sm" onClick={() => onDone(null)}>
          Cancel
        </Button>
        <Button type="submit" size="sm" disabled={!nameOk || create.isLoading}>
          {create.isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />}
          Create campaign
        </Button>
      </div>
    </form>
  )
}
