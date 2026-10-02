'use client'

/**
 * PRD-251 S2.4 (D6) — the Campaigns view, beside the List and the Board: the
 * workspace's campaigns, and the selected one with its posts and their statuses
 * (socials-campaign-detail.tsx), where "Approve series" lives.
 *
 * Series approval is optional per workspace: an owner or admin turns it on here
 * (PUT /api/workspaces/current/socials); everyone else sees whether it is on.
 * With it off, every post is approved on its own, whatever a campaign's mode.
 */
import { useState } from 'react'
import { Layers, Loader2, Plus } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { cn } from '@/lib/utils'
import { useWorkspace, type Workspace } from '@/components/workspace-provider'
import type { SocialCampaign, SocialPost } from '@/lib/api-client'
import { useSetSeriesApproval, useSocialCampaigns } from '@/hooks/use-socials-api'
import { SocialsCampaignDetail } from './socials-campaign-detail'
import { SocialsCampaignNew } from './socials-campaign-new'
import { canAuthorPosts, canTurnOnSocials } from './socials-status'
import { APPROVAL_MODE_LABELS } from './socials-series'

function SeriesSwitch({ role, seriesOn }: { role: Workspace['role']; seriesOn: boolean }) {
  const set = useSetSeriesApproval()
  if (!canTurnOnSocials(role)) {
    return <p className="text-sm text-muted-foreground">Series approval is {seriesOn ? 'on' : 'off'} for this workspace.</p>
  }
  return (
    <div className="flex items-center gap-2">
      <Switch
        id="socials-series-approval"
        checked={seriesOn}
        disabled={set.isLoading}
        onCheckedChange={(on) => set.mutate(on)}
      />
      <Label htmlFor="socials-series-approval">Series approval</Label>
    </div>
  )
}

interface CampaignListProps {
  campaigns: ReadonlyArray<SocialCampaign>
  selectedId: string | null
  onSelect: (campaignId: string) => void
}

function CampaignList({ campaigns, selectedId, onSelect }: CampaignListProps) {
  return (
    <ul aria-label="Campaigns" className="space-y-1.5">
      {campaigns.map((campaign) => (
        <li key={campaign.id}>
          <button
            type="button"
            aria-pressed={campaign.id === selectedId}
            onClick={() => onSelect(campaign.id)}
            className={cn(
              'flex w-full flex-col items-start gap-0.5 rounded-lg border border-border bg-card px-3 py-2 text-left transition',
              'hover:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/40',
              campaign.id === selectedId && 'border-primary/60',
            )}
          >
            <span className="max-w-full break-words text-sm font-medium text-foreground">{campaign.name}</span>
            <span className="text-xs text-muted-foreground">
              {APPROVAL_MODE_LABELS[campaign.approval_mode]} · {campaign.post_count ?? 0}{' '}
              {campaign.post_count === 1 ? 'post' : 'posts'}
            </span>
          </button>
        </li>
      ))}
    </ul>
  )
}

interface SocialsCampaignsProps {
  role: Workspace['role']
  /** The workspace's posts (the list's one query), to add to a campaign. */
  posts: ReadonlyArray<SocialPost>
}

export function SocialsCampaigns({ role, posts }: SocialsCampaignsProps) {
  const { workspace } = useWorkspace()
  const seriesOn = !!workspace?.socials?.series_approval
  const { data, isLoading, isError } = useSocialCampaigns()
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const campaigns = data?.campaigns ?? []
  const current = campaigns.find((c) => c.id === selectedId)?.id ?? campaigns[0]?.id ?? null

  const handleCreated = (campaign: SocialCampaign | null) => {
    setCreating(false)
    if (campaign) setSelectedId(campaign.id)
  }

  return (
    <div className="socials-campaigns space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SeriesSwitch role={role} seriesOn={seriesOn} />
        {canAuthorPosts(role) && !creating && (
          <Button size="sm" variant="outline" onClick={() => setCreating(true)}>
            <Plus className="mr-1.5 h-4 w-4" aria-hidden />
            New campaign
          </Button>
        )}
      </div>
      {creating && <SocialsCampaignNew onDone={handleCreated} />}
      {isLoading ? (
        <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          Loading campaigns…
        </div>
      ) : isError ? (
        <p role="alert" className="text-sm text-destructive">
          Could not load campaigns.
        </p>
      ) : campaigns.length === 0 ? (
        <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border/60 bg-card/20 px-6 py-10 text-center">
          <Layers className="h-7 w-7 text-muted-foreground" aria-hidden />
          <p className="text-sm font-medium text-foreground">No campaigns yet</p>
          <p className="text-sm text-muted-foreground">A campaign groups posts, to approve one by one or as a series.</p>
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
          <CampaignList campaigns={campaigns} selectedId={current} onSelect={setSelectedId} />
          {current && <SocialsCampaignDetail campaignId={current} role={role} posts={posts} seriesOn={seriesOn} />}
        </div>
      )}
    </div>
  )
}
