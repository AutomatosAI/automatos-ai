'use client'

/**
 * PRD-251 S2.4 — one campaign: its approval mode, its posts with their statuses,
 * adding and taking out posts (for a role that authors), and "Approve series"
 * (for a role that approves), offered only when the workspace's series approval
 * is on and the campaign approves as a series (D6). Otherwise it says why.
 */
import { useState } from 'react'
import { Loader2, X } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialCampaignApprovalMode, SocialCampaignWithPosts, SocialPost } from '@/lib/api-client'
import { useSocialCampaign, useSocialCampaignPost, useUpdateSocialCampaign } from '@/hooks/use-socials-api'
import { SocialsSeriesApprove } from './socials-series-approve'
import { SOCIAL_STATUS_LABELS, canAuthorPosts } from './socials-status'
import { APPROVAL_MODE_LABELS, canApproveSeries, seriesBlockedReason, seriesPlan } from './socials-series'

const SELECT_CLASS = 'h-9 rounded-md border border-input bg-background px-2 text-sm'

interface CampaignPostsProps {
  campaign: SocialCampaignWithPosts
  canEdit: boolean
}

function CampaignPosts({ campaign, canEdit }: CampaignPostsProps) {
  const change = useSocialCampaignPost()
  if (campaign.posts.length === 0) {
    return <p className="text-sm text-muted-foreground">No posts in this campaign yet.</p>
  }
  return (
    <ul aria-label="Campaign posts" className="space-y-1.5">
      {campaign.posts.map((post) => (
        <li key={post.id} className="flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2">
          <span className="min-w-0 flex-1 break-words text-sm font-medium text-foreground">{post.title}</span>
          <Badge variant="secondary">{SOCIAL_STATUS_LABELS[post.status]}</Badge>
          {canEdit && (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              aria-label={`Take ${post.title} out of the campaign`}
              disabled={change.isLoading}
              onClick={() => change.mutate({ campaignId: campaign.id, postId: post.id, action: 'remove' })}
            >
              <X className="h-4 w-4" aria-hidden />
            </Button>
          )}
        </li>
      ))}
    </ul>
  )
}

function AddPost({ campaign, posts }: { campaign: SocialCampaignWithPosts; posts: ReadonlyArray<SocialPost> }) {
  const change = useSocialCampaignPost()
  const [postId, setPostId] = useState('')
  const inCampaign = new Set(campaign.posts.map((post) => post.id))
  const choices = posts.filter((post) => !inCampaign.has(post.id))
  if (choices.length === 0) return null
  const add = () =>
    change.mutate({ campaignId: campaign.id, postId, action: 'add' }, { onSuccess: () => setPostId('') })
  return (
    <div className="flex flex-wrap items-center gap-2">
      <select aria-label="Post to add" value={postId} onChange={(e) => setPostId(e.target.value)} className={SELECT_CLASS}>
        <option value="">Choose a post</option>
        {choices.map((post) => (
          <option key={post.id} value={post.id}>
            {post.title}
          </option>
        ))}
      </select>
      <Button type="button" size="sm" variant="outline" disabled={!postId || change.isLoading} onClick={add}>
        Add to campaign
      </Button>
    </div>
  )
}

function ModePicker({ campaign }: { campaign: SocialCampaignWithPosts }) {
  const update = useUpdateSocialCampaign()
  return (
    <select
      aria-label="Approval mode"
      value={campaign.approval_mode}
      disabled={update.isLoading}
      onChange={(e) =>
        update.mutate({ campaignId: campaign.id, changes: { approval_mode: e.target.value as SocialCampaignApprovalMode } })
      }
      className={SELECT_CLASS}
    >
      {(Object.keys(APPROVAL_MODE_LABELS) as SocialCampaignApprovalMode[]).map((mode) => (
        <option key={mode} value={mode}>
          {APPROVAL_MODE_LABELS[mode]}
        </option>
      ))}
    </select>
  )
}

function SeriesAction({ campaign, seriesOn }: { campaign: SocialCampaignWithPosts; seriesOn: boolean }) {
  const [open, setOpen] = useState(false)
  const blocked = seriesBlockedReason(seriesOn, campaign, seriesPlan(campaign.posts))
  if (open) return <SocialsSeriesApprove campaign={campaign} onClose={() => setOpen(false)} />
  return (
    <div className="space-y-1">
      <Button type="button" size="sm" disabled={blocked !== null} onClick={() => setOpen(true)}>
        Approve series
      </Button>
      {blocked && <p className="text-xs text-muted-foreground">{blocked}</p>}
    </div>
  )
}

interface SocialsCampaignDetailProps {
  campaignId: string
  role: Workspace['role']
  /** The workspace's posts, to add to the campaign. */
  posts: ReadonlyArray<SocialPost>
  /** The workspace's series approval switch (D6). */
  seriesOn: boolean
}

export function SocialsCampaignDetail({ campaignId, role, posts, seriesOn }: SocialsCampaignDetailProps) {
  const { data: campaign, isLoading, isError } = useSocialCampaign(campaignId)
  const author = canAuthorPosts(role)

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
        Loading the campaign…
      </div>
    )
  }
  if (isError || !campaign) {
    return (
      <p role="alert" className="text-sm text-destructive">
        Could not load the campaign.
      </p>
    )
  }
  return (
    <section aria-label={`Campaign ${campaign.name}`} className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-foreground">{campaign.name}</h3>
        {author ? (
          <ModePicker campaign={campaign} />
        ) : (
          <span className="text-xs text-muted-foreground">{APPROVAL_MODE_LABELS[campaign.approval_mode]}</span>
        )}
      </div>
      <CampaignPosts campaign={campaign} canEdit={author} />
      {author && <AddPost campaign={campaign} posts={posts} />}
      {canApproveSeries(role) && <SeriesAction key={campaign.id} campaign={campaign} seriesOn={seriesOn} />}
    </section>
  )
}
