/**
 * PRD-251 S2.4 (D6) — the Campaigns view's reading of series approval: what an
 * "Approve series" will approve, what it leaves, and why a series cannot be
 * approved now. The server decides all of it (the workspace switch, the
 * campaign's mode, each post's hash at approval, D7's unsourced rule); this only
 * decides what to show and what to send.
 */
import type {
  SocialCampaign,
  SocialCampaignApprovalMode,
  SocialPost,
  SocialSeriesLeftPost,
  SocialSeriesLeftReason,
  SocialSeriesShownPost,
} from '@/lib/api-client'
import type { Workspace } from '@/components/workspace-provider'
import { postActions } from './socials-status'

// modules/socials/campaigns.py CAMPAIGN_NAME_MAX_CHARS.
export const SOCIAL_CAMPAIGN_NAME_MAX_CHARS = 200

export const APPROVAL_MODE_LABELS: Record<SocialCampaignApprovalMode, string> = {
  per_post: 'Post by post',
  series: 'As a series',
}

/** Why a series approval left a post unapproved, as the result shows it. */
export const SERIES_LEFT_LABELS: Record<SocialSeriesLeftReason, string> = {
  changed: 'Changed since you saw it',
  unsourced: 'Unsourced claims to confirm',
  not_waiting: 'Not waiting for approval',
  not_in_campaign: 'Not in this campaign',
  not_shown: 'Not shown to you',
  not_in_batch: 'Not in this batch',  // PRD-251C (C2): approving a plan's week, a post of another batch
}

export const SERIES_OFF_MESSAGE =
  'Series approval is off for this workspace, so each post is approved on its own.'
export const SERIES_PER_POST_MESSAGE = 'This campaign approves post by post. Set it to "As a series" to approve it at once.'
export const SERIES_NOTHING_WAITING_MESSAGE = 'No post of this campaign is waiting for approval.'

export interface SeriesPlan {
  /** Waiting for approval: each is approved at the version on screen. */
  approve: SocialPost[]
  /** Any other status: not covered, and a later change needs its own approval. */
  notCovered: SocialPost[]
}

/** What approving the series now would approve, and what it would leave. */
export function seriesPlan(posts: ReadonlyArray<SocialPost>): SeriesPlan {
  return {
    approve: posts.filter((post) => post.status === 'needs_approval'),
    notCovered: posts.filter((post) => post.status !== 'needs_approval'),
  }
}

/** The request's list: each post with the content_hash of the version shown (D6). */
export function shownPosts(posts: ReadonlyArray<SocialPost>, overrideUnsourced = false): SocialSeriesShownPost[] {
  return posts.map((post) => ({
    post_id: post.id,
    content_hash: post.content_hash,
    override_unsourced: overrideUnsourced,
  }))
}

/** Why "Approve series" is not offered now, or null when it is. */
export function seriesBlockedReason(
  seriesOn: boolean,
  campaign: Pick<SocialCampaign, 'approval_mode'>,
  plan: SeriesPlan,
): string | null {
  if (!seriesOn) return SERIES_OFF_MESSAGE
  if (campaign.approval_mode !== 'series') return SERIES_PER_POST_MESSAGE
  if (plan.approve.length === 0) return SERIES_NOTHING_WAITING_MESSAGE
  return null
}

/** Whoever may approve a single post may approve a series (socials:approve). */
export function canApproveSeries(role: Workspace['role'] | undefined): boolean {
  return postActions(role, 'needs_approval').review
}

/** The posts a series approval left for their unsourced claims, which a second
 * confirmation may approve with override_unsourced: at the versions shown. */
export function unsourcedToConfirm(
  left: ReadonlyArray<SocialSeriesLeftPost>,
  shown: ReadonlyArray<SocialPost>,
): SocialPost[] {
  const ids = new Set(left.filter((entry) => entry.reason === 'unsourced').map((entry) => entry.post_id))
  return shown.filter((post) => ids.has(post.id))
}
