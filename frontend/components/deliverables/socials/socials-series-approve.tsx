'use client'

/**
 * PRD-251 S2.4 (D6) — "Approve series", the confirmation. It lists the posts it
 * will approve (each waiting for approval, at the version on screen when it
 * opened) and the posts it will not, and asks to confirm. The server approves
 * each as a single approval would, bound to that version's hash, and answers
 * which it approved and which it left, with why: this shows that answer. A post
 * left for its unsourced claims (D7) needs a second confirmation naming them.
 */
import { useState } from 'react'
import { Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { SocialCampaignWithPosts, SocialPost, SocialSeriesApproval } from '@/lib/api-client'
import { useApproveSocialSeries } from '@/hooks/use-socials-api'
import { SOCIAL_STATUS_LABELS } from './socials-status'
import { SERIES_LEFT_LABELS, seriesPlan, shownPosts, unsourcedToConfirm, type SeriesPlan } from './socials-series'

interface SeriesResultProps {
  result: SocialSeriesApproval
  toConfirm: SocialPost[]
  busy: boolean
  onConfirmUnsourced: () => void
  onClose: () => void
}

function SeriesResult({ result, toConfirm, busy, onConfirmUnsourced, onClose }: SeriesResultProps) {
  const unsourced = result.left.filter((entry) => entry.reason === 'unsourced')
  return (
    <div role="status" aria-label="Series approval result" className="space-y-2 rounded-lg border border-border bg-card/40 px-3 py-3">
      <p className="text-sm font-medium text-foreground">
        {result.approved.length} {result.approved.length === 1 ? 'post' : 'posts'} approved.
      </p>
      {result.left.length > 0 && (
        <ul aria-label="Left unapproved" className="space-y-1 text-sm">
          {result.left.map((entry) => (
            <li key={entry.post_id}>
              <span className="font-medium text-foreground">{entry.title ?? entry.post_id}</span>
              {' — '}
              <span className="text-muted-foreground">{SERIES_LEFT_LABELS[entry.reason]}: {entry.message}</span>
            </li>
          ))}
        </ul>
      )}
      {toConfirm.length > 0 && (
        <div role="alertdialog" aria-label="Approve with unsourced claims" className="space-y-2 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2">
          <p className="text-sm text-destructive">
            These claims have no source:{' '}
            {unsourced.map((entry) => `${entry.title ?? entry.post_id} (${(entry.claims ?? []).join(', ')})`).join('; ')}.
            Approve them anyway?
          </p>
          <div className="flex justify-end">
            <Button type="button" size="sm" variant="destructive" onClick={onConfirmUnsourced} disabled={busy}>
              Approve with unsourced claims
            </Button>
          </div>
        </div>
      )}
      <div className="flex justify-end">
        <Button type="button" size="sm" variant="ghost" onClick={onClose}>
          Done
        </Button>
      </div>
    </div>
  )
}

interface SeriesConfirmProps {
  plan: SeriesPlan
  busy: boolean
  onConfirm: () => void
  onCancel: () => void
}

function SeriesConfirm({ plan, busy, onConfirm, onCancel }: SeriesConfirmProps) {
  const count = plan.approve.length
  return (
    <div role="alertdialog" aria-label="Approve series" className="space-y-3 rounded-lg border border-primary/40 bg-primary/5 px-3 py-3">
      <p className="text-sm text-foreground">
        This approves {count} {count === 1 ? 'post' : 'posts'}, each exactly as you see it now:
      </p>
      <ul aria-label="Will be approved" className="list-disc space-y-0.5 pl-5 text-sm">
        {plan.approve.map((post) => (
          <li key={post.id}>{post.title}</li>
        ))}
      </ul>
      {plan.notCovered.length > 0 && (
        <>
          <p className="text-sm text-muted-foreground">Not covered, and left as they are:</p>
          <ul aria-label="Not covered" className="list-disc space-y-0.5 pl-5 text-sm text-muted-foreground">
            {plan.notCovered.map((post) => (
              <li key={post.id}>
                {post.title} · {SOCIAL_STATUS_LABELS[post.status]}
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="text-xs text-muted-foreground">
        A post that changes before you confirm is left for its own review. A post added to the campaign, or edited, later
        needs its own approval.
      </p>
      <div className="flex justify-end gap-2">
        <Button type="button" size="sm" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="button" size="sm" onClick={onConfirm} disabled={busy}>
          {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />}
          Approve {count} {count === 1 ? 'post' : 'posts'}
        </Button>
      </div>
    </div>
  )
}

interface SocialsSeriesApproveProps {
  campaign: SocialCampaignWithPosts
  onClose: () => void
}

export function SocialsSeriesApprove({ campaign, onClose }: SocialsSeriesApproveProps) {
  // The versions on screen when the confirmation opened: the approval binds to them (D6).
  const [plan] = useState(() => seriesPlan(campaign.posts))
  const [result, setResult] = useState<SocialSeriesApproval | null>(null)
  const approve = useApproveSocialSeries()

  const send = (posts: SocialPost[], overrideUnsourced: boolean) =>
    approve.mutate(
      { campaignId: campaign.id, posts: shownPosts(posts, overrideUnsourced) },
      { onSuccess: (answer) => setResult(answer) },
    )

  if (result) {
    const toConfirm = unsourcedToConfirm(result.left, plan.approve)
    return (
      <SeriesResult
        result={result}
        toConfirm={toConfirm}
        busy={approve.isLoading}
        onConfirmUnsourced={() => send(toConfirm, true)}
        onClose={onClose}
      />
    )
  }
  return <SeriesConfirm plan={plan} busy={approve.isLoading} onConfirm={() => send(plan.approve, false)} onCancel={onClose} />
}
