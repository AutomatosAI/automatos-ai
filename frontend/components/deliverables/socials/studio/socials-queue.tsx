'use client'

/**
 * PRD-251B US-B111 — the Queue (Queue.dc.html): the posts waiting for approval, by the day of
 * their slot, today first; the heading says how many need you today. The selected post shows
 * in the pane exactly as it will go out, with the approver's actions. "Approve all shown"
 * approves each shown post by the hash on screen, offered only while the workspace's series
 * approval is on (PRD-251 D6); nothing is approved implicitly.
 *
 * PRD-251C US-C204: a weekly or monthly plan's batch comes first, as one section in slot order
 * ("Week of 19 Oct · Countdown · 7 posts"), with "Approve the week" (or month) for whoever may
 * approve: each post by its hash on screen, whatever the series switch says (O2). A post that
 * changed since it was shown is left and named.
 */
import { useMemo } from 'react'
import { Inbox, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { useWorkspace, type Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useSocialCampaigns } from '@/hooks/use-socials-api'
import { useApproveBatch, useApproveShown } from '@/hooks/use-socials-queue'
import { hasChannels } from '../socials-review'
import { postActions } from '../socials-status'
import { QueueList } from './queue-list'
import { QueuePane } from './queue-pane'
import { inBatch, isMonthBatch, queueBatches, queueGroups, queueHeading, waitingToday, type QueueBatch, type QueueGroup } from './queue-model'
import { SocialsPostDetail } from '../socials-post-detail'

export { queuedPosts } from './queue-model'

export const QUEUE_HINT = 'Approve each one before its slot. Anything not approved by then is skipped and nothing posts.'
export const APPROVE_ALL = 'Approve all shown'
export const APPROVE_WEEK = 'Approve the week'
export const APPROVE_MONTH = 'Approve the month'

function ApproveBatch({ batch }: { batch: QueueBatch }) {
  const approve = useApproveBatch()
  // F256: a post with no channel publishes nothing: the batch's approval leaves it out.
  const approvable = batch.posts.filter(hasChannels)
  return (
    <Button type="button" size="sm" variant="secondary" disabled={approve.isLoading || approvable.length === 0}
      onClick={() => approve.mutate({ planId: batch.planId, batchKey: batch.batchKey, posts: approvable })}>
      {approve.isLoading && <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden />}
      {isMonthBatch(batch.batchKey) ? APPROVE_MONTH : APPROVE_WEEK}
    </Button>
  )
}

function batchAction(group: QueueGroup) {
  return 'batchKey' in group ? <ApproveBatch batch={group as QueueBatch} /> : null
}

interface SocialsQueueProps {
  role: Workspace['role']
  posts: ReadonlyArray<SocialPost>
  selectedId: string | null
  onSelect: (postId: string) => void
  /** Opens a post in the editor (its channels, words and look). */
  onEdit?: (postId: string) => void
}

export function SocialsQueue({ role, posts, selectedId, onSelect, onEdit }: SocialsQueueProps) {
  const { workspace } = useWorkspace()
  const { data: campaignData } = useSocialCampaigns()
  const campaigns = useMemo(() => campaignData?.campaigns ?? [], [campaignData])
  const approveAll = useApproveShown(campaigns)
  const now = new Date()
  const groups = useMemo<QueueGroup[]>(() => [
    ...queueBatches(posts, campaigns),
    ...queueGroups(posts.filter((post) => !inBatch(post)), new Date()),
  ], [posts, campaigns])
  const shown = groups.flatMap((group) => group.posts)
  // F256: a post with no channel publishes nothing: Approve all shown leaves it out.
  const approvable = shown.filter(hasChannels)
  const selected = shown.find((post) => post.id === selectedId) ?? shown[0] ?? null
  const today = waitingToday(posts, now)
  const seriesOn = workspace?.socials?.series_approval === true
  const campaignName = campaigns.find((c) => c.id === selected?.campaign_id)?.name ?? null
  // Who may approve (socials:approve): the others read the post as it is.
  const reviewer = postActions(role, 'needs_approval').review

  return (
    <section aria-label="Queue" className="flex flex-col gap-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <h1 className="font-serif text-[32px] font-normal leading-[1.1] tracking-[-0.01em] text-foreground max-md:text-[26px]">
            {queueHeading(today)}
          </h1>
          <p className="m-0 max-w-[760px] text-[12.5px] text-muted-foreground">{QUEUE_HINT}</p>
        </div>
        {seriesOn && reviewer && approvable.length > 1 && (
          <Button type="button" variant="secondary" onClick={() => approveAll.mutate(approvable)} disabled={approveAll.isLoading}>
            {approveAll.isLoading && <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden />}
            {APPROVE_ALL}
          </Button>
        )}
      </div>
      {shown.length === 0 ? (
        <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border px-6 py-10 text-center">
          <Inbox className="h-7 w-7 text-muted-foreground" aria-hidden />
          <p className="text-sm text-muted-foreground">Nothing is waiting for your approval.</p>
        </div>
      ) : (
        <div className="grid items-start gap-5 lg:grid-cols-[320px_minmax(0,1fr)]">
          <QueueList groups={groups} selectedId={selected?.id ?? null} onSelect={onSelect} action={reviewer ? batchAction : undefined} />
          {selected && (reviewer
            ? <QueuePane post={selected} campaignName={campaignName} now={now} role={role}
                onEdit={onEdit ? () => onEdit(selected.id) : undefined} />
            : <SocialsPostDetail post={selected} role={role} />)}
        </div>
      )}
    </section>
  )
}
