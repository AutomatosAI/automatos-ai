'use client'

/**
 * PRD-251B US-B111 — the Queue (Queue.dc.html): the posts waiting for approval, by the day of
 * their slot, today first; the heading says how many need you today. The selected post shows
 * in the pane exactly as it will go out, with the approver's actions. "Approve all shown"
 * approves each shown post by the hash on screen, offered only while the workspace's series
 * approval is on (PRD-251 D6); nothing is approved implicitly.
 */
import { useMemo } from 'react'
import { Inbox, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { useWorkspace, type Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useSocialCampaigns } from '@/hooks/use-socials-api'
import { useApproveShown } from '@/hooks/use-socials-queue'
import { hasChannels } from '../socials-review'
import { postActions } from '../socials-status'
import { QueueList } from './queue-list'
import { QueuePane } from './queue-pane'
import { queueGroups, queueHeading } from './queue-model'
import { SocialsPostDetail } from '../socials-post-detail'

export { queuedPosts } from './queue-model'

export const QUEUE_HINT = 'Approve each one before its slot. Anything not approved by then is skipped and nothing posts.'
export const APPROVE_ALL = 'Approve all shown'

interface SocialsQueueProps {
  role: Workspace['role']
  posts: ReadonlyArray<SocialPost>
  selectedId: string | null
  onSelect: (postId: string) => void
}

export function SocialsQueue({ role, posts, selectedId, onSelect }: SocialsQueueProps) {
  const { workspace } = useWorkspace()
  const { data: campaignData } = useSocialCampaigns()
  const campaigns = useMemo(() => campaignData?.campaigns ?? [], [campaignData])
  const approveAll = useApproveShown(campaigns)
  const now = new Date()
  const groups = useMemo(() => queueGroups(posts, new Date()), [posts])
  const shown = groups.flatMap((group) => group.posts)
  // F256: a post with no channel publishes nothing: Approve all shown leaves it out.
  const approvable = shown.filter(hasChannels)
  const selected = shown.find((post) => post.id === selectedId) ?? shown[0] ?? null
  const today = groups.find((group) => group.today)?.posts.length ?? 0
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
          <QueueList groups={groups} selectedId={selected?.id ?? null} onSelect={onSelect} />
          {selected && (reviewer
            ? <QueuePane post={selected} campaignName={campaignName} now={now} />
            : <SocialsPostDetail post={selected} role={role} />)}
        </div>
      )}
    </section>
  )
}
