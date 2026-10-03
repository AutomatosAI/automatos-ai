'use client'

/**
 * PRD-251B US-B111 — the Queue's pane: the post to approve, shown exactly (PRD-251's approval
 * view: the media through FilePreview, each channel's copy with its count, the claims and
 * their sources), with when it publishes, and the approver's actions: "Approve · publishes
 * HH:MM" (the approve that schedules it into its slot, US-B105), Request changes, which sends
 * it back to Auto with what should change, Make another take, and Reject.
 */
import { Loader2, RefreshCw } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { SocialPost } from '@/lib/api-client'
import { useRetakeSocialPost, useSendBackToAuto } from '@/hooks/use-socials-queue'
import { SocialsPostEvidence } from '../socials-post-evidence'
import { SocialsPostReview } from '../socials-post-review'
import { postTime } from './socials-calendar-model'
import { approveLabel, metaLine, slotOfQueued, timeLeft } from './queue-model'
import { CARD, Hint } from './editor-ui'

export const EXACT_HINT = 'You approve exactly this file and this copy. Changing either afterwards sends the post back here.'
export const SEND_BACK_HINT = 'Auto redrafts it and it comes back here.'

interface QueuePaneProps {
  post: SocialPost
  campaignName: string | null
  now: Date
}

export function QueuePane({ post, campaignName, now }: QueuePaneProps) {
  const retake = useRetakeSocialPost()
  const sendBack = useSendBackToAuto()
  const slot = slotOfQueued(post)
  const anotherTake = (
    <Button size="sm" variant="outline" onClick={() => retake.mutate(post.id)} disabled={retake.isLoading}>
      {retake.isLoading ? <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden /> : <RefreshCw className="mr-1.5 h-4 w-4" aria-hidden />}
      Make another take
    </Button>
  )
  return (
    <section aria-label="Post to approve" className={CARD}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <span className="text-[12.5px] text-muted-foreground">{metaLine(post, campaignName)}</span>
          <h2 className="font-serif text-[26px] font-normal leading-[1.1] text-foreground">{post.title}</h2>
        </div>
        {slot && (
          <Badge variant="outline" className="h-6 rounded-full px-2.5">
            Publishes {postTime(post, slot)} · {timeLeft(slot, now)}
          </Badge>
        )}
      </div>
      <SocialsPostEvidence post={post} />
      <Hint>{EXACT_HINT}</Hint>
      <SocialsPostReview
        post={post}
        dirty={false}
        approveLabel={approveLabel(post)}
        extra={anotherTake}
        sendBack={{
          label: 'What should change?',
          submit: 'Send back to Auto',
          hint: SEND_BACK_HINT,
          busy: sendBack.isLoading,
          onSend: (comment) => sendBack.mutate({ postId: post.id, comment }),
        }}
      />
    </section>
  )
}
