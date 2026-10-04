'use client'

/**
 * 3 Oct 2026 (Gerard: "I might have something I want out ASAP") — Approve & post now: the
 * approve of the version on screen (D6, its content_hash), then Publish now, after one
 * confirmation that names the channels. A post with unsourced claims goes through Approve,
 * whose second confirmation names them; a 409 says the post changed and reloads it.
 */
import { useState } from 'react'
import { Send } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { SocialPost } from '@/lib/api-client'
import { useApproveSocialPost } from '@/hooks/use-socials-api'
import { usePublishSocialPostNow } from '@/hooks/use-socials-publish-api'
import { channelNames } from './socials-publish-controls'
import { unsourcedClaims } from './socials-review'

export const POST_NOW_LABEL = 'Approve & post now'
export const CLAIMS_FIRST = 'Its claims need a source or your confirmation first: use Approve.'

export function postNowQuestion(post: Pick<SocialPost, 'targets'>): string {
  return `Approve and post now to ${channelNames(post) || 'its channels'}? It goes live at once.`
}

export function ApproveAndPostNow({ post, disabled }: { post: SocialPost; disabled: boolean }) {
  const [asking, setAsking] = useState(false)
  const approve = useApproveSocialPost({ onStale: () => setAsking(false), onUnsourced: () => setAsking(false) })
  const publish = usePublishSocialPostNow()
  const busy = approve.isLoading || publish.isLoading
  const claimsFirst = unsourcedClaims(post).length > 0
  const postNow = () => approve.mutate(
    { postId: post.id, contentHash: post.content_hash, overrideUnsourced: false },
    { onSuccess: () => publish.mutate({ postId: post.id }, { onSettled: () => setAsking(false) }) },
  )
  if (!asking) {
    return (
      <Button size="sm" variant="outline" onClick={() => setAsking(true)} disabled={disabled || busy || claimsFirst}
        title={claimsFirst ? CLAIMS_FIRST : undefined}>
        <Send className="mr-1.5 h-4 w-4" aria-hidden />
        {POST_NOW_LABEL}
      </Button>
    )
  }
  return (
    <div role="alertdialog" aria-label={POST_NOW_LABEL} className="flex w-full flex-wrap items-center gap-2 rounded-lg border border-border px-3 py-2">
      <span className="text-sm text-foreground">{postNowQuestion(post)}</span>
      <Button size="sm" onClick={postNow} disabled={busy}>Post now</Button>
      <Button size="sm" variant="ghost" onClick={() => setAsking(false)} disabled={busy}>Cancel</Button>
    </div>
  )
}
