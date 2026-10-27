'use client'

/**
 * P256-FIX-SOC-2 (F433) — the Queue's "Waits for you": a plan's drafts the render refused
 * (a figure the topic did not give, a required photo), each with what to fill in. Read-only:
 * a row opens the post's editor, where the owner fills the fields and renders it; nothing
 * here approves it, and the week's approval never sends it.
 */
import type { SocialPost } from '@/lib/api-client'
import { CHANNEL_BADGE } from './socials-calendar-chip'
import { formatLabel, postBadges } from './socials-calendar-model'
import { inputAsked } from './queue-model'

export const WAITS_FOR_YOU = 'Waits for you'

interface QueueWaitingProps {
  posts: ReadonlyArray<SocialPost>
  /** Opens a post in the editor. */
  onEdit?: (postId: string) => void
}

export function QueueWaiting({ posts, onEdit }: QueueWaitingProps) {
  if (posts.length === 0) return null
  return (
    <section aria-label={WAITS_FOR_YOU} className="flex flex-col gap-2.5">
      <h2 className="text-[11.5px] font-semibold uppercase tracking-[.07em] text-muted-foreground">{WAITS_FOR_YOU}</h2>
      {posts.map((post) => (
        <button
          key={post.id}
          type="button"
          onClick={() => onEdit?.(post.id)}
          disabled={!onEdit}
          className="flex w-full flex-col gap-1.5 rounded-xl border border-dashed border-border bg-card px-3.5 py-3 text-left"
        >
          <span className="flex items-center gap-1.5">
            {postBadges(post).map((badge) => (
              <span key={badge} className={CHANNEL_BADGE}>{badge}</span>
            ))}
            <span className="text-[12.5px] text-muted-foreground">{formatLabel(post)}</span>
          </span>
          <span className="text-[15px] font-semibold text-foreground">{post.title}</span>
          <span className="text-[12.5px] text-muted-foreground">{inputAsked(post)}</span>
        </button>
      ))}
    </section>
  )
}
