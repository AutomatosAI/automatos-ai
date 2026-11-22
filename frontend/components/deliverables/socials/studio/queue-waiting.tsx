'use client'

/**
 * P256-FIX-SOC-2 (F433) — the Queue's "Waits for you": a plan's drafts the render refused
 * (a figure the topic did not give, a required photo), each with what to fill in. Read-only:
 * a row opens the post's editor, where the owner fills the fields and renders it; nothing
 * here approves it, and the week's approval never sends it.
 *
 * "Approved, no time yet": approved posts with neither a planned slot nor a schedule, which
 * post nothing until a person gives them a time or publishes them now (queue-model.ts
 * `approvedWithNoTime`). The same rows: one opens the post's page, where its Schedule and
 * Publish now are (socials-publish-controls.tsx).
 */
import type { SocialPost } from '@/lib/api-client'
import { CHANNEL_BADGE } from './socials-calendar-chip'
import { formatLabel, postBadges } from './socials-calendar-model'
import { inputAsked } from './queue-model'

export const WAITS_FOR_YOU = 'Waits for you'
export const APPROVED_NO_TIME = 'Approved, no time yet'
export const APPROVED_NO_TIME_HINT = 'These post nothing until they get a time or are posted now.'

interface QueueAsideProps {
  label: string
  /** One line under the heading, if any. */
  hint?: string
  posts: ReadonlyArray<SocialPost>
  /** What each row says under its title, if anything. */
  line?: (post: SocialPost) => string
  /** Opens a post's page. */
  onOpen?: (postId: string) => void
}

/** A Queue section apart from the approvals: its heading, its hint and a row per post that opens the post. */
function QueueAside({ label, hint, posts, line, onOpen }: QueueAsideProps) {
  if (posts.length === 0) return null
  return (
    <section aria-label={label} className="flex flex-col gap-2.5">
      <h2 className="text-[11.5px] font-semibold uppercase tracking-[.07em] text-muted-foreground">{label}</h2>
      {hint && <p className="m-0 text-[12.5px] text-muted-foreground">{hint}</p>}
      {posts.map((post) => (
        <button
          key={post.id}
          type="button"
          onClick={() => onOpen?.(post.id)}
          disabled={!onOpen}
          className="flex w-full flex-col gap-1.5 rounded-xl border border-dashed border-border bg-card px-3.5 py-3 text-left"
        >
          <span className="flex items-center gap-1.5">
            {postBadges(post).map((badge) => (
              <span key={badge} className={CHANNEL_BADGE}>{badge}</span>
            ))}
            <span className="text-[12.5px] text-muted-foreground">{formatLabel(post)}</span>
          </span>
          <span className="text-[15px] font-semibold text-foreground">{post.title}</span>
          {line && <span className="text-[12.5px] text-muted-foreground">{line(post)}</span>}
        </button>
      ))}
    </section>
  )
}

interface QueueWaitingProps {
  posts: ReadonlyArray<SocialPost>
  /** Opens a post in the editor. */
  onEdit?: (postId: string) => void
}

export function QueueWaiting({ posts, onEdit }: QueueWaitingProps) {
  return <QueueAside label={WAITS_FOR_YOU} posts={posts} line={inputAsked} onOpen={onEdit} />
}

export function QueueApprovedNoTime({ posts, onEdit }: QueueWaitingProps) {
  return <QueueAside label={APPROVED_NO_TIME} hint={APPROVED_NO_TIME_HINT} posts={posts} onOpen={onEdit} />
}
