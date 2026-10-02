'use client'

/**
 * PRD-251B US-B111 — the Queue's list (Queue.dc.html): the posts waiting for approval by the
 * day of their slot, each with its channel badges, its time, "Needs you", its title and its
 * format.
 */
import { cn } from '@/lib/utils'
import type { SocialPost } from '@/lib/api-client'
import { CHANNEL_BADGE, CHIP_KEY, TONES } from './socials-calendar-chip'
import { formatLabel, postBadges, postTime } from './socials-calendar-model'
import { slotOfQueued, type QueueGroup } from './queue-model'

interface QueueListProps {
  groups: ReadonlyArray<QueueGroup>
  selectedId: string | null
  onSelect: (postId: string) => void
}

function QueueItem({ post, selected, onSelect }: { post: SocialPost; selected: boolean; onSelect: () => void }) {
  const slot = slotOfQueued(post)
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onSelect}
      className={cn(
        'flex w-full flex-col gap-1.5 rounded-xl border bg-card px-3.5 py-3 text-left',
        selected ? 'border-accent ring-1 ring-accent' : 'border-border',
      )}
    >
      <span className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5">
          {postBadges(post).map((badge) => (
            <span key={badge} className={CHANNEL_BADGE}>{badge}</span>
          ))}
          {slot && <span className="font-mono text-xs text-muted-foreground">{postTime(post, slot)}</span>}
        </span>
        <span className={cn('rounded-lg border px-2 py-0.5', TONES.review.chip)}>
          <span className={cn(CHIP_KEY, TONES.review.key)}>Needs you</span>
        </span>
      </span>
      <span className="text-[15px] font-semibold text-foreground">{post.title}</span>
      <span className="text-[12.5px] text-muted-foreground">{formatLabel(post)}</span>
    </button>
  )
}

export function QueueList({ groups, selectedId, onSelect }: QueueListProps) {
  return (
    <aside aria-label="Waiting for approval" className="flex flex-col gap-2.5">
      {groups.map((group) => (
        <section key={group.label} aria-label={group.label} className="flex flex-col gap-2.5">
          <h2 className="text-[11.5px] font-semibold uppercase tracking-[.07em] text-muted-foreground">{group.label}</h2>
          {group.posts.map((post) => (
            <QueueItem key={post.id} post={post} selected={post.id === selectedId} onSelect={() => onSelect(post.id)} />
          ))}
        </section>
      ))}
    </aside>
  )
}
