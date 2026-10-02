'use client'

/**
 * PRD-251B US-B107 — the Queue view: the posts waiting for approval, and the selected one
 * in the PRD-251 approval view (SocialsPostDetail: the exact media, each channel's copy, the
 * sources, and the approver's actions). US-B111 adds the day grouping, the slot line and
 * another take.
 */
import { useMemo } from 'react'
import { Inbox } from 'lucide-react'

import { cn } from '@/lib/utils'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { SocialsPostDetail } from '../socials-post-detail'

export const QUEUE_STATUS = 'needs_approval'

/** The posts waiting for approval, the earliest slot first, then the oldest. */
export function queuedPosts(posts: ReadonlyArray<SocialPost>): SocialPost[] {
  const slot = (post: SocialPost) => post.planned_for ?? post.scheduled_for ?? '9999'
  return posts
    .filter((post) => post.status === QUEUE_STATUS)
    .sort((a, b) => slot(a).localeCompare(slot(b)) || a.created_at.localeCompare(b.created_at))
}

export function queueHeading(count: number): string {
  if (count === 0) return 'All caught up'
  return count === 1 ? '1 post needs you' : `${count} posts need you`
}

interface SocialsQueueProps {
  role: Workspace['role']
  posts: ReadonlyArray<SocialPost>
  selectedId: string | null
  onSelect: (postId: string) => void
}

export function SocialsQueue({ role, posts, selectedId, onSelect }: SocialsQueueProps) {
  const queued = useMemo(() => queuedPosts(posts), [posts])
  const selected = queued.find((post) => post.id === selectedId) ?? queued[0] ?? null

  return (
    <section aria-label="Queue" className="space-y-4">
      <h2 className="font-serif text-[26px] font-normal leading-[1.1] tracking-[-0.01em] text-foreground">
        {queueHeading(queued.length)}
      </h2>
      {queued.length === 0 ? (
        <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border px-6 py-10 text-center">
          <Inbox className="h-7 w-7 text-muted-foreground" aria-hidden />
          <p className="text-sm text-muted-foreground">Nothing is waiting for your approval.</p>
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
          <ul aria-label="Waiting for approval" className="space-y-2">
            {queued.map((post) => (
              <li key={post.id}>
                <button
                  type="button"
                  aria-pressed={post.id === selected?.id}
                  onClick={() => onSelect(post.id)}
                  className={cn(
                    'flex w-full flex-col items-start gap-0.5 rounded-xl border bg-card px-3 py-2.5 text-left',
                    post.id === selected?.id ? 'border-accent ring-1 ring-accent' : 'border-border',
                  )}
                >
                  <span className="max-w-full truncate text-sm font-medium text-foreground">{post.title}</span>
                  <span className="font-mono text-[10.5px] text-muted-foreground">{post.format ?? 'No format yet'}</span>
                </button>
              </li>
            ))}
          </ul>
          {selected && <SocialsPostDetail post={selected} role={role} />}
        </div>
      )}
    </section>
  )
}
