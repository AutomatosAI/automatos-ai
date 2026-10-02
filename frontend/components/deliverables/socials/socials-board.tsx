'use client'

/**
 * PRD-251 S2.1 — the Board view: one column per status in SOCIAL_STATUS_ORDER,
 * each with its count and its posts newest first, from the same posts as the
 * List (`boardColumns` is the list's grouping with the empty statuses kept).
 *
 * The board is READ-ONLY. A post changes status only through the actions in its
 * detail, which the server checks (the status machine, the compare-and-set, 409
 * when the post changed): nothing here drags or drops. The columns scroll
 * sideways; on a phone they snap one by one (the compact region of globals.css).
 */
import { useMemo } from 'react'

import { badgeVariants } from '@/components/ui/badge'
import type { SocialPost } from '@/lib/api-client'
import { SocialsPostCard } from './socials-post-card'
import { boardColumns, type StatusGroup } from './socials-status'

interface SocialsBoardProps {
  posts: ReadonlyArray<SocialPost>
  selectedId: string | null
  onSelect: (postId: string) => void
}

interface BoardColumnProps {
  column: StatusGroup
  selectedId: string | null
  onSelect: (postId: string) => void
}

function BoardColumn({ column, selectedId, onSelect }: BoardColumnProps) {
  return (
    <section
      aria-label={`${column.label} column`}
      className="socials-board-col flex w-72 shrink-0 flex-col gap-2 rounded-xl border border-border/60 bg-card/20 p-2"
    >
      <h3 className="flex items-center gap-2 px-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {column.label}{' '}
        <span className={badgeVariants({ variant: 'secondary' })} data-testid="socials-status-count">
          {column.posts.length}
        </span>
      </h3>
      {column.posts.length === 0 ? (
        <p className="px-1 py-6 text-center text-xs text-muted-foreground">No posts</p>
      ) : (
        <ul className="space-y-1.5">
          {column.posts.map((post) => (
            <li key={post.id}>
              <SocialsPostCard post={post} selected={post.id === selectedId} onSelect={onSelect} />
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export function SocialsBoard({ posts, selectedId, onSelect }: SocialsBoardProps) {
  const columns = useMemo(() => boardColumns(posts), [posts])
  return (
    <div className="socials-board flex gap-3 overflow-x-auto pb-2" role="group" aria-label="Posts by status">
      {columns.map((column) => (
        <BoardColumn key={column.status} column={column} selectedId={selectedId} onSelect={onSelect} />
      ))}
    </div>
  )
}
