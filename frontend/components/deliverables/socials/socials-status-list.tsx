'use client'

/**
 * PRD-251 S0.5 — the List view: the posts grouped by status, what needs a person
 * first, each group with its count and its posts newest first (S2.1 moved it out
 * of socials-post-list.tsx, beside the Board view).
 */
import { badgeVariants } from '@/components/ui/badge'
import { SocialsPostCard } from './socials-post-card'
import type { StatusGroup } from './socials-status'

interface SocialsStatusListProps {
  groups: StatusGroup[]
  selectedId: string | null
  onSelect: (postId: string) => void
}

export function SocialsStatusList({ groups, selectedId, onSelect }: SocialsStatusListProps) {
  return (
    <div className="space-y-5">
      {groups.map((group) => (
        <section key={group.status} aria-label={`${group.label} posts`} className="space-y-2">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground">
            {group.label}{' '}
            <span className={badgeVariants({ variant: 'secondary' })} data-testid="socials-status-count">
              {group.posts.length}
            </span>
          </h3>
          <ul className="space-y-1.5">
            {group.posts.map((post) => (
              <li key={post.id}>
                <SocialsPostCard post={post} selected={post.id === selectedId} onSelect={onSelect} />
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}
