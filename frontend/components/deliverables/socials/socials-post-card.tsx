'use client'

/**
 * PRD-251 S2.1 — one post as the list and the board show it: its title and when
 * it last changed. Selecting it opens the post's detail, where its actions are;
 * a card never moves a post itself.
 */
import { formatDistanceToNow } from 'date-fns'

import { cn } from '@/lib/utils'
import type { SocialPost } from '@/lib/api-client'

function updatedAgo(post: SocialPost): string {
  try {
    return formatDistanceToNow(new Date(post.updated_at || post.created_at), { addSuffix: true })
  } catch {
    return ''
  }
}

interface SocialsPostCardProps {
  post: SocialPost
  selected: boolean
  onSelect: (postId: string) => void
}

export function SocialsPostCard({ post, selected, onSelect }: SocialsPostCardProps) {
  return (
    <button
      type="button"
      onClick={() => onSelect(post.id)}
      aria-pressed={selected}
      className={cn(
        'flex w-full flex-col items-start gap-0.5 rounded-lg border border-border bg-card px-3 py-2 text-left transition',
        'hover:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/40',
        selected && 'border-primary/60',
      )}
    >
      <span className="max-w-full break-words text-sm font-medium text-foreground">{post.title}</span>
      <span className="text-xs text-muted-foreground">Updated {updatedAgo(post)}</span>
    </button>
  )
}
