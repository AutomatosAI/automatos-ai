/**
 * DeliverableTags (7 Oct)
 * =======================
 *
 * A Deliverable's tags as chips on its panel: the tags the agent or the owner gave it,
 * and a board card's tags when the card produced it. Nothing renders when it has none.
 */

import { Tag } from 'lucide-react'

import { cn } from '@/lib/utils'

export function DeliverableTags({ tags, className }: { tags?: string[] | null; className?: string }) {
  if (!tags || tags.length === 0) return null
  return (
    <ul className={cn('flex flex-wrap items-center gap-1.5', className)} aria-label="Tags">
      {tags.map((tag) => (
        <li
          key={tag}
          className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] font-medium text-muted-foreground"
        >
          <Tag className="h-3 w-3" aria-hidden />
          {tag}
        </li>
      ))}
    </ul>
  )
}
