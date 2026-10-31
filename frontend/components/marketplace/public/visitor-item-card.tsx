'use client'

/**
 * PRD-270 US-270-08b (§5.7, K20) — one item of a visitor's read-only tab (an agent, a Playbook, a
 * package, an application, a plugin, a skill or an LLM): its icon, name, publisher, description,
 * category and first tags, as one button that opens its detail. Every value is the public read's
 * display field, rendered as text; an icon is drawn only as an `https` logo (the app's img-src
 * decides where from) or a short text such as an emoji, else the tab's own icon.
 */
import type { LucideIcon } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { ToolLogo } from '@/components/ui/tool-logo'
import { HTTPS_PREFIX } from '@/lib/marketplace/public-items'
import type { PublicItem } from '@/lib/marketplace/public-items'

import { byPublisher, itemDetails } from './copy'

const CARD_TAGS = 3
const ICON_SIZE = 40

export function ItemIcon({ item, fallback: Fallback }: { item: PublicItem; fallback: LucideIcon }) {
  if (item.icon?.startsWith(HTTPS_PREFIX)) return <ToolLogo logo={item.icon} name={item.name} size={ICON_SIZE} />
  if (item.icon) {
    return (
      <span className="flex h-10 w-10 items-center justify-center text-2xl" aria-hidden>
        {item.icon}
      </span>
    )
  }
  return <Fallback className="h-10 w-10 shrink-0 text-primary" aria-hidden />
}

interface VisitorItemCardProps {
  readonly item: PublicItem
  readonly icon: LucideIcon
  readonly onDetails: (key: string) => void
}

export function VisitorItemCard({ item, icon, onDetails }: VisitorItemCardProps) {
  return (
    <button
      type="button"
      aria-label={itemDetails(item.name)}
      data-testid={`visitor-item-${item.key}`}
      onClick={() => onDetails(item.key)}
      className="flex h-full flex-col gap-3 rounded-2xl border border-border bg-card p-5 text-left text-foreground ring-offset-background transition-colors hover:border-primary/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
    >
      <span className="flex items-center gap-3">
        <ItemIcon item={item} fallback={icon} />
        <span className="flex min-w-0 flex-col">
          <span className="truncate text-base font-semibold">{item.name}</span>
          {item.publisher && <span className="truncate text-xs text-muted-foreground">{byPublisher(item.publisher)}</span>}
        </span>
      </span>
      {item.description && <span className="line-clamp-2 text-sm text-muted-foreground">{item.description}</span>}
      <span className="mt-auto flex flex-wrap gap-1.5">
        {item.category && <Badge variant="outline">{item.category}</Badge>}
        {item.tags.slice(0, CARD_TAGS).map((tag, index) => (
          <Badge key={`${index}:${tag}`} variant="secondary">
            {tag}
          </Badge>
        ))}
      </span>
    </button>
  )
}
