'use client'

/**
 * PRD-251B US-B107 — the Socials Studio's header (docs/PRDS/prd251b-reference/Nav.dc.html):
 * the sub-navigation Calendar · Queue (how many posts need you) · Plans · Brand kit, and the
 * two actions New plan and New post. Every item is a 44 px target; the active one is
 * underlined in the accent (TOKENS.md). Brand kit opens the brand kit dialog in place, for a
 * role that edits it; the actions are for a role that drafts posts.
 */
import { Plus } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { SocialsStudioView } from './studio-route'

const VIEW_ITEMS: ReadonlyArray<{ view: SocialsStudioView; label: string }> = [
  { view: 'calendar', label: 'Calendar' },
  { view: 'queue', label: 'Queue' },
  { view: 'plans', label: 'Plans' },
]

const ITEM = 'inline-flex min-h-[44px] items-center gap-2 border-b-2 px-3.5 text-sm font-medium transition-colors'
const ON = 'border-accent text-foreground'
const OFF = 'border-transparent text-muted-foreground hover:text-foreground'
/** The mockup's amber count: the warning token, ink text (TOKENS.md). */
const COUNT =
  'inline-flex h-5 min-w-[20px] items-center justify-center rounded-full bg-[hsl(var(--warning))] px-1.5 text-[11.5px] font-semibold text-accent-foreground'
/** Owner choice 2: the Studio's primary actions are orange, as the mockup has them. */
export const PRIMARY_ACTION = 'bg-accent text-accent-foreground hover:bg-accent/90'

function QueueCount({ count }: { count: number }) {
  if (count <= 0) return null
  return (
    <>
      <span className={COUNT} aria-hidden>
        {count}
      </span>
      <span className="sr-only">{count === 1 ? '1 post needs you' : `${count} posts need you`}</span>
    </>
  )
}

interface SocialsStudioNavProps {
  view: SocialsStudioView
  /** Posts waiting for approval: the Queue's badge. */
  queueCount: number
  canAuthor: boolean
  canBrand: boolean
  onView: (view: SocialsStudioView) => void
  onBrandKit: () => void
  onNewPlan: () => void
  onNewPost: () => void
}

export function SocialsStudioNav(props: SocialsStudioNavProps) {
  const { view, queueCount, canAuthor, canBrand, onView, onBrandKit, onNewPlan, onNewPost } = props
  return (
    <div className="socials-studio-head flex flex-wrap items-end justify-between gap-x-4 gap-y-2 border-b border-border">
      <nav aria-label="Socials" className="flex flex-wrap gap-0.5">
        {VIEW_ITEMS.map((item) => (
          <button
            key={item.view}
            type="button"
            className={cn(ITEM, view === item.view ? ON : OFF)}
            aria-current={view === item.view ? 'page' : undefined}
            onClick={() => onView(item.view)}
          >
            {item.label}
            {item.view === 'queue' && <QueueCount count={queueCount} />}
          </button>
        ))}
        {canBrand && (
          <button type="button" className={cn(ITEM, OFF)} aria-haspopup="dialog" onClick={onBrandKit}>
            Brand kit
          </button>
        )}
      </nav>
      {canAuthor && (
        <div className="socials-studio-actions flex flex-wrap items-center gap-2 pb-2">
          <Button size="sm" variant="secondary" onClick={onNewPlan}>
            New plan
          </Button>
          <Button size="sm" className={PRIMARY_ACTION} onClick={onNewPost}>
            <Plus className="mr-1.5 h-4 w-4" aria-hidden />
            New post
          </Button>
        </div>
      )}
    </div>
  )
}
