'use client'

/**
 * PRD-261 — the hero's "Looks and screens": one toggle chip per view (storefront-views.ts), with a
 * small thumbnail of its screen. Choosing one swaps the browser frame's and the phone frame's
 * screenshots together. A listing with one view has nothing to choose, so it shows no chips.
 */
import { useId } from 'react'

import { cn } from '@/lib/utils'

import { LOOKS_HEADING } from './copy'
import type { StorefrontView } from './storefront-views'

interface LooksChipsProps {
  readonly views: readonly StorefrontView[]
  readonly look: number
  readonly onLook: (look: number) => void
}

function LookChip({ view, pressed, onPick }: { view: StorefrontView; pressed: boolean; onPick: () => void }) {
  const thumbnail = view.web ?? view.phone
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={onPick}
      className={cn(
        'inline-flex min-h-11 items-center gap-2 rounded-xl border py-1 pl-1 pr-3.5 text-sm text-foreground',
        'ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
        pressed ? 'border-primary bg-primary/10' : 'border-border bg-background/50 hover:border-primary/40',
      )}
    >
      {thumbnail && (
        // A theme file on the app's own origin (A-08b proxy); next/image would re-encode it.
        // eslint-disable-next-line @next/next/no-img-element
        <img src={thumbnail} alt="" loading="lazy" className="h-[34px] w-[52px] rounded-lg object-cover" />
      )}
      {view.label}
    </button>
  )
}

export function LooksChips({ views, look, onLook }: LooksChipsProps) {
  const headingId = useId()
  if (views.length < 2) return null
  return (
    <div className="space-y-2.5">
      <p id={headingId} className="text-sm font-semibold text-muted-foreground">
        {LOOKS_HEADING}
      </p>
      <div role="group" aria-labelledby={headingId} className="flex flex-wrap gap-2" data-testid="storefront-looks">
        {views.map((view, index) => (
          <LookChip key={view.key} view={view} pressed={index === look} onPick={() => onLook(index)} />
        ))}
      </div>
    </div>
  )
}
