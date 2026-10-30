'use client'

/**
 * PRD-261 — the hero's Previous / Next arrows and its dots, one per listing the search shows. Real
 * buttons with labels; the dot of the listing in the hero is `aria-current`. Previous and Next wrap
 * around (use-storefront-selection.ts). One listing has nowhere to go, so it shows none of them.
 */
import { ChevronLeft, ChevronRight } from 'lucide-react'

import type { ThemeListing } from '@/lib/api-client'
import { cn } from '@/lib/utils'

import { listingName } from '../listing-format'
import { NEXT_THEME, PREVIOUS_THEME, showTheme } from './copy'

const ARROW =
  'absolute top-1/2 flex h-10 w-10 -translate-y-1/2 items-center justify-center rounded-full border border-border bg-background/70 text-foreground hover:bg-background sm:h-12 sm:w-12'
const FOCUS = 'ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2'

interface HeroNavProps {
  readonly listings: readonly ThemeListing[]
  /** The hero's listing among them; -1 when the search leaves it out. */
  readonly index: number
  readonly onSelect: (listingId: string) => void
  readonly onStep: (delta: number) => void
}

function Dot({ listing, current, onSelect }: { listing: ThemeListing; current: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      aria-label={showTheme(listingName(listing))}
      aria-current={current ? 'true' : undefined}
      onClick={onSelect}
      className={cn('flex h-8 w-8 items-center justify-center rounded-full', FOCUS)}
    >
      <span
        className={cn(
          'block h-1.5 rounded-full motion-safe:transition-all',
          current ? 'w-[22px] bg-foreground' : 'w-1.5 bg-foreground/40',
        )}
      />
    </button>
  )
}

export function HeroNav({ listings, index, onSelect, onStep }: HeroNavProps) {
  if (listings.length < 2) return null
  return (
    <>
      <button type="button" aria-label={PREVIOUS_THEME} onClick={() => onStep(-1)} className={cn(ARROW, FOCUS, 'left-2 sm:left-5')}>
        <ChevronLeft className="h-5 w-5" aria-hidden />
      </button>
      <button type="button" aria-label={NEXT_THEME} onClick={() => onStep(1)} className={cn(ARROW, FOCUS, 'right-2 sm:right-5')}>
        <ChevronRight className="h-5 w-5" aria-hidden />
      </button>
      <div className="absolute bottom-2 left-1/2 flex max-w-[80%] -translate-x-1/2 flex-wrap justify-center" data-testid="storefront-dots">
        {listings.map((listing, at) => (
          <Dot
            key={listing.listingId}
            listing={listing}
            current={at === index}
            onSelect={() => onSelect(listing.listingId)}
          />
        ))}
      </div>
    </>
  )
}
