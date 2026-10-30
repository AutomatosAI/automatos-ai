'use client'

/**
 * PRD-261 — "All themes": a side-scrolling strip of every listing the search shows, one snap card
 * each (strip-card.tsx), ending on "Make your own". The arrows scroll it by a card (the Marketplace's
 * widgets idiom, app/marketplace/widgets/page.tsx); when the hero moves to a card that is out of
 * the strip's sight, the strip scrolls to it, sideways only, so the page never jumps. Scrolling is
 * smooth unless the person asks for reduced motion.
 */
import { useEffect, useId, useRef } from 'react'
import type { RefObject } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { usePrefersReducedMotion } from '@/hooks/use-media-preferences'
import type { ThemeListing } from '@/lib/api-client'

import { ALL_THEMES, SCROLL_BACK, SCROLL_ON, SLIDE_FOR_MORE } from './copy'
import { MakeYourOwnCard } from './make-your-own'
import { StripCard } from './strip-card'

/** One card and the gap after it (strip-card.tsx `sm:w-[360px]`, `gap-5`). */
const CARD_STEP_PX = 380
const SELECTED_CARD = '[aria-pressed="true"]'

interface StripProps {
  readonly listings: readonly ThemeListing[]
  readonly selectedId: string | null
  readonly onSelect: (listingId: string) => void
}

function scrollBehavior(reduced: boolean): ScrollBehavior {
  return reduced ? 'auto' : 'smooth'
}

/** Bring the hero's card into the strip's sight when it is outside it. */
function useSelectedInSight(scroller: RefObject<HTMLDivElement>, selectedId: string | null, reduced: boolean) {
  useEffect(() => {
    const strip = scroller.current
    const card = strip?.querySelector<HTMLElement>(SELECTED_CARD)
    if (!strip || !card) return
    const before = card.offsetLeft < strip.scrollLeft
    const after = card.offsetLeft + card.offsetWidth > strip.scrollLeft + strip.clientWidth
    if (before || after) strip.scrollTo?.({ left: card.offsetLeft, behavior: scrollBehavior(reduced) })
  }, [scroller, selectedId, reduced])
}

export function Strip({ listings, selectedId, onSelect }: StripProps) {
  const headingId = useId()
  const scroller = useRef<HTMLDivElement>(null)
  const reduced = usePrefersReducedMotion()
  useSelectedInSight(scroller, selectedId, reduced)
  const scrollStrip = (left: number) => scroller.current?.scrollBy?.({ left, behavior: scrollBehavior(reduced) })

  return (
    <section aria-labelledby={headingId} className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 id={headingId} className="text-xl font-semibold tracking-tight text-foreground">
          {ALL_THEMES}
        </h3>
        <div className="flex items-center gap-1">
          <span className="mr-2 text-sm text-muted-foreground">{SLIDE_FOR_MORE}</span>
          <Button variant="ghost" size="icon" aria-label={SCROLL_BACK} onClick={() => scrollStrip(-CARD_STEP_PX)}>
            <ChevronLeft className="h-4 w-4" aria-hidden />
          </Button>
          <Button variant="ghost" size="icon" aria-label={SCROLL_ON} onClick={() => scrollStrip(CARD_STEP_PX)}>
            <ChevronRight className="h-4 w-4" aria-hidden />
          </Button>
        </div>
      </div>
      <div
        ref={scroller}
        className="relative -mx-1 flex snap-x snap-mandatory gap-5 overflow-x-auto px-1 pb-4 pt-1"
        data-testid="theme-listing-strip"
      >
        {listings.map((listing) => (
          <StripCard
            key={listing.listingId}
            listing={listing}
            selected={listing.listingId === selectedId}
            onSelect={onSelect}
          />
        ))}
        <MakeYourOwnCard />
      </div>
    </section>
  )
}
