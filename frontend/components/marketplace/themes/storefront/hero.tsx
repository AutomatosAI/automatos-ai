'use client'

/**
 * PRD-261 — the storefront's hero stage: the listing the selection points at (use-storefront-
 * selection.ts), on a blurred backdrop of its current screenshot, with its words and install on one
 * side, its web and phone screens on the other, and Previous / Next and the dots over the stage.
 *
 * The listing's detail (C-04a `GET /api/marketplace/themes/{listingId}`) is fetched for the hero's
 * listing only, when it is shown: its presets give the looks (storefront-views.ts). Until it
 * answers, the hero shows the card's own cover, name and badges, and a failed detail leaves it
 * there. Once it answers, the detail's fresher state (installed, revoked) is what the hero shows.
 */
import { useMemo } from 'react'
import type { RefObject } from 'react'

import type { ThemeListing } from '@/lib/api-client'

import { listingName } from '../listing-format'
import { useThemeListing } from '../listing-hooks'
import { FEATURED_THEME } from './copy'
import { HeroDevices } from './hero-devices'
import { HeroNav } from './hero-nav'
import { HeroText } from './hero-text'
import { clampLook, storefrontViews } from './storefront-views'
import type { StorefrontSelection } from './use-storefront-selection'

interface HeroProps {
  readonly listing: ThemeListing
  /** The listings the search shows: the dots, and where Previous / Next go. */
  readonly shown: readonly ThemeListing[]
  readonly selection: StorefrontSelection
  readonly canManage: boolean
  readonly onDetails: (listingId: string) => void
  /** The stage, so an installed row's View can bring it into sight. */
  readonly stageRef: RefObject<HTMLElement>
}

function Backdrop({ src }: { src: string | null }) {
  return (
    <div className="pointer-events-none absolute inset-0" aria-hidden>
      {src && (
        // The hero's own screenshot again (already loading), blurred into the stage's backdrop.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt=""
          className="absolute -left-16 -top-16 h-[calc(100%+8rem)] w-[calc(100%+8rem)] max-w-none object-cover opacity-70 blur-3xl saturate-150"
        />
      )}
      <div className="absolute inset-0 bg-background/70" />
    </div>
  )
}

export function Hero({ listing, shown, selection, canManage, onDetails, stageRef }: HeroProps) {
  const detail = useThemeListing(listing.listingId)
  const current = detail.data ?? listing
  const views = useMemo(() => storefrontViews(current), [current])
  const look = clampLook(views, selection.look)
  const view = views[look]

  return (
    <section
      ref={stageRef}
      aria-label={FEATURED_THEME}
      data-testid="storefront-hero"
      className="relative scroll-mt-6 overflow-hidden rounded-3xl border border-border bg-card"
    >
      <Backdrop src={view.web ?? view.phone} />
      <div className="relative flex flex-wrap items-center gap-12 px-14 pb-16 pt-10 sm:px-20 lg:px-24 lg:pt-14">
        <HeroText
          listing={current}
          position={selection.index + 1}
          count={shown.length}
          views={views}
          look={look}
          onLook={selection.pickLook}
          canManage={canManage}
          onDetails={onDetails}
        />
        <HeroDevices view={view} name={listingName(current)} />
      </div>
      <HeroNav listings={shown} index={selection.index} onSelect={selection.select} onStep={selection.step} />
    </section>
  )
}
