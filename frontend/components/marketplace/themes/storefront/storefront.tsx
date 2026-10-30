'use client'

/**
 * PRD-261 — the Themes tab's storefront (the approved design, 30 Oct 2026), over C-04a's listings:
 * a header, the hero stage for the listing the selection points at, and the strip of every
 * listing the Marketplace's search shows, ending on "Make your own".
 *
 * Loading and a refusal read as before (the server's own sentence); with nothing listed, the empty
 * stage invites the first theme; when the search shows nothing, the strip keeps "Make your own".
 */
import type { RefObject } from 'react'
import type { UseQueryResult } from '@tanstack/react-query'

import type { ThemeListing } from '@/lib/api-client'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { LOADING_LISTINGS, NO_MATCHES, THEMES_TAB_LEDE } from '../listing-format'
import { STOREFRONT_TITLE, WEB_AND_PHONE, themeCount } from './copy'
import { EmptyStage } from './empty-stage'
import { Hero } from './hero'
import { Strip } from './strip'
import type { StorefrontSelection } from './use-storefront-selection'

interface StorefrontProps {
  readonly listings: UseQueryResult<ThemeListing[]>
  /** The listings the Marketplace's search shows. */
  readonly shown: readonly ThemeListing[]
  readonly selection: StorefrontSelection
  readonly canManage: boolean
  readonly onDetails: (listingId: string) => void
  readonly stageRef: RefObject<HTMLElement>
}

function StorefrontHeader({ listings }: { listings: readonly ThemeListing[] }) {
  const withPhone = listings.some((listing) => Boolean(listing.phoneCover))
  return (
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div className="space-y-2">
        <h2 className="text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">{STOREFRONT_TITLE}</h2>
        <p className="max-w-2xl text-base leading-relaxed text-muted-foreground">{THEMES_TAB_LEDE}</p>
      </div>
      {listings.length > 0 && (
        <p className="font-mono text-xs text-muted-foreground">
          {themeCount(listings.length)}
          {withPhone && ` · ${WEB_AND_PHONE}`}
        </p>
      )}
    </header>
  )
}

export function Storefront({ listings, shown, selection, canManage, onDetails, stageRef }: StorefrontProps) {
  if (listings.isLoading) return <p className="text-sm text-muted-foreground">{LOADING_LISTINGS}</p>
  if (listings.isError) {
    return (
      <p className="text-sm text-destructive" role="alert">
        {themeErrorMessage(listings.error)}
      </p>
    )
  }
  const all = listings.data ?? []
  return (
    <div className="space-y-8" data-testid="theme-storefront">
      <StorefrontHeader listings={all} />
      {all.length === 0 && <EmptyStage />}
      {selection.current && (
        <Hero
          listing={selection.current}
          shown={shown}
          selection={selection}
          canManage={canManage}
          onDetails={onDetails}
          stageRef={stageRef}
        />
      )}
      {all.length > 0 && shown.length === 0 && <p className="text-sm text-muted-foreground">{NO_MATCHES}</p>}
      {all.length > 0 && <Strip listings={shown} selectedId={selection.current?.listingId ?? null} onSelect={selection.select} />}
    </div>
  )
}
