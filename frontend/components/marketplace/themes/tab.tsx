'use client'

/**
 * PRD-261 C-04b — the Marketplace's Themes tab (named "Themes" by Gerard, D3/Q11), as a storefront
 * (the approved design, 30 Oct 2026; storefront/):
 *
 * - the hero stage for one listing (its looks on the web and on Automatos Go, Add to my workspace,
 *   Details), Previous / Next and dots, and the strip of every approved theme (C-04a), filtered by
 *   the Marketplace's search box, ending on "Make your own" (the Design Kit, and C-01b's Publish
 *   theme form in a dialog, for every member: the submit route gates on members:read);
 * - the detail of a listing (Details), with its facts, versions and documents;
 * - **Installed in this workspace**: A-07d's `InstallTheme` (owners and admins) and `InstalledList`,
 *   reused, which is where a listing install is updated, rolled back and uninstalled (the same
 *   A-07b routes as a private one); a private install of a theme with an Official listing can View
 *   it in the hero.
 *
 * While the platform switch `THEMES_MARKETPLACE` is off (`ThemeSettings.marketplace`), the
 * storefront gives way to one plain note, so no request is made that the server would refuse;
 * private uploads and the installed list have their own switches and stay. Until the settings
 * answer, the storefront waits; if they fail, it shows and the server decides.
 *
 * The role decides which controls show (`canManageThemes`); the server enforces every gate and
 * step-up itself. Both editions: in the local edition the operator is the workspace owner.
 * Radix mounts a tab's content only while it is selected; Themes is the tab the Marketplace opens
 * on, so its listings are asked for as the Marketplace opens.
 */
import { useMemo, useRef, useState } from 'react'

import { canManageThemes } from '@/components/settings/themes/theme-labels'
import { useWorkspace } from '@/components/workspace-provider'
import { usePrefersReducedMotion } from '@/hooks/use-media-preferences'
import { useThemeSettings } from '@/hooks/use-installed-themes'
import type { ThemeListing } from '@/lib/api-client'

import { DetailDialog } from './detail-dialog'
import { LOADING_LISTINGS, MARKETPLACE_OFF_NOTE, MARKETPLACE_SWITCH_OFF, filterListings } from './listing-format'
import { useThemeListings } from './listing-hooks'
import { StorefrontInstalled } from './storefront/installed-rows'
import { Storefront } from './storefront/storefront'
import { useStorefrontSelection } from './storefront/use-storefront-selection'

export { THEMES_TAB_LABEL, THEMES_TAB_VALUE } from './listing-format'

const NONE_LISTED: readonly ThemeListing[] = []

export function MarketplaceThemesTab({ searchQuery = '' }: { searchQuery?: string }) {
  const { workspace } = useWorkspace()
  const canManage = canManageThemes(workspace?.role)
  const settings = useThemeSettings()
  const marketplaceOff = settings.data?.marketplace === MARKETPLACE_SWITCH_OFF
  const waiting = settings.isInitialLoading
  const listed = !waiting && !marketplaceOff
  const listings = useThemeListings(listed)
  const all = listed ? listings.data ?? NONE_LISTED : NONE_LISTED
  const shown = useMemo(() => filterListings(all, searchQuery), [all, searchQuery])
  const selection = useStorefrontSelection(all, shown, searchQuery)
  const [openListing, setOpenListing] = useState<string | null>(null)
  const stage = useRef<HTMLElement>(null)
  const reducedMotion = usePrefersReducedMotion()

  const viewInHero = (listingId: string) => {
    selection.select(listingId)
    stage.current?.scrollIntoView?.({ block: 'start', behavior: reducedMotion ? 'auto' : 'smooth' })
  }

  return (
    <div className="space-y-10" data-testid="marketplace-themes-tab">
      {waiting && <p className="text-sm text-muted-foreground">{LOADING_LISTINGS}</p>}
      {marketplaceOff && (
        <p className="text-sm text-muted-foreground" data-testid="themes-marketplace-off">
          {MARKETPLACE_OFF_NOTE}
        </p>
      )}
      {listed && (
        <Storefront
          listings={listings}
          shown={shown}
          selection={selection}
          canManage={canManage}
          onDetails={setOpenListing}
          stageRef={stage}
        />
      )}
      <StorefrontInstalled canManage={canManage} settings={settings.data} listings={all} onView={viewInHero} />
      {!marketplaceOff && (
        <DetailDialog listingId={openListing} canManage={canManage} onClose={() => setOpenListing(null)} />
      )}
    </div>
  )
}
