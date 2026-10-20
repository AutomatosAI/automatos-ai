'use client'

/**
 * PRD-261 C-04b — the Marketplace's Themes tab (the minimal C-04, plan §2.6; named "Themes" by
 * Gerard, D3/Q11).
 *
 * - the listing grid of approved themes (C-04a), filtered by the Marketplace's search box;
 * - the detail of a listing, with "Add to my workspace" and then "Use this theme";
 * - **Publish theme**: C-01b's form, for every member (the submit route gates on members:read);
 * - **Install theme** for a private upload: A-07d's `InstallTheme`, reused (owners and admins);
 * - **Installed in this workspace**: A-07d's `InstalledList`, reused, which is where a listing
 *   install is updated, rolled back and uninstalled (the same A-07b routes as a private one).
 *
 * While the platform switch `THEMES_MARKETPLACE` is off (`ThemeSettings.marketplace`), the grid
 * and Publish theme give way to one plain note, so no request is made that the server would
 * refuse; private uploads and the installed list have their own switches and stay. Until the
 * settings answer, the Marketplace sections wait; if it fails, they show and the server decides.
 *
 * The role decides which controls show (`canManageThemes`); the server enforces every gate and
 * step-up itself. Both editions: in the local edition the operator is the workspace owner.
 * Radix mounts a tab's content only while it is selected, so the Marketplace asks for no theme
 * data until this tab opens.
 */
import { useState } from 'react'

import { InstallTheme } from '@/components/settings/themes/install-theme'
import { InstalledList } from '@/components/settings/themes/installed-list'
import { canManageThemes } from '@/components/settings/themes/theme-labels'
import { useWorkspace } from '@/components/workspace-provider'
import { useThemeSettings } from '@/hooks/use-installed-themes'
import type { ThemeSettings } from '@/lib/api-client'

import { DetailDialog } from './detail-dialog'
import { ListingGrid } from './listing-grid'
import {
  INSTALLED_HEADING,
  LOADING_LISTINGS,
  MARKETPLACE_OFF_NOTE,
  MARKETPLACE_SWITCH_OFF,
  THEMES_TAB_LEDE,
} from './listing-format'
import { PUBLISH_THEME, PublishThemeForm } from './publish-theme'

export { THEMES_TAB_LABEL, THEMES_TAB_VALUE } from './listing-format'

function ManageThemes({ canManage, settings }: { canManage: boolean; settings: ThemeSettings | undefined }) {
  return (
    <section aria-label={INSTALLED_HEADING} className="space-y-4">
      <h3 className="text-base font-semibold text-foreground">{INSTALLED_HEADING}</h3>
      {canManage && <InstallTheme settings={settings} />}
      <InstalledList canManage={canManage} />
    </section>
  )
}

interface MarketplaceSectionsProps {
  readonly searchQuery: string
  readonly onOpen: (listingId: string) => void
}

function MarketplaceSections({ searchQuery, onOpen }: MarketplaceSectionsProps) {
  return (
    <>
      <div className="space-y-4">
        <p className="text-sm text-muted-foreground">{THEMES_TAB_LEDE}</p>
        <ListingGrid searchQuery={searchQuery} onOpen={onOpen} />
      </div>
      {/* The form is the "Publish theme" region itself (publish-theme.tsx). */}
      <div className="space-y-4">
        <h3 className="text-base font-semibold text-foreground">{PUBLISH_THEME}</h3>
        <PublishThemeForm />
      </div>
    </>
  )
}

export function MarketplaceThemesTab({ searchQuery = '' }: { searchQuery?: string }) {
  const { workspace } = useWorkspace()
  const canManage = canManageThemes(workspace?.role)
  const settings = useThemeSettings()
  const [openListing, setOpenListing] = useState<string | null>(null)
  const marketplaceOff = settings.data?.marketplace === MARKETPLACE_SWITCH_OFF
  const waiting = settings.isInitialLoading

  return (
    <div className="space-y-8" data-testid="marketplace-themes-tab">
      {waiting && <p className="text-sm text-muted-foreground">{LOADING_LISTINGS}</p>}
      {marketplaceOff && (
        <p className="text-sm text-muted-foreground" data-testid="themes-marketplace-off">
          {MARKETPLACE_OFF_NOTE}
        </p>
      )}
      {!waiting && !marketplaceOff && <MarketplaceSections searchQuery={searchQuery} onOpen={setOpenListing} />}
      <ManageThemes canManage={canManage} settings={settings.data} />
      {!marketplaceOff && (
        <DetailDialog listingId={openListing} canManage={canManage} onClose={() => setOpenListing(null)} />
      )}
    </div>
  )
}
