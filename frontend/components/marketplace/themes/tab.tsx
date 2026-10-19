'use client'

/**
 * PRD-261 C-04b — the Marketplace's Themes tab (the minimal C-04, plan §2.6; named "Themes" by
 * Gerard, D3/Q11).
 *
 * - the listing grid of approved themes (C-04a), filtered by the Marketplace's search box;
 * - the detail of a listing, with "Add to my workspace" and then "Use this theme";
 * - **Install theme** for a private upload: A-07d's `InstallTheme`, reused (owners and admins);
 * - **Installed in this workspace**: A-07d's `InstalledList`, reused, which is where a listing
 *   install is updated, rolled back and uninstalled (the same A-07b routes as a private one).
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

import { DetailDialog } from './detail-dialog'
import { ListingGrid } from './listing-grid'
import { INSTALLED_HEADING, THEMES_TAB_LEDE } from './listing-format'

export { THEMES_TAB_LABEL, THEMES_TAB_VALUE } from './listing-format'

function ManageThemes({ canManage }: { canManage: boolean }) {
  const settings = useThemeSettings()
  return (
    <section aria-label={INSTALLED_HEADING} className="space-y-4">
      <h3 className="text-base font-semibold text-foreground">{INSTALLED_HEADING}</h3>
      {canManage && <InstallTheme settings={settings.data} />}
      <InstalledList canManage={canManage} />
    </section>
  )
}

export function MarketplaceThemesTab({ searchQuery = '' }: { searchQuery?: string }) {
  const { workspace } = useWorkspace()
  const canManage = canManageThemes(workspace?.role)
  const [openListing, setOpenListing] = useState<string | null>(null)

  return (
    <div className="space-y-8" data-testid="marketplace-themes-tab">
      <div className="space-y-4">
        <p className="text-sm text-muted-foreground">{THEMES_TAB_LEDE}</p>
        <ListingGrid searchQuery={searchQuery} onOpen={setOpenListing} />
      </div>
      <ManageThemes canManage={canManage} />
      <DetailDialog listingId={openListing} canManage={canManage} onClose={() => setOpenListing(null)} />
    </div>
  )
}
