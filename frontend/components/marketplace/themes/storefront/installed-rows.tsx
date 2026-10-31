'use client'

/**
 * PRD-261 — "Installed in this workspace" on the storefront: A-07d's Install theme (owners and
 * admins) and A-07d's installed list, reused with every action and gate it has (Use this theme for
 * every member; default, review update, roll back and uninstall for `workspace:manage`), laid out
 * as compact rows.
 *
 * Each row gets the thumbnail of its theme's listing (the listing's cover: an install row carries
 * no image of its own), else none. A PRIVATE install (the workspace's own upload) whose theme id
 * has an Official listing says "Official x.y.z is listed", with View: it puts that listing in
 * the hero. While the Marketplace is off there are no listings, so the rows are A-07d's own.
 */
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { InstallTheme } from '@/components/settings/themes/install-theme'
import { InstalledList } from '@/components/settings/themes/installed-list'
import type { InstalledRowDecoration } from '@/components/settings/themes/installed-row'
import type { InstalledTheme, ThemeListing, ThemeSettings } from '@/lib/api-client'

import { INSTALLED_HEADING, listingName, themeFileUrl } from '../listing-format'
import { VIEW, officialListed, viewOfficial } from './copy'

const ROWS_GRID = 'grid grid-cols-1 gap-3 xl:grid-cols-2'

interface StorefrontInstalledProps {
  readonly canManage: boolean
  readonly settings: ThemeSettings | undefined
  /** Every listing (none while the Marketplace is off). */
  readonly listings: readonly ThemeListing[]
  /** Put a listing in the hero. */
  readonly onView: (listingId: string) => void
}

function Thumbnail({ src }: { src: string | null }) {
  if (!src) return null
  return (
    // A theme file on the app's own origin (A-08b proxy); next/image would re-encode it.
    // eslint-disable-next-line @next/next/no-img-element
    <img src={src} alt="" loading="lazy" className="h-[60px] w-24 shrink-0 rounded-lg border border-border object-cover" />
  )
}

function OfficialListed({ listing, onView }: { listing: ThemeListing; onView: (listingId: string) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-2 pt-1" data-testid="installed-official-listed">
      <Badge variant="outline" className="border-primary/60">
        {officialListed(listing.version)}
      </Badge>
      <Button
        size="sm"
        variant="outline"
        aria-label={viewOfficial(listingName(listing))}
        onClick={() => onView(listing.listingId)}
      >
        {VIEW}
      </Button>
    </div>
  )
}

/** The row's thumbnail, and for a private install of a theme with an Official listing, the notice. */
export function installDecoration(
  install: InstalledTheme,
  listings: readonly ThemeListing[],
  onView: (listingId: string) => void,
): InstalledRowDecoration {
  const listing = listings.find((candidate) => candidate.themeId === install.themeId)
  if (!listing) return {}
  const officialListing = install.private && listing.publisher.official
  return {
    leading: <Thumbnail src={themeFileUrl(listing.cover)} />,
    extra: officialListing ? <OfficialListed listing={listing} onView={onView} /> : undefined,
  }
}

export function StorefrontInstalled({ canManage, settings, listings, onView }: StorefrontInstalledProps) {
  return (
    <section aria-label={INSTALLED_HEADING} className="space-y-4">
      <h3 className="text-xl font-semibold tracking-tight text-foreground">{INSTALLED_HEADING}</h3>
      {canManage && <InstallTheme settings={settings} />}
      <InstalledList
        canManage={canManage}
        decorate={(install) => installDecoration(install, listings, onView)}
        listClassName={ROWS_GRID}
      />
    </section>
  )
}
