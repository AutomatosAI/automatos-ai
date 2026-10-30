/**
 * PRD-261 C-04b — the words the Marketplace's Themes tab shows, and the small rules it reads them by.
 *
 * The tab is named "Themes" (Gerard, D3/Q11, 19 Oct), not "Skins". Install counts, ratings and
 * pricing are not part of the minimal C-04 (plan §2.6 C-04b), so nothing here shows them.
 */
import type { ThemeListing } from '@/lib/api-client'
import { THEME_FILES_PATH } from '@/lib/theme-host/constants'

export const THEMES_TAB_VALUE = 'themes'
export const THEMES_TAB_LABEL = 'Themes'

/** The storefront's lede (the approved design, 30 Oct 2026). */
export const THEMES_TAB_LEDE =
  'Themes change how Automatos looks on the web and on your phone. Official themes are reviewed and signed by Automatos, and they update themselves.'
export const ADD_TO_WORKSPACE = 'Add to my workspace'
export const ADDING = 'Adding…'
export const ADDED_TO_WORKSPACE = 'Added to this workspace. Each member chooses it for themselves.'
export const ALREADY_IN_WORKSPACE = 'This version is already in this workspace.'
export const OFFICIAL_BADGE = 'Official'
export const INSTALLED_BADGE = 'In this workspace'
export const UPDATE_BADGE = 'Update available'
export const NOT_COMPATIBLE = 'Needs a newer Automatos'
export const NOT_COMPATIBLE_HINT = 'No version of this theme runs on this Automatos yet.'
export const OWN_UPLOAD_NOTE =
  'This workspace runs its own upload of this theme. Uninstall it below to add the Marketplace version.'
/**
 * A listing install with a newer version waiting: it is pinned, it was added at an older version, or the
 * server could not move it (an unpinned install otherwise updates itself, services/themes/auto_update.py).
 */
export const UPDATE_NOTE =
  'A newer version is listed and this install has not moved to it: review it under "Installed in this workspace".'
export const MANAGERS_ADD = 'A workspace owner or admin adds themes to this workspace.'
export const NO_LISTINGS = 'No themes are listed in the Marketplace yet.'
export const NO_MATCHES = 'No theme matches your search.'
export const INSTALLED_HEADING = 'Installed in this workspace'
export const MARKETPLACE_OFF_NOTE =
  'The theme Marketplace is turned off on this platform, so nothing is listed or published here. Private uploads and the themes in this workspace are below.'
/** `ThemeSettings.marketplace` while THEMES_MARKETPLACE is off. */
export const MARKETPLACE_SWITCH_OFF = 'off'
export const LOADING_LISTINGS = 'Loading themes…'
export const LOADING_LISTING = 'Loading the theme…'

export function listingName(listing: Pick<ThemeListing, 'name' | 'themeId'>): string {
  return listing.name || listing.themeId
}

/**
 * A preview or document URL from the server, only when it is the app's own files route
 * (`serve.module_url`, reached through the A-08b same-origin proxy). Anything else is dropped,
 * so a listing can never point an image or a link at another origin.
 */
export function themeFileUrl(url: string | null | undefined): string | null {
  if (typeof url !== 'string' || !url.startsWith(THEME_FILES_PATH)) return null
  return url.includes('..') || url.includes('\\') ? null : url
}

function searchText(listing: ThemeListing): string {
  return [listing.name, listing.themeId, listing.description, listing.publisher.name, ...listing.uses]
    .filter((part): part is string => typeof part === 'string')
    .join(' ')
    .toLowerCase()
}

/** The listings whose name, id, description, publisher or `uses` hold every word of the search. */
export function filterListings(listings: readonly ThemeListing[], query: string): ThemeListing[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (words.length === 0) return [...listings]
  return listings.filter((listing) => {
    const text = searchText(listing)
    return words.every((word) => text.includes(word))
  })
}

/** True when the workspace's install of this theme is its own upload, not the listed package. */
export function runsOwnUpload(listing: Pick<ThemeListing, 'installed'>): boolean {
  return listing.installed !== null && !listing.installed.fromListing
}
