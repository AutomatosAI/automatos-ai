/**
 * PRD-261 C-04b — the Themes tab's small rules: the tab's name (D3/Q11), search, the files-route
 * guard on every preview and document URL, and "runs its own upload".
 */
import { describe, expect, it } from 'vitest'

import {
  THEMES_TAB_LABEL,
  THEMES_TAB_VALUE,
  filterListings,
  runsOwnUpload,
  themeFileUrl,
} from '../listing-format'
import { FILES, LISTING_INSTALL_ID, LISTED_PACKAGE_ID, listing } from './listing-fixtures'

describe('the tab name', () => {
  it('is "Themes", not "Skins" (Gerard, D3/Q11)', () => {
    expect(THEMES_TAB_LABEL).toBe('Themes')
    expect(THEMES_TAB_VALUE).toBe('themes')
  })
})

describe('filterListings', () => {
  const aurora = listing()
  const night = listing({
    listingId: 'nightco.owl', themeId: 'nightco/owl', name: 'Night Owl', description: null,
    publisher: { name: 'nightco', official: true }, uses: ['mascot'],
  })

  it('keeps every listing for an empty search', () => {
    expect(filterListings([aurora, night], '  ')).toEqual([aurora, night])
  })

  it('matches the name, the publisher, the id and `uses`, every word, any case', () => {
    expect(filterListings([aurora, night], 'night')).toEqual([night])
    expect(filterListings([aurora, night], 'ACME')).toEqual([aurora])
    expect(filterListings([aurora, night], 'nightco/owl')).toEqual([night])
    expect(filterListings([aurora, night], 'mascot')).toEqual([night])
    expect(filterListings([aurora, night], 'soft sessions')).toEqual([aurora])
    expect(filterListings([aurora, night], 'soft owl')).toEqual([])
  })

  it('returns a new array, never the one it was given', () => {
    const all = [aurora]
    expect(filterListings(all, '')).not.toBe(all)
  })
})

describe('themeFileUrl', () => {
  it('keeps the app’s own files route', () => {
    expect(themeFileUrl(`${FILES}/assets/previews/default.png`)).toBe(`${FILES}/assets/previews/default.png`)
  })

  it('drops another origin, a protocol-relative URL, a climb out and nothing at all', () => {
    expect(themeFileUrl('https://evil.example/x.png')).toBeNull()
    expect(themeFileUrl('//evil.example/api/themes/files/x.png')).toBeNull()
    expect(themeFileUrl('javascript:alert(1)')).toBeNull()
    expect(themeFileUrl(`${FILES}/../../admin`)).toBeNull()
    expect(themeFileUrl(`${FILES}\\..\\x`)).toBeNull()
    expect(themeFileUrl(null)).toBeNull()
    expect(themeFileUrl(undefined)).toBeNull()
  })
})

describe('runsOwnUpload', () => {
  const install = { installId: LISTING_INSTALL_ID, packageId: LISTED_PACKAGE_ID, version: '1.2.0', updatePackageId: null }

  it('is true only for this workspace’s own upload of the listed theme', () => {
    expect(runsOwnUpload(listing())).toBe(false)
    expect(runsOwnUpload(listing({ installed: { ...install, fromListing: true } }))).toBe(false)
    expect(runsOwnUpload(listing({ installed: { ...install, fromListing: false } }))).toBe(true)
  })
})
