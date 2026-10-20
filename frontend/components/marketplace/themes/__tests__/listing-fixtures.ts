/**
 * PRD-261 C-04b — listing rows for the Themes tab tests, in the wire shapes of lib/api-client.ts
 * (the answers of orchestrator/services/themes/listings.py). The install rows and the render
 * helper are A-07d's (components/settings/themes/__tests__/theme-fixtures.tsx), reused.
 */
import type { ThemeListing, ThemeListingDetail } from '@/lib/api-client'

export const LISTING_ID = 'acme.aurora'
export const LISTED_PACKAGE_ID = '44444444-4444-4444-8444-444444444444'
export const LISTING_INSTALL_ID = '55555555-5555-4555-8555-555555555555'
export const DIGEST = 'b'.repeat(64)
export const FILES = `/api/themes/files/${DIGEST}`

export function listing(overrides: Partial<ThemeListing> = {}): ThemeListing {
  return {
    listingId: LISTING_ID,
    themeId: 'acme/aurora',
    name: 'Aurora',
    description: 'Soft gradients for long sessions.',
    publisher: { name: 'acme', official: false },
    version: '1.2.0',
    packageId: LISTED_PACKAGE_ID,
    compatible: true,
    uses: ['frame', 'looks'],
    pricing: 'Free',
    cover: `${FILES}/assets/previews/default.png`,
    revoked: null,
    installed: null,
    ...overrides,
  }
}

export function listingDetail(overrides: Partial<ThemeListingDetail> = {}): ThemeListingDetail {
  return {
    ...listing(),
    presets: [
      {
        id: 'default',
        default: true,
        previews: [
          { screen: null, url: `${FILES}/assets/previews/default.png` },
          { screen: 'chat', url: `${FILES}/assets/previews/default-chat.png` },
        ],
      },
    ],
    otherPreviews: [],
    readme: `${FILES}/README.md`,
    changelog: null,
    base: 'studio',
    tone: 'from-look',
    minWidth: 1024,
    pages: ['/chat'],
    routes: [],
    latest: '1.2.0',
    versions: {
      '1.2.0': { packageId: LISTED_PACKAGE_ID, engines: { automatos: '^1.0.0', sdk: '^1.0.0' }, compatible: true, revoked: null, createdAt: null },
      '2.0.0-beta': {
        packageId: '66666666-6666-4666-8666-666666666666', engines: { automatos: '^2.0.0', sdk: '^2.0.0' },
        compatible: false, revoked: null, createdAt: null,
      },
    },
    ...overrides,
  }
}
