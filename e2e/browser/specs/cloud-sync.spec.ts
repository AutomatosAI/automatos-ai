import { expect, test, type Locator, type Page, type Route, type TestInfo } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseCoreStyle, STUDIO_STYLE } from '../support/hello-theme'

// PRD-270 US-270-05c (@host, K30): Sync with Automatos Cloud on the local edition, in a real browser.
//
// The backend's answers are fulfilled here (support/backend-route.ts fulfillBackendJson): the Sync's
// status (GET /api/marketplace/cloud-sync/status), the Sync itself (POST /api/marketplace/cloud-sync)
// and the theme listings and the listing's detail (GET /api/marketplace/themes[/{listingId}]), one
// listing with source 'cloud' once the Sync has run. Every request the browser makes to Automatos
// Cloud's origin is answered here too (a catalogue screenshot with a 1×1 PNG, anything else aborted),
// so nothing reaches the real Automatos Cloud, and the backend never pulls (its POST never arrives).
//
// 1. /marketplace: the header shows Sync with Automatos Cloud and its status line; a click shows the
//    spinner while the POST is held, then the toast with the counts and the new status line; the
//    listings are read again and the Cloud listing carries From Automatos Cloud, its screenshot
//    loaded from Automatos Cloud's catalogue files route (K15: themeFileUrl and img-src allow it).
// 2. Settings › Automatos Cloud: the URL in use, the last Sync and Sync now.
//
// A screenshot of each is attached. Both browsers: the answers are this page's own, so nothing is
// shared through the lane's database. Studio is pinned with the style cookie and GET /api/themes/active
// answered with it (chooseCoreStyle), so another spec's stored theme choice never moves this page.
//
// The SaaS visitor flow (signed-out browse, Add -> the sign-in URL, the return) is NOT run here: the
// lane builds the local edition only (isSaaS is fixed at build time, and the lane has no Clerk keys).
// Vitest covers it:
//   US-270-08a frontend/lib/__tests__/marketplace-public-middleware.test.ts,
//              frontend/components/marketplace/public/__tests__/marketplace-page.test.tsx
//   US-270-08b frontend/components/marketplace/public/__tests__/visitor-tabs.test.tsx,
//              frontend/lib/marketplace/__tests__/public-catalog.test.ts
//   US-270-09a frontend/components/marketplace/public/__tests__/visitor-add.test.tsx,
//              frontend/components/auth/__tests__/resume-after-sign-in.test.tsx,
//              frontend/lib/marketplace/__tests__/return-after-sign-in.test.ts
//   US-270-09b frontend/components/marketplace/__tests__/marketplace-deep-link.test.tsx,
//              frontend/components/marketplace/themes/__tests__/themes-tab-add-link.test.tsx

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const MARKETPLACE_PATH = '/marketplace'
const SETTINGS_CLOUD_PATH = '/settings?tab=automatos-cloud'
const OK = 200

const SYNC_PATH = '/api/marketplace/cloud-sync'
const STATUS_PATH = `${SYNC_PATH}/status`
const LISTINGS_PATH = '/api/marketplace/themes'

/** lib/marketplace/cloud-files.ts AUTOMATOS_CLOUD_ORIGIN and CLOUD_FILES_PATH. */
const AUTOMATOS_CLOUD_ORIGIN = 'https://api.automatos.app'
const CLOUD_FILES_PATH = '/api/marketplace/catalog/files/'
const DIGEST = '0123456789abcdef'.repeat(4)
const CLOUD_FILES_PREFIX = `${AUTOMATOS_CLOUD_ORIGIN}${CLOUD_FILES_PATH}${DIGEST}/`
const CLOUD_SHOT = `${CLOUD_FILES_PREFIX}previews/web.png`
/** A valid 1×1 PNG: Automatos Cloud's screenshot, answered here. */
const PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==', 'base64')

const THEME_ID = 'automatos-e2e/cloud-c05c'
const LISTING_ID = 'automatos-e2e.cloud-c05c'
const THEME_NAME = 'Cloud Sync (05c)'
const VERSION = '1.0.0'
const PACKAGE_ID = '00000000-0000-4000-8000-0000000270c5'
const DETAIL_PATH = `${LISTINGS_PATH}/${LISTING_ID}`
const HOUR_MS = 3_600_000

// The words the page shows (components/marketplace/cloud/copy.ts, themes/listing-format.ts).
const SYNC_WITH_CLOUD = 'Sync with Automatos Cloud'
const NOT_SYNCED_YET = 'Not synced yet'
const SYNCED_TOAST = '1 new, 0 updated from Automatos Cloud'
const SYNCED_LINE = 'Last synced just now · 1 theme'
const CLOUD_BADGE = 'From Automatos Cloud'
const CLOUD_TAB = 'Automatos Cloud'
const SYNC_NOW = 'Sync now'
const COUNTS_LINE = '1 new, 0 updated, 0 unchanged, 0 refused'

const NO_COUNTS = { new: 0, updated: 0, unchanged: 0, refused: 0 }
const SYNC_COUNTS = { ...NO_COUNTS, new: 1 }
const SIGNATURE = { state: 'verified', keyId: 'automatos-e2e' }

/** pull.sync_status before any Sync: on, the URL in use, nothing held. */
const NOT_SYNCED = {
  enabled: true,
  cloudUrl: AUTOMATOS_CLOUD_ORIGIN,
  lastAttemptAt: null,
  lastSuccessAt: null,
  lastError: null,
  generatedAt: null,
  counts: NO_COUNTS,
  themes: 0,
  stale: true,
}

/** The listing a Sync wrote (listings.py listing_summary, source 'cloud'). */
const CLOUD_LISTING = {
  listingId: LISTING_ID,
  themeId: THEME_ID,
  name: THEME_NAME,
  description: 'A theme a Sync with Automatos Cloud listed.',
  publisher: { name: 'Automatos', official: true },
  version: VERSION,
  packageId: PACKAGE_ID,
  signature: SIGNATURE,
  compatible: true,
  uses: [],
  pricing: 'free',
  cover: CLOUD_SHOT,
  source: 'cloud',
  revoked: null,
  installed: null,
  phoneCover: null,
}

/** Its detail: one preset whose screenshot is on Automatos Cloud; a Cloud row has no readme or changelog. */
const CLOUD_DETAIL = {
  ...CLOUD_LISTING,
  presets: [{ id: 'default', default: true, previews: [{ screen: null, url: CLOUD_SHOT }] }],
  otherPreviews: [],
  readme: null,
  changelog: null,
  base: null,
  tone: null,
  minWidth: null,
  pages: [],
  routes: [],
  latest: VERSION,
  versions: { [VERSION]: { packageId: PACKAGE_ID, engines: null, compatible: true, revoked: null, signature: SIGNATURE, createdAt: null } },
}

/** pull.sync_status after one Sync at `syncedAt`: one new theme held. */
function syncedStatus(syncedAt: string) {
  return { ...NOT_SYNCED, lastAttemptAt: syncedAt, lastSuccessAt: syncedAt, generatedAt: syncedAt, counts: SYNC_COUNTS, themes: 1, stale: false }
}

interface SyncStub {
  /** Lets the held POST answer, once the spec has seen the spinner. */
  readonly release: () => void
  /** How many POSTs of the Sync the spec answered (none reached the backend). */
  readonly posts: () => number
}

function onPath(pathname: string): (url: URL) => boolean {
  return (url) => url.pathname === pathname
}

/** Answers the Sync's status and POST, and the listings: none before the Sync, the Cloud listing after. */
async function serveCloudSync(page: Page): Promise<SyncStub> {
  const state: { posts: number; syncedAt: string | null } = { posts: 0, syncedAt: null }
  let release: () => void = () => undefined
  const held = new Promise<void>((resolve) => {
    release = resolve
  })
  await page.route(onPath(STATUS_PATH), (route) =>
    fulfillBackendJson(route, OK, state.syncedAt ? syncedStatus(state.syncedAt) : NOT_SYNCED))
  await page.route(onPath(SYNC_PATH), async (route) => {
    if (route.request().method() === 'POST') {
      state.posts += 1
      await held
      state.syncedAt = new Date().toISOString()
    }
    const status = state.syncedAt ? syncedStatus(state.syncedAt) : NOT_SYNCED
    await fulfillBackendJson(route, OK, { counts: SYNC_COUNTS, updatesApplied: 0, updatesFailed: 0, status })
  })
  await page.route(onPath(LISTINGS_PATH), (route) =>
    fulfillBackendJson(route, OK, { listings: state.syncedAt ? [CLOUD_LISTING] : [] }))
  await page.route(onPath(DETAIL_PATH), (route) => fulfillBackendJson(route, OK, CLOUD_DETAIL))
  return { release: () => release(), posts: () => state.posts }
}

/** Settings reads a status after one Sync an hour ago. */
async function serveSyncedStatus(page: Page): Promise<void> {
  const syncedAt = new Date(Date.now() - HOUR_MS).toISOString()
  await page.route(onPath(STATUS_PATH), (route) => fulfillBackendJson(route, OK, syncedStatus(syncedAt)))
}

async function answerCloud(route: Route, asked: string[]): Promise<void> {
  const url = route.request().url()
  asked.push(url)
  if (url.startsWith(CLOUD_FILES_PREFIX)) {
    await route.fulfill({ status: OK, contentType: 'image/png', body: PNG })
    return
  }
  await route.abort()
}

/** Answers every request the browser makes to Automatos Cloud's origin; returns the URLs it asked for. */
async function stubAutomatosCloud(page: Page): Promise<string[]> {
  const asked: string[] = []
  await page.route((url) => url.origin === AUTOMATOS_CLOUD_ORIGIN, (route) => answerCloud(route, asked))
  return asked
}

async function attachShot(testInfo: TestInfo, name: string, body: Buffer): Promise<void> {
  await testInfo.attach(name, { body, contentType: 'image/png' })
}

function loaded(element: Element): boolean {
  return element instanceof HTMLImageElement && element.complete && element.naturalWidth > 0
}

/** `image` is the Cloud screenshot, and it loaded (the CSP's img-src let it through). */
async function expectCloudScreenshot(image: Locator): Promise<void> {
  await expect(image).toHaveAttribute('src', CLOUD_SHOT)
  await expect.poll(() => image.evaluate(loaded)).toBe(true)
}

test.describe('Sync with Automatos Cloud (local edition)', () => {
  test.beforeEach(async ({ page }) => {
    await chooseCoreStyle(page, STUDIO_STYLE, APP_URL)
  })

  test('the Marketplace header syncs: spinner, toast with the counts, the Cloud listing badged', { tag: '@host' }, async ({ page }, testInfo) => {
    const cloud = await stubAutomatosCloud(page)
    const sync = await serveCloudSync(page)
    await page.goto(MARKETPLACE_PATH)

    const controls = page.getByTestId('cloud-sync')
    const button = controls.getByRole('button', { name: SYNC_WITH_CLOUD })
    await expect(button).toBeEnabled()
    await expect(controls.getByTestId('cloud-sync-status')).toHaveText(NOT_SYNCED_YET)
    await expect(page.getByTestId('storefront-empty')).toBeVisible()
    await attachShot(testInfo, 'marketplace-sync-button', await page.screenshot())

    await button.click()
    await expect(controls.getByTestId('cloud-sync-spinner')).toBeVisible()
    await expect(button).toBeDisabled()
    await attachShot(testInfo, 'marketplace-sync-spinner', await page.screenshot())
    sync.release()

    await expect(page.locator('[data-sonner-toast]').filter({ hasText: SYNCED_TOAST })).toBeVisible()
    await expect(controls.getByTestId('cloud-sync-status')).toHaveText(SYNCED_LINE)
    await expect(button).toBeEnabled()
    await attachShot(testInfo, 'marketplace-sync-toast', await page.screenshot())
    expect(sync.posts(), 'one Sync, answered here').toBe(1)

    const card = page.getByTestId(`theme-listing-${LISTING_ID}`)
    await expect(card).toContainText(THEME_NAME)
    await expect(card.getByText(CLOUD_BADGE, { exact: true })).toBeVisible()
    const hero = page.getByTestId('storefront-hero')
    await expect(hero.getByText(CLOUD_BADGE, { exact: true })).toBeVisible()
    await expectCloudScreenshot(hero.getByTestId('storefront-web-screen'))
    await card.scrollIntoViewIfNeeded()
    await expectCloudScreenshot(card.locator('img').first()) // the card's cover is lazy
    await attachShot(testInfo, 'marketplace-cloud-listing', await card.screenshot())

    expect(cloud, 'Automatos Cloud was asked for the screenshot, answered here').toContain(CLOUD_SHOT)
    expect(cloud.filter((url) => !url.startsWith(CLOUD_FILES_PREFIX)), 'nothing else asked of Automatos Cloud').toEqual([])
  })

  test('Settings › Automatos Cloud shows the URL in use and Sync now', { tag: '@host' }, async ({ page }, testInfo) => {
    const cloud = await stubAutomatosCloud(page)
    await serveSyncedStatus(page)
    await page.goto(SETTINGS_CLOUD_PATH)

    const tab = page.getByRole('navigation', { name: 'Sections' }).getByRole('button', { name: CLOUD_TAB, exact: true })
    await expect(tab).toHaveAttribute('aria-current', 'page')
    const panel = page.getByTestId('automatos-cloud')
    await expect(panel.getByTestId('automatos-cloud-url')).toHaveText(AUTOMATOS_CLOUD_ORIGIN)
    await expect(panel.getByTestId('automatos-cloud-counts')).toHaveText(COUNTS_LINE)
    await expect(panel.getByTestId('automatos-cloud-themes')).toHaveText('1')
    await expect(panel.getByRole('button', { name: SYNC_NOW })).toBeEnabled()
    await attachShot(testInfo, 'settings-automatos-cloud', await page.screenshot())

    expect(cloud, 'nothing asked of Automatos Cloud').toEqual([])
  })
})
