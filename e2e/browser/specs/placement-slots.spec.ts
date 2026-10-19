import { expect, test } from '@playwright/test'
import { chooseStyle, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-261 A-12b (@host): a placement slot a theme registers draws on a live page. The placement
// hello theme (fixtures/hello-a12b-live) registers page-header.mascot@1; /documents renders the
// shared PageHeader under both bases (docs/theme-host/placement-reach.md), so the mascot draws
// inside the header's slot host, in the same header row as the page title. The registrations
// reach PageHeader through the theme host context (theme-host-provider.tsx), nothing else.

const HELLO_A12B_LIVE: InstalledTheme = { installId: '00000000-0000-4000-8000-0000000a12b1', fixture: 'hello-a12b-live' }
const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const PAGE_WITH_HEADER = '/documents'
const STATUS_ATTR = 'data-theme-host'
const SLOT_ATTR = 'data-theme-slot'
const MASCOT_SLOT = 'page-header.mascot@1'
const ART_SLOT = 'page-header.art@1'
const FRAME_TEST_ID = 'hello-a12b-live-frame'
const MASCOT_TEST_ID = 'hello-a12b-live-mascot'
const PAGE_TITLE_TEST_ID = 'page-title'
const HEADER_TEXT = 'Knowledge Bases'

test.describe('placement slots in the browser', () => {
  test('a registered page-header.mascot@1 draws in the PageHeader of a live page', { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    await chooseStyle(page, `theme:install:${HELLO_A12B_LIVE.installId}`, APP_URL)
    await serveInstalledTheme(page, HELLO_A12B_LIVE)
    await page.goto(PAGE_WITH_HEADER)

    await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
    await expect(page.getByTestId(FRAME_TEST_ID)).toBeVisible()

    const slotHost = page.locator(`[${SLOT_ATTR}="${MASCOT_SLOT}"]`)
    await expect(slotHost).toHaveCount(1)
    const mascot = slotHost.getByTestId(MASCOT_TEST_ID)
    await expect(mascot).toBeVisible()
    await expect(mascot).toHaveText(HEADER_TEXT)

    // The slot host sits in the header row, beside the head that holds the page title.
    const row = slotHost.locator('xpath=..')
    await expect(row.getByTestId(PAGE_TITLE_TEST_ID)).toBeVisible()
    // A slot the theme did not register draws nothing: no host element at all.
    await expect(page.locator(`[${SLOT_ATTR}="${ART_SLOT}"]`)).toHaveCount(0)

    expect(watch.pageErrors).toEqual([])
  })
})
