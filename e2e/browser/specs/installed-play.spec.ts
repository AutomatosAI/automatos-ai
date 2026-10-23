import { expect, test } from '@playwright/test'
import { THEME_E2E_PATH } from '../support/hello-theme'
import { chooseKitPackage, installKitPackage, readKitPackage, resetKitPackage } from '../support/kit-package'
import { watchPage } from '../support/watch'

// PRD-261 A-17a (@package): Play installed from the Design Kit's package, end to end on the local
// edition. The lane's `package` job packs automatos.play-1.1.0.autotheme from the kit at its
// pinned commit; it is uploaded, opted in to and chosen through the Install theme UI's routes
// (support/kit-package.ts), with nothing intercepted: the server's active answer, boot and files
// route serve it. On a Studio page the host loads the package and reports `active`, and the page
// is in Play's frame, not the Studio shell.
//
// The specs store the member's choice on the lane's one backend, so the workflow runs @package in
// its own step, after the other specs and one test at a time. Each test starts and ends with the
// choice on Studio and no install of Play's package.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const STATUS_ATTR = 'data-theme-host'
const PLAY_CLASS = /\bplay\b/
/** Play's frame: its sidebar (the kit's themes/play/src/components/play/shell/play-sidebar.tsx). */
const PLAY_FRAME = 'aside[aria-label="Sidebar"]'
/** The Studio desktop shell (components/layout/main-layout.tsx). */
const STUDIO_SHELL = '.sh-shell'
/** The package's entry, stylesheet and fonts come through the files proxy on a cold load. */
const THEME_LOAD_TIMEOUT_MS = 30_000

test.describe('PRD-261 A-17a — Play installed from the Design Kit package', () => {
  test.beforeEach(async ({ request }) => {
    await resetKitPackage(request)
  })

  test.afterEach(async ({ request }) => {
    await resetKitPackage(request)
  })

  test("the lane downloaded the kit's Play package", { tag: '@package' }, () => {
    expect(readKitPackage().length).toBeGreaterThan(0)
  })

  test("uploaded, opted in to and chosen, the package runs: the host is active in Play's frame", { tag: '@package' }, async ({ page, request }) => {
    const watch = watchPage(page)
    const install = await installKitPackage(request)
    await chooseKitPackage(page, install, APP_URL)
    await page.goto(THEME_E2E_PATH)

    const html = page.locator('html')
    await expect(html).toHaveAttribute(STATUS_ATTR, 'active', { timeout: THEME_LOAD_TIMEOUT_MS })
    await expect(html).toHaveClass(PLAY_CLASS)
    await expect(page.locator(PLAY_FRAME)).toBeVisible()
    await expect(page.locator(STUDIO_SHELL)).toHaveCount(0)
    expect(watch.pageErrors).toEqual([])
  })
})
