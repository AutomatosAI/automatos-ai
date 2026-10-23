import fs from 'fs'
import path from 'path'
import { expect, test, type Locator, type Page } from '@playwright/test'
import { PLAY_THEME_ID } from '../support/kit-package'
import { describeParity, THEME_LOAD_TIMEOUT_MS, type AvailablePlay } from '../support/play-parity'

// PRD-261 A-09b (@package): Play's looks are kept on the server, so a look saved in one browser
// appears in another. Two browser contexts share nothing but the local edition's one user and
// workspace: context A imports fixtures/hello-a09b/look.json on Play's Looks page and waits for the
// PUT to /api/themes/looks/automatos/play; context B, opened before the save with none of A's
// storage, shows the look after a reload, read from the server.
//
// A-19f4 (W-A8): until then this ran under builtin Play; it runs under the installed package now
// (support/play-parity.ts), whose look runtime keeps the looks under the same theme id. The
// package and the member's choice live on the lane's one backend, so the spec is @package and runs
// in the package step, one test at a time, which also keeps a second project from saving at the
// same moment (a save replaces the user's whole list): both projects run it.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const LOOKS_URL = `${APP_URL}/looks`
const LOOKS_API_PATH = `/api/themes/looks/${PLAY_THEME_ID}`
const FIXTURE = path.resolve(__dirname, '..', 'fixtures', 'hello-a09b', 'look.json')
const DESKTOP_VIEWPORT = { width: 1440, height: 900 }
const OK = 200
/** Room for the store's pause before it sends (LOOK_SEND_DELAY_MS) and the round trip. */
const SAVE_TIMEOUT_MS = 20_000

/** The fixture look under a name only this run uses. */
function uniqueLook(): { readonly name: string; readonly json: string } {
  const look = JSON.parse(fs.readFileSync(FIXTURE, 'utf8')) as Record<string, unknown>
  const name = `Hello a09b ${Date.now().toString(36)}`
  return { name, json: JSON.stringify({ ...look, name }) }
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** The card for `name` in the Looks row (its accessible name starts with the look's name). */
function lookCard(page: Page, name: string): Locator {
  return page.getByRole('region', { name: 'Looks' }).getByRole('button', { name: new RegExp(`^${escapeRegExp(name)}\\b`) })
}

/** Puts the page on installed Play and opens its Looks page, once the host has loaded the package. */
async function openLooks(page: Page, play: AvailablePlay): Promise<void> {
  await play.choose(page)
  await page.goto(LOOKS_URL)
  await play.ready(page)
  await expect(page.getByRole('region', { name: 'Import a look' })).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
}

function isLooksSave(method: string, url: string): boolean {
  return method === 'PUT' && new URL(url).pathname === LOOKS_API_PATH
}

describeParity("Play's looks kept on the server", (variant) => {
  test('a look saved in one browser appears in another after reload', { tag: ['@package', '@looks'] }, async ({ page, browser, request }) => {
    const play = await variant.makeAvailable(request)
    const look = uniqueLook()
    const other = await browser.newContext({ viewport: DESKTOP_VIEWPORT })
    try {
      const second = await other.newPage()
      await openLooks(second, play)
      await expect(lookCard(second, look.name)).toHaveCount(0)

      await openLooks(page, play)
      const importer = page.getByRole('region', { name: 'Import a look' })
      await importer.getByLabel("Paste a Look's JSON").fill(look.json)
      const saved = page.waitForResponse((response) => isLooksSave(response.request().method(), response.url()), { timeout: SAVE_TIMEOUT_MS })
      await importer.getByRole('button', { name: 'Import pasted JSON' }).click()
      await expect(importer.getByText(`Added ${look.name} to your Looks.`)).toBeVisible()
      expect((await saved).status()).toBe(OK)

      await second.reload()
      await play.ready(second)
      await expect(lookCard(second, look.name)).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
    } finally {
      await other.close()
    }
  })
})
