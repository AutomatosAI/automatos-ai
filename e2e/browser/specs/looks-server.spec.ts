import fs from 'fs'
import path from 'path'
import { expect, test, type Locator, type Page } from '@playwright/test'
import { chooseStyle } from '../support/hello-theme'

// PRD-261 A-09b (@host): builtin Play's looks are kept on the server, so a look saved in one
// browser appears in another. Two browser contexts share nothing but the local edition's one
// user and workspace: context A imports fixtures/hello-a09b/look.json on Play's Looks page and
// waits for the PUT to /api/themes/looks/automatos/play; context B, opened before the save with
// none of A's storage, shows the look after a reload, read from the server.
//
// Chromium only: the projects run in parallel against one backend user, and a save replaces the
// user's whole list, so a second project saving at the same moment could replace this one's.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const PLAY_STYLE = 'theme:builtin:play'
const LOOKS_URL = `${APP_URL}/looks`
const LOOKS_API_PATH = '/api/themes/looks/automatos/play'
const FIXTURE = path.resolve(__dirname, '..', 'fixtures', 'hello-a09b', 'look.json')
const DESKTOP_VIEWPORT = { width: 1440, height: 900 }
const ONLY_PROJECT = 'chromium'
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

async function openLooks(page: Page): Promise<void> {
  await chooseStyle(page, PLAY_STYLE, APP_URL)
  await page.goto(LOOKS_URL)
  await expect(page.getByRole('region', { name: 'Import a look' })).toBeVisible()
}

function isLooksSave(method: string, url: string): boolean {
  return method === 'PUT' && new URL(url).pathname === LOOKS_API_PATH
}

test('a look saved in one browser appears in another after reload', { tag: ['@host', '@looks'] }, async ({ page, browser }, testInfo) => {
  test.skip(testInfo.project.name !== ONLY_PROJECT, 'one project: a save replaces the shared user\'s whole list')
  const look = uniqueLook()
  const other = await browser.newContext({ viewport: DESKTOP_VIEWPORT })
  try {
    const second = await other.newPage()
    await openLooks(second)
    await expect(lookCard(second, look.name)).toHaveCount(0)

    await openLooks(page)
    const importer = page.getByRole('region', { name: 'Import a look' })
    await importer.getByLabel("Paste a Look's JSON").fill(look.json)
    const saved = page.waitForResponse((response) => isLooksSave(response.request().method(), response.url()), { timeout: SAVE_TIMEOUT_MS })
    await importer.getByRole('button', { name: 'Import pasted JSON' }).click()
    await expect(importer.getByText(`Added ${look.name} to your Looks.`)).toBeVisible()
    expect((await saved).status()).toBe(OK)

    await second.reload()
    await expect(lookCard(second, look.name)).toBeVisible()
  } finally {
    await other.close()
  }
})
