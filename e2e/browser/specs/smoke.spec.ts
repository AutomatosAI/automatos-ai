import { expect, test, type Locator, type Page } from '@playwright/test'
import { chooseCoreStyle, CORE_STYLES, STUDIO_STYLE, type CoreStyle } from '../support/hello-theme'
import { watchPage } from '../support/watch'

type Box = NonNullable<Awaited<ReturnType<Locator['boundingBox']>>>

// PRD-261 A-00: the lane's sample test. A fresh browser on the local edition gets Studio, the
// app reaches the local backend for its workspace, and the Studio desktop shell draws.
// PRD-261 A-19f4 (W-A8): the same under the Classic and under the Studio style cookie, each style
// drawing its own desktop chrome (support/hello-theme.ts CORE_STYLES).
// Theme voice: on a 390 px phone, Studio's chat composer keeps its box, the quick ask's voice button
// (components/chatbot/composer-voice.tsx) and Send in one row inside the screen, the button enabled
// because app/chat/layout.tsx mounts the host voice provider.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const SMOKE_PATH = '/agents'
const CHAT_PATH = '/chat'
const WORKSPACE_PATH = '/api/workspaces/current'
const STUDIO_CLASS = /\bstudio\b/
const OK = 200
const PHONE = { width: 390, height: 844 }
/** The core chat's composer (components/chatbot/multimodal-input.tsx), its voice button and Send. */
const CORE_COMPOSER = 'textarea[placeholder="Send a message..."]'
const HOLD_TO_TALK = 'Hold to talk'
const SEND = 'Send message'
/** Rounding between two boxes laid out on the same row. */
const SAME_ROW_PX = 1

/** Opens /agents and checks the workspace answered, `core`'s chrome drew and nothing threw. */
async function expectShellDraws(page: Page, core: CoreStyle): Promise<void> {
  const pageErrors: Error[] = []
  page.on('pageerror', (error) => pageErrors.push(error))

  // Registered before the navigation so the response cannot be missed. The method check skips
  // the CORS preflight (OPTIONS), which the browser sends first because :3000 calls :8000.
  const workspaceResponse = page.waitForResponse(
    (response) => response.url().endsWith(WORKSPACE_PATH) && response.request().method() === 'GET',
  )
  await page.goto(SMOKE_PATH)

  expect((await workspaceResponse).status()).toBe(OK)
  if (core.studio) await expect(page.locator('html')).toHaveClass(STUDIO_CLASS)
  else await expect(page.locator('html')).not.toHaveClass(STUDIO_CLASS)
  await expect(page.locator(core.shell)).toBeVisible()
  expect(pageErrors.map((error) => error.message)).toEqual([])
}

test('the local edition draws the Studio shell on /agents', { tag: '@smoke' }, async ({ page }) => {
  await expectShellDraws(page, STUDIO_STYLE)
})

for (const core of CORE_STYLES) {
  test(`under the '${core.style}' cookie the local edition draws the ${core.style} chrome on /agents`, { tag: '@smoke' }, async ({ page }) => {
    await chooseCoreStyle(page, core, APP_URL)
    await expectShellDraws(page, core)
  })
}

/** The element's box, which a visible element always has. */
async function boxOf(locator: Locator): Promise<Box> {
  const box = await locator.boundingBox()
  expect(box).not.toBeNull()
  return box ?? { x: 0, y: 0, width: 0, height: 0 }
}

test("on a 390 px phone Studio's chat composer has its box, the voice button and Send in one row on screen", { tag: '@smoke' }, async ({ page }) => {
  const watch = watchPage(page)
  await page.setViewportSize(PHONE)
  await page.goto(CHAT_PATH)

  const composer = page.locator(CORE_COMPOSER).filter({ visible: true }).first()
  const voice = page.getByRole('button', { name: HOLD_TO_TALK, exact: true }).filter({ visible: true }).first()
  const send = page.getByRole('button', { name: SEND, exact: true }).filter({ visible: true }).first()
  await expect(composer).toBeVisible()
  await expect(voice).toBeVisible()
  await expect(voice).toBeEnabled()
  await expect(send).toBeVisible()

  const [box, voiceBox, sendBox] = await Promise.all([boxOf(composer), boxOf(voice), boxOf(send)])
  expect(box.x).toBeGreaterThanOrEqual(0)
  expect(box.x + box.width).toBeLessThanOrEqual(PHONE.width)
  expect(voiceBox.x).toBeGreaterThanOrEqual(box.x)
  expect(voiceBox.x + voiceBox.width).toBeLessThanOrEqual(sendBox.x)
  expect(sendBox.x + sendBox.width).toBeLessThanOrEqual(box.x + box.width)
  expect(Math.abs(voiceBox.height - sendBox.height)).toBeLessThanOrEqual(SAME_ROW_PX)
  expect(Math.abs(voiceBox.y - sendBox.y)).toBeLessThanOrEqual(SAME_ROW_PX)
  expect(watch.pageErrors).toEqual([])
})
