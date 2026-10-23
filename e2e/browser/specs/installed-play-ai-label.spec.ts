import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { noStoredConversations } from '../support/chat-stream'
import { chooseKitPackage, installKitPackage, resetKitPackage } from '../support/kit-package'
import { MESSAGE_BOX, messageBox } from '../support/play-chat'
import { THEME_LOAD_TIMEOUT_MS } from '../support/play-parity'
import { watchPage } from '../support/watch'

// PRD-261 A-19e1 (@package), PRD-264 EUAI-11a (EU AI Act Art. 50): installed Play's /chat carries the
// host's AI label, and a style that hides it pauses Play's composer. Play is the Design Kit's package,
// uploaded, opted in to and chosen as installed-play.spec.ts does (support/kit-package.ts); the
// conversations list is answered with none, so the chat opens on a new chat, and nothing is sent.
//
// The label is the host's (components/ai-label/ai-label.tsx, drawn by app/chat/layout.tsx after the
// theme's page): role="note", "Auto is an AI", once, outside Play's page, frame and composer. Play's
// message box sits in the SDK's cards.AiLabelFence (the kit's themes/play/src/components/play/chat/
// message-box.tsx), so the host's check of the label disables it while a style keeps the label from
// being seen, and says why. This is the browser half of lib/theme-host/__tests__/ai-label.test.ts,
// on installed Play, the only Play since W-A8 took builtin Play out of core.
//
// The specs store the member's choice on the lane's one backend, so they run in the @package step,
// one test at a time; each starts and ends on Studio with no install of Play's package.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const CHAT_PATH = '/chat'
const STATUS_ATTR = 'data-theme-host'
const PLAY_CLASS = /\bplay\b/
/** The host's label (lib/ai-label/copy.ts aiLabelText, with no agent picked). */
const AI_LABEL = 'Auto is an AI'
/** components/ai-label/ai-label.tsx AI_LABEL_ATTR, and ai-label-fence.tsx AI_LABEL_FENCE_ATTR. */
const LABEL_SELECTOR = '[data-ai-label]'
const FENCE_SELECTOR = '[data-ai-label-fence]'
/** lib/ai-label/copy.ts AI_LABEL_PAUSED_TEXT. */
const PAUSED_TEXT = 'This theme hides the AI notice, so chat is paused. Switch theme in Settings.'
/** Play's frame: its sidebar (the kit's themes/play/src/components/play/shell/play-sidebar.tsx). */
const PLAY_FRAME = 'aside[aria-label="Sidebar"]'
/** A stylesheet that hides the host's label under installed Play's own <html> class, as a hostile theme's would. */
const HIDING_STYLE = `html.play ${LABEL_SELECTOR} { display: none !important; }`

/** Installs Play's package, chooses it, and opens /chat on a new chat; the message box is on screen. */
async function openInstalledPlayChat(page: Page, request: APIRequestContext): Promise<void> {
  const install = await installKitPackage(request)
  await chooseKitPackage(page, install, APP_URL)
  await noStoredConversations(page)
  await page.goto(CHAT_PATH)
  const html = page.locator('html')
  await expect(html).toHaveAttribute(STATUS_ATTR, 'active', { timeout: THEME_LOAD_TIMEOUT_MS })
  await expect(html).toHaveClass(PLAY_CLASS)
  await expect(messageBox(page)).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
}

/** The pause notice the host's gate shows while the label is hidden (Next's route announcer is an alert too). */
function pauseNotice(page: Page) {
  return page.getByRole('alert').filter({ hasText: PAUSED_TEXT })
}

test.describe("PRD-261 A-19e1 — installed Play's /chat carries the host's AI label (PRD-264)", () => {
  test.beforeEach(async ({ request }) => {
    await resetKitPackage(request)
  })

  test.afterEach(async ({ request }) => {
    await resetKitPackage(request)
  })

  test("shows the label once, outside Play's page, frame and composer, and the message box runs", { tag: '@package' }, async ({ page, request }) => {
    const watch = watchPage(page)
    await openInstalledPlayChat(page, request)

    const label = page.getByRole('note', { name: AI_LABEL, exact: true })
    await expect(label).toBeVisible()
    await expect(label).toHaveText(AI_LABEL)
    await expect(label).toBeInViewport()
    await expect(page.locator(LABEL_SELECTOR)).toHaveCount(1)
    await expect(page.locator(PLAY_FRAME)).toBeVisible()
    await expect(page.locator(`${PLAY_FRAME} ${LABEL_SELECTOR}`)).toHaveCount(0)
    await expect(page.locator(`main ${LABEL_SELECTOR}`)).toHaveCount(0)
    await expect(page.locator(`${FENCE_SELECTOR} ${LABEL_SELECTOR}`)).toHaveCount(0)
    await expect(page.locator(FENCE_SELECTOR).getByRole('textbox', { name: MESSAGE_BOX, exact: true })).toHaveCount(1)
    await expect(messageBox(page)).toBeEnabled()
    await expect(pauseNotice(page)).toHaveCount(0)
    expect(watch.pageErrors).toEqual([])
  })

  test('a style that hides the label pauses the message box and says why; taking it away lifts the pause', { tag: '@package' }, async ({ page, request }) => {
    await openInstalledPlayChat(page, request)
    await expect(messageBox(page)).toBeEnabled()

    const hiding = await page.addStyleTag({ content: HIDING_STYLE })
    await expect(page.locator(LABEL_SELECTOR)).toBeHidden()
    await expect(messageBox(page)).toBeDisabled()
    await expect(pauseNotice(page)).toBeVisible()

    await hiding.evaluate((node: Node) => {
      node.parentNode?.removeChild(node)
    })
    await expect(page.getByRole('note', { name: AI_LABEL, exact: true })).toBeVisible()
    await expect(messageBox(page)).toBeEnabled()
    await expect(pauseNotice(page)).toHaveCount(0)
  })
})
