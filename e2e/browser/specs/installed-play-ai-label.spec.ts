import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { noStoredConversations } from '../support/chat-stream'
import { chooseKitPackage, installKitPackage, resetKitPackage } from '../support/kit-package'
import { messageBox, sendButton, voiceButton } from '../support/play-chat'
import { THEME_LOAD_TIMEOUT_MS } from '../support/play-parity'
import { watchPage } from '../support/watch'

// PRD-261 A-19e1 (@package), PRD-264 EUAI-11a (EU AI Act Art. 50): installed Play's /chat carries the
// host's AI label, and no style can hide it: a style that hides the label brings the host's own copy of it,
// and Play's chat keeps running. Play is the Design Kit's package, uploaded, opted in to and chosen as
// installed-play.spec.ts does (support/kit-package.ts); the conversations list is answered with none, so the
// chat opens on a new chat, and nothing is sent.
//
// The label is the host's (components/ai-label/ai-label.tsx, drawn by app/chat/layout.tsx after the
// theme's page): role="note", "Auto is an AI", once, outside Play's page, frame and composer. Since
// Play 1.2.0 its message box (the kit's themes/play/src/components/play/chat/message-box.tsx) holds the
// words, then the host's voice button (the SDK's voice.VoiceButton), then Send, the words and Send each in
// the SDK's cards.AiLabelFence, which now draws its children alone. While a style keeps the label from being
// seen, the host draws the same words in its own layer (components/ai-label/protected-caption.tsx): a shadow
// root on an element of the host's at the end of <body>, guarded inline, the words a popover in the browser's
// top layer. A second style aimed at that copy, its element and anything like it leaves it on screen. The
// words, Send and the voice button stay enabled throughout. This is the browser half of
// lib/theme-host/__tests__/ai-label.test.ts (jsdom has no top layer and matches a document's rules inside
// shadow roots), on installed Play, the only Play since W-A8 took builtin Play out of core.
//
// The specs store the member's choice on the lane's one backend, so they run in the @package step,
// one test at a time; each starts and ends on Studio with no install of Play's package.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const CHAT_PATH = '/chat'
const STATUS_ATTR = 'data-theme-host'
const PLAY_CLASS = /\bplay\b/
/** The host's label (lib/ai-label/copy.ts aiLabelText, with no agent picked). */
const AI_LABEL = 'Auto is an AI'
/** components/ai-label/ai-label.tsx AI_LABEL_ATTR, and protected-caption.tsx AI_LABEL_CAPTION_ATTR. */
const LABEL_SELECTOR = '[data-ai-label]'
const CAPTION_SELECTOR = '[data-ai-label-caption]'
/** Play's frame: its sidebar (the kit's themes/play/src/components/play/shell/play-sidebar.tsx). */
const PLAY_FRAME = 'aside[aria-label="Sidebar"]'
/** A stylesheet that hides the host's label under installed Play's own <html> class, as a hostile theme's would. */
const HIDING_STYLE = `html.play ${LABEL_SELECTOR} { display: none !important; }`
/** A stylesheet aimed at the host's copy: its element by name, and anything a generic rule could catch it by. */
const AIMED_AT_COPY =
  `html.play ${CAPTION_SELECTOR}, html.play ${CAPTION_SELECTOR} *, html.play [popover], html.play p[role="note"] {` +
  ' display: none !important; visibility: hidden !important; opacity: 0 !important;' +
  ' transform: translateY(-200vh) !important; }'
/** Words in the box, so Send is enabled (Send stays disabled on an empty box). */
const DRAFT = 'Hello Auto'

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

/** The host's copy of the label in its own layer (the role engine reads inside the open shadow root). */
function hostCopy(page: Page) {
  return page.locator(CAPTION_SELECTOR).getByRole('note', { name: AI_LABEL, exact: true })
}

/** Play's words, Send and the voice button all run, the draft still in the box. */
async function expectChatRuns(page: Page): Promise<void> {
  await expect(messageBox(page)).toBeEnabled()
  await expect(messageBox(page)).toHaveValue(DRAFT)
  await expect(sendButton(page)).toBeEnabled()
  await expect(voiceButton(page)).toBeEnabled()
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
    await expect(page.locator(CAPTION_SELECTOR)).toHaveCount(0)
    await messageBox(page).fill(DRAFT)
    await expectChatRuns(page)
    expect(watch.pageErrors).toEqual([])
  })

  test("a style that hides the label brings the host's own copy, which a style aimed at it can't hide; the chat runs throughout", { tag: '@package' }, async ({ page, request }) => {
    await openInstalledPlayChat(page, request)
    await messageBox(page).fill(DRAFT)
    await expectChatRuns(page)

    const hiding = await page.addStyleTag({ content: HIDING_STYLE })
    await expect(page.locator(LABEL_SELECTOR)).toBeHidden()
    await expect(hostCopy(page)).toBeVisible()
    await expect(hostCopy(page)).toHaveText(AI_LABEL)
    await expect(hostCopy(page)).toBeInViewport()
    expect(await hostCopy(page).evaluate((node: Element) => node.matches(':popover-open'))).toBe(true)
    await expectChatRuns(page)

    const aimed = await page.addStyleTag({ content: AIMED_AT_COPY })
    await expect(hostCopy(page)).toBeVisible()
    await expect(hostCopy(page)).toBeInViewport()
    expect(await hostCopy(page).evaluate((node: Element) => window.getComputedStyle(node).opacity)).toBe('1')
    // the host's element keeps its inline guard against the important rules aimed at it
    const guard = await page.locator(CAPTION_SELECTOR).evaluate((node: Element) => {
      const style = window.getComputedStyle(node)
      return [style.display, style.visibility, style.opacity, style.transform]
    })
    expect(guard).toEqual(['block', 'visible', '1', 'none'])
    await expectChatRuns(page)

    for (const style of [aimed, hiding]) {
      await style.evaluate((node: Node) => {
        node.parentNode?.removeChild(node)
      })
    }
    await expect(page.locator(LABEL_SELECTOR)).toBeVisible()
    await expect(page.locator(CAPTION_SELECTOR)).toHaveCount(0)
    await expectChatRuns(page)
  })
})
