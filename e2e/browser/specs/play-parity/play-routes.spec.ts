import { expect, test, type Locator, type Page } from '@playwright/test'
import { countMounts, mountsOf } from '../../support/hello-theme'
import { messageBox, openPlayChat, sendButton, voiceButton } from '../../support/play-chat'
import { describeParity, openPlay, PARITY_TAG, THEME_LOAD_TIMEOUT_MS } from '../../support/play-parity'
import { watchPage } from '../../support/watch'

type Box = NonNullable<Awaited<ReturnType<Locator['boundingBox']>>>

// PRD-261 A-04b, A-19f4 (@parity): Play's pages and its route through ThemeSlot, under the installed
// package. Until W-A8, specs/theme-routes.spec.ts proved these under builtin Play (the `play`
// cookie); that spec now proves the same mechanics with the hello theme (fixtures/hello-a04b), and
// these tests hold Play itself to them, as the package registers them (the kit's
// themes/play/src/entry.ts: registerPage('/chat'), registerRoute('conversations')).
//
// 1. /chat is Play's chat inside Play's frame at 1440 px, and inside Play's phone frame at 800 px
//    (Play 1.2.0 draws from its minWidth, 360 px, up; its phone layout below 1024 px). At 390 x 844
//    its message box holds the words, the host's voice button and Send, on the screen, with no core
//    composer and no sideways scroll. At 340 px, below Play's minWidth, /chat is the core chat.
// 2. On a cold load of /chat, Play's frame mounts once: never the Studio shell or the core composer
//    first, never a second frame (P260-FX-25, M69). An init script counts every element of each
//    kind the page ever inserts.
// 3. /x/conversations renders Play's Conversations.

const CHAT_PATH = '/chat'
const CONVERSATIONS_PATH = '/x/conversations'
/** Play's phone layout: narrower than 1024 px (the kit's themes/play/src/lib/play/phone-layout.ts). */
const PHONE_FRAME_VIEWPORT = { width: 800, height: 900 }
const PHONE_VIEWPORT = { width: 390, height: 844 }
/** Below Play's minWidth (360 px, the kit's themes/play/theme.json): the core's own chrome and pages. */
const BELOW_MIN_WIDTH = { width: 340, height: 800 }
/** Play's desktop frame: its sidebar (the kit's themes/play/src/components/play/shell/play-sidebar.tsx). */
const PLAY_FRAME = 'aside[aria-label="Sidebar"]'
/** Play's phone frame: its top bar (the kit's themes/play/src/components/play/shell/phone-bar.tsx TOP_BAR_LABEL). */
const PHONE_TOP_BAR = 'header[aria-label="Top bar"]'
/** The Studio desktop shell (components/layout/main-layout.tsx). */
const STUDIO_SHELL = '.sh-shell'
/** The core chat's composer (components/chatbot/multimodal-input.tsx). */
const CORE_COMPOSER = 'textarea[placeholder="Send a message..."]'
/** Play's chat: Auto opening the conversation, the first-open hello, or Play's composer. */
const PLAY_OPENING = 'Opening this conversation…'
const PLAY_HELLO = 'Say hello'
const PLAY_COMPOSER = 'Message Auto'
const CONVERSATIONS_HEADING = 'Conversations'
const CONVERSATIONS_SEARCH = 'Search conversations'
/** How long a cold load has to show a second mount, once the first has drawn. */
const SETTLE_MS = 1_500
const COUNTED = [PLAY_FRAME, STUDIO_SHELL, CORE_COMPOSER]

/** The element's box, which a visible element always has. */
async function boxOf(locator: Locator): Promise<Box> {
  const box = await locator.boundingBox()
  expect(box).not.toBeNull()
  return box ?? { x: 0, y: 0, width: 0, height: 0 }
}

function playChat(page: Page): Locator {
  return page
    .getByText(PLAY_OPENING)
    .or(page.getByRole('region', { name: PLAY_HELLO }))
    .or(page.getByRole('textbox', { name: PLAY_COMPOSER }))
    .first()
}

describeParity("Play's pages and route through ThemeSlot", (variant) => {
  test("/chat is Play's chat in Play's frame at 1440 px", { tag: PARITY_TAG }, async ({ page, request }) => {
    const watch = watchPage(page)
    await openPlay(page, request, variant, CHAT_PATH)

    await expect(page.locator(PLAY_FRAME)).toBeVisible()
    await expect(playChat(page)).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
    await expect(page.locator(CORE_COMPOSER)).toHaveCount(0)
    expect(watch.pageErrors).toEqual([])
  })

  test("/chat is Play's chat in Play's phone frame at 800 px", { tag: PARITY_TAG }, async ({ page, request }) => {
    const watch = watchPage(page)
    await page.setViewportSize(PHONE_FRAME_VIEWPORT)
    await openPlay(page, request, variant, CHAT_PATH)

    await expect(page.locator(PHONE_TOP_BAR)).toBeVisible()
    await expect(page.locator(PLAY_FRAME)).toHaveCount(0)
    await expect(playChat(page)).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
    await expect(page.locator(CORE_COMPOSER)).toHaveCount(0)
    expect(watch.pageErrors).toEqual([])
  })

  test('at 390 x 844 /chat has the words, the voice button and Send on screen, no core composer and no sideways scroll', { tag: PARITY_TAG }, async ({ page, request }) => {
    const watch = watchPage(page)
    await page.setViewportSize(PHONE_VIEWPORT)
    await openPlayChat(page, request, variant, [])

    await expect(page.locator(PHONE_TOP_BAR)).toBeVisible()
    const [box, voice, send] = [messageBox(page), voiceButton(page), sendButton(page)]
    await expect(box).toBeVisible()
    await expect(voice).toBeVisible()
    await expect(voice).toBeEnabled()
    await expect(send).toBeVisible()
    await expect(page.locator(CORE_COMPOSER)).toHaveCount(0)

    const [boxAt, voiceAt, sendAt] = await Promise.all([boxOf(box), boxOf(voice), boxOf(send)])
    for (const at of [boxAt, voiceAt, sendAt]) {
      expect(at.x).toBeGreaterThanOrEqual(0)
      expect(at.x + at.width).toBeLessThanOrEqual(PHONE_VIEWPORT.width)
    }
    expect(voiceAt.x, 'the voice button stands after the words').toBeGreaterThanOrEqual(boxAt.x + boxAt.width)
    expect(voiceAt.x + voiceAt.width, 'and before Send').toBeLessThanOrEqual(sendAt.x)
    const widths = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, inner: window.innerWidth }))
    expect(widths.scroll, 'the page is no wider than the screen').toBeLessThanOrEqual(widths.inner)
    expect(watch.pageErrors).toEqual([])
  })

  test("/chat is the core chat at 340 px, below Play's minWidth", { tag: PARITY_TAG }, async ({ page, request }) => {
    const watch = watchPage(page)
    await page.setViewportSize(BELOW_MIN_WIDTH)
    await openPlay(page, request, variant, CHAT_PATH)

    await expect(page.locator(CORE_COMPOSER).filter({ visible: true }).first()).toBeVisible()
    await expect(page.locator(PLAY_FRAME)).toHaveCount(0)
    await expect(page.locator(PHONE_TOP_BAR)).toHaveCount(0)
    await expect(playChat(page)).toHaveCount(0)
    expect(watch.pageErrors).toEqual([])
  })

  test("a cold load of /chat mounts Play's frame once, and never the Studio shell or the core chat", { tag: PARITY_TAG }, async ({ page, request }) => {
    await countMounts(page, COUNTED)
    await openPlay(page, request, variant, CHAT_PATH)

    await expect(playChat(page)).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
    await page.waitForTimeout(SETTLE_MS)
    expect(await mountsOf(page)).toEqual({ [PLAY_FRAME]: 1 })
  })

  test("/x/conversations renders Play's Conversations", { tag: PARITY_TAG }, async ({ page, request }) => {
    const watch = watchPage(page)
    await openPlay(page, request, variant, CONVERSATIONS_PATH)

    await expect(page.locator(PLAY_FRAME)).toBeVisible()
    await expect(page.getByRole('heading', { level: 1, name: CONVERSATIONS_HEADING })).toBeVisible()
    await expect(page.getByRole('searchbox', { name: CONVERSATIONS_SEARCH }).or(page.getByLabel(CONVERSATIONS_SEARCH)).first()).toBeVisible()
    expect(watch.pageErrors).toEqual([])
  })
})
