import { expect, test, type Locator, type Page } from '@playwright/test'
import { countMounts, mountsOf } from '../../support/hello-theme'
import { describeParity, openPlay, PARITY_TAG, THEME_LOAD_TIMEOUT_MS } from '../../support/play-parity'
import { watchPage } from '../../support/watch'

// PRD-261 A-04b, A-19f4 (@parity): Play's pages and its route through ThemeSlot, under the installed
// package. Until W-A8, specs/theme-routes.spec.ts proved these under builtin Play (the `play`
// cookie); that spec now proves the same mechanics with the hello theme (fixtures/hello-a04b), and
// these tests hold Play itself to them, as the package registers them (the kit's
// themes/play/src/entry.ts: registerPage('/chat'), registerRoute('conversations')).
//
// 1. /chat is Play's chat inside Play's frame at 1440 px, and the core chat at 800 px, below Play's
//    width.
// 2. On a cold load of /chat, Play's frame mounts once: never the Studio shell or the core composer
//    first, never a second frame (P260-FX-25, M69). An init script counts every element of each
//    kind the page ever inserts.
// 3. /x/conversations renders Play's Conversations.

const CHAT_PATH = '/chat'
const CONVERSATIONS_PATH = '/x/conversations'
const NARROW_VIEWPORT = { width: 800, height: 900 }
/** Play's frame: its sidebar (the kit's themes/play/src/components/play/shell/play-sidebar.tsx). */
const PLAY_FRAME = 'aside[aria-label="Sidebar"]'
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

  test("/chat is the core chat at 800 px, below Play's width", { tag: PARITY_TAG }, async ({ page, request }) => {
    const watch = watchPage(page)
    await page.setViewportSize(NARROW_VIEWPORT)
    await openPlay(page, request, variant, CHAT_PATH)

    await expect(page.locator(CORE_COMPOSER).filter({ visible: true }).first()).toBeVisible()
    await expect(page.locator(PLAY_FRAME)).toHaveCount(0)
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
