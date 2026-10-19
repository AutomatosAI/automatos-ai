import { expect, test, type Page } from '@playwright/test'
import { chooseStyle, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-261 A-04b (@host): theme pages and routes through ThemeSlot.
//
// 1. Under the Play cookie, /chat is Play's chat inside Play's frame at 1440 px (builtin Play
//    registers it as its /chat page, lib/theme-host/builtin/play-slots.tsx), and the core chat
//    at 800 px, below Play's width.
// 2. On a cold load of /chat under Play, Play's frame mounts once: never the Studio shell or the
//    core composer first, never a second frame (P260-FX-25, M69). An init script counts every
//    element of each kind the page ever inserts.
// 3. /x/conversations renders Play's Conversations (builtin Play's route `conversations`).
// 4. The hello theme (fixtures/hello-a04b) registers the route `hello`; /x/hello renders it inside
//    the theme's frame.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const PLAY_COOKIE = 'play'
const CHAT_PATH = '/chat'
const CONVERSATIONS_PATH = '/x/conversations'
const HELLO_ROUTE_PATH = '/x/hello'
const NARROW_VIEWPORT = { width: 800, height: 900 }
const PLAY_CLASS = /\bplay\b/
const STATUS_ATTR = 'data-theme-host'

/** Play's frame: its sidebar (components/play/shell/play-sidebar.tsx). */
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

const HELLO_A04B: InstalledTheme = { installId: '00000000-0000-4000-8000-0000000a04b0', fixture: 'hello-a04b' }
const HELLO_FRAME_TEST_ID = 'hello-a04b-frame'
const HELLO_ROUTE_TEST_ID = 'hello-a04b-route'

/** How long a cold load has to show a second mount, once the first has drawn. */
const SETTLE_MS = 1_500

const COUNTED = [PLAY_FRAME, STUDIO_SHELL, CORE_COMPOSER] as const
type MountWindow = Window & { __a04bMounts?: Record<string, number> }

/** Counts, from before the first byte, every distinct element matching each selector the page inserts. */
async function countMounts(page: Page): Promise<void> {
  await page.addInitScript((selectors: readonly string[]) => {
    const seen = new WeakSet<Element>()
    const counts: Record<string, number> = {}
    ;(window as MountWindow).__a04bMounts = counts
    const tally = (node: Node): void => {
      if (!(node instanceof Element)) return
      for (const selector of selectors) {
        const hits = Array.from(node.querySelectorAll(selector))
        if (node.matches(selector)) hits.push(node)
        for (const hit of hits.filter((element) => !seen.has(element))) {
          seen.add(hit)
          counts[selector] = (counts[selector] ?? 0) + 1
        }
      }
    }
    new MutationObserver((records) => records.forEach((record) => record.addedNodes.forEach(tally))).observe(document, {
      childList: true,
      subtree: true,
    })
  }, COUNTED)
}

async function mountsOf(page: Page): Promise<Record<string, number>> {
  return page.evaluate(() => ({ ...((window as MountWindow).__a04bMounts ?? {}) }))
}

function playChat(page: Page) {
  return page
    .getByText(PLAY_OPENING)
    .or(page.getByRole('region', { name: PLAY_HELLO }))
    .or(page.getByRole('textbox', { name: PLAY_COMPOSER }))
    .first()
}

test.describe('theme pages and routes through ThemeSlot', () => {
  test("under the Play cookie /chat is Play's chat in Play's frame at 1440 px", { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    await chooseStyle(page, PLAY_COOKIE, APP_URL)
    await page.goto(CHAT_PATH)

    await expect(page.locator('html')).toHaveClass(PLAY_CLASS)
    await expect(page.locator(PLAY_FRAME)).toBeVisible()
    await expect(playChat(page)).toBeVisible()
    await expect(page.locator(CORE_COMPOSER)).toHaveCount(0)
    expect(watch.pageErrors).toEqual([])
  })

  test('under the Play cookie /chat is the core chat at 800 px, below Play\'s width', { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    await page.setViewportSize(NARROW_VIEWPORT)
    await chooseStyle(page, PLAY_COOKIE, APP_URL)
    await page.goto(CHAT_PATH)

    await expect(page.locator('html')).toHaveClass(PLAY_CLASS)
    await expect(page.locator(CORE_COMPOSER).filter({ visible: true }).first()).toBeVisible()
    await expect(page.locator(PLAY_FRAME)).toHaveCount(0)
    await expect(playChat(page)).toHaveCount(0)
    expect(watch.pageErrors).toEqual([])
  })

  test("a cold load of /chat under Play mounts Play's frame once, and never the Studio shell or the core chat", { tag: '@host' }, async ({ page }) => {
    await countMounts(page)
    await chooseStyle(page, PLAY_COOKIE, APP_URL)
    await page.goto(CHAT_PATH)

    await expect(playChat(page)).toBeVisible()
    await page.waitForTimeout(SETTLE_MS)
    expect(await mountsOf(page)).toEqual({ [PLAY_FRAME]: 1 })
  })

  test("/x/conversations renders Play's Conversations under the Play cookie", { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    await chooseStyle(page, PLAY_COOKIE, APP_URL)
    await page.goto(CONVERSATIONS_PATH)

    await expect(page.locator(PLAY_FRAME)).toBeVisible()
    await expect(page.getByRole('heading', { level: 1, name: CONVERSATIONS_HEADING })).toBeVisible()
    await expect(page.getByRole('searchbox', { name: CONVERSATIONS_SEARCH }).or(page.getByLabel(CONVERSATIONS_SEARCH)).first()).toBeVisible()
    expect(watch.pageErrors).toEqual([])
  })

  test("an installed theme's registerRoute('hello') renders at /x/hello, inside its frame", { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    await chooseStyle(page, `theme:install:${HELLO_A04B.installId}`, APP_URL)
    await serveInstalledTheme(page, HELLO_A04B)
    await page.goto(HELLO_ROUTE_PATH)

    await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
    const frame = page.getByTestId(HELLO_FRAME_TEST_ID)
    await expect(frame).toBeVisible()
    await expect(frame.getByTestId(HELLO_ROUTE_TEST_ID)).toBeVisible()
    expect(watch.pageErrors).toEqual([])
  })
})
