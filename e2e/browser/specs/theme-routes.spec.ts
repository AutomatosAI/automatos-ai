import { expect, test, type Page } from '@playwright/test'
import { chooseStyle, countMounts, mountsOf, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-261 A-04b (@host): theme pages and routes through ThemeSlot, with the hello theme
// (fixtures/hello-a04b: a frame that only wraps the page, and the route `hello`), served as an
// installed theme through route interception (support/hello-theme.ts). A-19f4 (W-A8): until then
// tests 1 to 3 ran under builtin Play (the `play` cookie); they run on the hello theme now, and
// specs/play-parity/play-routes.spec.ts holds the installed Play package to the same.
//
// 1. Under the installed theme, /chat at 1440 px is the core chat inside the theme's frame, and at
//    800 px, below the width the host draws an installed theme's frame from, the core chat with no
//    frame.
// 2. On a cold load of /chat under the theme, the theme's frame mounts once: never the Studio shell,
//    never the core composer outside the frame, never a second frame (P260-FX-25, M69). An init
//    script counts every element of each kind the page ever inserts.
// 3. The theme registers the route `hello`; /x/hello renders it inside the theme's frame.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const CHAT_PATH = '/chat'
const HELLO_ROUTE_PATH = '/x/hello'
const NARROW_VIEWPORT = { width: 800, height: 900 }
const STATUS_ATTR = 'data-theme-host'

const HELLO_A04B: InstalledTheme = { installId: '00000000-0000-4000-8000-0000000a04b0', fixture: 'hello-a04b' }
const HELLO_FRAME_TEST_ID = 'hello-a04b-frame'
const HELLO_ROUTE_TEST_ID = 'hello-a04b-route'
/** The hello theme's frame (fixtures/hello-a04b/entry.js RouteFrame). */
const HELLO_FRAME = `[data-testid="${HELLO_FRAME_TEST_ID}"]`
/** The Studio desktop shell (components/layout/main-layout.tsx). */
const STUDIO_SHELL = '.sh-shell'
/** The core chat's composer (components/chatbot/multimodal-input.tsx). */
const CORE_COMPOSER = 'textarea[placeholder="Send a message..."]'
/** The core chat's composer anywhere but inside the theme's frame: drawn in a chrome before the frame. */
const CORE_COMPOSER_OUTSIDE_FRAME = `${CORE_COMPOSER}:not(${HELLO_FRAME} *)`

/** How long a cold load has to show a second mount, once the first has drawn. */
const SETTLE_MS = 1_500
const COUNTED = [HELLO_FRAME, STUDIO_SHELL, CORE_COMPOSER_OUTSIDE_FRAME]

/** Puts this browser on the hello theme's install and serves it (its active answer and its files). */
async function chooseHello(page: Page): Promise<void> {
  const served = await serveInstalledTheme(page, HELLO_A04B)
  await chooseStyle(page, served.style, APP_URL)
}

test.describe('theme pages and routes through ThemeSlot', () => {
  test("under an installed theme /chat at 1440 px is the core chat inside the theme's frame", { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    await chooseHello(page)
    await page.goto(CHAT_PATH)

    await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
    const frame = page.getByTestId(HELLO_FRAME_TEST_ID)
    await expect(frame).toBeVisible()
    await expect(frame.locator(CORE_COMPOSER).filter({ visible: true }).first()).toBeVisible()
    await expect(page.locator(STUDIO_SHELL)).toHaveCount(0)
    expect(watch.pageErrors).toEqual([])
  })

  test("under an installed theme /chat at 800 px is the core chat, with no theme frame", { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    await page.setViewportSize(NARROW_VIEWPORT)
    await chooseHello(page)
    await page.goto(CHAT_PATH)

    await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
    await expect(page.locator(CORE_COMPOSER).filter({ visible: true }).first()).toBeVisible()
    await expect(page.getByTestId(HELLO_FRAME_TEST_ID)).toHaveCount(0)
    expect(watch.pageErrors).toEqual([])
  })

  test("a cold load of /chat under an installed theme mounts its frame once, and never the Studio shell or the core chat outside it", { tag: '@host' }, async ({ page }) => {
    await countMounts(page, COUNTED)
    await chooseHello(page)
    await page.goto(CHAT_PATH)

    const frame = page.getByTestId(HELLO_FRAME_TEST_ID)
    await expect(frame.locator(CORE_COMPOSER).filter({ visible: true }).first()).toBeVisible()
    await page.waitForTimeout(SETTLE_MS)
    expect(await mountsOf(page)).toEqual({ [HELLO_FRAME]: 1 })
  })

  test("an installed theme's registerRoute('hello') renders at /x/hello, inside its frame", { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    await chooseHello(page)
    await page.goto(HELLO_ROUTE_PATH)

    await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
    const frame = page.getByTestId(HELLO_FRAME_TEST_ID)
    await expect(frame).toBeVisible()
    await expect(frame.getByTestId(HELLO_ROUTE_TEST_ID)).toBeVisible()
    expect(watch.pageErrors).toEqual([])
  })
})
