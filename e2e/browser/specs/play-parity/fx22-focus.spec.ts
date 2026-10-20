import { expect, test, type Locator, type Page, type Route } from '@playwright/test'
import { fulfillBackendJson } from '../../support/backend-route'
import { chooseStyle } from '../../support/hello-theme'
import { ask, autoReplies, messageBox, openPlayChat } from '../../support/play-chat'
import {
  APP_URL,
  describeParity,
  openPlay,
  PARITY_TAG,
  THEME_LOAD_TIMEOUT_MS,
} from '../../support/play-parity'

// PRD-261 A-17b (@parity), P260-FX-22 (M20, M21): keyboard focus survives what Play unmounts and
// swaps, under builtin Play and the installed package.
//
// 1. M20: Try again on a turn that failed before Auto's reply began hands focus to the message box.
// 2. M21: when the backend stops answering, "Automatos is asleep" takes the focus the frame swap
//    dropped; when it answers again, the frame's <main> takes it, and a status line says so.
//    GET /health is answered 503 here until Check again.
// 3. M20: in Make it yours, This look's toolbar keeps focus through Rename, Save name, Delete,
//    Keep it and Delete look.
// 4. M21: Switch to Play on Studio's /looks note hands focus to the screen's heading. Onto the
//    package the switch reloads the page (PA-13); the note leaves word for the next load in
//    sessionStorage (lib/theme-host/switch-focus.ts, G3), and the heading takes focus once the
//    host has loaded the package, so the wait is the package's cold load.

const ASK = 'Find me a laptop under 1,000 euros'
const ANSWER = 'Here is your laptop shortlist.'
const SNAG = 'I hit a snag'
const TRY_AGAIN = 'Try again'
const HEALTH_PATH = '/health'
const SERVICE_UNAVAILABLE = 503
/** Four failed asks of GET /health (the first and BACKEND_HEALTH_RETRIES = 3, 2.5 s apart), with room. */
const ASLEEP_TIMEOUT_MS = 20_000
const ASLEEP_TITLE = 'Automatos is asleep'
const AWAKE_LINE = 'Automatos is back'
const CHECK_AGAIN = 'Check again'
const FRAME_PATH = '/agents'
const LOOKS_PATH = '/looks'
const STUDIO = 'studio'
const MANAGE = 'This look'
const RENAMED = 'Focus look'
const LOOKS_HEADING = 'Make it yours.'
const LOOKS_NOTE_REGION = 'Make it yours'
const SWITCH_TO_PLAY = 'Switch to Play'
const INSTALLED_PATH = '/api/themes/installed'
/** Play's frame puts its page in a <main tabIndex={-1}> (components/play/shell/play-frame.tsx), which takes focus after a nap. */
const FRAME_MAIN_TAG = 'MAIN'

/** The tag and tabindex of the focused element. */
function focusedElement(page: Page): Promise<{ readonly tag: string | null; readonly tabIndex: number | null }> {
  return page.evaluate(() => {
    const active = document.activeElement
    return { tag: active?.tagName ?? null, tabIndex: active instanceof HTMLElement ? active.tabIndex : null }
  })
}

function isHealth(url: URL): boolean {
  return url.pathname === HEALTH_PATH
}

async function answerUnavailable(route: Route): Promise<void> {
  await fulfillBackendJson(route, SERVICE_UNAVAILABLE, { detail: 'asleep' })
}

/** The button named `name` in This look: its toolbar and its rename and delete forms. */
function inThisLook(page: Page, name: string): Locator {
  return page.getByRole('region', { name: MANAGE, exact: true }).getByRole('button', { name, exact: true })
}

describeParity('keyboard focus survives what Play swaps (FX-22)', (variant) => {
  test('Try again on a failed turn hands focus to the message box', { tag: PARITY_TAG }, async ({ page, request }) => {
    await openPlayChat(page, request, variant, ['refused', 'answer'])
    await ask(page, ASK)
    const snag = page.getByRole('region', { name: SNAG, exact: true })
    await snag.getByRole('button', { name: TRY_AGAIN, exact: true }).click()
    await expect(messageBox(page)).toBeFocused()
    await expect(autoReplies(page).last()).toContainText(ANSWER)
  })

  test("asleep, the heading takes focus; awake again, the frame's main does", { tag: PARITY_TAG }, async ({ page, request }) => {
    await page.route(isHealth, answerUnavailable)
    await openPlay(page, request, variant, FRAME_PATH)
    await expect(page.getByRole('heading', { name: ASLEEP_TITLE, level: 1 })).toBeFocused({ timeout: ASLEEP_TIMEOUT_MS })

    await page.unroute(isHealth, answerUnavailable)
    await page.getByRole('button', { name: CHECK_AGAIN, exact: true }).click()
    await expect.poll(() => focusedElement(page), { timeout: ASLEEP_TIMEOUT_MS }).toEqual({ tag: FRAME_MAIN_TAG, tabIndex: -1 })
    await expect(page.getByRole('status').filter({ hasText: AWAKE_LINE })).toHaveCount(1)
  })

  test("This look's toolbar keeps focus through Rename, Delete and Keep it", { tag: PARITY_TAG }, async ({ page, request }) => {
    await openPlay(page, request, variant, LOOKS_PATH)
    await page.getByRole('complementary', { name: 'Edit the look', exact: true }).getByRole('button', { name: 'Save as new look', exact: true }).click()

    await inThisLook(page, 'Rename').click()
    const field = page.getByRole('region', { name: MANAGE, exact: true }).getByLabel('New name', { exact: true })
    await expect(field).toBeFocused()
    await field.fill(RENAMED)
    await page.keyboard.press('Enter')
    await expect(inThisLook(page, 'Rename')).toBeFocused()

    await inThisLook(page, 'Delete').click()
    await expect(inThisLook(page, 'Keep it')).toBeFocused()
    await page.keyboard.press('Enter')
    await expect(inThisLook(page, 'Delete')).toBeFocused()

    await inThisLook(page, 'Delete').click()
    await inThisLook(page, 'Delete look').click()
    await expect(inThisLook(page, 'Duplicate')).toBeFocused()
  })

  test("Switch to Play on Studio's note hands focus to the screen's heading", { tag: PARITY_TAG }, async ({ page, request }) => {
    await variant.makeAvailable(request)
    await chooseStyle(page, STUDIO, APP_URL)
    const installs = page.waitForResponse((response) => new URL(response.url()).pathname === INSTALLED_PATH && response.request().method() === 'GET')
    await page.goto(LOOKS_PATH)
    const note = page.getByRole('region', { name: LOOKS_NOTE_REGION, exact: true })
    await expect(note.getByRole('button', { name: SWITCH_TO_PLAY })).toBeVisible()
    await installs

    await note.getByRole('button', { name: SWITCH_TO_PLAY }).click()
    await expect(page.getByRole('heading', { name: LOOKS_HEADING, level: 1 })).toBeFocused({ timeout: THEME_LOAD_TIMEOUT_MS })
    await expect(page.locator('html')).toHaveClass(/\bplay\b/)
  })
})
