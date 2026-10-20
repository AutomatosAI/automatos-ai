import { expect, test, type Page } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseStyle, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-261 A-12a (@host): a theme places the host's cards through '@automatos/theme-sdk'.
//
// The host-cards hello theme (fixtures/hello-a12a) renders the SDK's HeaderUtilities in its frame's
// header, and its route /x/hello-a12a places the SDK's ToolApproval for one grant. The card is the
// host's own protected surface, and Approve & run posts the host's own decision (answered here),
// after which the card shows the host's outcome line and the theme hears 'approved' through
// onDecided. The theme imports nothing of the host but the SDK.

const HELLO_A12A: InstalledTheme = { installId: '00000000-0000-4000-8000-0000000a12a0', fixture: 'hello-a12a' }
const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const ROUTE_PATH = '/x/hello-a12a'
const STATUS_ATTR = 'data-theme-host'
const GRANT_ID = 12101
const GRANT_ROUTE = '**/api/v1/approval-grants/*/grant'
const GRANT_PATH = `/api/v1/approval-grants/${GRANT_ID}/grant`
const GRANT_ANSWER = { grant: { id: GRANT_ID, status: 'granted', details: {} } }
const ACTION = 'platform_delete_document'
const HELP_BUTTON = 'Help & documentation'
const APPROVE_BUTTON = 'Approve & run'
/** The host's outcome line for a grant whose answer reports no run (ToolApprovalWidget executedOutcomeLine). */
const APPROVED_LINE = 'Approved.'
const APPROVED = 'approved'
const WAITING = 'waiting'
const CARD = '[data-protected-surface="tool-approval"]'
const SHARING_FAILURE = /Invalid hook call|Minified React error|not in this SDK build/i
const OK = 200

/** Answers the grant decision and records the path of every POST of it. */
async function recordGrants(page: Page): Promise<string[]> {
  const granted: string[] = []
  await page.route(GRANT_ROUTE, async (route) => {
    if (route.request().method() === 'POST') granted.push(new URL(route.request().url()).pathname)
    await fulfillBackendJson(route, OK, GRANT_ANSWER)
  })
  return granted
}

test.describe("a theme places the host's cards", () => {
  test("the frame shows HeaderUtilities, and approving the host's ToolApproval posts the host's decision", { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    const granted = await recordGrants(page)
    const served = await serveInstalledTheme(page, HELLO_A12A)
    await chooseStyle(page, served.style, APP_URL)
    await page.goto(ROUTE_PATH)

    await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
    const frame = page.getByTestId('hello-a12a-frame')
    await expect(frame).toBeVisible()
    await expect(frame.getByTestId('hello-a12a-header').getByRole('button', { name: HELP_BUTTON })).toBeVisible()

    const route = frame.getByTestId('hello-a12a-route')
    const card = route.locator(CARD)
    await expect(card).toHaveCount(1)
    await expect(card).toBeVisible()
    await expect(card).toContainText(ACTION)
    await expect(route.getByTestId('hello-a12a-decision')).toHaveText(WAITING)

    await card.getByRole('button', { name: APPROVE_BUTTON }).click()
    await expect.poll(() => granted).toEqual([GRANT_PATH])
    await expect(card.getByRole('status')).toHaveText(APPROVED_LINE)
    await expect(route.getByTestId('hello-a12a-decision')).toHaveText(APPROVED)

    expect(watch.pageErrors).toEqual([])
    expect(watch.consoleErrors.filter((text) => SHARING_FAILURE.test(text))).toEqual([])
  })
})
