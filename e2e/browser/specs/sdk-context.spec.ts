import { expect, test } from '@playwright/test'
import { chooseStyle, serveInstalledTheme, THEME_E2E_PATH, type InstalledTheme } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-261 A-10a (@host): a theme reads the host through '@automatos/theme-sdk' alone. The
// SDK-context hello theme (fixtures/hello-a10a) draws its frame from useWorkspace(), useRole(),
// useCoreNav() and HostLink, and its Notify button calls notify(): the workspace it shows is the
// one the app's own WorkspaceProvider loaded, its first nav item is the Studio menu's, and the
// notice is drawn by the host's sonner Toaster, which the theme never imports.

const HELLO_A10A: InstalledTheme = { installId: '00000000-0000-4000-8000-00000000a10a', fixture: 'hello-a10a' }
const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const STATUS_ATTR = 'data-theme-host'
const WORKSPACE_KEY = 'last_active_workspace'
const NOTICE_TITLE = 'Hello from the SDK'
const FIRST_NAV_LABEL = 'Chat'
const SHARING_FAILURE = /Invalid hook call|Minified React error|not in this SDK build/i

test.describe('the theme SDK in the browser', () => {
  test('a theme reads the workspace, the nav and the toaster through the SDK', { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    await chooseStyle(page, `theme:install:${HELLO_A10A.installId}`, APP_URL)
    await serveInstalledTheme(page, HELLO_A10A)
    await page.goto(THEME_E2E_PATH)

    await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
    await expect(page.getByTestId('hello-a10a-frame')).toBeVisible()
    await expect(page.getByTestId('hello-a10a-content').locator('*:visible').first()).toBeVisible()

    const workspace = page.getByTestId('hello-a10a-workspace')
    await expect(workspace).toHaveAttribute('data-loading', 'false')
    await expect(workspace).not.toHaveAttribute('data-workspace-id', '')
    const shownId = await workspace.getAttribute('data-workspace-id')
    const providerId = await page.evaluate((key) => window.localStorage.getItem(key), WORKSPACE_KEY)
    expect(shownId).toBe(providerId)

    await expect(page.getByTestId('hello-a10a-first-nav')).toHaveText(FIRST_NAV_LABEL)
    await expect(page.getByTestId('hello-a10a-first-nav')).toHaveAttribute('href', '/chat')

    await page.getByTestId('hello-a10a-notify').click()
    await expect(page.locator('[data-sonner-toast]').filter({ hasText: NOTICE_TITLE })).toBeVisible()

    expect(watch.pageErrors).toEqual([])
    expect(watch.consoleErrors.filter((text) => SHARING_FAILURE.test(text))).toEqual([])
  })
})
