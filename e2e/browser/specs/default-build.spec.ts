import { expect, test } from '@playwright/test'
import { THEME_E2E_PATH } from '../support/hello-theme'
import '../support/page-globals'
import { themeRequestPaths, watchPage } from '../support/watch'

// PRD-261 A-01 (@off): the build every image ships, with the theme-host spike off. None of the
// host is on the page: no import map, no request for theme code, no global, no status attribute.

const IMPORT_MAP_TAG = 'type="importmap"'

test('the default build carries none of the theme host', { tag: '@off' }, async ({ page, request }) => {
  const html = await (await request.get(THEME_E2E_PATH)).text()
  expect(html).not.toContain(IMPORT_MAP_TAG)

  const watch = watchPage(page)
  await page.goto(THEME_E2E_PATH)
  await expect(page.locator('.sh-shell')).toBeVisible()

  expect(await page.evaluate(() => typeof window.__AUTOMATOS_THEME_HOST__)).toBe('undefined')
  await expect(page.locator('html')).not.toHaveAttribute('data-theme-host')
  expect(themeRequestPaths(watch.requests)).toEqual([])
})
