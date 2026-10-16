import { expect, test, type Page } from '@playwright/test'
import { serveHelloTheme, THEME_E2E_PATH } from '../support/hello-theme'
import '../support/page-globals'
import { expectedImportMap, readSharedModules } from '../support/shared-modules'
import { themeRequestPaths, watchPage } from '../support/watch'

// PRD-261 A-01 (@spike): a build with the theme-host spike on. The hello theme loads at runtime
// through the import map against the host's own React, survives a reload, falls back to Studio on
// every failure, and loads no theme code in safe mode or under Classic. The import map comes
// before every module script. Each test opens its own page.

const STATUS_ATTR = 'data-theme-host'
const REASON_ATTR = 'data-theme-host-reason'
const THEME_CLASS = /\bhello-spike\b/
const STUDIO_CLASS = /\bstudio\b/
const FALLBACK_NOTICE = /Studio is showing instead/
const SAFE_MODE_NOTICE = /Safe mode: themes are off in this tab/
const SHARING_FAILURE = /Invalid hook call|Minified React error|import ?map/i
const ACCENT_RGB = 'rgb(47, 111, 237)'
const SHIMS_THE_THEME_IMPORTS = ['react.js', 'react-jsx-runtime.js', 'react-dom.js', 'sdk.js']
const SAFE_MODE_PATH = `${THEME_E2E_PATH}?safe_mode=themes`
const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const STYLE_COOKIE = 'automatos-style'
const APPEARANCE_DEFAULTS = { key: 'automatos-appearance-defaults', value: '2026-09-17-studio-dark' }
const IMPORT_MAP_OPEN = '<script type="importmap">'
const SCRIPT_CLOSE = '</script>'
const MODULE_SCRIPT_TAG = /<script\b[^>]*\btype="module"[^>]*>/gi
const SCRIPT_TAG = /<script\b[^>]*>/gi

/**
 * Runs in the page as a module script. For each shared specifier, imports the shim by its bare
 * name (resolved through the import map) and compares it with the host's own instance on the
 * global. A constant: the specifiers come in through window.__shimSpecs.
 */
const SHIM_CHECK_MODULE = `
const results = [];
let error = null;
try {
  for (const spec of window.__shimSpecs) {
    const shim = await import(spec);
    const shared = window.__AUTOMATOS_THEME_HOST__.modules[spec];
    const keys = Object.keys(shared).filter((key) => key !== 'default' && key !== '__esModule');
    const missing = keys.filter((key) => !(key in shim));
    const differing = keys.filter((key) => key in shim && shim[key] !== shared[key]);
    results.push({ spec, missing, differing });
  }
} catch (caught) {
  error = String(caught && caught.message ? caught.message : caught);
}
window.__shimCheck = { results, error };
`

async function expectActive(page: Page): Promise<void> {
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
  await expect(page.getByTestId('hello-theme-frame')).toBeVisible()
}

/** The Studio fallback after a failure: the shell and the notice, and nothing of the theme left. */
async function expectStudioFallback(page: Page, reason: string): Promise<void> {
  const html = page.locator('html')
  await expect(html).toHaveAttribute(STATUS_ATTR, 'failed')
  await expect(html).toHaveAttribute(REASON_ATTR, reason)
  await expect(page.locator('.sh-shell')).toBeVisible()
  await expect(page.getByTestId('hello-theme-frame')).toHaveCount(0)
  await expect(page.getByText(FALLBACK_NOTICE)).toBeVisible()
  await expect(html).not.toHaveClass(THEME_CLASS)
  await expect(page.locator('link[data-theme-host-style]')).toHaveCount(0)
}

function hostGlobalType(page: Page): Promise<string> {
  return page.evaluate(() => typeof window.__AUTOMATOS_THEME_HOST__)
}

test.describe('the theme host (spike build)', () => {
  test('renders the hello theme on the host React', { tag: '@spike' }, async ({ page }) => {
    const watch = watchPage(page)
    await serveHelloTheme(page, 'ok')
    await page.goto(THEME_E2E_PATH)
    await expectActive(page)

    await expect(page.locator('.sh-shell')).toHaveCount(0)
    await expect(page.getByTestId('hello-theme-content').locator('*:visible').first()).toBeVisible()
    const count = page.getByTestId('hello-theme-count')
    await expect(count).toHaveText('Clicks: 0')
    await count.click()
    await expect(count).toHaveText('Clicks: 1')
    await expect(page.locator('body > [data-testid="hello-theme-portal"]')).toBeAttached()
    await expect(page.locator('html')).toHaveClass(THEME_CLASS)
    await expect(page.locator('[data-hello-bar]')).toHaveCSS('background-color', ACCENT_RGB)

    for (const file of SHIMS_THE_THEME_IMPORTS) {
      const seen = watch.responses.filter((response) => new URL(response.url).pathname === `/theme-sdk/v1/${file}`)
      expect({ file, statuses: seen.map((response) => response.status) }).toEqual({ file, statuses: [200] })
    }
    expect(watch.pageErrors).toEqual([])
    expect(watch.consoleErrors.filter((text) => SHARING_FAILURE.test(text))).toEqual([])
  })

  test('shares the host React: every shim export is the instance on the global', { tag: '@spike' }, async ({ page }) => {
    const table = readSharedModules()
    await serveHelloTheme(page, 'ok')
    await page.goto(THEME_E2E_PATH)
    await expectActive(page)

    const specs = table.modules.map((entry) => entry.specifier)
    await page.evaluate((list) => {
      window.__shimSpecs = list
    }, specs)
    await page.addScriptTag({ type: 'module', content: SHIM_CHECK_MODULE })
    await page.waitForFunction(() => window.__shimCheck !== undefined)
    const check = await page.evaluate(() => window.__shimCheck)

    expect(check?.error ?? null).toBeNull()
    expect(check?.results.map((result) => result.spec)).toEqual(specs)
    for (const result of check?.results ?? []) {
      expect(result).toEqual({ spec: result.spec, missing: [], differing: [] })
    }
    const liveReact = await page.evaluate(() => String(window.__AUTOMATOS_THEME_HOST__?.modules.react?.version))
    expect(liveReact).toBe(process.env.EXPECTED_REACT_VERSION)
  })

  test('survives a reload', { tag: '@spike' }, async ({ page }) => {
    const watch = watchPage(page)
    await serveHelloTheme(page, 'ok')
    await page.goto(THEME_E2E_PATH)
    await expectActive(page)
    await page.getByTestId('hello-theme-count').click()
    await expect(page.getByTestId('hello-theme-count')).toHaveText('Clicks: 1')

    await page.reload()
    await expectActive(page)
    await expect(page.getByTestId('hello-theme-count')).toHaveText('Clicks: 0')
    expect(await hostGlobalType(page)).toBe('object')
    expect(watch.pageErrors).toEqual([])
  })

  test('falls back to Studio when the entry throws', { tag: '@spike' }, async ({ page }) => {
    await serveHelloTheme(page, 'entry-throws')
    await page.goto(THEME_E2E_PATH)
    await expectStudioFallback(page, 'load')
  })

  test('falls back to Studio when the frame throws while drawing', { tag: '@spike' }, async ({ page }) => {
    await serveHelloTheme(page, 'render-throws')
    await page.goto(THEME_E2E_PATH)
    await expectStudioFallback(page, 'render')
  })

  test('refuses an SDK mismatch without requesting the entry', { tag: '@spike' }, async ({ page }) => {
    const requested = await serveHelloTheme(page, 'engines-mismatch')
    await page.goto(THEME_E2E_PATH)
    await expectStudioFallback(page, 'engines')
    expect(requested).toContain('/theme-spike/hello/theme.json')
    expect(requested).not.toContain('/theme-spike/hello/entry.js')
  })

  test('falls back to Studio when no theme is served', { tag: '@spike' }, async ({ page }) => {
    await serveHelloTheme(page, 'missing')
    await page.goto(THEME_E2E_PATH)
    await expectStudioFallback(page, 'manifest')
  })

  test('puts the import map first: no module script before it', { tag: '@spike' }, async ({ page, request }, testInfo) => {
    const html = await (await request.get(THEME_E2E_PATH)).text()
    const mapAt = html.indexOf(IMPORT_MAP_OPEN)
    expect(html.split(IMPORT_MAP_OPEN).length - 1).toBe(1)
    expect(html.indexOf('<head>')).toBeGreaterThanOrEqual(0)
    expect(html.indexOf('<head>')).toBeLessThan(mapAt)
    expect(mapAt).toBeLessThan(html.indexOf('</head>'))
    const moduleScriptsBefore = Array.from(html.matchAll(MODULE_SCRIPT_TAG)).filter((match) => (match.index ?? 0) < mapAt)
    expect(moduleScriptsBefore.map((match) => match[0])).toEqual([])
    const mapEnd = html.indexOf(SCRIPT_CLOSE, mapAt)
    expect(JSON.parse(html.slice(mapAt + IMPORT_MAP_OPEN.length, mapEnd))).toEqual(expectedImportMap(readSharedModules()))

    const tags = Array.from(html.matchAll(SCRIPT_TAG), (match) => ({ at: match.index ?? 0, tag: match[0] }))
    const before = tags.filter((entry) => entry.at < mapAt).map((entry) => entry.tag)
    const after = tags.filter((entry) => entry.at > mapAt).slice(0, 2).map((entry) => entry.tag)
    await testInfo.attach('script-tags-around-the-import-map', {
      body: ['before the map:', ...before, '', 'right after it:', ...after].join('\n'),
      contentType: 'text/plain',
    })

    await serveHelloTheme(page, 'ok')
    await page.goto(THEME_E2E_PATH)
    await expectActive(page)
    const order = await page.evaluate(() => {
      const maps = Array.from(document.scripts).filter((script) => script.type === 'importmap')
      const modules = Array.from(document.scripts).filter((script) => script.type === 'module')
      const first = maps[0]
      const allAfter = first !== undefined && modules.every((script) => (first.compareDocumentPosition(script) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0)
      return { maps: maps.length, modules: modules.length, allAfter }
    })
    expect(order.maps).toBe(1)
    expect(order.allAfter).toBe(true)
  })

  test('safe mode loads no theme code, and stays on in the tab', { tag: '@spike' }, async ({ page }) => {
    const watch = watchPage(page)
    await serveHelloTheme(page, 'ok')
    await page.goto(SAFE_MODE_PATH)
    await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'safe-mode')
    await expect(page.locator('.sh-shell')).toBeVisible()
    await expect(page.getByText(SAFE_MODE_NOTICE)).toBeVisible()
    expect(themeRequestPaths(watch.requests)).toEqual([])
    expect(await hostGlobalType(page)).toBe('undefined')

    await page.goto(THEME_E2E_PATH)
    await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'safe-mode')
    await expect(page.locator('.sh-shell')).toBeVisible()
    expect(themeRequestPaths(watch.requests)).toEqual([])
    expect(await hostGlobalType(page)).toBe('undefined')
  })

  test('loads no theme code under Classic', { tag: '@spike' }, async ({ page }) => {
    const watch = watchPage(page)
    // Without the defaults marker, a fresh browser is moved to Studio once (ui-style-context.tsx).
    await page.addInitScript(({ key, value }) => {
      window.localStorage.setItem(key, value)
    }, APPEARANCE_DEFAULTS)
    await page.context().addCookies([{ name: STYLE_COOKIE, value: 'classic', url: APP_URL }])
    await serveHelloTheme(page, 'ok')
    await page.goto(THEME_E2E_PATH)
    await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'off')
    await expect(page.locator('html')).not.toHaveClass(STUDIO_CLASS)
    expect(themeRequestPaths(watch.requests)).toEqual([])
    expect(await hostGlobalType(page)).toBe('undefined')
  })
})
