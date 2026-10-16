import { createHash } from 'crypto'
import { expect, test } from '@playwright/test'
import { serveHelloTheme, THEME_E2E_PATH } from '../support/hello-theme'
import '../support/page-globals'
import type { ProbeOutcome } from '../support/page-globals'
import { writeRecord } from '../support/records'

// PRD-261 A-01 (@record): measured facts, written to records/ and summarised in the job summary.
// R1: the React the App Router really renders with, against next/dist/compiled/react in the CI
// install. R2: whether this browser enforces `integrity` in an import map (PRD-261 §6 signing
// rests on it). Runs on Chromium, WebKit and Firefox.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const STATUS_ATTR = 'data-theme-host'
const LOADER_FINISHED = /^(active|failed)$/
const PROBE_ROUTE = `${APP_URL}/__e2e__/**`
const PROBE_PAGE = '/__e2e__/integrity-probe.html'
const PROBE_NAMES = ['probe-ok', 'probe-bad', 'probe-plain'] as const

/** A page of our own on the app's origin (never served by the app), with no import map in it. */
const PROBE_HTML = '<!doctype html><html><head><meta charset="utf-8"><title>Import map integrity probe</title></head><body></body></html>'
const OK_BODY = "export const probe = 'ok';\n"
const BAD_BODY = "export const probe = 'bad';\n"
const PLAIN_BODY = "export const probe = 'plain';\n"
const PROBE_BODIES: Readonly<Record<string, string | undefined>> = {
  '/__e2e__/ok.js': OK_BODY,
  '/__e2e__/bad.js': BAD_BODY,
  '/__e2e__/plain.js': PLAIN_BODY,
}

const OK_DIGEST = `sha384-${createHash('sha384').update(OK_BODY).digest('base64')}`
// 64 base64 characters: a well-formed sha384 value that cannot match bad.js. A malformed value
// would be dropped silently by integrity parsing and read as "no integrity". Low entropy on
// purpose, so a secret scanner (gitleaks) does not take it for a credential.
const MISMATCHED_DIGEST = 'sha384-' + 'A'.repeat(64)

const PROBE_MAP = {
  imports: { 'probe-ok': '/__e2e__/ok.js', 'probe-bad': '/__e2e__/bad.js', 'probe-plain': '/__e2e__/plain.js' },
  integrity: { '/__e2e__/ok.js': OK_DIGEST, '/__e2e__/bad.js': MISMATCHED_DIGEST },
}

/** Runs in the probe page as a module script, after the map is in place. A constant. */
const PROBE_MODULE = `
const outcome = {};
for (const name of ['probe-ok', 'probe-bad', 'probe-plain']) {
  try {
    await import(name);
    outcome[name] = 'loaded';
  } catch (caught) {
    outcome[name] = 'refused';
  }
}
window.__probeResult = outcome;
`

type IntegritySupport = 'enforced' | 'ignored' | 'inconclusive'

function classify(outcome: Readonly<Record<string, ProbeOutcome>>): IntegritySupport {
  if (outcome['probe-plain'] !== 'loaded') return 'inconclusive'
  if (outcome['probe-ok'] === 'loaded' && outcome['probe-bad'] === 'refused') return 'enforced'
  if (outcome['probe-ok'] === 'loaded' && outcome['probe-bad'] === 'loaded') return 'ignored'
  return 'inconclusive'
}

test('records the React the App Router renders with', { tag: '@record' }, async ({ page, browser }, testInfo) => {
  await serveHelloTheme(page, 'ok')
  await page.goto(THEME_E2E_PATH)
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, LOADER_FINISHED)
  const live = await page.evaluate(() => ({
    react: String(window.__AUTOMATOS_THEME_HOST__?.modules.react?.version),
    importMapSupported: typeof HTMLScriptElement.supports === 'function' && HTMLScriptElement.supports('importmap'),
    userAgent: navigator.userAgent,
    themeHostStatus: document.documentElement.getAttribute('data-theme-host'),
  }))
  await writeRecord(testInfo, 'app', {
    project: testInfo.project.name,
    browserVersion: browser.version(),
    userAgent: live.userAgent,
    importMapSupported: live.importMapSupported,
    themeHostStatus: live.themeHostStatus,
    reactLive: live.react,
    reactExpected: process.env.EXPECTED_REACT_VERSION ?? null,
    next: process.env.NEXT_VERSION ?? null,
  })
  expect(live.react).toBe(process.env.EXPECTED_REACT_VERSION)
})

test('records whether import-map integrity is enforced', { tag: '@record' }, async ({ page, browser }, testInfo) => {
  await page.route(PROBE_ROUTE, async (route) => {
    const { pathname } = new URL(route.request().url())
    if (pathname === PROBE_PAGE) {
      await route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: PROBE_HTML })
      return
    }
    const body = PROBE_BODIES[pathname]
    if (body === undefined) {
      await route.fulfill({ status: 404, contentType: 'text/plain; charset=utf-8', body: 'not found' })
      return
    }
    await route.fulfill({ status: 200, contentType: 'text/javascript; charset=utf-8', headers: { 'cache-control': 'no-store' }, body })
  })
  await page.goto(PROBE_PAGE)
  await page.evaluate((map) => {
    const script = document.createElement('script')
    script.type = 'importmap'
    script.textContent = JSON.stringify(map)
    document.head.appendChild(script)
  }, PROBE_MAP)
  await page.addScriptTag({ type: 'module', content: PROBE_MODULE })
  await page.waitForFunction(() => window.__probeResult !== undefined)
  const outcome = (await page.evaluate(() => window.__probeResult)) ?? {}

  await writeRecord(testInfo, 'integrity', {
    project: testInfo.project.name,
    browserVersion: browser.version(),
    integrity: classify(outcome),
    outcome: Object.fromEntries(PROBE_NAMES.map((name) => [name, outcome[name] ?? null])),
  })
  expect(outcome['probe-plain']).toBe('loaded')
  expect(outcome['probe-ok']).toBe('loaded')
})
