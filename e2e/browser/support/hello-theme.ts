import { createHash } from 'crypto'
import fs from 'fs'
import path from 'path'
import type { Page, Route } from '@playwright/test'
import { fulfillBackendJson } from './backend-route'

// PRD-261 A-01, A-04a: serves a test theme as an installed one, through route interception.
// GET /api/themes/active (on the backend's origin) answers with the install and the fixture's
// theme.json as its manifest; /api/themes/files/<digest>/** (on the app's origin) serves the
// fixture's files. Nothing in the app or the local backend serves either yet (A-07b, A-08a), so a
// theme reaches the browser only here.
//
// PA-17: read-only after A-04a. A later story adds its own folder under fixtures/
// (fixtures/hello-<story>/) and calls serveInstalledTheme with it.
//
// PRD-261 A-19f4 (W-A8) adds the two core styles the first-paint, default-build and smoke specs
// run under (CORE_STYLES, chooseCoreStyle), and the mount counter the route specs share
// (countMounts), beside the serving above, which it leaves as it was.

/** The page the theme-host specs open: it renders inside MainLayout's Studio chrome. */
export const THEME_E2E_PATH = '/agents'
export const THEME_SDK_ROOT = '/theme-sdk/'
/** Every theme route (the active install, the files): a page with no theme style asks for none. */
export const THEME_API_ROOT = '/api/themes/'

const ACTIVE_ROUTE = '**/api/themes/active'
const FILES_ROUTE = '**/api/themes/files/**'
const FILES_PATH = '/api/themes/files/'
const FIXTURES_DIR = path.resolve(__dirname, '..', 'fixtures')
const DEFAULT_MANIFEST_FILE = 'theme.json'
const STYLE_COOKIE = 'automatos-style'
const INSTALLED_STYLE_PREFIX = 'theme:install:'
const DIGEST_SEED_PREFIX = 'automatos-e2e:'
// Without this marker a fresh browser is moved to Studio once (contexts/ui-style-context.tsx).
const APPEARANCE_DEFAULTS = { key: 'automatos-appearance-defaults', value: '2026-09-17-studio-dark' }
const OK = 200
const NOT_FOUND = 404

const CONTENT_TYPES: Readonly<Record<string, string>> = {
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
}

/** A fixture folder served as the workspace's active install. */
export interface InstalledTheme {
  /** A lowercase uuid; the style is `theme:install:<installId>`. */
  readonly installId: string
  /** The folder under fixtures/ whose files are served. */
  readonly fixture: string
  /** The fixture file sent as the install's manifest (default theme.json). */
  readonly manifestFile?: string
  /** Served path → fixture file, for the paths a variant serves differently. */
  readonly overrides?: Readonly<Record<string, string>>
  /** Every theme route answers 404: no theme is served. */
  readonly missing?: boolean
}

export interface ServedTheme {
  readonly style: string
  readonly digest: string
  /** `/api/themes/files/<digest>/` */
  readonly filesRoot: string
  /** The path of every theme request the page made (GET only), filled in as they arrive. */
  readonly requested: string[]
}

export const HELLO_INSTALL_ID = '00000000-0000-4000-8000-00000000e2e1'
export const HELLO_STYLE = `${INSTALLED_STYLE_PREFIX}${HELLO_INSTALL_ID}`

export type HelloThemeVariant = 'ok' | 'entry-throws' | 'render-throws' | 'engines-mismatch' | 'missing'

const HELLO_VARIANTS: Readonly<Record<HelloThemeVariant, InstalledTheme>> = {
  ok: { installId: HELLO_INSTALL_ID, fixture: 'hello-theme' },
  'entry-throws': { installId: HELLO_INSTALL_ID, fixture: 'hello-theme', overrides: { 'entry.js': 'entry.throws.js' } },
  'render-throws': { installId: HELLO_INSTALL_ID, fixture: 'hello-theme', overrides: { 'entry.js': 'entry.render-throws.js' } },
  'engines-mismatch': { installId: HELLO_INSTALL_ID, fixture: 'hello-theme', manifestFile: 'theme.engines-mismatch.json' },
  missing: { installId: HELLO_INSTALL_ID, fixture: 'hello-theme', missing: true },
}

/** A stable sha256 hex digest per install, standing in for the package digest (A-06a). */
export function digestOf(installId: string): string {
  return createHash('sha256').update(`${DIGEST_SEED_PREFIX}${installId}`).digest('hex')
}

/** The fixture file for a served path, or null: outside the folder, not a file, or of an unserved type. */
function fixtureFile(theme: InstalledTheme, servedPath: string): string | null {
  const dir = path.join(FIXTURES_DIR, theme.fixture)
  const file = path.resolve(dir, theme.overrides?.[servedPath] ?? servedPath)
  if (!file.startsWith(dir + path.sep) || CONTENT_TYPES[path.extname(file)] === undefined) return null
  return fs.existsSync(file) && fs.statSync(file).isFile() ? file : null
}

function readManifest(theme: InstalledTheme): unknown {
  const file = path.join(FIXTURES_DIR, theme.fixture, theme.manifestFile ?? DEFAULT_MANIFEST_FILE)
  return JSON.parse(fs.readFileSync(file, 'utf8'))
}

/** The body of GET /api/themes/active for the install (plan §2.2's shape, A-07b). */
function activeBody(theme: InstalledTheme, digest: string): unknown {
  const manifest = readManifest(theme) as { id?: unknown; version?: unknown }
  return {
    choice: theme.installId,
    theme: {
      installId: theme.installId,
      themeId: manifest.id,
      version: manifest.version,
      digest,
      manifest,
      reviewed: false,
      optIn: { required: false, wording: null, packageId: null },
      revoked: false,
      needsUpdate: false,
    },
  }
}

function recordGet(route: Route, requested: string[]): void {
  if (route.request().method() === 'GET') requested.push(new URL(route.request().url()).pathname)
}

async function serveFile(route: Route, theme: InstalledTheme, filesRoot: string): Promise<void> {
  const { pathname } = new URL(route.request().url())
  const file = theme.missing === true || !pathname.startsWith(filesRoot) ? null : fixtureFile(theme, pathname.slice(filesRoot.length))
  if (file === null) {
    await route.fulfill({ status: NOT_FOUND, contentType: 'text/plain; charset=utf-8', body: 'not found' })
    return
  }
  await route.fulfill({
    status: OK,
    contentType: CONTENT_TYPES[path.extname(file)],
    headers: { 'cache-control': 'no-store' },
    body: fs.readFileSync(file),
  })
}

/** Routes the active-install answer and the files of `theme`; returns what was served and asked for. */
export async function serveInstalledTheme(page: Page, theme: InstalledTheme): Promise<ServedTheme> {
  const digest = digestOf(theme.installId)
  const filesRoot = `${FILES_PATH}${digest}/`
  const requested: string[] = []
  await page.route(ACTIVE_ROUTE, async (route) => {
    recordGet(route, requested)
    if (theme.missing === true) await fulfillBackendJson(route, NOT_FOUND, { detail: 'Not Found' })
    else await fulfillBackendJson(route, OK, activeBody(theme, digest))
  })
  await page.route(FILES_ROUTE, async (route) => {
    recordGet(route, requested)
    await serveFile(route, theme, filesRoot)
  })
  return { style: `${INSTALLED_STYLE_PREFIX}${theme.installId}`, digest, filesRoot, requested }
}

/** Serves the hello theme (fixtures/hello-theme) in one of its variants. */
export function serveHelloTheme(page: Page, variant: HelloThemeVariant): Promise<ServedTheme> {
  return serveInstalledTheme(page, HELLO_VARIANTS[variant])
}

/** Puts this browser on `style` before the first navigation: the style cookie and the defaults marker. */
export async function chooseStyle(page: Page, style: string, appUrl: string): Promise<void> {
  await page.addInitScript(({ key, value }) => {
    window.localStorage.setItem(key, value)
  }, APPEARANCE_DEFAULTS)
  await page.context().addCookies([{ name: STYLE_COOKIE, value: style, url: appUrl }])
}

/** The two core styles (lib/ui-style.ts): Classic and Studio. */
export type CoreStyleName = 'classic' | 'studio'

/** A core style, and what only its desktop chrome draws (components/layout/main-layout.tsx). */
export interface CoreStyle {
  readonly style: CoreStyleName
  /** Whether <html> carries `studio`: Studio does; Classic carries no style class. */
  readonly studio: boolean
  /** A selector only this style's desktop chrome matches. */
  readonly shell: string
  /** The start of that chrome's tag, as the server writes it in the page's HTML. */
  readonly shellInHtml: string
}

/** Classic's desktop chrome: its Header (components/layout/header.tsx), the only <header> with glass-card. */
export const CLASSIC_STYLE: CoreStyle = {
  style: 'classic',
  studio: false,
  shell: 'header.glass-card',
  shellInHtml: '<header class="sticky top-0 z-30 glass-card',
}

/** Studio's desktop chrome: the Studio shell (MainLayout's StudioDesktopShell). */
export const STUDIO_STYLE: CoreStyle = { style: 'studio', studio: true, shell: '.sh-shell', shellInHtml: 'class="sh-shell' }

/** PRD-261 A-19f4: the styles the first-paint, default-build and smoke specs each run under. */
export const CORE_STYLES: readonly CoreStyle[] = [CLASSIC_STYLE, STUDIO_STYLE]

/** Answers GET /api/themes/active with `style` as the member's choice and no theme (A-07b). */
export async function answerCoreChoice(page: Page, style: CoreStyleName): Promise<void> {
  await page.route(ACTIVE_ROUTE, (route) => fulfillBackendJson(route, OK, { choice: style, theme: null }))
}

/**
 * Puts this browser on a core style before the first navigation (the cookie and the defaults
 * marker), with a server that agrees: the lane's projects share one backend user, whose stored
 * choice another spec may change, and ThemeChoiceSync moves a tab the server disagrees with (A-02c).
 */
export async function chooseCoreStyle(page: Page, core: CoreStyle, appUrl: string): Promise<void> {
  await answerCoreChoice(page, core.style)
  await chooseStyle(page, core.style, appUrl)
}

/** The window property countMounts keeps its tally on. */
const MOUNTS_KEY = '__e2eMounts'

/**
 * Counts, from before the first byte, every distinct element matching each of `selectors` that the
 * page ever inserts, parsed or rendered; the selectors match distinct elements (an element is
 * counted once, for the first that matches it). Register it before the first navigation; mountsOf
 * reads it.
 */
export async function countMounts(page: Page, selectors: readonly string[]): Promise<void> {
  await page.addInitScript(({ key, wanted }) => {
    const seen = new WeakSet<Element>()
    const counts: Record<string, number> = {}
    Object.defineProperty(window, key, { value: counts, configurable: true })
    const tally = (node: Node): void => {
      if (!(node instanceof Element)) return
      for (const selector of wanted) {
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
  }, { key: MOUNTS_KEY, wanted: selectors })
}

/** What countMounts has counted so far: selector → elements inserted (a selector never seen is absent). */
export function mountsOf(page: Page): Promise<Record<string, number>> {
  return page.evaluate((key) => ({ ...((window as unknown as Record<string, Record<string, number> | undefined>)[key] ?? {}) }), MOUNTS_KEY)
}
