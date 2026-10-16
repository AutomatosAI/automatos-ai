import fs from 'fs'
import path from 'path'
import type { Page } from '@playwright/test'

// PRD-261 A-01: serves the hello theme (fixtures/hello-theme) at the spike's fixed path through
// route interception. Nothing under frontend/ serves /theme-spike/, so the theme reaches the
// browser only here.

export type HelloThemeVariant = 'ok' | 'entry-throws' | 'render-throws' | 'engines-mismatch' | 'missing'

/** The page the theme-host specs open: it renders inside MainLayout's Studio chrome. */
export const THEME_E2E_PATH = '/agents'
export const THEME_SPIKE_ROOT = '/theme-spike/hello/'
export const THEME_SDK_ROOT = '/theme-sdk/'

const THEME_ROUTE = '**/theme-spike/hello/**'
const FIXTURE_DIR = path.resolve(__dirname, '..', 'fixtures', 'hello-theme')
const NOT_FOUND = 404
const OK = 200

/** The theme's files a browser may ask for, each mapped to its fixture file. */
const SERVED_FILES: Readonly<Record<string, string>> = {
  'theme.json': 'theme.json',
  'entry.js': 'entry.js',
  'theme.css': 'theme.css',
  'presets/default.look.json': 'presets/default.look.json',
}

/** What a variant serves instead, path by path. */
const VARIANT_FILES: Readonly<Record<HelloThemeVariant, Readonly<Record<string, string>>>> = {
  ok: {},
  'entry-throws': { 'entry.js': 'entry.throws.js' },
  'render-throws': { 'entry.js': 'entry.render-throws.js' },
  'engines-mismatch': { 'theme.json': 'theme.engines-mismatch.json' },
  missing: {},
}

const CONTENT_TYPES: Readonly<Record<string, string>> = {
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
}

function fixtureFor(variant: HelloThemeVariant, servedPath: string): string | null {
  if (variant === 'missing') return null
  const override = VARIANT_FILES[variant][servedPath]
  return override ?? SERVED_FILES[servedPath] ?? null
}

/**
 * Routes the theme's folder to the fixture files for this variant and returns the list of paths
 * the page asks for under it, filled in as requests arrive. An unknown path, or every path of the
 * 'missing' variant, answers 404.
 */
export async function serveHelloTheme(page: Page, variant: HelloThemeVariant): Promise<string[]> {
  const requested: string[] = []
  await page.route(THEME_ROUTE, async (route) => {
    const { pathname } = new URL(route.request().url())
    requested.push(pathname)
    const file = pathname.startsWith(THEME_SPIKE_ROOT) ? fixtureFor(variant, pathname.slice(THEME_SPIKE_ROOT.length)) : null
    if (file === null) {
      await route.fulfill({ status: NOT_FOUND, contentType: 'text/plain; charset=utf-8', body: 'not found' })
      return
    }
    await route.fulfill({
      status: OK,
      contentType: CONTENT_TYPES[path.extname(file)],
      headers: { 'cache-control': 'no-store' },
      body: fs.readFileSync(path.join(FIXTURE_DIR, file)),
    })
  })
  return requested
}
