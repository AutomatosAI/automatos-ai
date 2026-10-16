import type { Page } from '@playwright/test'
import { THEME_SDK_ROOT, THEME_SPIKE_ROOT } from './hello-theme'

// PRD-261 A-01: what a page did while a spec watched it: uncaught errors, console errors, and
// every request and response.

export interface ResponseSeen {
  readonly url: string
  readonly status: number
}

export interface PageWatch {
  readonly pageErrors: string[]
  readonly consoleErrors: string[]
  readonly requests: string[]
  readonly responses: ResponseSeen[]
}

/** Starts watching; register it before the first navigation so nothing is missed. */
export function watchPage(page: Page): PageWatch {
  const watch: PageWatch = { pageErrors: [], consoleErrors: [], requests: [], responses: [] }
  page.on('pageerror', (error) => watch.pageErrors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') watch.consoleErrors.push(message.text())
  })
  page.on('request', (request) => watch.requests.push(request.url()))
  page.on('response', (response) => watch.responses.push({ url: response.url(), status: response.status() }))
  return watch
}

/** The paths of the requests that reached for theme code: the shims or the theme's folder. */
export function themeRequestPaths(urls: readonly string[]): string[] {
  return urls
    .map((url) => new URL(url).pathname)
    .filter((pathname) => pathname.startsWith(THEME_SDK_ROOT) || pathname.startsWith(THEME_SPIKE_ROOT))
}
