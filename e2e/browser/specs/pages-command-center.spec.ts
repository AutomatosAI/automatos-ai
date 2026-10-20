import { expect, test, type Page, type Route } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseStyle, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-261 A-12h (@host): a theme's own Command Center page reads the page's data through the SDK.
//
// The hello theme (fixtures/hello-a12h) registers a page on /command-center that reads the stats,
// the Needs-you number, the feed, the schedule, the open questions and the watches through
// '@automatos/theme-sdk' (pages/command-center), opens the board's event stream through it, and
// counts the board changes the stream pushes. Every one of those backend answers is intercepted
// here, so what the page shows can only have come through the SDK's hooks, the host's backend
// client and its query cache:
//
// 1. Each value the theme's page shows is the intercepted answer's, asked with the core page's
//    parameters (the stats for 1d, the schedule for 7d, the open questions, five feed items).
// 2. A board change pushed on the stream reaches the theme as the SDK's board-changed event, and
//    refreshes the Needs-you number, as on the core page.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const STATUS_ATTR = 'data-theme-host'
const HELLO_A12H: InstalledTheme = { installId: '00000000-0000-4000-8000-0000000a1280', fixture: 'hello-a12h' }
const OK = 200
const SHARING_FAILURE = /Invalid hook call|Minified React error|not in this SDK build/i

const STATS_PATH = '/api/activity/stats'
const NEEDS_YOU_PATH = '/api/activity/needs-you'
const FEED_PATH = '/api/activity/feed'
const SCHEDULE_PATH = '/api/activity/schedule'
const QUESTIONS_PATH = '/api/v1/approval-grants'
const WATCHES_PATH = '/api/v1/watches'
const STREAM_PATH = '/api/v1/tasks/stream'

const WORKING_NOW = 7
const NEEDS_YOU_BEFORE = 3
const NEEDS_YOU_AFTER = 4
const FEED_ITEM = 'Weekly report run'
const SCHEDULED = 'Morning brief'
const QUESTION = 'Which tone should the launch post take?'
const WATCHED = 'Launch post quality'
const BOARD_CHANGED_FRAME = 'event: board_changed\ndata: {"task_id":42,"status":"review"}\n\n'

const STATS = { working_now: WORKING_NOW, channels_live: 1, completed_today: 2, needs_attention: 3, agents_active: 4, tasks_in_queue: 5, period: '1d' }
const FEED = {
  items: [{ id: 'recipe-1', type: 'recipe', name: FEED_ITEM, status: 'completed', started_at: null, completed_at: null, duration_seconds: null, agent: null, agents: [], summary: '', source_id: null, source_url: null, trigger: 'scheduled', error_message: null }],
  total: 1,
  limit: 5,
  offset: 0,
}
const SCHEDULE = { scheduled: [{ id: 'routine-1', name: SCHEDULED, type: 'routine', next_run_at: null, frequency: 'daily', agent_name: null, agent_id: null }] }
const QUESTIONS = { grants: [{ id: 12, kind: 'question', status: 'pending', subject_type: 'board_task', subject_id: '42', question_md: QUESTION }] }
const WATCHES = { watches: [{ id: 'watch-1', title: WATCHED, status: 'active' }], total: 1 }

function needsYou(total: number): unknown {
  return {
    total,
    counts: { review: total, question: 0, approval: 0, stuck: 0, failed: 0 },
    rows: { review: [], question: [], approval: [], stuck: [], failed: [] },
  }
}

function onPath(pathname: string): (url: URL) => boolean {
  return (url) => url.pathname === pathname
}

/** The board's event stream: one board_changed frame, then the stream ends (the host reconnects). */
async function fulfillStream(route: Route): Promise<void> {
  const request = route.request()
  if (request.method() === 'OPTIONS') return fulfillBackendJson(route, OK, null)
  const origin = (await request.headerValue('origin')) ?? new URL(APP_URL).origin
  await route.fulfill({
    status: OK,
    contentType: 'text/event-stream',
    headers: { 'access-control-allow-origin': origin, 'access-control-allow-credentials': 'true', vary: 'Origin', 'cache-control': 'no-store' },
    body: BOARD_CHANGED_FRAME,
  })
}

/** Answers every read the theme's page makes; returns the URLs asked, filled in as they arrive. */
async function serveCommandCenterData(page: Page): Promise<string[]> {
  const asked: string[] = []
  let needsYouReads = 0
  const answer = (pathname: string, body: () => unknown) =>
    page.route(onPath(pathname), async (route) => {
      if (route.request().method() === 'GET') asked.push(route.request().url())
      await fulfillBackendJson(route, OK, body())
    })
  await answer(STATS_PATH, () => STATS)
  await answer(FEED_PATH, () => FEED)
  await answer(SCHEDULE_PATH, () => SCHEDULE)
  await answer(QUESTIONS_PATH, () => QUESTIONS)
  await answer(WATCHES_PATH, () => WATCHES)
  await page.route(onPath(NEEDS_YOU_PATH), async (route) => {
    const isRead = route.request().method() === 'GET'
    if (isRead) needsYouReads += 1
    await fulfillBackendJson(route, OK, needsYou(isRead && needsYouReads > 1 ? NEEDS_YOU_AFTER : NEEDS_YOU_BEFORE))
  })
  await page.route(onPath(STREAM_PATH), async (route) => {
    if (route.request().method() === 'GET') asked.push(route.request().url())
    await fulfillStream(route)
  })
  return asked
}

async function openThemePage(page: Page): Promise<string[]> {
  await chooseStyle(page, `theme:install:${HELLO_A12H.installId}`, APP_URL)
  await serveInstalledTheme(page, HELLO_A12H)
  const asked = await serveCommandCenterData(page)
  await page.goto('/command-center')
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
  await expect(page.getByTestId('hello-a12h-frame').getByTestId('hello-a12h-page')).toBeVisible()
  return asked
}

function searchOf(asked: readonly string[], pathname: string): string[] {
  return asked.map((url) => new URL(url)).filter((url) => url.pathname === pathname).map((url) => url.search)
}

test.describe("a theme's Command Center page reads its data through the SDK", () => {
  test('each value is the intercepted answer, asked as the core page asks', { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    const asked = await openThemePage(page)

    await expect(page.getByTestId('hello-a12h-working-now')).toHaveText(String(WORKING_NOW))
    await expect(page.getByTestId('hello-a12h-feed')).toHaveText(FEED_ITEM)
    await expect(page.getByTestId('hello-a12h-schedule')).toHaveText(SCHEDULED)
    await expect(page.getByTestId('hello-a12h-question')).toHaveText(QUESTION)
    await expect(page.getByTestId('hello-a12h-watch')).toHaveText(WATCHED)
    await expect(page.getByTestId('hello-a12h-needs-you')).toHaveText(/^[34]$/)
    await expect(page.getByRole('heading', { level: 1, name: /^Command Center$/ })).toHaveCount(0)

    expect(searchOf(asked, STATS_PATH)).toContain('?period=1d')
    expect(searchOf(asked, SCHEDULE_PATH)).toContain('?range=7d')
    expect(searchOf(asked, FEED_PATH)).toContain('?limit=5')
    expect(searchOf(asked, QUESTIONS_PATH)).toContain('?status=pending&kind=question')
    expect(searchOf(asked, WATCHES_PATH)).toContain('')
    expect(watch.pageErrors).toEqual([])
    expect(watch.consoleErrors.filter((text) => SHARING_FAILURE.test(text))).toEqual([])
  })

  test('a board change on the stream reaches the theme and refreshes Needs you', { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    const asked = await openThemePage(page)

    await expect(page.getByTestId('hello-a12h-board-changes')).toHaveText(/^[1-9]\d*$/)
    await expect(page.getByTestId('hello-a12h-needs-you')).toHaveText(String(NEEDS_YOU_AFTER))
    expect(searchOf(asked, STREAM_PATH).length).toBeGreaterThan(0)
    expect(watch.pageErrors).toEqual([])
  })
})
