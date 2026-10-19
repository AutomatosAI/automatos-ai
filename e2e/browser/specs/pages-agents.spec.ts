import fs from 'fs'
import path from 'path'
import { expect, test, type Page } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseStyle, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-261 A-12d (@host): a theme's agents page reads its data through the SDK's pages/agents area.
// The hello theme (fixtures/hello-a12d) registers a page on /agents and draws it from useAgents()
// and useAgentStats() alone. This spec answers the two reads the core page makes, GET /api/agents/
// (the roster) and GET /api/system/agent-statistics (the counts), from fixtures/hello-a12d/responses.json
// (test data), so what the theme draws can only have come through the SDK's hooks, the core hooks
// they wrap and the host's backend client:
//
// 1. /agents renders the theme's page inside its frame, with the intercepted roster (names and
//    statuses) and the counts (seven agents, three active: the stats read, not the roster's two),
//    and never the core Agent Management page.
// 2. Refresh calls the roster query's refetch, and the roster is asked for again.

interface AgentRow {
  readonly id: number
  readonly name: string
  readonly status: string
}

interface Responses {
  readonly agents: readonly AgentRow[]
  readonly stats: { readonly total_agents: number; readonly active_agents: number }
}

/** How many times the roster was read: GETs of /api/agents/ without the system agents. */
interface RosterReads {
  count: number
}

const HELLO_A12D: InstalledTheme = { installId: '00000000-0000-4000-8000-00000000a12d', fixture: 'hello-a12d' }
const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const RESPONSES_FILE = path.resolve(__dirname, '..', 'fixtures', HELLO_A12D.fixture, 'responses.json')
const RESPONSES = JSON.parse(fs.readFileSync(RESPONSES_FILE, 'utf8')) as Responses
const AGENTS_PAGE = '/agents'
const AGENTS_PATH = '/api/agents/'
const STATS_PATH = '/api/system/agent-statistics'
/** The query parameter the assignee pickers add (useAssignableAgents); the roster never sends it. */
const SYSTEM_AGENTS_PARAM = 'include_workspace_system'
const STATUS_ATTR = 'data-theme-host'
const FRAME_TEST_ID = 'hello-a12d-frame'
const PAGE_TEST_ID = 'hello-a12d-page'
const CORE_HEADING = /^Agent Management$/
const OK = 200
const SHARING_FAILURE = /Invalid hook call|Minified React error|not in this SDK build/i

/** Answers the agents page's two reads; counts the roster reads. */
async function serveAgentData(page: Page): Promise<RosterReads> {
  const reads: RosterReads = { count: 0 }
  await page.route((url) => url.pathname === AGENTS_PATH, async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (request.method() === 'GET' && !url.searchParams.has(SYSTEM_AGENTS_PARAM)) reads.count += 1
    await fulfillBackendJson(route, OK, RESPONSES.agents)
  })
  await page.route((url) => url.pathname === STATS_PATH, (route) => fulfillBackendJson(route, OK, RESPONSES.stats))
  return reads
}

test.describe("a theme's agents page reads its data through the SDK", () => {
  test('the roster and the counts come from the intercepted reads, and Refresh reads the roster again', { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    const reads = await serveAgentData(page)
    await chooseStyle(page, `theme:install:${HELLO_A12D.installId}`, APP_URL)
    await serveInstalledTheme(page, HELLO_A12D)
    await page.goto(AGENTS_PAGE)

    await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
    const frame = page.getByTestId(FRAME_TEST_ID).first()
    await expect(frame).toBeVisible()
    const themePage = frame.getByTestId(PAGE_TEST_ID)
    await expect(themePage).toHaveCount(1)
    await expect(themePage).toHaveAttribute('data-loading', 'false')
    await expect(frame.getByRole('heading', { level: 1, name: CORE_HEADING })).toHaveCount(0)

    const roster = themePage.getByTestId('hello-a12d-roster').locator('li')
    await expect(roster).toHaveText(RESPONSES.agents.map((agent) => agent.name))
    for (const agent of RESPONSES.agents) {
      await expect(roster.filter({ hasText: agent.name })).toHaveAttribute('data-status', agent.status)
    }
    await expect(themePage.getByTestId('hello-a12d-total')).toHaveText(String(RESPONSES.stats.total_agents))
    await expect(themePage.getByTestId('hello-a12d-active')).toHaveText(String(RESPONSES.stats.active_agents))

    await expect.poll(() => reads.count).toBeGreaterThan(0)
    const before = reads.count
    await themePage.getByTestId('hello-a12d-refresh').click()
    await expect.poll(() => reads.count).toBeGreaterThan(before)
    await expect(roster).toHaveCount(RESPONSES.agents.length)

    expect(watch.pageErrors).toEqual([])
    expect(watch.consoleErrors.filter((text) => SHARING_FAILURE.test(text))).toEqual([])
  })
})
