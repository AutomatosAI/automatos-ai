import fs from 'fs'
import path from 'path'
import { expect, test, type Locator, type Page } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseStyle, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage, type PageWatch } from '../support/watch'

// PRD-261 A-12g (@host): a theme's playbooks page reads its data through the SDK's pages/playbooks
// area. The hello theme (fixtures/hello-a12g) registers a page on /playbooks; this spec answers the
// Playbook endpoints from fixtures/hello-a12g/responses.json (test data), so what the theme draws can
// only have come through the SDK's hooks, the core hooks they wrap and the host's backend client:
//
// 1. /playbooks: the list (useWorkflowPlaybooks), in the order and with the total the backend gave,
//    asked for with the theme's limit and sort.
// 2. Open a playbook: its steps (useWorkflowPlaybook) and its runs (usePlaybookExecutions); Run
//    (useExecutePlaybook) posts to that playbook's execute endpoint and shows the new run's id.

interface PlaybookRow {
  readonly template_id: string
  readonly name: string
  readonly steps: readonly unknown[]
}

interface Responses {
  readonly playbookList: { readonly items: readonly PlaybookRow[]; readonly total: number; readonly limit: number }
  readonly playbookDetail: PlaybookRow
  readonly runs: { readonly items: readonly { readonly execution_id: string; readonly status: string }[] }
  readonly runStarted: { readonly recipe_execution_id: string }
}

const HELLO_A12G: InstalledTheme = { installId: '00000000-0000-4000-8000-0000000a12a7', fixture: 'hello-a12g' }
const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const RESPONSES_FILE = path.resolve(__dirname, '..', 'fixtures', 'hello-a12g', 'responses.json')
const RESPONSES = JSON.parse(fs.readFileSync(RESPONSES_FILE, 'utf8')) as Responses
const STATUS_ATTR = 'data-theme-host'
const FRAME_TEST_ID = 'hello-a12g-frame'
const PLAYBOOKS_PATH = '/api/workflow-recipes'
const EXECUTIONS_SUFFIX = '/executions'
const EXECUTE_SUFFIX = '/execute'
const PAGE_ADDRESS = '/playbooks'
/** The entry's useWorkflowPlaybooks({ limit: 10, sort_by: 'name' }). */
const LIST_QUERY = { limit: '10', sort_by: 'name' }
const OK = 200
const SHARING_FAILURE = /Invalid hook call|Minified React error|not in this SDK build/i

interface PlaybookTraffic {
  /** The query string of every list read. */
  readonly listQueries: URLSearchParams[]
  /** The path of every run posted. */
  readonly runs: string[]
}

/** Answers the endpoints the theme's page reads; returns what it saw. */
async function servePlaybookData(page: Page): Promise<PlaybookTraffic> {
  const detailPath = `${PLAYBOOKS_PATH}/${RESPONSES.playbookDetail.template_id}`
  const executePath = `${detailPath}${EXECUTE_SUFFIX}`
  const traffic: PlaybookTraffic = { listQueries: [], runs: [] }
  await page.route((url) => url.pathname === PLAYBOOKS_PATH, async (route) => {
    if (route.request().method() === 'GET') traffic.listQueries.push(new URL(route.request().url()).searchParams)
    await fulfillBackendJson(route, OK, RESPONSES.playbookList)
  })
  await page.route((url) => url.pathname === detailPath, (route) => fulfillBackendJson(route, OK, RESPONSES.playbookDetail))
  await page.route(
    (url) => url.pathname === `${detailPath}${EXECUTIONS_SUFFIX}`,
    (route) => fulfillBackendJson(route, OK, RESPONSES.runs),
  )
  await page.route((url) => url.pathname === executePath, async (route) => {
    if (route.request().method() === 'POST') traffic.runs.push(executePath)
    await fulfillBackendJson(route, OK, RESPONSES.runStarted)
  })
  return traffic
}

async function openUnderTheme(page: Page): Promise<Locator> {
  await chooseStyle(page, `theme:install:${HELLO_A12G.installId}`, APP_URL)
  await serveInstalledTheme(page, HELLO_A12G)
  await page.goto(PAGE_ADDRESS)
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
  const frame = page.getByTestId(FRAME_TEST_ID).first()
  await expect(frame).toBeVisible()
  return frame
}

function expectNoSharingFailure(watch: PageWatch): void {
  expect(watch.pageErrors).toEqual([])
  expect(watch.consoleErrors.filter((text) => SHARING_FAILURE.test(text))).toEqual([])
}

test.describe("a theme's playbooks page reads its data through the SDK", () => {
  test('the page lists the playbooks the backend answered', { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    const traffic = await servePlaybookData(page)
    const frame = await openUnderTheme(page)

    const themePage = frame.getByTestId('hello-a12g-page')
    await expect(themePage).toBeVisible()
    await expect(page).toHaveURL(new RegExp(`${PAGE_ADDRESS}$`))
    const rows = themePage.getByTestId('hello-a12g-playbook')
    await expect(rows).toHaveCount(RESPONSES.playbookList.items.length)
    for (const [index, playbook] of RESPONSES.playbookList.items.entries()) {
      await expect(rows.nth(index)).toHaveAttribute('data-template-id', playbook.template_id)
      await expect(rows.nth(index).getByTestId('hello-a12g-playbook-name')).toHaveText(playbook.name)
    }
    await expect(themePage.getByTestId('hello-a12g-total')).toHaveText(String(RESPONSES.playbookList.total))
    expect(traffic.listQueries.length).toBeGreaterThan(0)
    expect(Object.fromEntries(traffic.listQueries[0].entries())).toEqual(LIST_QUERY)
    expectNoSharingFailure(watch)
  })

  test('an open playbook shows its steps and runs, and Run starts a run', { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    const traffic = await servePlaybookData(page)
    const detail = RESPONSES.playbookDetail
    const frame = await openUnderTheme(page)

    await frame.getByRole('button', { name: `Open ${detail.name}` }).click()
    const open = frame.getByTestId('hello-a12g-open')
    await expect(open.getByTestId('hello-a12g-open-name')).toHaveText(detail.name)
    await expect(open.getByTestId('hello-a12g-steps')).toHaveText(String(detail.steps.length))
    const runs = open.getByTestId('hello-a12g-run')
    await expect(runs).toHaveText(RESPONSES.runs.items.map((run) => run.execution_id))
    for (const [index, run] of RESPONSES.runs.items.entries()) {
      await expect(runs.nth(index)).toHaveAttribute('data-status', run.status)
    }

    await open.getByTestId('hello-a12g-run-now').click()
    await expect(open.getByTestId('hello-a12g-started')).toHaveText(RESPONSES.runStarted.recipe_execution_id)
    expect(traffic.runs).toEqual([`${PLAYBOOKS_PATH}/${detail.template_id}${EXECUTE_SUFFIX}`])
    expectNoSharingFailure(watch)
  })
})
