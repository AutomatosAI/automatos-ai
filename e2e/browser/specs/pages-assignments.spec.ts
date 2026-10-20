import fs from 'fs'
import path from 'path'
import { expect, test, type Locator, type Page } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseStyle, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage, type PageWatch } from '../support/watch'

// PRD-261 A-12e (@host): a theme's missions pages read their data through the SDK's
// pages/assignments area. The hello theme (fixtures/hello-a12e) registers a page on /assignments
// (the Missions tab) and on /missions/[id]; this spec answers the Mission endpoints from
// fixtures/hello-a12e/responses.json (test data), so what the theme draws can only have come
// through the SDK's hooks, the core hooks they wrap and the host's backend client:
//
// 1. /assignments?tab=missions: the list (useMissions) with each Mission linked to its page, its
//    total, and the recommendations (useRecommendedAssignments).
// 2. /missions/<id>: the Mission (useMission, the id from useHostPathname), and Pause
//    (usePauseMission) posts to that Mission's pause endpoint and hands back its answer.

interface MissionRow {
  readonly id: string
  readonly goal: string
  readonly state: string
}

interface Responses {
  readonly missionList: { readonly missions: readonly MissionRow[]; readonly total: number }
  readonly missionDetail: MissionRow & { readonly tasks: readonly unknown[] }
  readonly pausedMission: MissionRow
  readonly recommended: { readonly items: readonly { readonly name: string }[] }
}

const HELLO_A12E: InstalledTheme = { installId: '00000000-0000-4000-8000-00000000a12e', fixture: 'hello-a12e' }
const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const RESPONSES_FILE = path.resolve(__dirname, '..', 'fixtures', 'hello-a12e', 'responses.json')
const RESPONSES = JSON.parse(fs.readFileSync(RESPONSES_FILE, 'utf8')) as Responses
const STATUS_ATTR = 'data-theme-host'
const FRAME_TEST_ID = 'hello-a12e-frame'
const MISSIONS_PATH = '/api/missions'
const RECOMMENDED_PATH = '/api/assignments/recommended'
const PAUSE_SUFFIX = '/pause'
const MISSIONS_TAB = '/assignments?tab=missions'
const OK = 200
const SHARING_FAILURE = /Invalid hook call|Minified React error|not in this SDK build/i

/** Answers the endpoints the theme's pages read; returns the paths of the pause posts it saw. */
async function serveMissionData(page: Page): Promise<string[]> {
  const detailPath = `${MISSIONS_PATH}/${RESPONSES.missionDetail.id}`
  const pausePath = `${detailPath}${PAUSE_SUFFIX}`
  const pauses: string[] = []
  await page.route((url) => url.pathname === MISSIONS_PATH, (route) => fulfillBackendJson(route, OK, RESPONSES.missionList))
  await page.route((url) => url.pathname === RECOMMENDED_PATH, (route) => fulfillBackendJson(route, OK, RESPONSES.recommended))
  await page.route((url) => url.pathname === detailPath, (route) => fulfillBackendJson(route, OK, RESPONSES.missionDetail))
  await page.route((url) => url.pathname === pausePath, async (route) => {
    if (route.request().method() === 'POST') pauses.push(pausePath)
    await fulfillBackendJson(route, OK, RESPONSES.pausedMission)
  })
  return pauses
}

async function openUnderTheme(page: Page, address: string): Promise<Locator> {
  await chooseStyle(page, `theme:install:${HELLO_A12E.installId}`, APP_URL)
  await serveInstalledTheme(page, HELLO_A12E)
  await page.goto(address)
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
  const frame = page.getByTestId(FRAME_TEST_ID).first()
  await expect(frame).toBeVisible()
  return frame
}

function expectNoSharingFailure(watch: PageWatch): void {
  expect(watch.pageErrors).toEqual([])
  expect(watch.consoleErrors.filter((text) => SHARING_FAILURE.test(text))).toEqual([])
}

test.describe("a theme's missions pages read their data through the SDK", () => {
  test('the Missions tab lists the Missions and the recommendations', { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    await serveMissionData(page)
    const frame = await openUnderTheme(page, MISSIONS_TAB)

    const themePage = frame.getByTestId('hello-a12e-missions-page')
    await expect(themePage).toBeVisible()
    const rows = themePage.getByTestId('hello-a12e-mission')
    await expect(rows).toHaveCount(RESPONSES.missionList.missions.length)
    for (const [index, mission] of RESPONSES.missionList.missions.entries()) {
      await expect(rows.nth(index)).toHaveAttribute('data-mission-id', mission.id)
      await expect(rows.nth(index)).toHaveAttribute('data-state', mission.state)
      await expect(rows.nth(index).getByRole('link', { name: mission.goal })).toHaveAttribute('href', `/missions/${mission.id}`)
    }
    await expect(themePage.getByTestId('hello-a12e-total')).toHaveText(String(RESPONSES.missionList.total))
    const names = RESPONSES.recommended.items.map((item) => item.name)
    await expect(themePage.getByTestId('hello-a12e-recommended-item')).toHaveText(names)
    expectNoSharingFailure(watch)
  })

  test("one Mission's page reads the Mission and pauses it", { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    const pauses = await serveMissionData(page)
    const detail = RESPONSES.missionDetail
    const frame = await openUnderTheme(page, `/missions/${detail.id}`)

    const themePage = frame.getByTestId('hello-a12e-mission-page')
    await expect(themePage.getByTestId('hello-a12e-goal')).toHaveText(detail.goal)
    await expect(themePage.getByTestId('hello-a12e-state')).toHaveText(detail.state)
    await expect(themePage.getByTestId('hello-a12e-tasks')).toHaveText(String(detail.tasks.length))

    await themePage.getByTestId('hello-a12e-pause').click()
    await expect(themePage.getByTestId('hello-a12e-paused-state')).toHaveText(RESPONSES.pausedMission.state)
    expect(pauses).toEqual([`${MISSIONS_PATH}/${detail.id}${PAUSE_SUFFIX}`])
    expectNoSharingFailure(watch)
  })
})
