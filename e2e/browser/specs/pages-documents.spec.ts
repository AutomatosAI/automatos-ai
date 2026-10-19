import fs from 'fs'
import path from 'path'
import { expect, test, type Locator, type Page } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseStyle, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage, type PageWatch } from '../support/watch'

// PRD-261 A-12f (@host): a theme's knowledge page reads its data through the SDK's pages/documents
// area. The hello theme (fixtures/hello-a12f) registers a page on /documents; this spec answers the
// Knowledge page's endpoints from fixtures/hello-a12f/responses.json (test data), so what the theme
// draws can only have come through the SDK's hooks, the core hooks they wrap and the host's
// backend client:
//
// 1. The documents (useDocuments), the total (useDocumentStats), each team with its count
//    (useTeams, useDocumentTeamCounts), the cloud connections (useCloudConnections) and the
//    database sources (useDatabaseKnowledge), in place of the core Knowledge page.
// 2. Delete (useDeleteDocument) sends DELETE /api/documents/<id>, and the list the core hook
//    refetches after it no longer has that document.

interface DocumentRow {
  readonly id: number
  readonly filename: string
  readonly status: string
}

interface Responses {
  readonly documents: readonly DocumentRow[]
  readonly stats: { readonly total_documents: number }
  readonly teams: readonly { readonly name: string; readonly normalized_name: string }[]
  readonly teamCounts: { readonly counts: Readonly<Record<string, number>> }
  readonly connections: readonly { readonly app_name: string }[]
  readonly databaseSources: readonly { readonly name: string; readonly schema_tables_count: number }[]
  readonly templates: readonly unknown[]
}

const HELLO_A12F: InstalledTheme = { installId: '00000000-0000-4000-8000-00000000a12f', fixture: 'hello-a12f' }
const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const RESPONSES_FILE = path.resolve(__dirname, '..', 'fixtures', 'hello-a12f', 'responses.json')
const RESPONSES = JSON.parse(fs.readFileSync(RESPONSES_FILE, 'utf8')) as Responses
const STATUS_ATTR = 'data-theme-host'
const FRAME_TEST_ID = 'hello-a12f-frame'
const KNOWLEDGE_PATH = '/documents'
const DOCUMENTS_LIST_PATH = '/api/documents/'
const DOCUMENT_PATH = /^\/api\/documents\/(\d+)$/
const STATIC_ANSWERS: Readonly<Record<string, unknown>> = {
  '/api/documents/analytics': RESPONSES.stats,
  '/api/teams': RESPONSES.teams,
  '/api/documents/team-counts': RESPONSES.teamCounts,
  '/api/cloud-documents/connections': RESPONSES.connections,
  '/api/knowledge/sources/database/': RESPONSES.databaseSources,
  '/api/knowledge/sources/database/templates/list': RESPONSES.templates,
}
const OK = 200
/** The core Knowledge page's heading (theme-pages.spec.ts): it must not be drawn under the theme's page. */
const CORE_HEADING = /Knowledge/
const SHARING_FAILURE = /Invalid hook call|Minified React error|not in this SDK build/i

/** Answers the endpoints the theme's page reads; returns the paths of the DELETEs it saw. */
async function serveKnowledgeData(page: Page): Promise<string[]> {
  const deleted = new Set<number>()
  const deletes: string[] = []
  await page.route((url) => url.pathname === DOCUMENTS_LIST_PATH, (route) =>
    fulfillBackendJson(route, OK, RESPONSES.documents.filter((row) => !deleted.has(row.id))),
  )
  await page.route((url) => Object.prototype.hasOwnProperty.call(STATIC_ANSWERS, url.pathname), (route) =>
    fulfillBackendJson(route, OK, STATIC_ANSWERS[new URL(route.request().url()).pathname]),
  )
  await page.route((url) => DOCUMENT_PATH.test(url.pathname), async (route) => {
    const { pathname } = new URL(route.request().url())
    if (route.request().method() === 'DELETE') {
      deletes.push(pathname)
      deleted.add(Number(DOCUMENT_PATH.exec(pathname)?.[1]))
    }
    await fulfillBackendJson(route, OK, { message: 'Document deleted' })
  })
  return deletes
}

async function openUnderTheme(page: Page): Promise<Locator> {
  await chooseStyle(page, `theme:install:${HELLO_A12F.installId}`, APP_URL)
  await serveInstalledTheme(page, HELLO_A12F)
  await page.goto(KNOWLEDGE_PATH)
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
  const frame = page.getByTestId(FRAME_TEST_ID).first()
  await expect(frame).toBeVisible()
  return frame
}

function expectNoSharingFailure(watch: PageWatch): void {
  expect(watch.pageErrors).toEqual([])
  expect(watch.consoleErrors.filter((text) => SHARING_FAILURE.test(text))).toEqual([])
}

test.describe("a theme's knowledge page reads its data through the SDK", () => {
  test('the page draws the documents, the totals, the teams and the sources', { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    await serveKnowledgeData(page)
    const frame = await openUnderTheme(page)

    const themePage = frame.getByTestId('hello-a12f-page')
    await expect(themePage).toBeVisible()
    await expect(frame.getByRole('heading', { level: 1, name: CORE_HEADING })).toHaveCount(0)
    await expect(themePage.getByTestId('hello-a12f-filename')).toHaveText(RESPONSES.documents.map((row) => row.filename))
    const rows = themePage.getByTestId('hello-a12f-document')
    for (const [index, row] of RESPONSES.documents.entries()) {
      await expect(rows.nth(index)).toHaveAttribute('data-document-id', String(row.id))
      await expect(rows.nth(index)).toHaveAttribute('data-status', row.status)
    }
    await expect(themePage.getByTestId('hello-a12f-total')).toHaveText(String(RESPONSES.stats.total_documents))
    const teams = RESPONSES.teams.map((team) => `${team.name} (${RESPONSES.teamCounts.counts[team.normalized_name] ?? 0})`)
    await expect(themePage.getByTestId('hello-a12f-team')).toHaveText(teams)
    await expect(themePage.getByTestId('hello-a12f-connection')).toHaveText(RESPONSES.connections.map((row) => row.app_name))
    const sources = RESPONSES.databaseSources.map((row) => `${row.name} (${row.schema_tables_count} tables)`)
    await expect(themePage.getByTestId('hello-a12f-source')).toHaveText(sources)
    expectNoSharingFailure(watch)
  })

  test('Delete sends the DELETE and the refetched list drops the document', { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    const deletes = await serveKnowledgeData(page)
    const frame = await openUnderTheme(page)
    const [first, ...rest] = RESPONSES.documents

    const themePage = frame.getByTestId('hello-a12f-page')
    await expect(themePage.getByTestId('hello-a12f-document')).toHaveCount(RESPONSES.documents.length)
    await themePage.getByRole('button', { name: `Delete ${first.filename}` }).click()

    await expect(themePage.getByTestId('hello-a12f-filename')).toHaveText(rest.map((row) => row.filename))
    expect(deletes).toEqual([`/api/documents/${first.id}`])
    expectNoSharingFailure(watch)
  })
})
