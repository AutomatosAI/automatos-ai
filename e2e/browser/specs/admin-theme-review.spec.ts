import { execFileSync } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { expect, test, type APIRequestContext } from '@playwright/test'
import { chooseStyle } from '../support/hello-theme'

// PRD-261 C-03b (@host): the admin review queue, end to end on the local edition, against the real
// backend (no route interception). The local operator is the instance's super admin
// (orchestrator/core/auth/hybrid.py:642), so the queue is theirs:
//
//   submit fixtures/hello-c03b's package to the Marketplace → /admin/themes lists it → its report
//   shows the checks and "LLM stage did not run" (the lane's backend has no LLM key) → Approve →
//   the version is approved, the signature request is shown, and it has left the pending list.
//
// Chromium only: the lane's backend is one database for the whole run, and an approved version can't
// be approved again, so a second browser would find nothing to approve.

const BACKEND_URL = process.env.E2E_BACKEND_URL ?? 'http://localhost:8000'
const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const BUILDER = path.resolve(__dirname, '..', 'fixtures', 'hello-c03b', 'build.py')
const PACKAGE_NAME = 'hello-c03b.autotheme'
const PACKAGE_TYPE = 'application/zip'
const THEME_NAME = 'Hello review queue (C-03b)'
const QUEUE_PATH = '/admin/themes'
const STUDIO = 'studio'
const LLM_DID_NOT_RUN = 'LLM stage did not run'
const LOCAL_REVIEW = /your approval is the review/
const SPEC_TIMEOUT_MS = 150_000
// Q9 (one open submission per author) and the review slots are shared with the other specs that
// submit as the same local operator: a 409 "waiting for review" or a 503 is waited out, not failed.
const SUBMIT_WAIT_MS = 90_000
const SUBMIT_RETRY_MS = 3_000
const BUSY = 503
const CONFLICT = 409
const WAITING_FOR_REVIEW = 'waiting for review'

interface Submitted {
  readonly id: string
  readonly status: string
  readonly digest: string
}

function buildPackage(): Buffer {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'hello-c03b-'))
  const out = path.join(folder, PACKAGE_NAME)
  execFileSync('python3', [BUILDER, '--out', out], { stdio: 'pipe' })
  return fs.readFileSync(out)
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/** Submits the package (201, or 200 for the same bytes), waiting out another spec's open submission. */
async function submit(request: APIRequestContext, buffer: Buffer): Promise<Submitted> {
  const deadline = Date.now() + SUBMIT_WAIT_MS
  for (;;) {
    const response = await request.post(`${BACKEND_URL}/api/marketplace/themes/submit`, {
      multipart: { file: { name: PACKAGE_NAME, mimeType: PACKAGE_TYPE, buffer } },
    })
    if (response.ok()) return ((await response.json()) as { package: Submitted }).package
    const body = await response.text()
    const waiting = response.status() === BUSY || (response.status() === CONFLICT && body.includes(WAITING_FOR_REVIEW))
    if (!waiting || Date.now() > deadline) throw new Error(`POST …/submit answered ${response.status()}: ${body}`)
    await sleep(SUBMIT_RETRY_MS)
  }
}

async function reportStatus(request: APIRequestContext, packageId: string): Promise<string> {
  const response = await request.get(`${BACKEND_URL}/api/admin/themes/${packageId}/report`)
  expect(response.status()).toBe(200)
  return ((await response.json()) as { status: string }).status
}

test.describe('Admin › Theme review queue', () => {
  test.beforeEach(async ({ page, browserName }) => {
    test.skip(browserName !== 'chromium', 'one database per run: an approved version is proved once (see the header)')
    await chooseStyle(page, STUDIO, APP_URL)
  })

  test('approve the submitted hello package in the queue', { tag: '@host' }, async ({ page }) => {
    test.setTimeout(SPEC_TIMEOUT_MS)
    const submitted = await submit(page.request, buildPackage())
    expect(submitted.status, 'the fixture version is new in this run').toBe('pending')

    await page.goto(QUEUE_PATH)
    const row = page.getByTestId(`pending-theme-${submitted.id}`)
    await expect(row).toContainText(THEME_NAME)
    await row.click()

    const report = page.getByTestId('theme-review-report')
    await expect(report.getByTestId('theme-check-sizes')).toContainText('Pass')
    await expect(report.getByTestId('theme-llm-not-run')).toContainText(LLM_DID_NOT_RUN)
    await expect(report.getByTestId('theme-local-review')).toHaveText(LOCAL_REVIEW)
    await expect(report).toContainText(submitted.digest)

    const approve = report.getByRole('button', { name: 'Approve', exact: true })
    await expect(approve).toBeEnabled()
    await approve.click()

    await expect(report.getByTestId('theme-review-status')).toHaveText('Approved')
    await expect(report.getByTestId('theme-signature-request')).toContainText(submitted.digest)
    await expect(page.getByTestId(`pending-theme-${submitted.id}`)).toHaveCount(0)
    expect(await reportStatus(page.request, submitted.id)).toBe('approved')
  })
})
