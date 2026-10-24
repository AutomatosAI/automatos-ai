/**
 * PRD-261 C-05j — the report's signature panel and the review job's inputs:
 *
 * - a stored signature an active key verifies reads "Verified · <keyId>", one none verifies says
 *   so, and a pasted signature the server refuses shows the server's 400 sentence (nothing says a
 *   check is pending: the report checks the stored signature on every read);
 * - on the hosted edition a pending version's report shows "Review job inputs" from
 *   `GET …/review-package` (C-03c), asked for only when the reviewer wants them; the local edition,
 *   with no isolated job, shows none;
 * - nothing under components/admin/themes calls the backend with a raw `fetch('/api…')`.
 */
import { readdirSync, readFileSync } from 'fs'
import path from 'path'
import { fireEvent, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  getThemeReport: vi.fn(),
  storeThemeSignature: vi.fn(),
  getThemeReviewPackage: vi.fn(),
  addThemeReportNote: vi.fn(),
  approveTheme: vi.fn(),
  rejectTheme: vi.fn(),
  deactivateTheme: vi.fn(),
}))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@clerk/nextjs', () => ({
  useReverification: (fetcher: (...args: unknown[]) => Promise<unknown>) => fetcher,
}))

import { ReviewReport } from '../review-report'
import { GET_REVIEW_JOB_INPUTS, REVIEW_JOB_INPUTS, SIGNATURE_INVALID } from '../review-words'
import { SAVE_SIGNATURE } from '../signature-panel'
import {
  DIGEST,
  KEY_ID,
  PACKAGE_ID,
  REQUEST_ID,
  httpError,
  renderWithQuery,
  reportOf,
  reviewLink,
  signatureRequest,
} from './review-fixtures'

// signing.py NOT_VERIFIED: the 400 a pasted signature no active key verifies answers.
const NOT_VERIFIED =
  "No Automatos signing key verifies that signature for this version, so it wasn't stored. " +
  "Paste the signature the kit's sign workflow printed for this version's request."
const PENDING_WORDS = /verification is pending|pending verification/i
const ADMIN_THEMES_DIR = path.resolve(__dirname, '..')

async function openReport(hosted: boolean) {
  renderWithQuery(<ReviewReport packageId={PACKAGE_ID} hosted={hosted} />)
  return screen.findByTestId('theme-review-report')
}

beforeEach(() => {
  for (const mock of Object.values(api)) mock.mockReset()
  api.getThemeReport.mockResolvedValue(reportOf())
  api.getThemeReviewPackage.mockResolvedValue(reviewLink())
})

describe('the signature panel (C-05j)', () => {
  it('a signature an active key verifies reads "Verified · <keyId>"', async () => {
    api.getThemeReport.mockResolvedValue(
      reportOf({ status: 'approved', reviewed: true, signed: true, signatureVerified: true, keyId: KEY_ID }),
    )
    const report = await openReport(false)
    expect(screen.getByTestId('theme-signed')).toHaveTextContent(`Verified · ${KEY_ID}`)
    expect(report).not.toHaveTextContent(PENDING_WORDS)
  })

  it('one no active key verifies says so, never pending', async () => {
    api.getThemeReport.mockResolvedValue(reportOf({ status: 'approved', reviewed: true, signed: true }))
    const report = await openReport(false)
    expect(screen.getByTestId('theme-signed')).toHaveTextContent(SIGNATURE_INVALID)
    expect(report).not.toHaveTextContent(PENDING_WORDS)
  })

  it("a pasted signature the server refuses shows the server's 400 sentence", async () => {
    api.getThemeReport.mockResolvedValue(
      reportOf({ status: 'approved', reviewed: true, signatureRequest: signatureRequest() }),
    )
    api.storeThemeSignature.mockRejectedValue(httpError(400, NOT_VERIFIED))
    await openReport(false)
    fireEvent.change(screen.getByLabelText(/Paste the signature/), { target: { value: 'c2lnbmF0dXJl' } })
    fireEvent.click(screen.getByRole('button', { name: SAVE_SIGNATURE }))
    await waitFor(() => expect(api.storeThemeSignature).toHaveBeenCalledWith(PACKAGE_ID, REQUEST_ID, 'c2lnbmF0dXJl'))
    expect(await screen.findByRole('alert')).toHaveTextContent(NOT_VERIFIED)
    expect(screen.queryByTestId('theme-signed')).toBeNull()
  })
})

describe('the review job inputs (C-03c, hosted)', () => {
  it('shows package_url and expected_digest from the review-package route, once asked for', async () => {
    await openReport(true)
    expect(screen.getByRole('region', { name: REVIEW_JOB_INPUTS })).toBeInTheDocument()
    expect(api.getThemeReviewPackage).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: GET_REVIEW_JOB_INPUTS }))
    expect(await screen.findByTestId('theme-review-input-package_url')).toHaveTextContent(reviewLink().url)
    expect(screen.getByTestId('theme-review-input-expected_digest')).toHaveTextContent(DIGEST)
    expect(api.getThemeReviewPackage).toHaveBeenCalledWith(PACKAGE_ID)
  })

  it("shows the route's refusal as it is", async () => {
    const missing = 'The stored file dist/entry.js of this version is missing, so its package can\'t be rebuilt for review.'
    api.getThemeReviewPackage.mockRejectedValue(httpError(409, missing))
    await openReport(true)
    fireEvent.click(screen.getByRole('button', { name: GET_REVIEW_JOB_INPUTS }))
    expect(await screen.findByRole('alert')).toHaveTextContent(missing)
  })

  it('the local edition, which has no isolated job, and an approved version show none', async () => {
    await openReport(false)
    expect(screen.queryByRole('region', { name: REVIEW_JOB_INPUTS })).toBeNull()
    api.getThemeReport.mockResolvedValue(reportOf({ status: 'approved', reviewed: true }))
    renderWithQuery(<ReviewReport packageId="other" hosted />)
    await waitFor(() => expect(screen.getAllByTestId('theme-review-report')).toHaveLength(2))
    expect(screen.queryByRole('region', { name: REVIEW_JOB_INPUTS })).toBeNull()
  })
})

describe('no raw backend fetch in the admin theme UI', () => {
  it("calls the backend only through apiClient (no fetch('/api…'))", () => {
    const sources = readdirSync(ADMIN_THEMES_DIR).filter((name) => /\.tsx?$/.test(name))
    expect(sources.length).toBeGreaterThan(0)
    for (const name of sources) {
      const text = readFileSync(path.join(ADMIN_THEMES_DIR, name), 'utf8')
      expect(text, name).not.toMatch(/fetch\(\s*[`'"][^`'"]*\/api/)
    }
  })
})
