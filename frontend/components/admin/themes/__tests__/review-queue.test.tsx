/**
 * PRD-261 C-03b — the admin theme review queue: the pending list, the report (Error, Warning,
 * Pass; "LLM stage did not run"), Approve (disabled for a blocked or not-run scan and while the
 * server's `approval` says no), Reject with a reason, the hosted edition's isolated-job note that
 * must name the digest, the signature request and paste-back, take-down, and who sees the queue.
 *
 * `@clerk/nextjs` is the double install-update.test.tsx uses: it does what Clerk's
 * `createReverificationHandler` does with no `onNeedsReverification` (call the fetcher; on Clerk's
 * own `isReverificationHint(result)`, open the reverification, call the fetcher ONE more time), so
 * a step-up 403 on approve is proved to reach Clerk from the button.
 */
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  getPendingThemes: vi.fn(),
  getThemeReport: vi.fn(),
  addThemeReportNote: vi.fn(),
  approveTheme: vi.fn(),
  rejectTheme: vi.fn(),
  deactivateTheme: vi.fn(),
  storeThemeSignature: vi.fn(),
}))
const clerk = vi.hoisted(() => ({ openReverification: vi.fn() }))
const role = vi.hoisted(() => ({
  current: { systemRole: 'super_admin', isAdmin: true, isLoading: false } as
    | { systemRole: string; isAdmin: boolean; isLoading: boolean }
    | null,
}))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/contexts/role-context', () => ({ useSystemRoleOptional: () => role.current }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@clerk/nextjs', async () => {
  const { isReverificationHint } = await import('@clerk/shared/authorization-errors')
  type Fetcher = (...args: unknown[]) => Promise<unknown>
  return {
    useReverification: (fetcher: Fetcher) => async (...args: unknown[]) => {
      let result = await fetcher(...args)
      if (isReverificationHint(result)) {
        await clerk.openReverification()
        result = await fetcher(...args)
      }
      return result
    },
  }
})

import { DEACTIVATE } from '../deactivate-form'
import { NOT_READY, READY } from '../pending-list'
import { SAVE_NOTE } from '../report-note-form'
import { APPROVE, REJECT } from '../review-actions'
import { ThemeReviewQueue } from '../review-queue'
import { ADMIN_ONLY, LOCAL_REVIEW_NOTE, QUEUE_EMPTY, SCAN_BLOCKED, SCAN_NOT_RUN, SIGNATURE_INVALID } from '../review-words'
import { SAVE_SIGNATURE } from '../signature-panel'
import {
  DIGEST,
  KEY_ID,
  PACKAGE_ID,
  REQUEST_ID,
  RETIRED_KEY_ID,
  httpError,
  kitSummary,
  pendingRow,
  renderWithQuery,
  reportOf,
  signatureRequest,
  stepUpError,
} from './review-fixtures'

const NOTE_MISSING = 'Run the isolated review job in the kit repo and paste its summary first.'

async function openReport(hosted = false) {
  renderWithQuery(<ThemeReviewQueue hosted={hosted} />)
  fireEvent.click(await screen.findByTestId(`pending-theme-${PACKAGE_ID}`))
  return screen.findByTestId('theme-review-report')
}

const OTHER_ID = '66666666-6666-4666-8666-666666666666'
const OTHER_NAME = 'Other theme'

function rejectReason() {
  return screen.getByLabelText(/Reason \(the submitter sees it\)/)
}

function approveButton() {
  return screen.getByRole('button', { name: APPROVE })
}

beforeEach(() => {
  for (const mock of Object.values(api)) mock.mockReset()
  clerk.openReverification.mockReset()
  clerk.openReverification.mockResolvedValue(undefined)
  role.current = { systemRole: 'super_admin', isAdmin: true, isLoading: false }
  api.getPendingThemes.mockResolvedValue({ pending: [pendingRow()] })
  api.getThemeReport.mockResolvedValue(reportOf())
})

describe('the pending list', () => {
  it('lists each pending version with its verdict and whether it is ready', async () => {
    api.getPendingThemes.mockResolvedValue({
      pending: [pendingRow(), pendingRow({ id: 'other', scanVerdict: 'blocked', approval: { allowed: false, reason: 'x' } })],
    })
    renderWithQuery(<ThemeReviewQueue hosted={false} />)

    const ready = await screen.findByTestId(`pending-theme-${PACKAGE_ID}`)
    expect(within(ready).getByText('Review required')).toBeInTheDocument()
    expect(within(ready).getByText(READY)).toBeInTheDocument()
    const blocked = screen.getByTestId('pending-theme-other')
    expect(within(blocked).getByText('Blocked')).toBeInTheDocument()
    expect(within(blocked).getByText(NOT_READY)).toBeInTheDocument()
    expect(api.getThemeReport).not.toHaveBeenCalled()
  })

  it('says so when nothing is waiting', async () => {
    api.getPendingThemes.mockResolvedValue({ pending: [] })
    renderWithQuery(<ThemeReviewQueue hosted={false} />)
    expect(await screen.findByText(QUEUE_EMPTY)).toBeInTheDocument()
  })

  it("shows the server's 403 sentence to a caller the server does not take as an admin", async () => {
    api.getPendingThemes.mockRejectedValue(httpError(403, ADMIN_ONLY))
    renderWithQuery(<ThemeReviewQueue hosted={false} />)
    expect(await screen.findByRole('alert')).toHaveTextContent(ADMIN_ONLY)
  })

  it('a person whose system role is not admin sees the notice and no request is made', async () => {
    role.current = { systemRole: 'user', isAdmin: false, isLoading: false }
    renderWithQuery(<ThemeReviewQueue hosted />)
    expect(screen.getByTestId('theme-queue-admin-only')).toHaveTextContent(ADMIN_ONLY)
    expect(api.getPendingThemes).not.toHaveBeenCalled()
  })
})

describe('the report', () => {
  it('shows every check as Error, Warning or Pass, and the LLM stage that did not run', async () => {
    const report = await openReport()

    expect(api.getThemeReport).toHaveBeenCalledWith(PACKAGE_ID)
    expect(within(within(report).getByTestId('theme-check-sizes')).getByText('Pass')).toBeInTheDocument()
    const uses = within(report).getByTestId('theme-check-uses')
    expect(within(uses).getByText('Error')).toBeInTheDocument()
    expect(within(uses).getByText('useSecret is not an SDK name.')).toBeInTheDocument()
    expect(within(within(report).getByTestId('theme-check-contrast')).getByText('Not measured')).toBeInTheDocument()
    const llm = within(report).getByTestId('theme-check-llm')
    expect(within(llm).getByText('Warning')).toBeInTheDocument()
    expect(within(llm).getByTestId('theme-llm-not-run')).toHaveTextContent('LLM stage did not run')
    expect(within(report).getByText(DIGEST)).toBeInTheDocument()
  })

  it('an LLM check with no reason still says "LLM stage did not run", never Pass', async () => {
    const llm = { level: 'warning', status: 'partial', findings: [] }
    api.getThemeReport.mockResolvedValue(reportOf({ scanReport: { verdict: 'review_required', checks: { llm } } }))
    const report = await openReport()
    expect(within(report).getByTestId('theme-llm-not-run')).toHaveTextContent('LLM stage did not run')
  })
})

describe('Approve', () => {
  it('approves an allowed version; a step-up 403 opens Clerk once and approve is sent again', async () => {
    api.approveTheme.mockRejectedValueOnce(stepUpError()).mockResolvedValueOnce({ ...pendingRow({ status: 'approved' }) })
    await openReport()
    fireEvent.click(approveButton())

    await waitFor(() => expect(api.approveTheme).toHaveBeenCalledTimes(2))
    expect(api.approveTheme).toHaveBeenCalledWith(PACKAGE_ID)
    expect(clerk.openReverification).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(api.getThemeReport).toHaveBeenCalledTimes(2))
  })

  it('is disabled for a blocked scan, with the reason', async () => {
    api.getThemeReport.mockResolvedValue(reportOf({ scanVerdict: 'blocked', approval: { allowed: false, reason: 'blocked' } }))
    await openReport()
    expect(approveButton()).toBeDisabled()
    expect(screen.getByTestId('theme-approve-refusal')).toHaveTextContent(SCAN_BLOCKED)
  })

  it('is disabled for a not-run scan even when the approval state says allowed', async () => {
    api.getThemeReport.mockResolvedValue(reportOf({ scanVerdict: 'not_run', approval: { allowed: true, reason: null } }))
    await openReport()
    expect(approveButton()).toBeDisabled()
    expect(screen.getByTestId('theme-approve-refusal')).toHaveTextContent(SCAN_NOT_RUN)
    fireEvent.click(approveButton())
    expect(api.approveTheme).not.toHaveBeenCalled()
  })

  it("is disabled while the server's approval says no, with the server's reason", async () => {
    api.getThemeReport.mockResolvedValue(reportOf({ approval: { allowed: false, reason: NOTE_MISSING } }))
    await openReport(true)
    expect(approveButton()).toBeDisabled()
    expect(screen.getByTestId('theme-approve-refusal')).toHaveTextContent(NOTE_MISSING)
  })
})

describe('Reject', () => {
  it('needs a reason and sends it trimmed', async () => {
    api.rejectTheme.mockResolvedValue({ ...pendingRow({ status: 'rejected' }), reason: 'Copies a core page.' })
    await openReport()
    const reject = screen.getByRole('button', { name: REJECT })
    expect(reject).toBeDisabled()
    fireEvent.change(screen.getByLabelText(/Reason \(the submitter sees it\)/), { target: { value: '  Copies a core page. ' } })
    fireEvent.click(reject)

    await waitFor(() => expect(api.rejectTheme).toHaveBeenCalledWith(PACKAGE_ID, 'Copies a core page.'))
    expect(clerk.openReverification).not.toHaveBeenCalled()
  })

  it('a reason typed for one version does not carry over to another whose report is cached', async () => {
    api.getPendingThemes.mockResolvedValue({ pending: [pendingRow(), pendingRow({ id: OTHER_ID, name: OTHER_NAME })] })
    api.getThemeReport.mockImplementation(async (id: string) => reportOf(id === OTHER_ID ? { id: OTHER_ID, name: OTHER_NAME } : {}))
    renderWithQuery(<ThemeReviewQueue hosted={false} />)
    const pick = async (id: string, name: string) => {
      fireEvent.click(await screen.findByTestId(`pending-theme-${id}`))
      await within(await screen.findByTestId('theme-review-report')).findByRole('heading', { name })
    }
    await pick(PACKAGE_ID, 'Hello theme')
    await pick(OTHER_ID, OTHER_NAME) // both reports are cached from here on
    await pick(PACKAGE_ID, 'Hello theme')
    fireEvent.change(rejectReason(), { target: { value: 'Copies a core page.' } })

    await pick(OTHER_ID, OTHER_NAME)
    expect(rejectReason()).toHaveValue('')
    expect(screen.getByRole('button', { name: REJECT })).toBeDisabled()
    expect(api.rejectTheme).not.toHaveBeenCalled()
  })
})

describe('the isolated review note', () => {
  it('hosted: a passing note must hold the digest line before it can be saved', async () => {
    api.addThemeReportNote.mockResolvedValue({ ...pendingRow(), reportNote: { summary: 'x', outcome: 'pass', by: 7 } })
    await openReport(true)
    const save = screen.getByRole('button', { name: SAVE_NOTE })
    const summary = screen.getByLabelText('Paste the Theme review job summary')
    fireEvent.change(summary, { target: { value: kitSummary('e'.repeat(64)) } })
    expect(save).toBeDisabled()
    expect(screen.getByTestId('theme-note-digest-hint')).toHaveTextContent(`Package digest: ${DIGEST}`)

    fireEvent.change(summary, { target: { value: kitSummary(DIGEST) } })
    expect(save).toBeEnabled()
    fireEvent.click(save)
    await waitFor(() =>
      expect(api.addThemeReportNote).toHaveBeenCalledWith(PACKAGE_ID, { summary: kitSummary(DIGEST), outcome: 'pass' }),
    )
  })

  it('hosted: a failing note needs no digest line', async () => {
    api.addThemeReportNote.mockResolvedValue({ ...pendingRow(), reportNote: { summary: 'x', outcome: 'fail', by: 7 } })
    await openReport(true)
    fireEvent.change(screen.getByLabelText('Paste the Theme review job summary'), { target: { value: 'build differs' } })
    fireEvent.click(screen.getByLabelText('Fail'))
    fireEvent.click(screen.getByRole('button', { name: SAVE_NOTE }))
    await waitFor(() =>
      expect(api.addThemeReportNote).toHaveBeenCalledWith(PACKAGE_ID, { summary: 'build differs', outcome: 'fail' }),
    )
  })

  it('local: no note form; the approval is the review', async () => {
    await openReport(false)
    expect(screen.getByTestId('theme-local-review')).toHaveTextContent(LOCAL_REVIEW_NOTE)
    expect(screen.queryByRole('button', { name: SAVE_NOTE })).toBeNull()
  })
})

describe('after approval', () => {
  it('shows the whole signature request and stores the pasted signature for it', async () => {
    api.getThemeReport.mockResolvedValue(
      reportOf({ status: 'approved', reviewed: true, signatureRequest: signatureRequest() }),
    )
    api.storeThemeSignature.mockResolvedValue({})
    await openReport()
    expect(screen.queryByRole('button', { name: APPROVE })).toBeNull()
    const request = screen.getByTestId('theme-signature-request')
    expect(request).toHaveTextContent(REQUEST_ID)
    expect(request).toHaveTextContent('"payloadSha256"')

    fireEvent.change(screen.getByLabelText(/Paste the signature/), { target: { value: ' c2lnbmF0dXJl ' } })
    fireEvent.click(screen.getByRole('button', { name: SAVE_SIGNATURE }))
    await waitFor(() => expect(api.storeThemeSignature).toHaveBeenCalledWith(PACKAGE_ID, REQUEST_ID, 'c2lnbmF0dXJl'))
  })

  it('a signed version an active Automatos key verifies says so, naming the key (C-05b)', async () => {
    api.getThemeReport.mockResolvedValue(
      reportOf({ status: 'approved', reviewed: true, signed: true, signatureVerified: true, keyId: KEY_ID }),
    )
    await openReport()
    const signed = screen.getByTestId('theme-signed')
    expect(signed).toHaveTextContent(`Signed. Verified with the active Automatos key ${KEY_ID}.`)
    expect(signed).not.toHaveTextContent(/not verified|does not verify/)
  })

  it.each([null, RETIRED_KEY_ID])(
    'a signed version no active key verifies says it does not verify against one (key %s)',
    async (keyId) => {
      api.getThemeReport.mockResolvedValue(reportOf({ status: 'approved', reviewed: true, signed: true, keyId }))
      await openReport()
      const signed = screen.getByTestId('theme-signed')
      expect(signed).toHaveTextContent(`Signed. ${SIGNATURE_INVALID}`)
      expect(signed.textContent?.includes(RETIRED_KEY_ID)).toBe(keyId !== null)
    },
  )

  it('deactivates an approved version with a reason', async () => {
    api.getThemeReport.mockResolvedValue(reportOf({ status: 'approved', reviewed: true }))
    api.deactivateTheme.mockResolvedValue({})
    await openReport()
    fireEvent.change(screen.getByLabelText('Reason'), { target: { value: 'Steals tokens' } })
    fireEvent.click(screen.getByRole('button', { name: DEACTIVATE }))
    await waitFor(() => expect(api.deactivateTheme).toHaveBeenCalledWith(PACKAGE_ID, 'Steals tokens'))
  })

  it('a pending version has no take-down and no signature panel', async () => {
    await openReport()
    expect(screen.queryByRole('button', { name: DEACTIVATE })).toBeNull()
    expect(screen.queryByTestId('theme-signature-request')).toBeNull()
  })
})
