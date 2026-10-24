/**
 * PRD-261 C-05j — the admin's theme revocations: the rows with their origin, Lift only where the
 * server lets it (a local row, or a feed row while sync is off or stale) and only after a
 * confirmation, the create form's body and the server's 400 sentence, and the local edition's sync
 * status (last success and error, the stale warning, Check now), which the hosted edition neither
 * shows nor asks for.
 *
 * `@clerk/nextjs` is the double review-queue.test.tsx uses (create and lift carry step-up).
 */
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  getThemeRevocations: vi.fn(),
  createThemeRevocation: vi.fn(),
  liftThemeRevocation: vi.fn(),
  getThemeRevocationSync: vi.fn(),
  pullThemeRevocations: vi.fn(),
}))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@clerk/nextjs', () => ({
  useReverification: (fetcher: (...args: unknown[]) => Promise<unknown>) => fetcher,
}))

import { RevocationsPanel, feedRowsLiftable, showsLift } from '../revocations-panel'
import { revocationRequest, type RevocationFields } from '../revocation-form'
import { CHECK_NOW, CONFIRM_LIFT, KEEP_IT, LIFT, NO_REVOCATIONS, REVOKE, SYNC_OFF, SYNC_STALE } from '../review-words'
import {
  DIGEST,
  FEED_REVOCATION_ID,
  REVOCATION_ID,
  daysAgo,
  httpError,
  renderWithQuery,
  revocationRow,
  syncStatus,
} from './review-fixtures'

// revocation_admin.py BAD_DIGEST, and revocation_pull.py UNREACHABLE as a 502 records it.
const BAD_DIGEST = 'A package digest is 64 lower-case hex characters (its sha256).'
const UNREACHABLE = "The revocation feed couldn't be reached (timed out)."
const LIFTED_ID = '99999999-9999-4999-8999-999999999999'

const LOCAL_ROW = revocationRow()
const FEED_ROW = revocationRow({ id: FEED_REVOCATION_ID, origin: 'feed', digest: null, themeId: 'acme/ink', versionRange: '^1.0.0', level: 'warn' })
const LIFTED_ROW = revocationRow({ id: LIFTED_ID, liftedAt: daysAgo(2) })

function serve(rows = [LOCAL_ROW, FEED_ROW, LIFTED_ROW], sync = syncStatus()) {
  api.getThemeRevocations.mockResolvedValue({ revocations: rows })
  api.getThemeRevocationSync.mockResolvedValue(sync)
}

async function rowOf(id: string) {
  return screen.findByTestId(`theme-revocation-${id}`)
}

beforeEach(() => {
  for (const mock of Object.values(api)) mock.mockReset()
  serve()
})

describe('the revocations list', () => {
  it('lists every row with its level and origin badge', async () => {
    renderWithQuery(<RevocationsPanel hosted={false} />)
    expect(await screen.findByTestId(`theme-revocation-origin-${REVOCATION_ID}`)).toHaveTextContent('Set here')
    expect(screen.getByTestId(`theme-revocation-origin-${FEED_REVOCATION_ID}`)).toHaveTextContent('From the Automatos feed')
    const local = await rowOf(REVOCATION_ID)
    expect(local).toHaveTextContent('Disable')
    expect(local).toHaveTextContent(DIGEST)
    expect(local).toHaveTextContent(LOCAL_ROW.reason)
    expect(await rowOf(FEED_REVOCATION_ID)).toHaveTextContent('acme/ink ^1.0.0')
    expect(await rowOf(LIFTED_ID)).toHaveTextContent(/Lifted/)
  })

  it('a live local row has Lift; a feed row has none while sync works; a lifted row has none', async () => {
    renderWithQuery(<RevocationsPanel hosted={false} />)
    expect(within(await rowOf(REVOCATION_ID)).getByRole('button', { name: LIFT })).toBeInTheDocument()
    await screen.findByTestId('revocation-sync-success')
    expect(within(await rowOf(FEED_REVOCATION_ID)).queryByRole('button', { name: LIFT })).toBeNull()
    expect(within(await rowOf(LIFTED_ID)).queryByRole('button', { name: LIFT })).toBeNull()
  })

  it.each([
    ['stale', syncStatus({ stale: true, lastSuccessAt: daysAgo(8) })],
    ['off', syncStatus({ enabled: false, lastSuccessAt: null, lastAttemptAt: null })],
  ])('a feed row gets Lift while sync is %s (the server lets it lift there)', async (_case, sync) => {
    serve([FEED_ROW], sync)
    renderWithQuery(<RevocationsPanel hosted={false} />)
    const feed = await rowOf(FEED_REVOCATION_ID)
    expect(await within(feed).findByRole('button', { name: LIFT })).toBeInTheDocument()
  })

  it('the Lift rule: never a lifted row; a feed row only while feed rows are liftable', () => {
    expect(showsLift(LOCAL_ROW, false)).toBe(true)
    expect(showsLift(FEED_ROW, false)).toBe(false)
    expect(showsLift(FEED_ROW, true)).toBe(true)
    expect(showsLift(LIFTED_ROW, true)).toBe(false)
    expect(feedRowsLiftable(false, undefined)).toBe(false)
    expect(feedRowsLiftable(false, syncStatus())).toBe(false)
    expect(feedRowsLiftable(false, syncStatus({ stale: true }))).toBe(true)
    expect(feedRowsLiftable(true, undefined)).toBe(true)
  })

  it('lift asks for a confirmation first, and Keep it leaves the row alone', async () => {
    api.liftThemeRevocation.mockResolvedValue({ revocation: { ...LOCAL_ROW, liftedAt: daysAgo(0) } })
    renderWithQuery(<RevocationsPanel hosted={false} />)
    const local = await rowOf(REVOCATION_ID)
    fireEvent.click(within(local).getByRole('button', { name: LIFT }))
    expect(api.liftThemeRevocation).not.toHaveBeenCalled()
    fireEvent.click(within(local).getByRole('button', { name: KEEP_IT }))
    expect(within(local).queryByRole('button', { name: CONFIRM_LIFT })).toBeNull()

    fireEvent.click(within(local).getByRole('button', { name: LIFT }))
    fireEvent.click(within(local).getByRole('button', { name: CONFIRM_LIFT }))
    await waitFor(() => expect(api.liftThemeRevocation).toHaveBeenCalledWith(REVOCATION_ID))
    expect(api.liftThemeRevocation).toHaveBeenCalledTimes(1)
  })

  it('says so when nothing is revoked', async () => {
    serve([])
    renderWithQuery(<RevocationsPanel hosted={false} />)
    expect(await screen.findByText(NO_REVOCATIONS)).toBeInTheDocument()
  })
})

describe('revoking', () => {
  function fill(reason: string) {
    fireEvent.change(screen.getByLabelText(/^Reason/), { target: { value: reason } })
  }

  it('sends {digest, level, reason} for a digest', async () => {
    api.createThemeRevocation.mockResolvedValue({ revocation: LOCAL_ROW, created: true })
    renderWithQuery(<RevocationsPanel hosted={false} />)
    const revoke = await screen.findByRole('button', { name: REVOKE })
    expect(revoke).toBeDisabled()
    fireEvent.change(screen.getByLabelText('Package digest'), { target: { value: ` ${DIGEST} ` } })
    fill(' Sends chat text away. ')
    fireEvent.click(revoke)
    await waitFor(() =>
      expect(api.createThemeRevocation).toHaveBeenCalledWith({ digest: DIGEST, level: 'disable', reason: 'Sends chat text away.' }),
    )
  })

  it('keeps every hex character of a digest pasted with whitespace around it', async () => {
    // user-event cuts a paste to the field's maxlength as a browser does (fireEvent.change does not).
    api.createThemeRevocation.mockResolvedValue({ revocation: LOCAL_ROW, created: true })
    const user = userEvent.setup()
    renderWithQuery(<RevocationsPanel hosted={false} />)
    await user.click(await screen.findByLabelText('Package digest'))
    await user.paste(`  ${DIGEST}  `)
    fill('Sends chat text away.')
    fireEvent.click(screen.getByRole('button', { name: REVOKE }))
    await waitFor(() =>
      expect(api.createThemeRevocation).toHaveBeenCalledWith({ digest: DIGEST, level: 'disable', reason: 'Sends chat text away.' }),
    )
  })

  it('sends {themeId, versionRange, level, reason} for a theme id, and no range when it is empty', async () => {
    api.createThemeRevocation.mockResolvedValue({ revocation: FEED_ROW, created: true })
    renderWithQuery(<RevocationsPanel hosted={false} />)
    fireEvent.click(await screen.findByLabelText('A theme id'))
    fireEvent.change(screen.getByLabelText('Theme id'), { target: { value: 'acme/ink' } })
    fireEvent.change(screen.getByLabelText('Version range (optional)'), { target: { value: '^1.0.0' } })
    fireEvent.click(screen.getByLabelText(/^Warn/))
    fill('Update to 2.0.0.')
    fireEvent.click(screen.getByRole('button', { name: REVOKE }))
    await waitFor(() =>
      expect(api.createThemeRevocation).toHaveBeenCalledWith({
        themeId: 'acme/ink', versionRange: '^1.0.0', level: 'warn', reason: 'Update to 2.0.0.',
      }),
    )
    const everyVersion: RevocationFields = {
      kind: 'theme', digest: '', themeId: 'acme/ink', versionRange: '  ', level: 'disable', reason: 'Gone.',
    }
    expect(revocationRequest(everyVersion)).toEqual({ themeId: 'acme/ink', level: 'disable', reason: 'Gone.' })
  })

  it("shows the server's 400 sentence", async () => {
    api.createThemeRevocation.mockRejectedValue(httpError(400, BAD_DIGEST))
    renderWithQuery(<RevocationsPanel hosted={false} />)
    fireEvent.change(await screen.findByLabelText('Package digest'), { target: { value: 'not-a-digest' } })
    fill('Bad theme.')
    fireEvent.click(screen.getByRole('button', { name: REVOKE }))
    expect(await screen.findByTestId('theme-revocation-error')).toHaveTextContent(BAD_DIGEST)
  })
})

describe('the revocation sync (local edition)', () => {
  it('shows the last success, the last error and the rows held', async () => {
    serve([], syncStatus({ lastSuccessAt: daysAgo(2), lastError: UNREACHABLE, rows: 3 }))
    renderWithQuery(<RevocationsPanel hosted={false} />)
    expect(await screen.findByTestId('revocation-sync-success')).toHaveTextContent('2d ago')
    expect(screen.getByTestId('revocation-sync-error')).toHaveTextContent(UNREACHABLE)
    expect(screen.getByTestId('revocation-sync-rows')).toHaveTextContent('3')
    expect(screen.queryByTestId('revocation-sync-stale')).toBeNull()
  })

  it('warns once no pull has worked for 7 days (the server says stale)', async () => {
    serve([], syncStatus({ lastSuccessAt: daysAgo(8), stale: true }))
    renderWithQuery(<RevocationsPanel hosted={false} />)
    expect(await screen.findByTestId('revocation-sync-stale')).toHaveTextContent(SYNC_STALE)
  })

  it('Check now calls the pull route', async () => {
    api.pullThemeRevocations.mockResolvedValue(syncStatus())
    renderWithQuery(<RevocationsPanel hosted={false} />)
    fireEvent.click(await screen.findByRole('button', { name: CHECK_NOW }))
    await waitFor(() => expect(api.pullThemeRevocations).toHaveBeenCalledTimes(1))
  })

  it("shows a failed pull's own error", async () => {
    api.pullThemeRevocations.mockRejectedValue(httpError(502, UNREACHABLE))
    renderWithQuery(<RevocationsPanel hosted={false} />)
    fireEvent.click(await screen.findByRole('button', { name: CHECK_NOW }))
    expect(await screen.findByTestId('revocation-sync-pull-error')).toHaveTextContent(UNREACHABLE)
  })

  it('says sync is off without a feed address, and Check now waits for one', async () => {
    serve([], syncStatus({ enabled: false }))
    renderWithQuery(<RevocationsPanel hosted={false} />)
    expect(await screen.findByTestId('revocation-sync-off')).toHaveTextContent(SYNC_OFF)
    expect(screen.getByRole('button', { name: CHECK_NOW })).toBeDisabled()
  })

  it('is hidden in the hosted edition, which never asks for it', async () => {
    renderWithQuery(<RevocationsPanel hosted />)
    expect(await rowOf(REVOCATION_ID)).toBeInTheDocument()
    expect(screen.queryByTestId('revocation-sync')).toBeNull()
    expect(screen.queryByRole('button', { name: CHECK_NOW })).toBeNull()
    expect(api.getThemeRevocationSync).not.toHaveBeenCalled()
  })
})
